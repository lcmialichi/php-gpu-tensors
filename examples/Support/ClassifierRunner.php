<?php
declare(strict_types=1);

use Cuda\CudaArray;
use Cuda\Fusion;
use Cuda\FusionGraph;

/**
 * -------------------------------------------------------------------------
 * Compiles each batch-size variant once and retains plans for repeated replay.
 * -------------------------------------------------------------------------
 *
 * @param array{params:list<CudaArray>,firstMoments:list<CudaArray>,secondMoments:list<CudaArray>,beta1Power:?CudaArray,beta2Power:?CudaArray} $initialState
 * @param array<int,list<array{CudaArray,CudaArray,CudaArray,int}>> $permutations
 * @param array{zero:CudaArray,one:CudaArray,upper:CudaArray,lower:CudaArray} $constants
 * @return array{
 *   training:array<int,array{0:FusionGraph,1:FusionGraph}>,
 *   evaluation:array<int,FusionGraph>
 * }
 */
function compileClassifierFusionPlans(
    ModelConfiguration $configuration,
    array $initialState,
    array $permutations,
    int $validationRows,
    CudaArray $validationFeatures,
    CudaArray $validationLabels,
    int $testRows,
    CudaArray $testFeatures,
    CudaArray $testLabels,
    array $constants,
    float $learningRate,
    bool $loadModel,
    bool $profile
): array {
    $trainingPlans = [];
    if (!$loadModel) {
        $firstByRows = [];
        foreach ($permutations[0] as [$features, $transposed, $labels, $rows]) {
            if (!isset($firstByRows[$rows])) {
                $firstByRows[$rows] = [$features, $transposed, $labels];
            }
        }
        $exampleRate = scalarTensor($learningRate);
        foreach ($firstByRows as $rows => [$features, $transposed, $labels]) {
            $inputs = trainStateInputs($initialState, $features, $transposed, $labels, $exampleRate);
            $trainingPlans[$rows] = [
                Fusion::compile(
                    fusionPositionalCallback(
                        trainingBody($configuration, $rows, false, $constants),
                        modelTrainArity($configuration)
                    ),
                    $inputs
                ),
                Fusion::compile(
                    fusionPositionalCallback(
                        trainingBody($configuration, $rows, true, $constants),
                        modelTrainArity($configuration)
                    ),
                    $inputs
                ),
            ];
            if ($profile) {
                $trainingPlans[$rows][0]->setProfiling(true);
                $trainingPlans[$rows][1]->setProfiling(true);
            }
        }
    }

    $evaluationPlans = [];
    foreach ([
        [$validationRows, $validationFeatures, $validationLabels],
        [$testRows, $testFeatures, $testLabels],
    ] as [$rows, $features, $labels]) {
        if (isset($evaluationPlans[$rows])) {
            continue;
        }
        $evaluationPlans[$rows] = Fusion::compile(
            fusionPositionalCallback(
                evaluationBody($configuration, $rows, $constants),
                modelEvaluationArity($configuration)
            ),
            [$features, $labels, ...$initialState['params']]
        );
        if ($profile) {
            $evaluationPlans[$rows]->setProfiling(true);
        }
    }

    return ['training' => $trainingPlans, 'evaluation' => $evaluationPlans];
}

/**
 * -------------------------------------------------------------------------
 * Compares compiled Fusion plans with their eager and CPU equivalents.
 * -------------------------------------------------------------------------
 *
 * @param array{params:list<CudaArray>,firstMoments:list<CudaArray>,secondMoments:list<CudaArray>,beta1Power:?CudaArray,beta2Power:?CudaArray} $initialState
 * @param array<int,array{0:FusionGraph,1:FusionGraph}> $trainingPlans
 * @param array<int,FusionGraph> $evaluationPlans
 * @param list<array{CudaArray,CudaArray,CudaArray,int}> $firstPermutation
 * @param list<array{status:string,name:string,detail:string}> $checks
 * @param list<array{float,float}> $cpuSample
 * @param array{zero:CudaArray,one:CudaArray,upper:CudaArray,lower:CudaArray} $constants
 */
function verifyClassifierFusion(
    ModelConfiguration $configuration,
    array $initialState,
    array $trainingPlans,
    array $evaluationPlans,
    CudaArray $validationFeatures,
    CudaArray $validationLabels,
    array $validationTargets,
    array $cpuSample,
    array $firstPermutation,
    array $constants,
    float $learningRate,
    bool $loadModel,
    array &$checks
): void {
    echo "\n" . C_BLD . "[3/4] Self-Tests" . C_RST . "\n";
    foreach ($trainingPlans as $rows => $pair) {
        foreach ([false, true] as $withLoss) {
            $stats = $pair[(int)$withLoss]->getStats();
            $name = sprintf('plan batch=%d%s', $rows, $withLoss ? ' (with loss)' : '');
            if ($stats['fusedKernels'] < 1) {
                addCheck($checks, 'FAIL', $name, 'no fused kernels were produced');
                continue;
            }
            $expected = modelExpectedBoundaries($configuration, $withLoss);
            addCheck($checks, $stats['boundaries'] === $expected ? 'PASS' : 'WARN', $name,
                sprintf('%d fused kernels, %d boundaries (expected %d)', $stats['fusedKernels'], $stats['boundaries'], $expected));
        }
    }

    $initial = evaluateSet(
        $evaluationPlans[count($validationTargets)],
        $validationFeatures,
        $validationLabels,
        $initialState['params'],
        $validationTargets
    );
    $cpu = cpuForward($configuration, $initialState['params'], $cpuSample);
    $gpu = $initial['logits']->toArray();
    $worst = 0.0;
    $scale = 1.0;
    foreach ($cpu as $row => $values) {
        foreach ($values as $column => $value) {
            $worst = max($worst, abs($gpu[$row][$column] - $value));
            $scale = max($scale, abs($value));
        }
    }
    addCheck($checks, $worst <= 2e-3 * $scale ? 'PASS' : 'FAIL', 'gpu-vs-cpu forward',
        sprintf('max |diff| %.3e over %d samples (logit scale %.2f)', $worst, count($cpu), $scale));

    if ($loadModel) {
        return;
    }

    [$features, $transposed, $labels, $rows] = $firstPermutation[0];
    $inputs = trainStateInputs($initialState, $features, $transposed, $labels, scalarTensor($learningRate * 0.1));
    $eagerBody = trainingBody($configuration, $rows, true, $constants);
    /** @var list<CudaArray> $eagerOutputs */
    $eagerOutputs = array_values($eagerBody($inputs));
    /** @var list<CudaArray> $fusedOutputs */
    $fusedOutputs = array_values($trainingPlans[$rows][1]->run(...$inputs));
    if (count($eagerOutputs) !== count($fusedOutputs)) {
        addCheck($checks, 'FAIL', 'eager-vs-fusion step',
            sprintf('output count differs (%d vs %d)', count($eagerOutputs), count($fusedOutputs)));
        return;
    }

    $worst = 0.0;
    $allMatch = true;
    foreach ($eagerOutputs as $index => $tensor) {
        [$difference, $matches] = compareTensors($tensor, $fusedOutputs[$index], 1e-5, 2e-3);
        $worst = max($worst, $difference);
        $allMatch = $allMatch && $matches;
    }
    addCheck($checks, $allMatch ? 'PASS' : 'FAIL', 'eager-vs-fusion step',
        sprintf('%d output tensors, max |diff| %.3e', count($eagerOutputs), $worst));
}

/**
 * -------------------------------------------------------------------------
 * Prints kernel and native-boundary counts for the retained Fusion plans.
 * -------------------------------------------------------------------------
 *
 * @param array<int,array{0:FusionGraph,1:FusionGraph}> $trainingPlans
 * @param array<int,FusionGraph> $evaluationPlans
 */
function printClassifierFusionProfile(array $trainingPlans, array $evaluationPlans): void
{
    echo "\n  " . C_BLD . "Fusion Plan Profile:" . C_RST . "\n";
    foreach ($trainingPlans as $rows => $pair) {
        foreach ($pair as $withLoss => $plan) {
            $stats = $plan->getStats();
            printf("    batch=%d%s: %d fused kernels, %d native boundaries, %.3f ms profiled execution\n",
                $rows, $withLoss ? ' + loss' : '', $stats['fusedKernels'], $stats['boundaries'],
                $stats['executeTimeNs'] / 1e6);
            foreach ($plan->getPlan() as $step) {
                if ($step['kind'] === 'native') {
                    printf("      step %d: %s\n", $step['step'], $step['reason']);
                }
            }
        }
    }
    foreach ($evaluationPlans as $rows => $plan) {
        $stats = $plan->getStats();
        printf("    eval=%d: %d fused kernels, %d native boundaries, %.3f ms profiled execution\n",
            $rows, $stats['fusedKernels'], $stats['boundaries'], $stats['executeTimeNs'] / 1e6);
    }
}

/**
 * -------------------------------------------------------------------------
 * Trains the classifier by replaying batch-shape-specific Fusion plans.
 * -------------------------------------------------------------------------
 *
 * @param array<string,mixed> $options
 * @param array{params:list<CudaArray>,firstMoments:list<CudaArray>,secondMoments:list<CudaArray>,beta1Power:?CudaArray,beta2Power:?CudaArray} $state
 * @param array<int,array{0:FusionGraph,1:FusionGraph}> $trainingPlans
 * @param array<int,FusionGraph> $evaluationPlans
 * @param array<int,list<array{CudaArray,CudaArray,CudaArray,int}>> $permutations
 * @param array{loss:float,accuracy:float} $initialValidation
 * @param list<int> $validationTargets
 * @return array{
 *   state:array{params:list<CudaArray>,firstMoments:list<CudaArray>,secondMoments:list<CudaArray>,beta1Power:?CudaArray,beta2Power:?CudaArray},
 *   history:list<array{epoch:int,trainLoss:float,valLoss:float,valAccuracy:float,lr:float}>,
 *   epochsRun:int,steps:int,trainSeconds:float,bestEpoch:int
 * }
 */
function trainClassifier(
    ModelConfiguration $configuration,
    array $options,
    array $state,
    array $initialValidation,
    array $trainingPlans,
    array $evaluationPlans,
    array $permutations,
    int $trainingRows,
    CudaArray $validationFeatures,
    CudaArray $validationLabels,
    array $validationTargets
): array {
    $history = [];
    $epochsRun = 0;
    $steps = 0;
    $trainSeconds = 0.0;
    $plainSeconds = 0.0;
    $plainSteps = 0;
    $bestEpoch = 0;

    if ($options['load']) {
        return compact('state', 'history', 'epochsRun', 'steps', 'trainSeconds', 'bestEpoch');
    }

    echo "\n" . C_BLD . "[4/4] Training Loop" . C_RST . "\n";
    printf("  Initial validation: loss %.4f, acc %.2f%%\n\n",
        $initialValidation['loss'], $initialValidation['accuracy']);
    $bestParams = null;
    $bestLoss = INF;
    $badReports = 0;
    for ($epoch = 0; $epoch < $options['epochs']; $epoch++) {
        $report = $epoch % $options['reportEvery'] === 0 || $epoch === $options['epochs'] - 1;
        $learningRate = scheduleLr($epoch, $options['epochs'], $options['warmup'], $options['lr']);
        $learningRateTensor = scalarTensor($learningRate);
        $batches = $permutations[$epoch % $options['permutations']];
        $lossSum = 0.0;
        $start = hrtime(true);

        foreach ($batches as [$features, $transposed, $labels, $rows]) {
            /** @var list<CudaArray> $stepOutputs */
            $stepOutputs = $trainingPlans[$rows][(int)$report]->run(
                ...trainStateInputs($state, $features, $transposed, $labels, $learningRateTensor)
            );
            $tail = applyTrainOutputs($state, $stepOutputs, $configuration);
            if ($report) {
                $lossSum += $tail[0][0];
            }
            $steps++;
        }

        $elapsed = (hrtime(true) - $start) / 1e9;
        $trainSeconds += $elapsed;
        $epochsRun = $epoch + 1;
        if (!$report) {
            $plainSeconds += $elapsed;
            $plainSteps += count($batches);
            continue;
        }

        $trainLoss = $lossSum / $trainingRows;
        if (!is_finite($trainLoss)) {
            throw new RuntimeException('Training diverged: non-finite cross-entropy.');
        }
        $validation = evaluateSet(
            $evaluationPlans[count($validationTargets)],
            $validationFeatures,
            $validationLabels,
            $state['params'],
            $validationTargets
        );
        $history[] = [
            'epoch' => $epoch + 1,
            'trainLoss' => $trainLoss,
            'valLoss' => $validation['loss'],
            'valAccuracy' => $validation['accuracy'],
            'lr' => $learningRate,
        ];
        $accuracyColor = $validation['accuracy'] >= 85.0 ? C_GRN : C_YLW;
        printf("  " . C_BLD . "Epoch %3d" . C_RST . ": train CE %.4f | val CE %.4f, acc %s%6.2f%%%s | lr %.5f | %.2f ms/step\n",
            $epoch + 1, $trainLoss, $validation['loss'], $accuracyColor, $validation['accuracy'],
            C_RST, $learningRate, $elapsed * 1000 / count($batches));

        if ($validation['loss'] < $bestLoss - 1e-6) {
            $bestLoss = $validation['loss'];
            $bestEpoch = $epoch + 1;
            $bestParams = $state['params'];
            $badReports = 0;
        } elseif (++$badReports >= $options['patience']) {
            printf(C_YLW . "  Early stopping: no validation improvement for %d reports (best epoch %d)." . C_RST . "\n",
                $options['patience'], $bestEpoch);
            break;
        }
    }

    if ($bestParams !== null) {
        $state['params'] = $bestParams;
    }
    $millisecondsPerStep = $trainSeconds * 1000 / max(1, $steps);
    $steadyMilliseconds = $plainSteps > 0 ? $plainSeconds * 1000 / $plainSteps : $millisecondsPerStep;
    $samplesPerSecond = $trainingRows * $epochsRun / max(1e-9, $trainSeconds);
    $gflops = modelStepFlops($configuration, $options['batchSize']) / ($steadyMilliseconds / 1000.0) / 1e9;
    printf("\n  " . C_BLD . "Training Stats:" . C_RST . " %.2f s, %d epochs, %d steps, %.3f ms/step\n",
        $trainSeconds, $epochsRun, $steps, $millisecondsPerStep);
    printf("  " . C_BLD . "Performance:" . C_RST . " %.0f samples/s, ~" . C_CYN . "%.1f GFLOP/s" . C_RST . " matmul work\n",
        $samplesPerSecond, $gflops);
    if ($options['profile']) {
        printClassifierFusionProfile($trainingPlans, $evaluationPlans);
    }

    return compact('state', 'history', 'epochsRun', 'steps', 'trainSeconds', 'bestEpoch');
}

/**
 * -------------------------------------------------------------------------
 * Reuses one compiled probability plan to classify rows of a fixed shape.
 * -------------------------------------------------------------------------
 *
 * @param list<CudaArray> $parameters
 * @param array{zero:CudaArray,one:CudaArray,upper:CudaArray,lower:CudaArray} $constants
 * @return list<float>
 */
function classifierProbabilities(
    string $row,
    ModelConfiguration $configuration,
    array $parameters,
    array $constants,
    ?FusionGraph &$probabilityPlan
): array {
    $features = CudaArray::fromBuffer($row, [1, $configuration->dimensions[0]]);
    if ($probabilityPlan === null) {
        $probabilityPlan = Fusion::compile(
            fusionPositionalCallback(
                probabilityBody($configuration, $constants),
                1 + modelParameterCount($configuration)
            ),
            [$features, ...$parameters]
        );
    }
    /** @var array{0:CudaArray} $outputs */
    $outputs = $probabilityPlan->run($features, ...$parameters);
    /** @var list<float> $probabilities */
    $probabilities = $outputs[0]->toArray()[0];
    return $probabilities;
}

/**
 * -------------------------------------------------------------------------
 * Displays the highest-probability classes and optional ground truth.
 * -------------------------------------------------------------------------
 *
 * @param list<float> $probabilities
 * @param list<string> $classNames
 */
function printClassifierPrediction(array $probabilities, ?int $truth, array $classNames): void
{
    $order = array_keys($probabilities);
    usort($order, static fn(int $a, int $b): int => $probabilities[$b] <=> $probabilities[$a]);
    $predicted = $order[0];
    printf("  Prediction: %s%s (%.1f%%)%s%s\n",
        C_BLD, $classNames[$predicted], 100 * $probabilities[$predicted], C_RST,
        $truth === null ? '' : ($truth === $predicted
            ? C_GRN . '  [correct]' . C_RST
            : C_RED . '  [truth: ' . $classNames[$truth] . ']' . C_RST));
    foreach (array_slice($order, 0, 3) as $index) {
        printf("    %-20s %6.2f%%\n", $classNames[$index], 100 * $probabilities[$index]);
    }
}

/**
 * -------------------------------------------------------------------------
 * Runs optional sample/file inference through a cached probability plan.
 * -------------------------------------------------------------------------
 *
 * @param array{predictIndex:?int,predictFile:?string,invert:bool,raw:bool} $options
 * @param array{params:list<CudaArray>} $state
 * @param array{classNames:list<string>,test:array{rows:list<string>},image:?array{h:int,w:int,mean:float,std:float,table:array}} $data
 * @param list<int> $testTargets
 * @param array{zero:CudaArray,one:CudaArray,upper:CudaArray,lower:CudaArray} $constants
 */
function runClassifierInference(
    array $options,
    ModelConfiguration $configuration,
    array $data,
    array $state,
    int $testRows,
    array $testTargets,
    array $constants
): void {
    if ($options['predictIndex'] === null && $options['predictFile'] === null) {
        return;
    }

    echo "\n" . C_BLD . "--- Inference Mode ---" . C_RST . "\n";
    /** @var FusionGraph|null $probabilityPlan */
    $probabilityPlan = null;
    if ($options['predictIndex'] !== null) {
        $index = $options['predictIndex'];
        if ($index >= $testRows) {
            throw new InvalidArgumentException("--predict-index must be below $testRows.");
        }
        $row = $data['test']['rows'][$index];
        $image = $data['image'];
        echo "  Test sample #$index:\n";
        if ($image !== null) {
            echo asciiImage($row, $image['h'], $image['w'], $image['mean'], $image['std']);
        }
        printClassifierPrediction(
            classifierProbabilities($row, $configuration, $state['params'], $constants, $probabilityPlan),
            $testTargets[$index],
            $data['classNames']
        );
    }

    if ($options['predictFile'] === null) {
        return;
    }
    $image = $data['image'];
    if ($image === null) {
        throw new InvalidArgumentException('--predict-file only works with the image datasets.');
    }
    [$width, $height, $pixels] = readPgm($options['predictFile']);
    $canvas = prepareDrawing($pixels, $width, $height, $image['w'], $image['h'], $options['invert'], $options['raw']);
    $row = strtr(implode('', array_map('chr', $canvas)), $image['table']);
    echo "  Model input from {$options['predictFile']}:\n",
        asciiImage($row, $image['h'], $image['w'], $image['mean'], $image['std']);
    printClassifierPrediction(
        classifierProbabilities($row, $configuration, $state['params'], $constants, $probabilityPlan),
        null,
        $data['classNames']
    );
}

/**
 * -------------------------------------------------------------------------
 * Runs dataset loading, plan compilation, training, evaluation, and reporting.
 * -------------------------------------------------------------------------
 */
function runClassifier(): int
{
    $cfg = parseConfig();
    if (!extension_loaded('cuda') || cuda_get_device_count() < 1) {
        throw new RuntimeException('This tool requires the CUDA extension and a visible NVIDIA GPU.');
    }
    @ini_set('memory_limit', '4G');
    $checks = [];
    $wallStart = hrtime(true);

    // Header
    printf("\n" . C_BLD . C_MAG . "=== PHP-CUDA-EXT MLP Classifier ===" . C_RST . "\n");
    printf(C_CYN . "PHP %s (%s) | cuda extension %s | %d GPU(s)" . C_RST . "\n\n",
        PHP_VERSION, PHP_ZTS ? 'ZTS' : 'NTS', phpversion('cuda') ?: 'unknown', cuda_get_device_count());

    // 1. Data Loading --------------------------------------------------------
    $start = hrtime(true);
    echo C_BLD . "[1/4] Data Loading" . C_RST . "\n";
    echo "  Loading dataset " . C_CYN . $cfg['dataset'] . C_RST . "...\n";
    $data = loadData($cfg);
    $nTrain = count($data['train']['rows']);
    $nVal = count($data['val']['rows']);
    $nTest = count($data['test']['rows']);
    printf("  %s: %d train / %d val / %d test | %d features, %d classes (%.2f s)\n", $data['name'], $nTrain, $nVal,
        $nTest, $data['features'], $data['classes'], (hrtime(true) - $start) / 1e9);

    // 2. Model Initialization -----------------------------------------------
    echo C_BLD . "\n[2/4] Model & JIT Compilation" . C_RST . "\n";
    if ($cfg['load']) {
        [$header, $buffers] = loadModelFile($cfg['model']);
        $dims = $header['dims'];
        $activation = (string)($header['activation'] ?? 'relu');
        if ($dims[0] !== $data['features'] || $dims[count($dims) - 1] !== $data['classes']) {
            throw new RuntimeException(sprintf('The model expects %d features and %d classes, the dataset has %d and %d.',
                $dims[0], $dims[count($dims) - 1], $data['features'], $data['classes']));
        }
        printf("  Loaded model %s (%s)\n", $cfg['model'], $header['dataset'] ?? 'unknown dataset');
    } else {
        $dims = array_merge([$data['features']], $cfg['hidden'], [$data['classes']]);
        $activation = $cfg['activation'];
    }
    $configuration = new ModelConfiguration(
        $dims,
        $activation,
        $cfg['optimizer'],
        $cfg['weightDecay'],
        $cfg['clip'],
        $cfg['labelSmoothing']
    );
    printf("  Architecture: %s | %d weight layers | %s parameters | %s%s\n", implode('-', $dims),
        modelLayerCount($configuration), number_format(modelParameterElements($configuration)), $activation,
        $cfg['load'] ? '' : ', ' . (modelUsesAdam($configuration) ? 'AdamW' : 'SGD+momentum'));

    if (!$cfg['load']) {
        $buffers = initializeModelBuffers($configuration, $cfg['seed'] + 1);
    }
    $consts = [
        'zero' => CudaArray::zeros([1]),
        'one' => CudaArray::full([1], 1.0),
        'upper' => CudaArray::full([1], $cfg['clip']),
        'lower' => CudaArray::full([1], -$cfg['clip']),
    ];

    // 3. Tensor Upload ------------------------------------------------------
    $start = hrtime(true);
    [$valX, $valY, $valTargets] = evalTensors($data['val'], $data['features'], $data['classes']);
    [$testX, $testY, $testTargets] = evalTensors($data['test'], $data['features'], $data['classes']);
    $cpuSample = array_slice($data['val']['rows'], 0, min(32, $nVal));
    $predictRows = $data['test']['rows'];
    $perms = [];
    if (!$cfg['load']) {
        $trainMb = $nTrain * $data['features'] * 4 / 1048576;
        printf("  Uploading %d epoch permutation(s) (~%.0f MB each)%s...\n",
            $cfg['permutations'], $trainMb, ($cfg['shift'] > 0 && $data['image'] !== null) ? ", shift aug up to {$cfg['shift']}px" : '');
        for ($p = 0; $p < $cfg['permutations']; $p++) {
            $perms[$p] = buildBatches($data, $cfg['batchSize'], $cfg['seed'] + 100 + $p, $cfg['shift']);
        }
    }
    $data['train'] = ['rows' => [], 'labels' => []];
    $data['val']['rows'] = [];
    $data['test']['rows'] = $cfg['predictIndex'] !== null ? $predictRows : [];
    unset($predictRows);

    // 4. JIT Compilation (Fusion Plans) -------------------------------------
    $state0 = createTrainState($configuration, $buffers);
    $compiledPlans = compileClassifierFusionPlans(
        $configuration,
        $state0,
        $perms,
        $nVal,
        $valX,
        $valY,
        $nTest,
        $testX,
        $testY,
        $consts,
        $cfg['lr'],
        $cfg['load'],
        $cfg['profile']
    );
    $plans = $compiledPlans['training'];
    $evalPlans = $compiledPlans['evaluation'];
    printf("  Compiled %d Fusion training shape(s) and %d evaluation plan(s); upload/JIT in %.2f s\n",
        count($plans), count($evalPlans), (hrtime(true) - $start) / 1e9);

    // 5. Self-tests (Verification) ------------------------------------------
    if ($cfg['selftest']) {
        verifyClassifierFusion(
            $configuration,
            $state0,
            $plans,
            $evalPlans,
            $valX,
            $valY,
            $valTargets,
            $cpuSample,
            $perms[0] ?? [],
            $consts,
            $cfg['lr'],
            $cfg['load'],
            $checks
        );
    }

    // 6. Training Loop ------------------------------------------------------
    $state = createTrainState($configuration, $buffers);
    $initialVal = evaluateSet($evalPlans[$nVal], $valX, $valY, $state['params'], $valTargets);
    $training = trainClassifier(
        $configuration,
        $cfg,
        $state,
        $initialVal,
        $plans,
        $evalPlans,
        $perms,
        $nTrain,
        $valX,
        $valY,
        $valTargets
    );
    $state = $training['state'];
    $history = $training['history'];
    $epochsRun = $training['epochsRun'];
    $steps = $training['steps'];
    $trainSeconds = $training['trainSeconds'];
    $bestEpoch = $training['bestEpoch'];

    // 7. Final Evaluation ---------------------------------------------------
    $finalVal = evaluateSet($evalPlans[$nVal], $valX, $valY, $state['params'], $valTargets);
    $test = evaluateSet($evalPlans[$nTest], $testX, $testY, $state['params'], $testTargets);
    $report = classificationReport($test['predictions'], $testTargets, $data['classes']);

    echo "\n" . C_BLD . C_MAG . "=== Final Evaluation ===" . C_RST . "\n";
    printf("  Validation: loss %.4f, accuracy %s%.2f%%%s.\n", $finalVal['loss'], C_GRN, $finalVal['accuracy'], C_RST);
    printf("  Test Set:   loss %.4f, accuracy %s%.2f%%%s (%d samples), macro-F1 %s%.4f%s\n\n",
        $test['loss'], C_GRN, $test['accuracy'], C_RST, $nTest, C_GRN, $report['macroF1'], C_RST);

    echo C_BLD . C_CYN . "  class                  support  precision  recall     F1" . C_RST . "\n";
    foreach ($report['perClass'] as $k => $row) {
        printf("  %-20s %8d  %9.3f  %6.3f  %5.3f\n", substr($data['classNames'][$k], 0, 20), $row['support'],
            $row['precision'], $row['recall'], $row['f1']);
    }

    if ($report['topConfusions'] !== []) {
        echo "\n" . C_BLD . "  Top Confusions (truth -> predicted):" . C_RST . "\n";
        foreach ($report['topConfusions'] as $pair) {
            printf("    %-14s -> %-14s %s%d%s\n", substr($data['classNames'][$pair['truth']], 0, 14),
                substr($data['classNames'][$pair['predicted']], 0, 14), C_YLW, $pair['count'], C_RST);
        }
    }

    if ($cfg['selftest'] && !$cfg['load']) {
        addCheck($checks, $finalVal['loss'] < $initialVal['loss'] ? 'PASS' : 'FAIL', 'validation loss improved',
            sprintf('%.4f -> %.4f', $initialVal['loss'], $finalVal['loss']));
    }
    if ($cfg['minAccuracy'] > 0.0) {
        addCheck($checks, $test['accuracy'] >= $cfg['minAccuracy'] ? 'PASS' : 'FAIL', 'test accuracy threshold',
            sprintf('%.2f%% vs. required %.2f%%', $test['accuracy'], $cfg['minAccuracy']));
    }

    // 8. Save Model ---------------------------------------------------------
    if (!$cfg['load'] && $cfg['save']) {
        saveModel($cfg['model'], $configuration, $state['params'], $data);
        printf("\n  " . C_GRN . "✔ Model saved:" . C_RST . " %s (%.1f KB)\n", $cfg['model'], filesize($cfg['model']) / 1024);
    }

    // 9. Inference Showcase -------------------------------------------------
    runClassifierInference($cfg, $configuration, $data, $state, $nTest, $testTargets, $consts);

    if ($cfg['json'] !== null) {
        $jsonPath = (string)$cfg['json'];
        if (!str_starts_with($jsonPath, '/') && !preg_match('/^[a-zA-Z]:\\\\/', $jsonPath)) { // Se não for caminho absoluto
            $jsonPath = dirname(__DIR__) . DIRECTORY_SEPARATOR . 'training' . DIRECTORY_SEPARATOR . $jsonPath;
        }

        $payload = [
            'schema' => 1,
            'generatedAt' => date(DATE_ATOM),
            'environment' => ['php' => PHP_VERSION, 'zts' => (bool)PHP_ZTS, 'extension' => phpversion('cuda') ?: null],
            'configuration' => $cfg,
            'dataset' => ['name' => $data['name'], 'train' => $nTrain, 'val' => $nVal, 'test' => $nTest,
                'features' => $data['features'], 'classes' => $data['classes']],
            'model' => ['dims' => $dims, 'parameters' => modelParameterElements($configuration)],
            'training' => $cfg['load'] ? null : ['epochsRun' => $epochsRun, 'steps' => $steps, 'seconds' => $trainSeconds,
                'bestEpoch' => $bestEpoch, 'history' => $history],
            'validation' => ['loss' => $finalVal['loss'], 'accuracy' => $finalVal['accuracy']],
            'test' => ['loss' => $test['loss'], 'accuracy' => $test['accuracy'], 'macroF1' => $report['macroF1'],
                'perClass' => $report['perClass'], 'topConfusions' => $report['topConfusions']],
            'checks' => $checks,
            'wallSeconds' => (hrtime(true) - $wallStart) / 1e9,
        ];
        file_put_contents($jsonPath, json_encode($payload, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR));
    }

    $failures = checkFailureCount($checks);
    $warnings = checkWarningCount($checks);
    printf("\nExecution finished in %.1f s: %d failure(s), %d warning(s).\n",
        (hrtime(true) - $wallStart) / 1e9, $failures, $warnings);
    return $failures > 0 ? 1 : 0;
}
