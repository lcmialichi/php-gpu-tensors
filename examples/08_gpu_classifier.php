<?php
declare(strict_types=1);

use PhpTensors\CudaArray;
use PhpTensors\Fusion;
use PhpTensors\FusionGraph;
use PhpTensors\Optimizer;

// 1. Download MNIST once, then turn its grayscale bytes into normalized FP32 rows.
function readMnist(string $split): array
{
    if (!in_array($split, ['train', 'test'], true)) {
        throw new InvalidArgumentException('MNIST split must be train or test.');
    }
    $directory = __DIR__ . '/training/data/mnist';
    if (!is_dir($directory) && !mkdir($directory, 0775, true) && !is_dir($directory)) {
        throw new RuntimeException("Cannot create $directory.");
    }
    $prefix = $split === 'train' ? 'train' : 't10k';
    $raw = [];
    foreach (['images-idx3-ubyte.gz', 'labels-idx1-ubyte.gz'] as $suffix) {
        $path = "$directory/$prefix-$suffix";
        if (!is_file($path)) {
            $url = "https://storage.googleapis.com/cvdf-datasets/mnist/$prefix-$suffix";
            echo "Downloading $prefix-$suffix\n";
            if (!copy($url, $path . '.part') || !rename($path . '.part', $path)) {
                throw new RuntimeException("Cannot download $url; remove any .part file before retrying.");
            }
        }
        $compressed = file_get_contents($path);
        $decoded = $compressed === false ? false : gzdecode($compressed);
        if ($decoded === false) {
            throw new RuntimeException("Cannot decode $path; remove it and retry.");
        }
        $raw[] = $decoded;
    }
    if (strlen($raw[0]) < 16 || strlen($raw[1]) < 8) {
        throw new RuntimeException('Truncated MNIST header.');
    }
    $images = unpack('Nmagic/Ncount/Nheight/Nwidth', $raw[0]);
    $labels = unpack('Nmagic/Ncount', $raw[1]);
    if ($images['magic'] !== 2051 || $labels['magic'] !== 2049 || $images['height'] !== 28 || $images['width'] !== 28
        || $images['count'] !== $labels['count'] || $images['count'] < 1
        || strlen($raw[0]) !== 16 + $images['count'] * 784 || strlen($raw[1]) !== 8 + $labels['count']) {
        throw new RuntimeException('Invalid MNIST dimensions or payload length.');
    }
    $targets = array_values(unpack('C*', substr($raw[1], 8)));
    if (max($targets) > 9) {
        throw new RuntimeException('MNIST labels must be digits from 0 to 9.');
    }
    $normalization = [];
    for ($pixel = 0; $pixel < 256; $pixel++) {
        $normalization[chr($pixel)] = pack('g', ($pixel / 255.0 - 0.1307) / 0.3081);
    }
    return [str_split(strtr(substr($raw[0], 16), $normalization), 784 * 4), $targets];
}

// 2. Pack one batch. Small random shifts teach the model to tolerate off-center digits.
function uploadBatch(array $images, array $targets, array $indices, bool $augment = false): array
{
    $features = '';
    $labels = '';
    $background = pack('g', -0.1307 / 0.3081);
    foreach ($indices as $index) {
        $image = $images[$index];
        if ($augment) {
            $dx = mt_rand(-2, 2);
            $dy = mt_rand(-2, 2);
            if ($dy !== 0) {
                $padding = str_repeat($background, 28 * abs($dy));
                $image = $dy > 0 ? $padding . substr($image, 0, (28 - $dy) * 112)
                    : substr($image, -$dy * 112) . $padding;
            }
            if ($dx !== 0) {
                $shifted = '';
                $padding = str_repeat($background, abs($dx));
                for ($row = 0; $row < 28; $row++) {
                    $line = substr($image, $row * 112, 112);
                    $shifted .= $dx > 0 ? $padding . substr($line, 0, (28 - $dx) * 4)
                        : substr($line, -$dx * 4) . $padding;
                }
                $image = $shifted;
            }
        }
        $features .= $image;
        $oneHot = array_fill(0, 10, 0.0);
        $oneHot[$targets[$index]] = 1.0;
        $labels .= pack('g*', ...$oneHot);
    }
    return [
        CudaArray::fromBuffer($features, [count($indices), 784]),
        CudaArray::fromBuffer($labels, [count($indices), 10]),
    ];
}

// 3. A fixed MLP: 784 pixels -> 512 ReLU units -> 256 ReLU units -> 10 scores.
function initializeParameters(): array
{
    // Box-Muller reproduces the original seeded Gaussian weights, unlike a different GPU RNG.
    mt_srand(2027);
    $parameters = [];
    foreach ([[784, 512], [512, 256], [256, 10]] as $layer => [$inputs, $outputs]) {
        $scale = sqrt(($layer < 2 ? 2.0 : 1.0) / $inputs);
        $weights = [];
        for ($index = 0; $index < $inputs * $outputs; $index += 2) {
            $radius = sqrt(-2.0 * log((mt_rand() + 0.5) / (mt_getrandmax() + 1.0)));
            $angle = 2.0 * M_PI * ((mt_rand() + 0.5) / (mt_getrandmax() + 1.0));
            $weights[] = $radius * cos($angle) * $scale;
            $weights[] = $radius * sin($angle) * $scale;
        }
        $parameters[] = CudaArray::fromFlatArray($weights, [$inputs, $outputs]);
        $parameters[] = CudaArray::zeros([1, $outputs]);
    }
    return $parameters;
}

function predict(CudaArray $features, array $parameters, CudaArray $zero): CudaArray
{
    [$w1, $b1, $w2, $b2, $w3, $b3] = $parameters;
    $hidden1 = $features->matmul($w1) + $b1;
    $hidden1 = CudaArray::where($hidden1->gt(0), $hidden1, $zero);
    $hidden2 = $hidden1->matmul($w2) + $b2;
    $hidden2 = CudaArray::where($hidden2->gt(0), $hidden2, $zero);
    return $hidden2->matmul($w3) + $b3;
}

function crossEntropy(CudaArray $scores, CudaArray $labels): CudaArray
{
    // Subtracting the largest score keeps exp() stable without changing softmax.
    $rows = $scores->getShape()[0];
    $shifted = $scores - $scores->max(1)->reshape([$rows, 1]);
    $logPartition = $shifted->exp()->sum(1)->reshape([$rows, 1])->log();
    $targetScore = ($shifted * $labels)->sum(1)->reshape([$rows, 1]);
    return ($logPartition - $targetScore)->sum(0);
}

// 4. Autograd computes the derivatives; AdamW returns new weights and moment estimates.
function trainBatch(CudaArray $features, CudaArray $labels, array $parameters,
    Optimizer $optimizer, array $state, CudaArray $learningRate): array
{
    $zero = CudaArray::zeros([1]);
    $one = CudaArray::ones([1]);
    $upper = CudaArray::full([1], 5.0);
    $lower = CudaArray::full([1], -5.0);
    $optimizer->zeroGrad($parameters);
    foreach ($parameters as $parameter) {
        $parameter->requiresGrad();
    }
    $loss = crossEntropy(predict($features, $parameters, $zero), $labels);
    // Average the gradients over the batch, including the smaller final batch.
    $loss->backward($one / $features->getShape()[0]);
    $gradients = array_map(fn($parameter) => $parameter->grad(), $parameters);
    foreach ($gradients as $index => $gradient) {
        $gradient = CudaArray::where($gradient->gt(5.0), $upper, $gradient);
        $gradients[$index] = CudaArray::where($gradient->lt(-5.0), $lower, $gradient);
    }
    // The mask applies weight decay to weights, not biases.
    return ['loss' => $loss, ...$optimizer->step(
        $parameters, $gradients, $state, $learningRate, [true, false, true, false, true, false]
    )];
}

// Fusion inputs are tensors, so pass AdamW's moment lists and powers explicitly.
function trainingInputs(CudaArray $features, CudaArray $labels, CudaArray $rate, array $parameters, array $state): array
{
    return [$features, $labels, $rate, $state['beta1Power'], $state['beta2Power'],
        ...$parameters, ...$state['firstMoment'], ...$state['secondMoment']];
}

function compileTrainingPlan(array $inputs, Optimizer $optimizer): FusionGraph
{
    return Fusion::compile(
        function ($features, $labels, $rate, $beta1Power, $beta2Power, ...$tensors) use ($optimizer) {
            // Three weight matrices and three biases: six tensors in each group.
            [$parameters, $firstMoment, $secondMoment] = array_chunk($tensors, 6);
            $state = ['firstMoment' => $firstMoment, 'secondMoment' => $secondMoment,
                'beta1Power' => $beta1Power, 'beta2Power' => $beta2Power];
            return trainBatch($features, $labels, $parameters, $optimizer, $state, $rate);
        },
        $inputs,
        cudaGraph: true
    );
}

function evaluate(CudaArray $features, CudaArray $labels, array $parameters): array
{
    $parameters = array_map(fn($parameter) => $parameter->detach(), $parameters);
    $zero = CudaArray::zeros([1]);
    $metrics = Fusion::run(function () use ($features, $labels, $parameters, $zero) {
        $scores = predict($features, $parameters, $zero);
        $correct = $scores->argMax(1)->eq($labels->argMax(1))->astype('float32')->sum();
        return ['loss' => crossEntropy($scores, $labels), 'correct' => $correct];
    });
    $rows = $features->getShape()[0];
    return ['loss' => $metrics['loss']->item() / $rows, 'accuracy' => 100 * $metrics['correct']->item() / $rows];
}

function trainMnist(): array
{
    if (!extension_loaded('phptensors') || cuda_get_device_count() < 1) {
        throw new RuntimeException('MNIST training requires phptensors with CUDA and a visible NVIDIA GPU.');
    }
    ini_set('memory_limit', '4G');
    [$images, $targets] = readMnist('train');
    if (count($images) <= 5000) {
        throw new RuntimeException('MNIST needs more than 5000 images to reserve a validation set.');
    }
    $order = range(0, count($images) - 1);
    mt_srand(2026);
    shuffle($order);
    [$validationX, $validationY] = uploadBatch($images, $targets, array_slice($order, 0, 5000));
    $trainingIndices = array_slice($order, 5000);
    $permutations = [];
    for ($copy = 0; $copy < 4; $copy++) {
        $order = $trainingIndices;
        mt_srand(2126 + $copy);
        shuffle($order);
        foreach (array_chunk($order, 128) as $indices) {
            $permutations[$copy][] = uploadBatch($images, $targets, $indices, augment: true);
        }
    }
    unset($images, $targets);

    $parameters = initializeParameters();
    $parameterCount = array_sum(array_map(fn(CudaArray $parameter): int => $parameter->getSize(), $parameters));
    $optimizer = Optimizer::adamW();
    $state = $optimizer->initState($parameters);
    $plans = [];
    $bestLoss = INF;
    $bestEpoch = 0;
    $bestParameters = $parameters;
    $history = [];
    printf("MNIST | MLP 784-512-256-10 | ReLU + AdamW | batch 128 | %s trainable parameters\n",
        number_format($parameterCount));
    for ($epoch = 0; $epoch < 40; $epoch++) {
        $epochStarted = hrtime(true);
        // Two warmup epochs, then cosine decay from 0.001 to 5% of that rate.
        $rate = 0.001 * ($epoch < 2 ? ($epoch + 1) / 2 : 0.05 + 0.475 * (1 + cos(M_PI * ($epoch - 2) / 37)));
        $learningRate = CudaArray::full([1], $rate);
        $lossSum = 0.0;
        foreach ($permutations[$epoch % 4] as [$features, $labels]) {
            $rows = $features->getShape()[0];
            $inputs = trainingInputs($features, $labels, $learningRate, $parameters, $state);
            // Capture once per shape (128 rows or the final partial batch); replay thereafter.
            $plans[$rows] ??= compileTrainingPlan($inputs, $optimizer);
            $step = $plans[$rows]->run(...$inputs);
            $parameters = $step['parameters'];
            $state = $step['state'];
            $lossSum += $step['loss']->item();
        }
        $validation = evaluate($validationX, $validationY, $parameters);
        $trainLoss = $lossSum / count($trainingIndices);
        if (!is_finite($trainLoss) || !is_finite($validation['loss'])) {
            throw new RuntimeException('Training diverged: non-finite cross-entropy.');
        }
        $history[] = ['trainLoss' => $trainLoss, 'valLoss' => $validation['loss'], 'valAccuracy' => $validation['accuracy']];
        $epochSeconds = (hrtime(true) - $epochStarted) / 1e9;
        printf("Epoch %2d/40 | train loss %.4f | validation loss %.4f, accuracy %.2f%% | time %.2f s\n",
            $epoch + 1, $trainLoss, $validation['loss'], $validation['accuracy'], $epochSeconds);
        if ($validation['loss'] < $bestLoss - 1e-6) {
            $bestLoss = $validation['loss'];
            $bestParameters = $parameters;
            $bestEpoch = $epoch + 1;
        } elseif ($epoch + 1 - $bestEpoch >= 8) {
            break;
        }
    }

    // 5. Only validation selects weights. The untouched test set measures generalization.
    [$images, $targets] = readMnist('test');
    [$testX, $testY] = uploadBatch($images, $targets, range(0, count($images) - 1));
    $validation = evaluate($validationX, $validationY, $bestParameters);
    $test = evaluate($testX, $testY, $bestParameters);
    printf("Best epoch: %d | validation accuracy %.2f%% | test accuracy %.2f%% (%d samples)\n",
        $bestEpoch, $validation['accuracy'], $test['accuracy'], count($targets));
    return ['bestEpoch' => $bestEpoch, 'history' => $history, 'validation' => $validation, 'test' => $test,
        'plans' => array_map(fn(FusionGraph $plan): array => $plan->getStats(), $plans)];
}

if (realpath($_SERVER['SCRIPT_FILENAME'] ?? '') === __FILE__) {
    try {
        if ($argc > 1) {
            throw new InvalidArgumentException('This example takes no CLI options; read and edit the training steps above.');
        }
        trainMnist();
    } catch (Throwable $error) {
        fwrite(STDERR, 'Error: ' . $error->getMessage() . PHP_EOL);
        exit(2);
    }
}
