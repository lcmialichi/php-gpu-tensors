--TEST--
Standalone MNIST training matches reference AdamW updates across batch shapes
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;
use PhpTensors\Optimizer;

$root = is_dir(__DIR__ . '/../examples') ? dirname(__DIR__) : dirname(__DIR__, 2);
require $root . '/examples/08_gpu_classifier.php';
require $root . '/examples/Support/ModelConfiguration.php';
require $root . '/examples/Support/functions.php';
require $root . '/examples/Support/model_functions.php';
require $root . '/examples/Support/training_functions.php';

$input = CudaArray::fromFlatArray([2.0], [1]);
$callback = fusionPositionalCallback(static fn(array $inputs): CudaArray => $inputs[0] + $inputs[1], 2);
$graph = Fusion::compile($callback, [$input, $input]);
if ($graph->run($input, $input)->item() !== 4.0) {
    throw new RuntimeException('Variadic Fusion callback failed.');
}
try {
    $callback($input);
    throw new RuntimeException('Callback accepted the wrong input count.');
} catch (InvalidArgumentException) {
}
echo "advanced runner uses a regular variadic callback, not eval\n";

$initial = initializeParameters();
$initialBuffers = initializeModelBuffers(new ModelConfiguration([784, 512, 256, 10], 'relu', 'adam', 0.01, 5.0, 0.0), 2027);
foreach ($initial as $index => $parameter) {
    if ($parameter->toBuffer() !== $initialBuffers[$index]) {
        throw new RuntimeException('Standalone initialization differs from the original recipe.');
    }
}
unset($initial, $initialBuffers);
echo "initial weights match the original recipe byte for byte\n";

$model = new ModelConfiguration([4, 8, 6, 3], 'relu', 'adam', 0.01, 5.0, 0.0);
$buffers = initializeModelBuffers($model, 2027);
$constants = [
    'zero' => CudaArray::zeros([1]),
    'one' => CudaArray::full([1], 1.0),
    'upper' => CudaArray::full([1], 5.0),
    'lower' => CudaArray::full([1], -5.0),
];
$reference = createTrainState($model, $buffers);
$parameters = createTrainState($model, $buffers)['params'];
$optimizer = Optimizer::adamW();
$state = $optimizer->initState($parameters);
$firstLoss = null;
$plans = [];
for ($epoch = 0; $epoch < 4; $epoch++) {
    $rate = scalarTensor(scheduleLr($epoch, 4, 2, 0.001));
    foreach ([4, 1] as $rows) {
        $features = CudaArray::fromFlatArray(array_fill(0, $rows * 4, 0.5), [$rows, 4]);
        $labels = CudaArray::fromBuffer(str_repeat(oneHotTemplates(3)[1], $rows), [$rows, 3]);
        $body = trainingBody($model, $rows, true, $constants);
        $inputs = trainStateInputs($reference, $features, $features->transpose(), $labels, $rate);
        $expected = $body($inputs);
        if ($epoch === 0) {
            $graph = Fusion::compile(fusionPositionalCallback($body, modelTrainArity($model)), $inputs);
            $replayed = $graph->run(...$inputs);
            foreach ($expected as $index => $tensor) {
                if (!compareTensors($tensor, $replayed[$index], 1e-5, 2e-3)[1]) {
                    throw new RuntimeException('Variadic AdamW capture differs from eager execution.');
                }
            }
        }
        $inputs = trainingInputs($features, $labels, $rate, $parameters, $state);
        $plans[$rows] ??= compileTrainingPlan($inputs, $optimizer);
        $step = $plans[$rows]->run(...$inputs);
        $parameters = $step['parameters'];
        $state = $step['state'];
        $actual = [...$parameters, ...$state['firstMoment'], ...$state['secondMoment'],
            $state['beta1Power'], $state['beta2Power'], $step['loss']];
        foreach ($expected as $index => $tensor) {
            if (!compareTensors($tensor, $actual[$index], 1e-7, 1e-5)[1]) {
                throw new RuntimeException("Reference mismatch at epoch $epoch, batch $rows, output $index.");
            }
        }
        applyTrainOutputs($reference, $expected, $model);
        $lastLoss = $step['loss']->item() / $rows;
        if (!is_finite($lastLoss)) {
            throw new RuntimeException('Non-finite training loss.');
        }
        $firstLoss ??= $lastLoss;
    }
}
if ($lastLoss >= $firstLoss) {
    throw new RuntimeException('Training loss did not improve.');
}
foreach ($plans as $plan) {
    $stats = $plan->getStats();
    if ($stats['backend'] !== 'cuda-graph' || $stats['executions'] !== 4 || $stats['graphLaunches'] !== 4) {
        throw new RuntimeException('Training did not reuse one CUDA Graph per batch shape.');
    }
}
$evaluation = evaluate($features, $labels, $parameters);
if (!is_finite($evaluation['loss']) || $evaluation['accuracy'] < 0 || $evaluation['accuracy'] > 100) {
    throw new RuntimeException('Invalid held-out evaluation metrics.');
}
echo "eight steps match the reference, including the partial batch\n";
echo "two retained CUDA Graphs replay four steps each\n";
echo "finite loss improved; detached evaluation returns valid metrics\n";
?>
--EXPECT--
advanced runner uses a regular variadic callback, not eval
initial weights match the original recipe byte for byte
eight steps match the reference, including the partial batch
two retained CUDA Graphs replay four steps each
finite loss improved; detached evaluation returns valid metrics
