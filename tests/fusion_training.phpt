--TEST--
Real Fusion training step matches eager execution, CPU finite differences and stable cross-entropy
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
$script = dirname(__DIR__) . '/fused.php';
if (!is_file($script)) $script = dirname(__DIR__, 2) . '/fused.php';
require $script;
function check(bool $ok): void {
    if (!$ok) throw new RuntimeException('Training check failed.');
}
function gpu(array $matrix): CudaArray {
    return CudaArray::fromBuffer(pack('g*', ...array_merge(...$matrix)), [count($matrix), count($matrix[0])]);
}
function cpuLoss(array $parameters, array $x, array $y): float {
    [$w1, $b1, $w2, $b2] = $parameters;
    $loss = 0.0;
    foreach ($x as $row => $features) {
        $hidden = [];
        for ($h = 0; $h < count($b1[0]); $h++) {
            $value = $b1[0][$h];
            foreach ($features as $i => $feature) $value += $feature * $w1[$i][$h];
            $hidden[] = max(0.0, $value);
        }
        $logits = [];
        for ($c = 0; $c < count($b2[0]); $c++) {
            $value = $b2[0][$c];
            foreach ($hidden as $h => $activation) $value += $activation * $w2[$h][$c];
            $logits[] = $value;
        }
        $maximum = max($logits);
        $denominator = array_sum(array_map(fn($value) => exp($value - $maximum), $logits));
        foreach ($y[$row] as $c => $target) {
            $loss += $target * (log($denominator) - ($logits[$c] - $maximum));
        }
    }
    return $loss / count($x);
}
$xHost = [[0.2, 0.8], [0.6, 0.1]];
$yHost = [[1.0, 0.0], [0.0, 1.0]];
$host = [
    [[0.2, -0.3, 0.1], [0.4, 0.2, -0.2]], [[0.1, 0.1, 0.3]],
    [[0.1, -0.2], [-0.3, 0.4], [0.2, 0.3]], [[0.1, -0.1]],
];
$x = gpu($xHost);
$y = gpu($yHost);
$parameters = array_map('gpu', $host);
$inputs = [$x, $x->transpose(), $y, ...$parameters];
$network = new NeuralNetwork(2, 3, 2);
$expression = $network->trainingExpression(2, 0.05, true);
$graph = Fusion::compile($expression, $inputs);
check($graph->getStats()['fusedKernels'] > 0);
check($graph->getStats()['backend'] === 'stream');
$fused = $graph->run(...$inputs);
$eager = $expression(...$inputs);
foreach ($fused as $index => $output) {
    $actual = unpack('g*', $output->toHost()->toBuffer());
    $expected = unpack('g*', $eager[$index]->toHost()->toBuffer());
    foreach ($actual as $element => $value) check(abs($value - $expected[$element]) < 2e-5);
}
check(abs($fused[4][0] / 2 - cpuLoss($host, $xHost, $yHost)) < 2e-5);
echo "eager and CPU loss matched\n";
for ($parameter = 0; $parameter < 4; $parameter++) {
    $updated = $fused[$parameter]->toArray();
    check($parameters[$parameter]->toArray() === gpu($host[$parameter])->toArray());
    foreach ($host[$parameter] as $row => $values) {
        foreach ($values as $column => $value) {
            $plus = $minus = $host;
            $plus[$parameter][$row][$column] += 1e-4;
            $minus[$parameter][$row][$column] -= 1e-4;
            $gradient = (cpuLoss($plus, $xHost, $yHost) - cpuLoss($minus, $xHost, $yHost)) / 2e-4;
            $actual = ($value - $updated[$row][$column]) / 0.05;
            check(abs($actual - $gradient) < 2e-4);
        }
    }
}
echo "all parameter gradients matched finite differences\n";
$large = $inputs;
$large[6] = gpu([[1000.0, -1000.0]]);
$extreme = $graph->run(...$large);
check(is_finite($extreme[4][0]));
$reference = $host;
$reference[3] = [[1000.0, -1000.0]];
check(abs($extreme[4][0] / 2 - cpuLoss($reference, $xHost, $yHost)) < 0.001);
echo "large logits remained stable\n";
$bigX = [[100.0, 80.0], [60.0, 10.0]];
$bigTensor = gpu($bigX);
$clipped = $graph->run($bigTensor, $bigTensor->transpose(), $y, ...$parameters);
$sawClipping = false;
for ($parameter = 0; $parameter < 4; $parameter++) {
    $updated = $clipped[$parameter]->toArray();
    foreach ($host[$parameter] as $row => $values) {
        foreach ($values as $column => $value) {
            $plus = $minus = $host;
            $plus[$parameter][$row][$column] += 1e-4;
            $minus[$parameter][$row][$column] -= 1e-4;
            $gradient = (cpuLoss($plus, $bigX, $yHost) - cpuLoss($minus, $bigX, $yHost)) / 2e-4;
            $sawClipping = $sawClipping || abs($gradient) > 5;
            check(abs(($value - $updated[$row][$column]) / 0.05 - max(-5, min(5, $gradient))) < 0.001);
        }
    }
}
check($sawClipping);
$nan = $graph->run($x, $x->transpose(), gpu([[NAN, NAN], [NAN, NAN]]), ...$parameters);
foreach ($parameters as $index => $parameter) check($nan[$index]->toArray() === $parameter->toArray());
echo "clipping and NaN-gradient policy preserved\n";
$retained = $fused[0]->toArray();
$next = $graph->run($x, $x->transpose(), $y, ...array_slice($fused, 0, 4));
check($fused[0]->toArray() === $retained);
check($next[0]->toArray() !== $retained);
echo "replay replaced weights without mutating prior outputs\n";
$one = $graph->runAsync(...$inputs);
$two = $graph->runAsync(...$inputs);
$asyncFirst = $one->wait();
$asyncSecond = $two->wait();
foreach ($fused as $index => $output) {
    check($asyncFirst[$index]->toArray() === $output->toArray());
    check($asyncSecond[$index]->toArray() === $output->toArray());
}
echo "concurrent full training replay checked\n";
$path = tempnam(sys_get_temp_dir(), 'fusion-training-');
if ($path === false) throw new RuntimeException('Cannot create model test file.');
try {
    ob_start();
    $network->saveModel($path, 'test-fusion-v2');
    ob_end_clean();
    $restored = new NeuralNetwork(2, 3, 2);
    $restored->loadModel($path, 'test-fusion-v2');
    foreach ($network->parameters() as $index => $parameter) {
        check($parameter->toArray() === $restored->parameters()[$index]->toArray());
    }
    try {
        $restored->loadModel($path, 'wrong-version');
        throw new RuntimeException('Wrong model version was accepted.');
    } catch (RuntimeException $error) {
        check($error->getMessage() === 'Saved model version or parameters are incompatible.');
    }
} finally {
    unlink($path);
}
echo "model round-trip and version check passed\n";
?>
--EXPECT--
eager and CPU loss matched
all parameter gradients matched finite differences
large logits remained stable
clipping and NaN-gradient policy preserved
replay replaced weights without mutating prior outputs
concurrent full training replay checked
model round-trip and version check passed
