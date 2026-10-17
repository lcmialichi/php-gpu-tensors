--TEST--
CudaArray repeat supports scalar and per-element repetition counts
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion;

function check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}

$input = A::fromFlatArray([1, 2, 3, 4], [2, 2]);
check($input->repeat([2, 0, 1, 3])->toArray() ===
    [1.0, 1.0, 3.0, 4.0, 4.0, 4.0],
    'flattened per-element repeat mismatch');
check($input->repeat([1, 2], 0)->toArray() ===
    [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0]],
    'axis-zero per-element repeat mismatch');
check($input->repeat([2, 0], 1)->toArray() ===
    [[1.0, 1.0], [3.0, 3.0]],
    'axis repeat counts mismatch');
check($input->transpose()->repeat([1, 2], 1)->toArray() ===
    [[1.0, 3.0, 3.0], [2.0, 4.0, 4.0]],
    'strided input per-element repeat mismatch');
check($input->repeat([0, 0, 0, 0])->getShape() === [0],
    'all-zero flattened counts mismatch');
check($input->repeat(2)->toArray() ===
    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    'scalar repeat behavior changed');

$fused = Fusion::run(fn() => $input->repeat([2, 0, 1, 3]));
check($fused->toArray() === [1.0, 1.0, 3.0, 4.0, 4.0, 4.0],
    'synchronous Fusion repeat mismatch');
$graph = Fusion::compile(fn($x) => $x->repeat([1, 2], 0), [$input]);
check($graph->run($input)->toArray() ===
    [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0]],
    'compiled Fusion repeat mismatch');
check($graph->getStats()['incompatibility'] === 'repeat-counts',
    'Fusion should report repeat-count async limitation');
try {
    $graph->runAsync($input);
    throw new RuntimeException('asynchronous repeat counts were accepted');
} catch (PhpTensors\RuntimeException $error) {
}

$tracked = A::fromFlatArray([1.0, 2.0, 3.0], [3])->requiresGrad();
$tracked->repeat([2, 0, 3])->sum()->backward();
check($tracked->grad()->toArray() === [2.0, 0.0, 3.0],
    'flattened repeat-count gradient mismatch');
$trackedZeroCounts = A::ones([3])->requiresGrad();
$trackedZeroCounts->repeat([0, 0, 0])->sum()->backward();
check($trackedZeroCounts->grad()->toArray() === [0.0, 0.0, 0.0],
    'all-zero repeat-count gradient mismatch');
$trackedRows = A::fromFlatArray([1.0, 2.0, 3.0, 4.0], [2, 2])->requiresGrad();
$trackedRows->repeat([1, 2], 1)->sum()->backward();
check($trackedRows->grad()->toArray() === [[1.0, 2.0], [1.0, 2.0]],
    'axis repeat-count gradient mismatch');
$trackedFusion = A::fromFlatArray([1.0, 2.0, 3.0], [3])->requiresGrad();
[, $fusionGradient] = Fusion::run(function () use ($trackedFusion) {
    $result = $trackedFusion->repeat([2, 0, 3]);
    $result->sum()->backward();
    return [$result, $trackedFusion->grad()];
});
check($fusionGradient->toArray() === [2.0, 0.0, 3.0],
    'Fusion repeat-count gradient mismatch');
$trackedLowPrecision = A::fromFlatArray([1.0, 2.0, 3.0], [3], 'float16')
    ->requiresGrad();
$trackedLowPrecision->repeat([2, 0, 3])->backward(A::ones([5], 'float16'));
check($trackedLowPrecision->grad()->dtype() === 'float32' &&
    $trackedLowPrecision->grad()->toArray() === [2.0, 0.0, 3.0],
    'low-precision repeat-count gradient mismatch');

foreach ([
    fn() => $input->repeat([1, 2, 3]),
    fn() => $input->repeat([1, -1, 1, 1]),
    fn() => $input->repeat([1, 2.0, 1, 1]),
    fn() => $input->repeat([1, 2147483647, 1, 1]),
] as $invalid) {
    try {
        $invalid();
        throw new RuntimeException('invalid repeat counts were accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
}

echo "per-element repeat counts passed\n";
?>
--EXPECT--
per-element repeat counts passed
