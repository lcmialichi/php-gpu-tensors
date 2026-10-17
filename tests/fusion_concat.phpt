--TEST--
Fusion captures concat with strided inputs, dtype preservation and gradients
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;

function check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}

$left = CudaArray::fromFlatArray([1.0, 2.0, 3.0, 4.0], [2, 2]);
$right = CudaArray::fromFlatArray([5.0, 6.0], [2, 1]);
$tail = CudaArray::fromFlatArray([7.0, 8.0], [2, 1]);
$graph = Fusion::compile(
    fn($a, $b, $c) => $a->concat([$b, $c], 1),
    [$left, $right, $tail]
);
check($graph->run($left, $right, $tail)->toArray() === [[1.0, 2.0, 5.0, 7.0], [3.0, 4.0, 6.0, 8.0]],
    'multi-input concat result');
check($graph->getStats()['asyncCompatible'] === false, 'concat async compatibility');
check($graph->getStats()['cudaGraphCompatible'] === false, 'concat CUDA Graph compatibility');
try {
    $graph->runAsync($left, $right, $tail);
    throw new RuntimeException('concat unexpectedly accepted asynchronous execution');
} catch (PhpTensors\RuntimeException $error) {
}

$transposed = CudaArray::fromFlatArray([1, 2, 3, 4, 5, 6], [2, 3])->transpose();
$column = CudaArray::fromFlatArray([7, 8, 9], [3, 1]);
$strided = Fusion::compile(fn($a, $b) => $a->concat([$b], 1), [$transposed, $column]);
check($strided->run($transposed, $column)->toArray() === [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]],
    'strided concat result');
check($transposed->concat([$column], 1)->toArray() === [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]],
    'eager strided concat result');

$empty = CudaArray::zeros([2, 0, 2]);
$nonEmpty = CudaArray::full([2, 1, 2], 9);
$emptyGraph = Fusion::compile(fn($a, $b) => $a->concat([$b], 1), [$empty, $nonEmpty]);
check($emptyGraph->run($empty, $nonEmpty)->toArray() === [[[9.0, 9.0]], [[9.0, 9.0]]],
    'empty input concat result');

$singleGraph = Fusion::compile(fn($a) => $a->concat([], 0), [$left]);
check($singleGraph->run($left)->toArray() === $left->toArray(), 'single-input concat result');

$doubleLeft = CudaArray::fromFlatArray([1.5, 2.5], [2], 'float64');
$doubleRight = CudaArray::fromFlatArray([3.5], [1], 'float64');
$doubleGraph = Fusion::compile(fn($a, $b) => $a->concat([$b]), [$doubleLeft, $doubleRight]);
check($doubleGraph->run($doubleLeft, $doubleRight)->toArray() === [1.5, 2.5, 3.5],
    'float64 concat result');

$intLeft = CudaArray::fromFlatArray([1, 2], [2], 'int64');
$intRight = CudaArray::fromFlatArray([3], [1], 'int64');
$intGraph = Fusion::compile(fn($a, $b) => $a->concat([$b]), [$intLeft, $intRight]);
check($intGraph->run($intLeft, $intRight)->toArray() === [1, 2, 3], 'int64 concat result');

$trackedLeft = CudaArray::fromFlatArray([1.0, 2.0], [1, 2])->requiresGrad();
$trackedRight = CudaArray::fromFlatArray([3.0], [1, 1])->requiresGrad();
$autograd = Fusion::compile(function ($a, $b) {
    $loss = $a->concat([$b], 1)->sum();
    $loss->backward();
    return [$loss, $a->grad(), $b->grad()];
}, [$trackedLeft, $trackedRight]);
[$loss, $leftGradient, $rightGradient] = $autograd->run($trackedLeft, $trackedRight);
check($loss->toArray() === [6.0], 'concat autograd loss');
check($leftGradient->toArray() === [[1.0, 1.0]], 'concat left gradient');
check($rightGradient->toArray() === [[1.0]], 'concat right gradient');
echo "Fusion concat, layout and gradients passed\n";
?>
--EXPECT--
Fusion concat, layout and gradients passed
