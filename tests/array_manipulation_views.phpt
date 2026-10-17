--TEST--
CudaArray copy, swapaxes and moveaxis preserve data, views and gradients
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

$matrix = A::fromFlatArray([0, 1, 2, 3, 4, 5], [2, 3], 'float32');
$swapped = $matrix->swapaxes(0, 1);
check($swapped->getShape() === [3, 2] &&
    $swapped->toArray() === [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
    'swapaxes result mismatch');
check($matrix->swapaxes(-2, -1)->toArray() === $swapped->toArray(),
    'negative swapaxes mismatch');

$tensor = A::fromFlatArray(range(0, 23), [2, 3, 4]);
$moved = $tensor->moveaxis(0, 2);
check($moved->getShape() === [3, 4, 2] &&
    $moved->toArray()[0][0] === [0.0, 12.0],
    'moveaxis scalar axes mismatch');
$multiMoved = A::fromFlatArray(range(0, 119), [2, 3, 4, 5])
    ->moveaxis([0, 2], [2, 0]);
check($multiMoved->getShape() === [4, 3, 2, 5],
    'moveaxis multiple axes mismatch');
check($tensor->moveaxis(-1, 0)->getShape() === [4, 2, 3],
    'moveaxis negative axes mismatch');

$transposed = $matrix->transpose();
$copy = $transposed->copy();
check($copy->getShape() === [3, 2] && $copy->getStrides() === [2, 1] &&
    $copy->toArray() === $transposed->toArray(),
    'copy of strided view mismatch');
$flatCopy = $copy->flatten();
$flatCopy[0] = 90.0;
check($copy->toArray()[0] === [90.0, 3.0] &&
    $transposed->toArray()[0] === [0.0, 3.0],
    'copy shares storage with its source');

$tracked = A::fromFlatArray([1, 2, 3, 4, 5, 6], [2, 3])->requiresGrad();
$tracked->moveaxis(0, 1)->copy()->multiply(2)->sum()->backward();
check($tracked->grad()->toArray() === [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
    'copy/moveaxis gradient mismatch');

$fused = Fusion::compile(fn($x) => $x->swapaxes(0, 1)->copy() + 1, [$matrix]);
check($fused->run($matrix)->toArray() === [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
    'Fusion copy/swapaxes result mismatch');

foreach ([
    fn() => $matrix->swapaxes(0, 2),
    fn() => $tensor->moveaxis([0, 0], [1, 2]),
    fn() => $tensor->moveaxis([0, 1], [1]),
    fn() => $tensor->moveaxis(0, 3),
] as $invalid) {
    try {
        $invalid();
        throw new RuntimeException('invalid axis arguments were accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
}

echo "CudaArray manipulation views checked\n";
?>
--EXPECT--
CudaArray manipulation views checked
