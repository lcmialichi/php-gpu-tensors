--TEST--
CPU HostArray works directly without CUDA or a backend registry
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (class_exists('PhpTensors\\NN', false)) die('skip CUDA-enabled build');
?>
--FILE--
<?php
use PhpTensors\HostArray;
use PhpTensors\RuntimeException;

function check(bool $condition): void
{
    if (!$condition) throw new RuntimeException('CPU HostArray check failed');
}

$left = new HostArray([[1, 2], [3, 4]], 'float64');
$right = HostArray::fromFlatArray([10, 20, 30, 40], [2, 2], 'float64');
check($left->getBackend() === 'cpu');
check($left instanceof \PhpTensors\OperatorOverloadable);
check($left->getShape() === [2, 2]);
check($left->add($right)->toArray() === [[11.0, 22.0], [33.0, 44.0]]);
check(($left + 2)->toArray() === [[3.0, 4.0], [5.0, 6.0]]);
check((10 - $left)->toArray() === [[9.0, 8.0], [7.0, 6.0]]);
check(($left * $right)->toArray() === [[10.0, 40.0], [90.0, 160.0]]);
check((2 + $left)->toArray() === [[3.0, 4.0], [5.0, 6.0]]);
check((2 * $left)->toArray() === [[2.0, 4.0], [6.0, 8.0]]);
check((2 / $left)->toArray() === [[2.0, 1.0], [2.0 / 3.0, 0.5]]);
check($left->subtract(2)->toArray() === [[-1.0, 0.0], [1.0, 2.0]]);
check($left->multiply(2)->toArray() === [[2.0, 4.0], [6.0, 8.0]]);
check($left->divide(2)->toArray() === [[0.5, 1.0], [1.5, 2.0]]);
check((HostArray::ones([2])->toArray()) === [1.0, 1.0]);
check(HostArray::full([2], 3)->toArray() === [3.0, 3.0]);
check(HostArray::onesLike($left)->toArray() === [[1.0, 1.0], [1.0, 1.0]]);
check($left->toHost() === $left);
check($left->getStrides() === [2, 1]);
$float32 = new HostArray([16777217, 1.25], 'float32');
check($float32->toBuffer() === pack('f*', 16777216, 1.25));
check(($float32 + 1)->toBuffer() === pack('f*', 16777216, 2.25));
$empty = HostArray::fromFlatArray([], [2, 0], 'float32');
check($empty->getShape() === [2, 0] && $empty->getSize() === 0);
check($empty->toArray() === [[], []]);
try {
    $left + new HostArray([[1, 2], [3, 4]], 'float32');
    throw new RuntimeException('HostArray arithmetic unexpectedly accepted different dtypes');
} catch (PhpTensors\InvalidArgumentException $error) {
    check(str_contains($error->getMessage(), 'identical dtypes'));
}
try {
    $left + new HostArray([[1, 2, 3], [4, 5, 6]], 'float64');
    throw new RuntimeException('HostArray arithmetic unexpectedly accepted different shapes');
} catch (PhpTensors\InvalidArgumentException $error) {
    check(str_contains($error->getMessage(), 'identical shapes'));
}
try {
    $left / 0;
    throw new RuntimeException('HostArray division by zero unexpectedly succeeded');
} catch (PhpTensors\InvalidArgumentException $error) {
    check(str_contains($error->getMessage(), 'Division by zero'));
}
foreach (['matmul', 'reshape', 'sum', 'fromFile'] as $method) {
    if (!method_exists(HostArray::class, $method)) {
        throw new RuntimeException("CPU HostArray is missing {$method}");
    }
}
try {
    $left->matmul($right);
    throw new RuntimeException('Unsupported host matmul unexpectedly succeeded');
} catch (RuntimeException $error) {
    check(str_contains($error->getMessage(), 'matmul'));
}
$restored = unserialize(serialize($left));
check($restored instanceof HostArray && $restored->toArray() === $left->toArray());

try {
    $left->toGpu();
    throw new RuntimeException('CPU-only HostArray unexpectedly transferred to CUDA');
} catch (RuntimeException $error) {
    check(str_contains($error->getMessage(), 'CUDA backend is unavailable'));
}
try {
    new PhpTensors\CudaArray([1]);
    throw new RuntimeException('CudaArray unexpectedly initialized without CUDA');
} catch (RuntimeException $error) {
    check(str_contains($error->getMessage(), 'CUDA backend is unavailable'));
}
echo "CPU HostArray direct construction and arithmetic\n";
?>
--EXPECT--
CPU HostArray direct construction and arithmetic
