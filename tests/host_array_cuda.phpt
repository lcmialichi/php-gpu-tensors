--TEST--
HostArray is the single CPU storage and transfer class beside CUDA tensors
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\HostArray;
use PhpTensors\RuntimeException;

function check(bool $condition): void
{
    if (!$condition) throw new RuntimeException('mixed CPU/CUDA HostArray check failed');
}

$cpu = new HostArray([[1, 2], [3, 4]], 'float32');
$cpuResult = $cpu * 2 + 1;
check($cpu->getBackend() === 'cpu');
check($cpuResult->toArray() === [[3.0, 5.0], [7.0, 9.0]]);
check($cpu->toGpu()->toArray() === [[1.0, 2.0], [3.0, 4.0]]);

$gpu = new CudaArray([[5, 6], [7, 8]], 'float32');
$host = $gpu->toHost();
check($host instanceof HostArray);
check($host->toArray() === [[5.0, 6.0], [7.0, 8.0]]);
check($host->toHost() === $host);
check($host->toGpu()->toArray() === [[5.0, 6.0], [7.0, 8.0]]);
check($host instanceof \PhpTensors\Numbers);
check(array_diff(get_class_methods(CudaArray::class), get_class_methods(HostArray::class)) === []);
try {
    $host->matmul($host);
    throw new RuntimeException('Unsupported host matmul unexpectedly succeeded');
} catch (RuntimeException $error) {
    check(str_contains($error->getMessage(), 'matmul'));
}
check($cpu instanceof HostArray && $gpu instanceof CudaArray);
echo "One HostArray class transfers to and from CUDA\n";
?>
--EXPECT--
One HostArray class transfers to and from CUDA
