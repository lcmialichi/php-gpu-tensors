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
check($cpu[1]->toBuffer() === pack('f*', 3.0, 4.0));
check($cpu[1]->toGpu()->toArray() === [3.0, 4.0]);
check(($cpu[0] + $cpu[1])->toArray() === [4.0, 6.0]);

$gpu = new CudaArray([[5, 6], [7, 8]], 'float32');
$host = $gpu->toHost();
check($host instanceof HostArray);
check($host->toArray() === [[5.0, 6.0], [7.0, 8.0]]);
check($host->toHost() === $host);
check($host->toGpu()->toArray() === [[5.0, 6.0], [7.0, 8.0]]);
check($host instanceof \PhpTensors\OperatorOverloadable);
check(array_diff(get_class_methods(CudaArray::class), get_class_methods(HostArray::class)) === []);
foreach (['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16',
          'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'] as $dtype) {
    $source = new CudaArray([[1, 2], [3, 4]], $dtype);
    $downloaded = $source->toHost();
    check($downloaded->dtype() === $dtype, "$dtype dtype after download");
    check($downloaded->getShape() === [2, 2], "$dtype shape after download");
    check($downloaded->toBuffer() === $source->toBuffer(), "$dtype bytes after download");
    $uploaded = $downloaded->toGpu();
    check($uploaded->dtype() === $dtype, "$dtype dtype after upload");
    check($uploaded->toBuffer() === $source->toBuffer(), "$dtype round trip bytes");
}
$strided = (new CudaArray([[1, 2, 3], [4, 5, 6]], 'float32'))->slice(':, ::2');
$stridedHost = $strided->toHost();
check($stridedHost->toArray() === [[1.0, 3.0], [4.0, 6.0]]);
check($stridedHost->toGpu()->toArray() === [[1.0, 3.0], [4.0, 6.0]]);
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
