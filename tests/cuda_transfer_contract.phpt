--TEST--
CUDA transfers preserve exact dtype bytes, view layouts and independent ownership for pinned and ordinary host storage
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA-enabled build required');
try {
    if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
} catch (PhpTensors\RuntimeException $error) {
    die('skip CUDA device query unavailable: ' . $error->getMessage());
}
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\HostArray;
function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}
$fixtures = require __DIR__ . '/fixtures/transfer_contract.php';
foreach ($fixtures['patterns'] as $dtype => $bytes) {
    $width = intdiv(strlen($bytes), 8);
    foreach ([false, true] as $pinned) {
        $source = HostArray::fromBuffer($bytes, [2, 4], $dtype, $pinned);
        check($source->isPinned() === $pinned, "$dtype pinning");
        $device = $source->toGpu();
        foreach ($fixtures['layouts'] as $name => [$shape, $indices, $makeView]) {
            $expected = '';
            foreach ($indices as $index) $expected .= substr($bytes, $index * $width, $width);
            $hostView = $makeView($source);
            $gpuView = $makeView($device);
            $upload = $hostView->toGpu();
            $download = $gpuView->toHost();
            check($upload->getShape() === $shape && $upload->dtype() === $dtype, "$dtype $name upload metadata");
            check($download->getShape() === $shape && $download->dtype() === $dtype, "$dtype $name download metadata");
            check($upload->toBuffer() === $expected, "$dtype $name host view upload bytes");
            check($gpuView->toBuffer() === $expected, "$dtype $name device view export bytes");
            check($download->toBuffer() === $expected, "$dtype $name device view download bytes");
            check(!$download->isPinned(), "$dtype $name download owns ordinary host storage");
            check($download->toGpu()->toBuffer() === $expected, "$dtype $name second upload bytes");
            $upload->__construct([], $dtype);
            check($hostView->toBuffer() === $expected, "$dtype $name upload owns independent storage");
            $download->__construct([], $dtype);
            check($gpuView->toBuffer() === $expected, "$dtype $name download owns independent storage");
        }
        $snapshot = $device->toHost();
        $source->__construct([[0, 0, 0, 0], [0, 0, 0, 0]], $dtype);
        check($device->toBuffer() === $bytes, "$dtype host replacement cannot mutate upload");
        $device->__construct([], $dtype);
        unset($source, $device, $hostView, $gpuView, $upload, $download);
        check($snapshot->toBuffer() === $bytes, "$dtype download survives device replacement and release");
        unset($snapshot);
    }

    $large = CudaArray::fromBuffer(str_repeat($bytes, 8192), [8192, 8], $dtype);
    $expected = '';
    for ($i = 0; $i < 8; $i++) $expected .= str_repeat(substr($bytes, $i * $width, $width), 8192);
    $transpose = $large->transpose();
    check($transpose->toBuffer() === $expected, "$dtype device-packed dense transpose");
    check($transpose->toHost()->toBuffer() === $expected, "$dtype device-packed transpose to host");
    $column = $large->slice(':, 3');
    $expected = str_repeat(substr($bytes, 3 * $width, $width), 8192);
    check($column->toBuffer() === $expected, "$dtype device-packed sparse offset");
    check($column->toHost()->toBuffer() === $expected, "$dtype device-packed sparse offset to host");
    unset($large, $transpose, $column);
}
echo "All 13 dtypes round-trip nine layouts from pinned and ordinary host storage byte-exactly\n";
echo "Transfers own independent storage and preserve large device-packed dense/sparse views\n";
?>
--EXPECT--
All 13 dtypes round-trip nine layouts from pinned and ordinary host storage byte-exactly
Transfers own independent storage and preserve large device-packed dense/sparse views
