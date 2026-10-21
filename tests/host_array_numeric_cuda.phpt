--TEST--
HostArray copy, safe casts and broadcasting agree with CUDA on dtypes, logical contents and floating arithmetic
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
use PhpTensors\HostArray;
function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}
$dtypes = ['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16',
    'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'];
foreach ($dtypes as $from) {
    $values = str_contains($from, 'float') ? [-1.5, 0.0, 3.25]
        : (str_starts_with($from, 'int') ? [-7, 0, 3] : [0, 1, 3]);
    $host = new HostArray([$values], $from);
    $gpu = $host->toGpu();
    check($host->copy()->toBuffer() === $gpu->copy()->toHost()->toBuffer(), "$from copy parity");
    foreach ($dtypes as $to) {
        $hostCast = null;
        try {
            $hostCast = $host->astype($to);
        } catch (PhpTensors\InvalidArgumentException $expected) {}
        if ($hostCast === null) {
            try {
                $gpu->astype($to);
                throw new RuntimeException("CUDA accepted $from to $to rejected by CPU");
            } catch (PhpTensors\InvalidArgumentException | PhpTensors\RuntimeException $expected) {}
        } else {
            $gpuCast = $gpu->astype($to)->toHost();
            check($hostCast->dtype() === $gpuCast->dtype(), "$from to $to dtype parity");
            check($hostCast->getShape() === $gpuCast->getShape(), "$from to $to shape parity");
            check($hostCast->toBuffer() === $gpuCast->toBuffer(), "$from to $to value parity");
        }
    }
    $hostView = $host[0]->broadcastTo([2, 3]);
    $gpuView = $host[0]->toGpu()->broadcastTo([2, 3]);
    check($hostView->getStrides() === $gpuView->getStrides(), "$from zero-stride parity");
    check($hostView->copy()->toBuffer() === $gpuView->copy()->toHost()->toBuffer(), "$from logical copy parity");
    check($hostView->toGpu()->toHost()->toBuffer() === $hostView->toBuffer(), "$from strided upload");
    $hostWide = $hostView->astype('float64');
    $gpuWide = $gpuView->astype('float64')->toHost();
    check($hostWide->toBuffer() === $gpuWide->toBuffer(), "$from strided cast parity");
    $emptyHost = HostArray::ones([1], $from)->broadcastTo([2, 0]);
    $emptyGpu = HostArray::ones([1], $from)->toGpu()->broadcastTo([2, 0]);
    check($emptyHost->copy()->toArray() === $emptyGpu->copy()->toHost()->toArray(), "$from empty copy parity");
}
foreach (['int64', 'uint64'] as $dtype) {
    $bytes = $dtype === 'uint64' ? hex2bin('ffffffffffffffff') : pack('q*', PHP_INT_MIN, PHP_INT_MAX);
    $host = HostArray::fromBuffer($bytes, [intdiv(strlen($bytes), 8)], $dtype);
    check($host->astype('float64')->toBuffer() === $host->toGpu()->astype('float64')->toHost()->toBuffer(), "$dtype boundary cast parity");
}
echo "169 cast decisions, all storage dtypes, integer boundaries and strided transfers match CUDA\n";

foreach ([['float32', 'float32'], ['float32', 'float64'], ['float64', 'float32'], ['float64', 'float64']] as [$a, $b]) {
    $hostA = new HostArray([[8], [16]], $a);
    $hostB = new HostArray([2, 4, 8], $b);
    $gpuA = $hostA->toGpu();
    $gpuB = $hostB->toGpu();
    foreach (['add', 'subtract', 'multiply', 'divide'] as $method) {
        $cpu = $hostA->$method($hostB);
        $gpu = $gpuA->$method($gpuB)->toHost();
        check($cpu->dtype() === $gpu->dtype() && $cpu->getShape() === $gpu->getShape(), "$method mixed dtype metadata");
        check($cpu->toBuffer() === $gpu->toBuffer(), "$method mixed dtype values");
    }
}
$host = (new HostArray([-16777216]))->broadcastTo([2, 3]);
$gpu = (new HostArray([-16777216]))->toGpu()->broadcastTo([2, 3]);
check(($host + 16777217)->toBuffer() === ($gpu + 16777217)->toHost()->toBuffer(), 'FP32 scalar rounding parity');
$gpuResults = [(0 / $gpu), (2 - $gpu), (2 * $gpu), (2 / $gpu)];
foreach ([(0 / $host), (2 - $host), (2 * $host), (2 / $host)] as $i => $cpu) {
    check($cpu->toBuffer() === $gpuResults[$i]->toHost()->toBuffer(), "scalar-left operation $i");
}
$hostEmpty = HostArray::zeros([2, 0, 3]) + HostArray::ones([1, 3]);
$gpuEmpty = HostArray::zeros([2, 0, 3])->toGpu() + HostArray::ones([1, 3])->toGpu();
check($hostEmpty->getShape() === $gpuEmpty->getShape() && $gpuEmpty->toHost()->toArray() === [[], []], 'empty arithmetic parity');
$pinned = HostArray::fromBuffer(pack('f*', 1, 2, 3), [3], 'float32', true);
$view = $pinned->broadcastTo([2, 3]);
check($view->isPinned() && $view->copy()->isPinned() && $view->astype('float64')->isPinned(), 'pinning retained by views, copy and cast');
check($view->toGpu()->toHost()->toArray() === [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], 'pinned strided transfer');
echo "Broadcast arithmetic, scalar operand order, empty axes and pinned views match CUDA\n";
?>
--EXPECT--
169 cast decisions, all storage dtypes, integer boundaries and strided transfers match CUDA
Broadcast arithmetic, scalar operand order, empty axes and pinned views match CUDA
