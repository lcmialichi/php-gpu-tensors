--TEST--
Complete minimum HostArray numerical core agrees with CUDA on view contracts, masks, reductions, unary math and batched matmul
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
function parity(HostArray $cpu, PhpTensors\CudaArray $gpu, string $message): void
{
    $actual = $gpu->toHost();
    check($cpu->dtype() === $actual->dtype() && $cpu->getShape() === $actual->getShape(), "$message metadata");
    if (!str_contains($cpu->dtype(), 'float')) {
        check($cpu->toBuffer() === $actual->toBuffer(), "$message exact integer/bool bytes");
        return;
    }
    $expected = [];
    $values = [];
    $cpuValues = $cpu->toArray();
    $gpuValues = $actual->toArray();
    array_walk_recursive($cpuValues, function ($v) use (&$expected) { $expected[] = $v; });
    array_walk_recursive($gpuValues, function ($v) use (&$values) { $values[] = $v; });
    $eps = match ($cpu->dtype()) { 'float64' => 2e-12, 'float16' => 5e-3, 'bfloat16' => 3e-2, default => 3e-5 };
    check(count($expected) === count($values), "$message logical size");
    foreach ($expected as $i => $value) {
        if (is_nan($value)) check(is_nan($values[$i]), "$message NaN");
        elseif (is_infinite($value)) check($value === $values[$i], "$message infinity");
        else check(abs($value - $values[$i]) <= $eps * max(1.0, abs($value)), "$message numeric tolerance at $i");
    }
}
$dtypes = ['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16', 'int32',
    'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'];
foreach ($dtypes as $dtype) {
    $cpu = new HostArray([[0, 1, 0], [1, 0, 1]], $dtype);
    $gpu = $cpu->toGpu();
    parity($cpu->transpose(), $gpu->transpose(), "$dtype transpose");
    parity($cpu->slice('1, 0:3:2'), $gpu->slice('1, 0:3:2'), "$dtype stepped slice");
    parity($cpu->slice(-1, -1), $gpu->slice(-1, -1), "$dtype scalar slice");
    parity($cpu->slice([0, 0]), $gpu->slice([0, 0]), "$dtype empty slice");
    parity($cpu->slice(1)->reshape([3, 1]), $gpu->slice(1)->reshape([3, 1]), "$dtype offset reshape");
    foreach (['sum', 'mean', 'min', 'max'] as $method) {
        foreach ([null, 0, -1] as $axis) {
            parity($cpu->transpose()->$method($axis), $gpu->transpose()->$method($axis), "$dtype $method axis");
        }
    }
    $condition = new HostArray([[0], [1]], $dtype);
    parity(HostArray::where($condition, $cpu, HostArray::zeros([1], $dtype)),
        PhpTensors\CudaArray::where($condition->toGpu(), $gpu, HostArray::zeros([1], $dtype)->toGpu()), "$dtype where");
    $empty = HostArray::zeros([2, 0, 3], $dtype);
    parity($empty->sum(1), $empty->toGpu()->sum(1), "$dtype empty sum");
    parity($empty->mean(1), $empty->toGpu()->mean(1), "$dtype empty mean");
}
foreach ($dtypes as $a) foreach ($dtypes as $b) {
    $cpuA = new HostArray([[0], [1]], $a);
    $cpuB = new HostArray([0, 1, 0], $b);
    $gpuA = $cpuA->toGpu();
    $gpuB = $cpuB->toGpu();
    foreach (['gt', 'lt', 'eq', 'ne', 'ge', 'le'] as $method) {
        $result = null;
        try { $result = $cpuA->$method($cpuB); }
        catch (PhpTensors\InvalidArgumentException $expected) {}
        if ($result === null) {
            try {
                $gpuA->$method($gpuB);
                throw new RuntimeException("CUDA accepted $a/$b $method rejected by CPU");
            } catch (PhpTensors\InvalidArgumentException | PhpTensors\RuntimeException $expected) {}
        } else {
            parity($result, $gpuA->$method($gpuB), "$a/$b $method numeric masks");
        }
    }
}
echo "All dtype pairs cover six comparison operators; all storage types cover views, where and reductions\n";

$unary = ['sqrt', 'exp', 'expm1', 'exp2', 'log', 'log1p', 'log2', 'log10',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
    'abs', 'neg', 'floor', 'ceil', 'round'];
foreach (['float32', 'float64', 'float16', 'bfloat16'] as $dtype) {
    $cpu = (new HostArray([[0.25, 0.5], [0.75, 0.25]], $dtype))->transpose();
    $gpu = (new HostArray([[0.25, 0.5], [0.75, 0.25]], $dtype))->toGpu()->transpose();
    foreach ($unary as $method) parity($cpu->$method(), $gpu->$method(), "$dtype $method");
    $singular = new HostArray([1, -1, 0], $dtype);
    parity($singular / 0, $singular->toGpu() / 0, "$dtype IEEE scalar division");
    parity(0 / $singular, 0 / $singular->toGpu(), "$dtype IEEE scalar-left division");
    parity((new HostArray([-1, 0, INF], $dtype))->sqrt(),
        (new HostArray([-1, 0, INF], $dtype))->toGpu()->sqrt(), "$dtype IEEE sqrt");
    foreach (['float32', 'float64', 'float16', 'bfloat16'] as $other) {
        $left = new HostArray([[1], [2]], $dtype);
        $right = new HostArray([1, 2, 4], $other);
        foreach (['add', 'subtract', 'multiply', 'divide'] as $method) {
            parity($left->$method($right), $left->toGpu()->$method($right->toGpu()), "$dtype/$other $method");
        }
    }
}
foreach (['float32', 'float16', 'bfloat16'] as $dtype) {
    $a = new HostArray([[1, 2, 3], [4, 5, 6]], $dtype);
    $b = $a->transpose();
    parity($a->matmul($b), $a->toGpu()->matmul($a->toGpu()->transpose()), "$dtype strided matmul");
    parity($a->broadcastTo([2, 1, 2, 3])->matmul($b->broadcastTo([3, 3, 2])),
        $a->toGpu()->broadcastTo([2, 1, 2, 3])->matmul($b->toGpu()->broadcastTo([3, 3, 2])), "$dtype batched matmul");
    $x = HostArray::zeros([2, 0], $dtype);
    $y = HostArray::zeros([0, 3], $dtype);
    parity($x->matmul($y), $x->toGpu()->matmul($y->toGpu()), "$dtype empty inner matmul");
}
$pinned = HostArray::fromBuffer(pack('f*', 1, 2, 3, 4), [2, 2], 'float32', true);
$view = $pinned->slice(null, [0, 2])->transpose();
unset($pinned);
check($view->isPinned() && $view->copy()->isPinned(), 'pinned transform retains ownership');
parity($view->sum(1), $view->toGpu()->sum(1), 'pinned transformed upload');
echo "Floating unary math, mixed-dtype arithmetic, IEEE division and batched matmul agree within dtype tolerances\n";
?>
--EXPECT--
All dtype pairs cover six comparison operators; all storage types cover views, where and reductions
Floating unary math, mixed-dtype arithmetic, IEEE division and batched matmul agree within dtype tolerances
