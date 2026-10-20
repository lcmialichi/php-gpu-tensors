--TEST--
Typed factories and numeric imports validate every dtype without silent zero fills or integer overflow
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
if (PHP_INT_SIZE !== 8) die('skip 64-bit PHP integer boundaries required');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\HostArray as H;
use PhpTensors\Fusion as F;
function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}
function invalid(callable $operation): void
{
    try {
        $operation();
        throw new RuntimeException('Invalid numeric conversion was accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
}
$dtypes = ['float32', 'float64', 'float16', 'bfloat16',
           'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'];
foreach ($dtypes as $dtype) {
    $floating = str_starts_with($dtype, 'float') || $dtype === 'bfloat16';
    $zero = $dtype === 'bool' ? false : ($floating ? 0.0 : 0);
    $one = $dtype === 'bool' ? true : ($floating ? 1.0 : 1);
    $fill = $dtype === 'bool' ? true : ($floating ? 3.5 : 7);
    foreach ([A::class, H::class] as $class) {
        $prototype = new $class([[1, 2]], $dtype);
        check($class::zeros([2], $dtype)->toArray() === [$zero, $zero], "$class $dtype zeros");
        check($class::ones([2], $dtype)->toArray() === [$one, $one], "$class $dtype ones");
        check($class::full([2], $fill, $dtype)->toArray() === [$fill, $fill], "$class $dtype full");
        check($class::full([2], true, $dtype)->toArray() === [$one, $one], "$class $dtype bool fill");
        check($class::zerosLike($prototype)->toArray() === [[$zero, $zero]], "$class $dtype zerosLike");
        check($class::onesLike($prototype)->toArray() === [[$one, $one]], "$class $dtype onesLike");
        check($class::fullLike($prototype, $fill)->toArray() === [[$fill, $fill]], "$class $dtype fullLike");
        check($class::fullLike($prototype, true)->toArray() === [[$one, $one]], "$class $dtype bool fullLike");
    }
    $minimum = $dtype === 'bool' ? 0 : 2;
    $maximum = $dtype === 'bool' ? 1 : 4;
    $random = A::rand([64], $minimum, $maximum, $dtype, 123);
    check($random->toBuffer() === A::rand([64], $minimum, $maximum, $dtype, 123)->toBuffer(),
          "$dtype random seed");
    foreach ($random->toArray() as $value)
        check($value >= $minimum && $value <= $maximum, "$dtype random range");
    check(count(array_unique($random->toArray(), SORT_REGULAR)) > 1, "$dtype constant random output");
    check(A::rand([4], 1, 1, $dtype, 1)->toArray() === [$one, $one, $one, $one],
          "$dtype constant random range");
}
foreach (['int32' => 2147483647, 'int64' => PHP_INT_MAX, 'uint64' => PHP_INT_MAX] as $dtype => $edge) {
    check(A::full([2], $edge, $dtype)->toArray() === [$edge, $edge], "$dtype exact fill boundary");
    check(A::rand([8], $edge, $edge, $dtype, 1)->toArray() === array_fill(0, 8, $edge),
          "$dtype exact constant random boundary");
}
foreach (['float32' => 3e38, 'float64' => 1e308] as $dtype => $limit) {
    foreach (A::rand([32], -$limit, $limit, $dtype, 2)->toArray() as $value)
        check(is_finite($value) && $value >= -$limit && $value <= $limit, "$dtype stable interpolation");
}
$badValues = [
    'int8' => [-129, 128, NAN, INF, -INF],
    'int16' => [-32769, 32768, NAN, INF],
    'int32' => [-2147483649, 2147483648, NAN, INF],
    'int64' => [2 ** 63, -(2 ** 64), NAN, INF],
    'uint8' => [-1, -0.5, 256, NAN, INF],
    'uint16' => [-1, -0.5, 65536, NAN, INF],
    'uint32' => [-1, -0.5, 4294967296, NAN, INF],
    'uint64' => [-1, -0.5, 2 ** 64, NAN, INF],
];
foreach ($badValues as $dtype => $values) foreach ($values as $value) {
    foreach ([A::class, H::class] as $class) {
        invalid(fn() => $class::full([1], $value, $dtype));
        invalid(fn() => $class::full([0], $value, $dtype));
        invalid(fn() => new $class([$value], $dtype));
        invalid(fn() => $class::fromFlatArray([$value], [1], $dtype));
        $original = new $class([7], $dtype);
        invalid(fn() => $class::fullLike($original, $value));
        invalid(fn() => $original->__construct([$value], $dtype));
        check($original->toArray() === [7], "$class $dtype failed reconstruction changed storage");
    }
    invalid(fn() => A::rand([1], $value, $value, $dtype, 1));
}
foreach ($dtypes as $dtype) foreach ([[NAN, 1], [0, INF], [2, 1]] as [$minimum, $maximum]) {
    invalid(fn() => A::rand([4], $minimum, $maximum, $dtype, 1));
    invalid(fn() => A::rand([0], $minimum, $maximum, $dtype, 1));
}
check(A::rand([8], 0, 0, 'bool', 2)->toArray() === array_fill(0, 8, false), 'constant false random range');
invalid(fn() => A::rand([4], -1, 1, 'bool', 2));
invalid(fn() => A::rand([0], 0, 1e300, 'float32', 2));
invalid(fn() => A::rand([0], 0, 1e10, 'float16', 2));
check(A::full([2], -3.75, 'int16')->toArray() === [-3, -3], 'in-range integer truncation');
check((new H([-3.75], 'int16'))->toArray() === [-3], 'host integer truncation');
check((new A([-3.75], 'int16'))->toArray() === [-3], 'GPU integer truncation');
check(A::full([1], NAN, 'bool')->toArray() === [true], 'nonzero bool conversion');
$allowed = [
    'float32' => ['float32', 'float64'],
    'float64' => ['float64'],
    'float16' => ['float16', 'float32', 'float64'],
    'bfloat16' => ['bfloat16', 'float32', 'float64'],
    'int8' => ['int8', 'int16', 'int32', 'int64', 'float32', 'float64'],
    'int16' => ['int16', 'int32', 'int64', 'float32', 'float64'],
    'int32' => ['int32', 'int64', 'float32', 'float64'],
    'int64' => ['int64', 'float64'],
    'uint8' => ['uint8', 'uint16', 'uint32', 'uint64', 'int16', 'int32', 'int64', 'float32', 'float64'],
    'uint16' => ['uint16', 'uint32', 'uint64', 'int32', 'int64', 'float32', 'float64'],
    'uint32' => ['uint32', 'uint64', 'int64', 'float32', 'float64'],
    'uint64' => ['uint64', 'float64'],
    'bool' => ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64'],
];
foreach ($dtypes as $from) {
    $source = new A([7, 0], $from);
    foreach ($dtypes as $to) foreach ([false, true] as $fused) {
        $cast = fn() => $fused ? F::run(fn() => $source->astype($to)) : $source->astype($to);
        if (!in_array($to, $allowed[$from], true)) {
            invalid($cast);
            continue;
        }
        $floating = str_starts_with($to, 'float') || $to === 'bfloat16';
        $expected = $to === 'bool' ? [true, false]
            : ($floating ? [$from === 'bool' ? 1.0 : 7.0, 0.0]
                         : [$from === 'bool' ? 1 : 7, 0]);
        $converted = $cast();
        check($converted->dtype() === $to && $converted->toArray() === $expected,
              "$from to $to fused=$fused cast contract");
    }
}
foreach (['float16', 'bfloat16'] as $dtype) {
    $source = new A([NAN, INF, -INF, -0.0], $dtype);
    foreach ([$source->astype('float64'), F::run(fn() => $source->astype('float64'))] as $converted) {
        $values = $converted->toArray();
        check(is_nan($values[0]) && $values[1] === INF && $values[2] === -INF &&
              fdiv(1.0, $values[3]) === -INF, "$dtype widening preserves IEEE values");
    }
    $assigned = new A([0, 0], $dtype);
    $assigned[0] = 3.5;
    $assigned[1] = true;
    check($assigned->toArray() === [3.5, 1.0], "$dtype typed scalar assignment");
}
echo "All dtype factories, integer bounds and random ranges passed\n";
?>
--EXPECT--
All dtype factories, integer bounds and random ranges passed
