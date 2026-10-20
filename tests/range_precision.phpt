--TEST--
Typed ranges preserve exact integer endpoints and safe floating interpolation
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
if (PHP_INT_SIZE !== 8) die('skip 64-bit PHP required');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
function check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}
function invalid(callable $operation): void {
    try {
        $operation();
        throw new RuntimeException('Invalid range was accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {
    }
}
check(A::arange(9007199254740993, 9007199254740996, 1, 'uint64')->toArray() ===
      [9007199254740993, 9007199254740994, 9007199254740995], 'uint64 arange lost precision');
foreach (['int64', 'uint64'] as $dtype) {
    check(A::arange(PHP_INT_MAX - 2, PHP_INT_MAX, 1, $dtype)->toArray() ===
          [PHP_INT_MAX - 2, PHP_INT_MAX - 1], "$dtype arange boundary");
    check(A::linspace(PHP_INT_MAX - 2, PHP_INT_MAX, 3, true, $dtype)->toArray() ===
          [PHP_INT_MAX - 2, PHP_INT_MAX - 1, PHP_INT_MAX], "$dtype linspace boundary");
    check(A::linspace(PHP_INT_MAX, PHP_INT_MAX, 7, true, $dtype)->toArray() ===
          array_fill(0, 7, PHP_INT_MAX), "$dtype constant linspace");
}
check(A::linspace(PHP_INT_MIN, PHP_INT_MAX, 5, true, 'int64')->toArray() ===
      [PHP_INT_MIN, -4611686018427387904, 0, 4611686018427387903, PHP_INT_MAX],
      'full int64 span interpolation or truncation');
check(A::linspace(PHP_INT_MAX, PHP_INT_MIN, 5, true, 'int64')->toArray() ===
      [PHP_INT_MAX, 4611686018427387903, 0, -4611686018427387904, PHP_INT_MIN],
      'descending full int64 span');
check(A::linspace(PHP_INT_MIN, PHP_INT_MAX, 4, false, 'int64')->toArray() ===
      [PHP_INT_MIN, -4611686018427387904, 0, 4611686018427387903],
      'endpoint=false full int64 span');
foreach (['float32', 'float64', 'float16', 'bfloat16',
          'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'] as $dtype) {
    $floating = str_starts_with($dtype, 'float') || $dtype === 'bfloat16';
    $expected = $dtype === 'bool' ? [false, true, true]
        : ($floating ? [0.0, 1.0, 2.0] : [0, 1, 2]);
    check(A::arange(0, 3, 1, $dtype)->toArray() === $expected, "$dtype arange");
    check(A::linspace(0, 2, 3, true, $dtype)->toArray() === $expected, "$dtype linspace");
    check(A::linspace(0, 2, 0, true, $dtype)->toArray() === [], "$dtype empty linspace");
    check(A::linspace(0, 2, 1, true, $dtype)->toArray() === [$expected[0]], "$dtype singleton");
    check(A::eye(2, 3, 1, $dtype)->toArray() ===
          [[$expected[0], $expected[1], $expected[0]],
           [$expected[0], $expected[0], $expected[1]]], "$dtype eye");
    if ($floating) {
        $normal = A::randn([16], $dtype, 9);
        check($normal->dtype() === $dtype &&
              $normal->toBuffer() === A::randn([16], $dtype, 9)->toBuffer(), "$dtype normal seed");
        foreach ($normal->toArray() as $value) check(is_finite($value), "$dtype normal value");
    } else {
        invalid(fn() => A::randn([0], $dtype, 9));
    }
    foreach ([NAN, INF, -INF] as $bad) {
        invalid(fn() => A::arange($bad, 0, 1, $dtype));
        invalid(fn() => A::linspace(0, $bad, 0, true, $dtype));
    }
}
foreach (['int8', 'int16', 'int32', 'int64'] as $dtype) {
    check(A::linspace(-3, 3, 5, true, $dtype)->toArray() === [-3, -1, 0, 1, 3],
          "$dtype truncation toward zero");
    check(A::linspace(3, -3, 5, true, $dtype)->toArray() === [3, 1, 0, -1, -3],
          "$dtype descending truncation toward zero");
}
check(A::linspace(127.75, 127.75, 3, true, 'int8')->toArray() === [127, 127, 127],
      'fractional integer range agrees with fills');
foreach ([0, 1, 3] as $count) {
    invalid(fn() => A::linspace(0, 2 ** 63, $count, true, 'int64'));
    invalid(fn() => A::linspace(-0.5, 1, $count, true, 'uint64'));
}
foreach ([1e308, PHP_FLOAT_MAX, -PHP_FLOAT_MAX] as $edge) {
    check(A::linspace($edge, $edge, 7)->toArray() === array_fill(0, 7, $edge),
          'constant FP64 interpolation');
}
foreach (A::linspace(-PHP_FLOAT_MAX, PHP_FLOAT_MAX, 9)->toArray() as $value)
    check(is_finite($value), 'wide FP64 interpolation overflow');
check(A::arange(-1e308, 1.5e308, 1e308)->toArray() === [-1e308, 0.0, 1e308],
      'arange intermediate product overflow');
check(A::arange(PHP_INT_MAX, PHP_INT_MIN, PHP_INT_MIN)->toArray() === [PHP_INT_MAX, -1],
      'descending arange with INT64_MIN stride');
for ($start = -4; $start <= 4; $start++) for ($stop = -4; $stop <= 4; $stop++) {
    for ($count = 0; $count <= 8; $count++) foreach ([false, true] as $endpoint) {
        $expected = [];
        $denominator = $endpoint ? max(1, $count - 1) : $count;
        for ($index = 0; $index < $count; $index++)
            $expected[] = (int)(($start * $denominator + ($stop - $start) * $index) / $denominator);
        check(A::linspace($start, $stop, $count, $endpoint, 'int64')->toArray() === $expected,
              "rational interpolation $start $stop $count endpoint=$endpoint");
    }
}
echo "Typed range precision and boundary checks passed\n";
?>
--EXPECT--
Typed range precision and boundary checks passed
