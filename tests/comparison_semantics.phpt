--TEST--
CUDA eager and Fusion comparisons use exact equality across dtypes, layouts and non-finite values
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion as F;
function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}
function mask(array $values): array
{
    return array_map(fn($value) => is_array($value) ? mask($value) : match ($value) {
        0, 0.0, false => false,
        1, 1.0, true => true,
        default => throw new RuntimeException('Comparison produced a value other than zero or one'),
    }, $values);
}
function compare(A $left, $right, array $expected, string $label): void
{
    foreach (['eq', 'ne'] as $operation) {
        $values = $operation === 'eq' ? $expected : array_map(
            fn($value) => is_array($value) ? array_map(fn($item) => !$item, $value) : !$value,
            $expected
        );
        $eager = $left->$operation($right);
        $fused = F::run(fn() => $left->$operation($right));
        check($eager->dtype() === $fused->dtype(), "$label $operation dtype");
        check(mask($eager->toArray()) === $values, "$label eager $operation");
        check(mask($fused->toArray()) === $values, "$label Fusion $operation");
    }
}
foreach (['float32' => 1e-7, 'float64' => 1e-7,
          'float16' => 0.0009765625, 'bfloat16' => 0.0078125] as $dtype => $delta) {
    $left = new A([1, 1 + $delta, NAN, INF, -INF, 0.0, -0.0, 3], $dtype);
    $right = new A([1, 1, NAN, INF, INF, -0.0, 0.0, 4], $dtype);
    compare($left, $right, [true, false, false, true, false, true, true, false], $dtype);
    compare(new A([0, $delta, NAN, INF], $dtype), 0, [true, false, false, false], "$dtype scalar");
    compare(new A([[1], [1 + $delta]], $dtype),
            new A([[1, 1 + $delta]], $dtype),
            [[true, false], [false, true]], "$dtype broadcast");
    compare((new A([[1, 1 + $delta], [NAN, INF]], $dtype))->transpose(),
            new A([[1, NAN], [1, INF]], $dtype),
            [[true, false], [false, true]], "$dtype strided");
}
foreach (['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'] as $dtype) {
    compare(new A([0, 1, 1], $dtype), new A([0, 0, 1], $dtype), [true, false, true], $dtype);
}
compare(new A([9007199254740992, 9007199254740993], 'int64'),
        new A([9007199254740993, 9007199254740993], 'int64'), [false, true], 'wide integers');
compare(new A([1, 2, 3], 'int32'), new A([1.0, 2.5, 3.0], 'float64'),
        [true, false, true], 'mixed numeric dtypes');
$example = new A([1.0, 2.0], 'float64');
$graph = F::compile(fn($a, $b) => [$a->eq($b), $a->ne($b)], [$example, $example]);
[$equal, $different] = $graph->run(new A([NAN, INF], 'float64'), new A([NAN, INF], 'float64'));
check(mask($equal->toArray()) === [false, true] && mask($different->toArray()) === [true, false],
      'Comparison graph replay retained capture values');
echo "Exact eager/Fusion comparison semantics passed\n";
?>
--EXPECT--
Exact eager/Fusion comparison semantics passed
