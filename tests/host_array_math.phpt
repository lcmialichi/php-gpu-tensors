--TEST--
HostArray floating math, numeric masks, exact where selection and batched matmul form a usable CPU core
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
use PhpTensors\HostArray;
function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}
function closeTo(float $actual, float $expected, float $tolerance, string $message): void
{
    check(abs($actual - $expected) <= $tolerance * max(1.0, abs($expected)), $message);
}
$unary = ['sqrt' => 'sqrt', 'exp' => 'exp', 'expm1' => 'expm1', 'exp2' => fn($x) => 2 ** $x,
    'log' => 'log', 'log1p' => 'log1p', 'log2' => fn($x) => log($x, 2), 'log10' => 'log10',
    'sin' => 'sin', 'cos' => 'cos', 'tan' => 'tan', 'asin' => 'asin', 'acos' => 'acos',
    'atan' => 'atan', 'sinh' => 'sinh', 'cosh' => 'cosh', 'tanh' => 'tanh',
    'abs' => 'abs', 'neg' => fn($x) => -$x, 'floor' => 'floor', 'ceil' => 'ceil', 'round' => 'round'];
foreach (['float32' => 2e-6, 'float64' => 2e-14, 'float16' => 2e-3, 'bfloat16' => 1e-2] as $dtype => $tolerance) {
    $input = (new HostArray([[0.25, 0.5], [0.75, 0.25]], $dtype))->transpose();
    foreach ($unary as $method => $reference) {
        $result = $input->$method();
        check($result->dtype() === $dtype && $result->getShape() === [2, 2], "$dtype $method metadata");
        foreach ($input->toArray() as $i => $row) foreach ($row as $j => $x) {
            closeTo($result->at($i, $j), $reference($x), $tolerance, "$dtype $method value");
        }
        check(HostArray::zeros([0], $dtype)->$method()->getSize() === 0, "$dtype $method empty");
    }
    $division = new HostArray([1, -1, 0], $dtype);
    $values = ($division / 0)->toArray();
    check($values[0] === INF && $values[1] === -INF && is_nan($values[2]), "$dtype IEEE division");
    check(is_nan((new HostArray([-1], $dtype))->sqrt()->item()), "$dtype sqrt domain");
    check((new HostArray([0], $dtype))->log()->item() === -INF, "$dtype log singularity");
    check(is_nan((new HostArray([-2], $dtype))->log1p()->item()), "$dtype log1p domain");
    check(pack('d', (new HostArray([-0.0], $dtype))->abs()->item()) === pack('d', -0.0), "$dtype CUDA abs signed zero");
    $left = new HostArray([[1], [2]], $dtype);
    check(($left + new HostArray([2, 3], $dtype))->toArray() === [[3.0, 4.0], [4.0, 5.0]], "$dtype low-precision broadcasting");
}
echo "22 unary functions across four floating dtypes preserve layout, IEEE values and scalar rounding\n";

$dtypes = ['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16', 'int32',
    'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'];
foreach ($dtypes as $dtype) {
    $a = new HostArray([[0], [1]], $dtype);
    $b = new HostArray([0, 1], $dtype);
    $expected = [
        'gt' => [[0, 0], [1, 0]], 'lt' => [[0, 1], [0, 0]],
        'eq' => [[1, 0], [0, 1]], 'ne' => [[0, 1], [1, 0]],
        'ge' => [[1, 0], [1, 1]], 'le' => [[1, 1], [0, 1]],
    ];
    $maskDtype = $dtype === 'bool' ? 'int32' : $dtype;
    foreach ($expected as $method => $values) {
        $mask = $a->$method($b);
        check($mask->dtype() === $maskDtype, "$dtype $method numeric mask dtype");
        check($mask->toBuffer() === (new HostArray($values, $maskDtype))->toBuffer(), "$dtype $method broadcast mask");
        check($a->$method(0.0)->getShape() === [2, 1], "$dtype scalar comparison");
    }
    $x = new HostArray([[0], [1]], $dtype);
    $y = new HostArray([1, 0], $dtype);
    $selected = HostArray::where(new HostArray([false, true], 'bool'), $x, $y);
    check($selected->dtype() === $dtype && $selected->toBuffer() ===
        (new HostArray([[1, 0], [1, 1]], $dtype))->toBuffer(), "$dtype where broadcasting");
}
$unsigned = HostArray::fromBuffer(hex2bin('feffffffffffffffffffffffffffffff'), [2], 'uint64');
check($unsigned->gt($unsigned->slice(0)->reshape([1]))->toArray() === [0, 1], 'uint64 comparisons do not round through double');
$selected = HostArray::where(new HostArray([0, 1]), $unsigned, $unsigned->slice(0)->reshape([1]));
check($selected->toBuffer() === $unsigned->toBuffer(), 'uint64 where preserves exact bits');
check((new HostArray([NAN]))->ne(NAN)->item() === 1.0, 'NaN inequality');
check((new HostArray([NAN]))->eq(NAN)->item() === 0.0, 'NaN equality');
check(HostArray::where(new HostArray([NAN, -0.0]), new HostArray([1]), new HostArray([2]))->toArray() === [1.0, 2.0], 'where numeric truth');
echo "All storage dtypes support broadcast comparisons and byte-exact where selection\n";

foreach (['float32', 'float16', 'bfloat16'] as $dtype) {
    $a = new HostArray([[1, 2, 3], [4, 5, 6]], $dtype);
    $b = new HostArray([[1, 4], [2, 5], [3, 6]], $dtype);
    check($a->matmul($b)->toArray() === [[14.0, 32.0], [32.0, 77.0]], "$dtype matrix multiply");
    check($a->matmul($a->transpose())->toArray() === [[14.0, 32.0], [32.0, 77.0]], "$dtype strided matrix");
    $batched = $a->broadcastTo([2, 1, 2, 3])->matmul($b->broadcastTo([3, 3, 2]));
    check($batched->getShape() === [2, 3, 2, 2] && $batched->at(1, 2, 1, 1) === 77.0, "$dtype batched broadcasting");
    $zero = HostArray::zeros([2, 0], $dtype)->matmul(HostArray::zeros([0, 3], $dtype));
    check($zero->toArray() === [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "$dtype empty inner product identity");
    check(HostArray::zeros([0, 2, 3], $dtype)->matmul($b)->getShape() === [0, 2, 2], "$dtype empty batch");
}
$prediction = (new HostArray([[1, 2]]))->matmul(new HostArray([[1], [2]]))->add(-4)->tanh();
closeTo($prediction->item(), tanh(1), 2e-6, 'CPU inference pipeline');
foreach ([
    fn() => (new HostArray([1], 'int32'))->sqrt(),
    fn() => (new HostArray([1], 'int64'))->eq(new HostArray([1], 'uint64')),
    fn() => (new HostArray([1], 'uint8'))->eq(-1),
    fn() => (new HostArray([1]))->gt(new stdClass()),
    fn() => HostArray::where(new HostArray([1]), new HostArray([1]), new HostArray([1], 'float64')),
    fn() => HostArray::where(new HostArray([1, 0]), new HostArray([1, 2, 3]), new HostArray([1])),
    fn() => (new HostArray([1]))->matmul(new HostArray([1])),
    fn() => (new HostArray([[1]], 'float64'))->matmul(new HostArray([[1]], 'float64')),
    fn() => (new HostArray([[1, 2]]))->matmul(new HostArray([[1, 2]])),
    fn() => HostArray::zeros([2, 1, 1])->matmul(HostArray::zeros([3, 1, 1])),
] as $invalid) {
    try {
        $invalid();
        throw new RuntimeException('Invalid CPU numerical operation accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {}
}
echo "Matching FP32/FP16/BF16 batched matmul, strided operands and CPU inference are covered\n";
?>
--EXPECT--
22 unary functions across four floating dtypes preserve layout, IEEE values and scalar rounding
All storage dtypes support broadcast comparisons and byte-exact where selection
Matching FP32/FP16/BF16 batched matmul, strided operands and CPU inference are covered
