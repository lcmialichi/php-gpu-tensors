--TEST--
HostArray broadcasting supports zero-stride views, logical copies, mixed floating dtypes and scalar-left operators
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
$column = new HostArray([[10], [20]], 'float32');
$row = new HostArray([1, 2, 4], 'float64');
$sum = $column + $row;
check($sum->getShape() === [2, 3] && $sum->dtype() === 'float64', 'shape and dtype promotion');
check($sum->toArray() === [[11.0, 12.0, 14.0], [21.0, 22.0, 24.0]], 'broadcast sum');
check(($row + $column)->toBuffer() === $sum->toBuffer(), 'symmetric dtype promotion');
check(($column - $row)->toArray() === [[9.0, 8.0, 6.0], [19.0, 18.0, 16.0]], 'broadcast subtraction');
check(($column * $row)->toArray() === [[10.0, 20.0, 40.0], [20.0, 40.0, 80.0]], 'broadcast multiplication');
check(($column / $row)->toArray() === [[10.0, 5.0, 2.5], [20.0, 10.0, 5.0]], 'broadcast division');
$left = HostArray::fromFlatArray([1, 2, 3, 4, 5, 6], [2, 1, 3]);
$right = HostArray::fromFlatArray([10, 20, 30, 40], [4, 1]);
$expected = [];
foreach ([[1, 2, 3], [4, 5, 6]] as $values) {
    $block = [];
    foreach ([10, 20, 30, 40] as $bias) $block[] = array_map(fn($value) => (float)($value + $bias), $values);
    $expected[] = $block;
}
check(($left + $right)->getShape() === [2, 4, 3], 'leading rank alignment');
check(($left + $right)->toArray() === $expected, 'multiple broadcast axes');
check((0 / new HostArray([2, 4]))->toArray() === [0.0, 0.0], 'zero scalar numerator');
check((8 / new HostArray([2, 4]))->toArray() === [4.0, 2.0], 'scalar-left division');
check(((new HostArray([-16777216])) + 16777217)->toArray() === [0.0], 'FP32 scalar rounds before arithmetic');
echo "Broadcast arithmetic aligns trailing axes, promotes FP32/FP64 and rounds FP32 scalars\n";

$base = new HostArray([[1, 2, 3], [4, 5, 6]], 'int32');
$view = $base[1]->broadcastTo([2, 3]);
unset($base);
check($view->getStrides() === [0, 1], 'zero-stride metadata');
check($view->toArray() === [[4, 5, 6], [4, 5, 6]], 'view retains offset storage');
check($view->at(1, 2) === 6 && $view[1][1] === 5, 'broadcast indexing');
check(iterator_to_array($view[0]) === [4, 5, 6], 'broadcast row iteration');
check(array_map(fn($row) => $row->toArray(), iterator_to_array($view)) === $view->toArray(), 'broadcast outer iteration');
$nested = $view->broadcastTo([2, 2, 3]);
check($nested->getStrides() === [0, 0, 1], 'nested zero strides');
check($nested->toArray() === [$view->toArray(), $view->toArray()], 'nested view values');
$copy = $nested->copy();
check($copy->getStrides() === [6, 3, 1] && $copy->toBuffer() === $nested->toBuffer(), 'copy packs logical view');
$same = $nested->astype('int32');
$cast = $nested->astype('float64');
check($same->getStrides() === [0, 0, 1] && $same->toBuffer() === $copy->toBuffer(), 'identity cast retains view');
check($cast->getStrides() === [6, 3, 1] && $cast->at(1, 1, 2) === 6.0, 'cast packs logical view');
$restored = unserialize(serialize($nested));
check($restored->getStrides() === [6, 3, 1] && $restored->toArray() === $copy->toArray(), 'serialization packs logical view');
unset($view, $nested, $same);
check($copy->at(1, 1, 0) === 4 && $cast->at(1, 1, 1) === 5.0, 'copy and cast survive source release');

$scalar = new HostArray([3], 'float32');
$expanded = $scalar->broadcastTo([2, 3]);
$alias = $expanded;
$expanded += new HostArray([1, 2, 3], 'float64');
check($expanded->toArray() === [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]], 'compound broadcasting');
check($expanded->dtype() === 'float64' && $alias->dtype() === 'float32', 'compound dtype transition');
check($alias->getStrides() === [0, 0] && $alias->at(1, 2) === 3.0 && $scalar->at(0) === 3.0, 'compound assignment leaves aliases unchanged');
check((12 / $alias)->toArray() === [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]], 'scalar-left strided arithmetic');
$value = new HostArray([[2], [4]]);
$original = $value;
try {
    $value /= new HostArray([[1, 0], [1, 1], [1, 1]]);
    throw new RuntimeException('Incompatible broadcast division was accepted');
} catch (PhpTensors\InvalidArgumentException $expected) {}
check($value === $original && $value->toArray() === [[2.0], [4.0]], 'failed shape-changing assignment preserves its operand');
$maximumRank = $scalar->broadcastTo(array_fill(0, 10, 2));
check($maximumRank->getNdims() === 10 && $maximumRank->getSize() === 1024, 'maximum rank broadcast');
check($maximumRank->at(...array_fill(0, 10, 1)) === 3.0, 'maximum rank view indexing');
try {
    $maximumRank->broadcastTo([2]);
    throw new RuntimeException('Lower target rank was accepted');
} catch (PhpTensors\InvalidArgumentException $expected) {}
echo "Zero-stride views retain storage and work with indexing, iteration, casts, persistence and operators\n";

foreach (['float32', 'float64', 'int64', 'float16', 'bfloat16', 'bool'] as $dtype) {
    $empty = HostArray::ones([1], $dtype)->broadcastTo([2, 0]);
    check($empty->getSize() === 0 && $empty->toArray() === [[], []], "$dtype empty broadcast");
    check($empty->copy()->toBuffer() === '' && $empty->astype('float64')->toBuffer() === '', "$dtype empty materialization");
    check(unserialize(serialize($empty))->toArray() === [[], []], "$dtype empty persistence");
}
check((HostArray::zeros([2, 0, 3]) + HostArray::ones([1, 3]))->getShape() === [2, 0, 3], 'empty arithmetic shape');
check((HostArray::zeros([2, 0, 3]) + HostArray::ones([1, 3]))->toArray() === [[], []], 'empty arithmetic values');
foreach ([[2], [3, 2], [-1, 3], [2.0, 3], [], array_fill(0, 11, 1), [PHP_INT_MAX, 3], [2147483647, 2147483647, 3]] as $shape) {
    try {
        $row->broadcastTo($shape);
        throw new RuntimeException('Invalid broadcast target accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {}
}
try {
    HostArray::zeros([0])->broadcastTo([1]);
    throw new RuntimeException('Empty source broadcast to nonempty target accepted');
} catch (PhpTensors\InvalidArgumentException $expected) {}
try {
    HostArray::zeros([2]) + HostArray::zeros([3]);
    throw new RuntimeException('Incompatible broadcast accepted');
} catch (PhpTensors\InvalidArgumentException $expected) {}
try {
    new HostArray([1], 'int32') + $row;
    throw new RuntimeException('Unsupported integer arithmetic accepted');
} catch (PhpTensors\InvalidArgumentException $expected) {}
if (getenv('USE_ZEND_ALLOC') !== '0') {
    $before = memory_get_usage();
    $large = $scalar->broadcastTo([1048576]);
    check(memory_get_usage() - $before < 8192 && $large->at(1048575) === 3.0, 'broadcast does not allocate logical payload');
    unset($large);
    $before = memory_get_usage();
    for ($i = 0; $i < 500; $i++) {
        $temporary = $scalar->broadcastTo([2, 3])->astype('float64')->copy();
        unset($temporary);
    }
    check(memory_get_usage() - $before < 8192, 'broadcast/cast/copy storage is released');
}
echo "Empty axes, invalid shapes and bounded zero-copy view lifetimes are covered\n";
?>
--EXPECT--
Broadcast arithmetic aligns trailing axes, promotes FP32/FP64 and rounds FP32 scalars
Zero-stride views retain storage and work with indexing, iteration, casts, persistence and operators
Empty axes, invalid shapes and bounded zero-copy view lifetimes are covered
