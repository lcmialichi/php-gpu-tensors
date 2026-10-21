--TEST--
HostArray reshape, transpose and shared CUDA slice syntax preserve dtype, logical layout and storage lifetime
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
$dtypes = ['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16', 'int32',
    'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'];
foreach ($dtypes as $dtype) {
    $base = new HostArray([[0, 1, 0], [1, 0, 1]], $dtype);
    $transposed = $base->transpose();
    check($transposed->getShape() === [3, 2] && $transposed->getStrides() === [1, 3], "$dtype transpose metadata");
    $expected = new HostArray([[0, 1], [1, 0], [0, 1]], $dtype);
    check($transposed->toBuffer() === $expected->toBuffer(), "$dtype transpose values");
    check($transposed->copy()->reshape([6])->toBuffer() === $expected->toBuffer(), "$dtype packed reshape");
    $row = $base->slice(-1);
    $reshaped = $row->reshape([3, 1]);
    check($reshaped->getStrides() === [1, 1], "$dtype offset reshape");
    $stepped = $base->slice(' :, 0:3:2 ');
    check($stepped->getShape() === [2, 2] && $stepped->getStrides() === [3, 2], "$dtype stepped metadata");
    check($stepped->toBuffer() === (new HostArray([[0, 0], [1, 1]], $dtype))->toBuffer(), "$dtype stepped values");
    $scalar = $base->slice(1, 2);
    check($scalar->getShape() === [] && $scalar->getSize() === 1, "$dtype scalar metadata");
    check($scalar->item() === $base->at(1, 2) && $scalar->get([]) === $base->at(1, 2), "$dtype scalar access");
    check($scalar->toArray() === [$scalar->item()], "$dtype scalar array export");
    check($scalar->copy()->toBuffer() === $scalar->astype($dtype)->toBuffer(), "$dtype scalar copy/cast");
    check(unserialize(serialize($scalar))->item() === $scalar->item(), "$dtype scalar persistence");
    $empty = $base->slice([2, 2], null);
    check($empty->getShape() === [0, 3] && $empty->toArray() === [], "$dtype empty slice");
    check($empty->transpose()->copy()->reshape([0])->getSize() === 0, "$dtype empty transforms");
    $nested = $transposed->slice([0, 3, 2], 1);
    $iterator = $nested->getIterator();
    unset($base, $transposed, $nested);
    check($reshaped->getShape() === [3, 1] && $reshaped->at(2, 0) === $row->at(2), "$dtype retained source");
    check(iterator_to_array($iterator) === [$row->at(0), $row->at(2)], "$dtype retained stepped iterator");
    check($stepped->copy()->astype('float64')->toArray() === [[0.0, 0.0], [1.0, 1.0]], "$dtype materialized transform");
}
echo "13 dtypes cover transposed/offset/stepped/empty/scalar views and retained iterators\n";

$base = new HostArray([[1, 2, 3], [4, 5, 6]]);
foreach ([
    fn() => $base->reshape([7]),
    fn() => $base->reshape([-1, 3]),
    fn() => $base->reshape([]),
    fn() => $base->transpose([0, 0]),
    fn() => $base->transpose([1]),
    fn() => $base->transpose([-1, 0]),
    fn() => $base->transpose([0.5, 1]),
    fn() => $base->slice('::0'),
    fn() => $base->slice('::-1'),
    fn() => $base->slice('1:2:3:4'),
    fn() => $base->slice('999999999999999999999999999999'),
    fn() => $base->slice('0,0,0'),
    fn() => $base->slice(PHP_INT_MIN),
    fn() => $base->slice([0, 1.5]),
    fn() => $base->slice(0, 3),
    fn() => $base->item(),
    fn() => HostArray::zeros([0])->item(),
] as $invalid) {
    try {
        $invalid();
        throw new RuntimeException('Invalid transform accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {}
}
try {
    $base->transpose()->reshape([6]);
    throw new RuntimeException('Non-contiguous reshape accepted without copy');
} catch (PhpTensors\RuntimeException $expected) {}
check(iterator_to_array((new HostArray([7]))->slice([0, 1, 2147483647])) === [7.0], 'single-element huge stride iteration');
check($base->slice([PHP_INT_MIN, PHP_INT_MAX])->toArray() === $base->toArray(), 'slice bounds clamp without integer overflow');
check($base->transpose([1, 0])->transpose()->toBuffer() === $base->toBuffer(), 'inverse permutation');
check($base->reshape(shape: [3, 2])->getShape() === [3, 2], 'named reshape parameter');
check($base->transpose(axis: [1, 0])->getShape() === [3, 2], 'named transpose parameter');
if (getenv('USE_ZEND_ALLOC') !== '0') {
    $before = memory_get_usage();
    for ($i = 0; $i < 500; $i++) {
        $view = $base->slice(null, [0, 3, 2])->transpose()->copy()->reshape([4]);
        unset($view);
    }
    check(memory_get_usage() - $before < 8192, 'view chains release backing storage');
}
echo "Transform validation, integer limits, explicit dense reshape and view-chain lifetimes are covered\n";
?>
--EXPECT--
13 dtypes cover transposed/offset/stepped/empty/scalar views and retained iterators
Transform validation, integer limits, explicit dense reshape and view-chain lifetimes are covered
