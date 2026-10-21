--TEST--
HostArray copies logical storage and applies the shared safe-cast policy to all 13 dtypes
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
foreach ($allowed as $from => $targets) {
    $source = new HostArray([[0, 1], [1, 0]], $from);
    $copy = $source->copy();
    check($copy !== $source && $copy->dtype() === $from, "$from independent copy");
    check($copy->toBuffer() === $source->toBuffer(), "$from exact copy");
    check($copy->getShape() === [2, 2] && $copy->getStrides() === [2, 1], "$from copy metadata");
    foreach (array_keys($allowed) as $to) {
        if (!in_array($to, $targets, true)) {
            try {
                $source->astype($to);
                throw new RuntimeException("$from to $to unsafe cast accepted");
            } catch (PhpTensors\InvalidArgumentException $expected) {}
            continue;
        }
        $cast = $source->astype($to);
        $expected = new HostArray([[0, 1], [1, 0]], $to);
        check($cast !== $source && $cast->dtype() === $to, "$from to $to dtype");
        check($cast->toBuffer() === $expected->toBuffer(), "$from to $to native values");
        check($cast->getShape() === [2, 2], "$from to $to shape");
        $empty = HostArray::zeros([2, 0], $from)->astype($to);
        check($empty->getShape() === [2, 0] && $empty->toBuffer() === '', "$from to $to empty");
    }
    $row = $source[1];
    unset($source);
    check($row->copy()->toBuffer() === (new HostArray([1, 0], $from))->toBuffer(), "$from retained row copy");
    check($row->astype('float64')->toArray() === [1.0, 0.0], "$from retained row cast");
}
echo "169 dtype pairs, logical rows and empty casts follow the safe-cast contract\n";

$unsigned = HostArray::fromBuffer(hex2bin('ffffffffffffffff'), [1], 'uint64');
check($unsigned->copy()->toBuffer() === $unsigned->toBuffer(), 'uint64 copy loses no bits');
check($unsigned->astype('uint64')->toBuffer() === $unsigned->toBuffer(), 'uint64 identity cast loses no bits');
check($unsigned->astype('float64')->toArray() === [18446744073709551616.0], 'uint64 numeric widening');
$signed = new HostArray([PHP_INT_MIN, PHP_INT_MAX], 'int64');
check($signed->copy()->toBuffer() === $signed->toBuffer(), 'int64 boundaries preserved');
check($signed->astype('float64')->toArray() === [(float)PHP_INT_MIN, (float)PHP_INT_MAX], 'int64 numeric widening');
foreach (['float16', 'bfloat16', 'float32'] as $dtype) {
    $special = new HostArray([NAN, INF, -INF, -0.0, 1.5], $dtype);
    check($special->copy()->toBuffer() === $special->toBuffer(), "$dtype special bits copied");
    $wide = $special->astype('float64')->toArray();
    check(is_nan($wide[0]) && $wide[1] === INF && $wide[2] === -INF, "$dtype IEEE widening");
    check(pack('d', $wide[3]) === pack('d', -0.0) && $wide[4] === 1.5, "$dtype signed zero and finite widening");
}
foreach (['', 'invalid', "float64\0ignored"] as $dtype) {
    try {
        $signed->astype($dtype);
        throw new RuntimeException('Invalid dtype was accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {}
}
if (getenv('USE_ZEND_ALLOC') !== '0') {
    $large = HostArray::ones([262144]);
    $before = memory_get_usage();
    $independent = $large->copy();
    $bytes = memory_get_usage() - $before;
    check($bytes >= 1048576 && $bytes < 1048576 + 8192, 'copy owns a separate native-width allocation');
    unset($large, $independent);
}
echo "Copies allocate independent storage; casts preserve integer boundaries and IEEE values\n";
?>
--EXPECT--
169 dtype pairs, logical rows and empty casts follow the safe-cast contract
Copies allocate independent storage; casts preserve integer boundaries and IEEE values
