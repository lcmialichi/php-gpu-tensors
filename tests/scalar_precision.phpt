--TEST--
CUDA scalar boundaries preserve dtype, FP64 precision, half values and oversized unsigned integers
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
if (PHP_INT_SIZE !== 8) die('skip 64-bit PHP boundary cases');
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
$cases = [
    'float32' => [16777216.0, -2.5],
    'float64' => [16777217.0, 9007199254740991.0],
    'float16' => [1.5, -2.5],
    'bfloat16' => [1.5, -2.5],
    'int8' => [-128, 127], 'int16' => [-32768, 32767],
    'int32' => [-2147483648, 2147483647],
    'int64' => [PHP_INT_MIN, PHP_INT_MAX],
    'uint8' => [0, 255], 'uint16' => [0, 65535],
    'uint32' => [0, 4294967295],
    'uint64' => [0, PHP_INT_MAX],
    'bool' => [false, true],
];
foreach ($cases as $dtype => $expected) {
    $tensor = new A($expected, $dtype);
    check($tensor->toArray() === $expected, "$dtype export");
    check($tensor[0] === $expected[0] && $tensor[-1] === $expected[1], "$dtype indexed types/precision");
    check($tensor->slice('0')->item() === $expected[0], "$dtype scalar item");
    check($tensor->slice('1')->reshape([1])->item() === $expected[1], "$dtype offset item");
    $strided = (new A([$expected, $expected], $dtype))->transpose();
    check($strided[1][1] === $expected[1], "$dtype strided index");
}
$unsignedBytes = pack('Q*', PHP_INT_MAX, PHP_INT_MIN, -1);
$unsignedExpected = [PHP_INT_MAX, '9223372036854775808', '18446744073709551615'];
$unsigned = A::fromBuffer($unsignedBytes, [3], 'uint64');
foreach ([$unsigned, $unsigned->copy(), F::run(fn() => $unsigned->slice())] as $tensor) {
    check($tensor->toArray() === $unsignedExpected, 'uint64 GPU export');
    foreach ($unsignedExpected as $i => $value) {
        check($tensor[$i] === $value && $tensor->slice((string)$i)->item() === $value,
              'uint64 index/item decimal representation');
    }
}
foreach ([H::fromBuffer($unsignedBytes, [3], 'uint64'), $unsigned->toHost()] as $host) {
    check($host->toArray() === $unsignedExpected, 'uint64 host export');
    $iterated = [];
    foreach ($host as $value) $iterated[] = $value;
    check($iterated === $unsignedExpected, 'uint64 host iterator');
    foreach ($unsignedExpected as $i => $value) {
        check($host[$i] === $value && $host->at($i) === $value && $host->get([$i]) === $value,
              'uint64 host scalar representation');
    }
    check($host->toGpu()->toHost()->toBuffer() === $unsignedBytes, 'uint64 exact byte round trip');
}
foreach ([
    'float16' => [[0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001], 2 ** -24],
    'bfloat16' => [[0x8000, 0x7f80, 0xff80, 0x7fc0, 0x0001], 2 ** -133],
] as $dtype => [$bits, $smallest]) {
    $tensor = A::fromBuffer(pack('S*', ...$bits), [5], $dtype);
    foreach ([fn($i) => $tensor[$i], fn($i) => $tensor->slice((string)$i)->item()] as $read) {
        check(fdiv(1.0, $read(0)) === -INF, "$dtype signed zero");
        check($read(1) === INF && $read(2) === -INF && is_nan($read(3)), "$dtype non-finite values");
        check($read(4) === $smallest, "$dtype subnormal");
    }
}
$tiny = new A([2 ** -1074, -0.0], 'float64');
check($tiny[0] === 2 ** -1074 && $tiny->slice('0')->item() === 2 ** -1074, 'FP64 subnormal');
check(fdiv(1.0, $tiny[1]) === -INF, 'FP64 indexed signed zero');
$wide = new A([16777217.0, 1e200], 'float64');
foreach ([
    [fn() => $wide / 2, [16777217.0 / 2, 1e200 / 2]],
    [fn() => 2 / $wide, [2 / 16777217.0, 2 / 1e200]],
    [fn() => $wide ** -1, [1 / 16777217.0, 1 / 1e200]],
] as [$operation, $expected]) {
    $eager = $operation();
    $fused = F::run($operation);
    check($eager->dtype() === 'float64' && $fused->dtype() === 'float64', 'FP64 scalar promotion');
    check($eager->toArray() === $expected && $fused->toArray() === $expected, 'FP64 scalar arithmetic precision');
}
$itemType = (string)(new ReflectionMethod(A::class, 'item'))->getReturnType();
check(str_contains($itemType, 'string'), 'item return type omits exact wide-integer exports');
echo "Typed scalar extraction and lossless integer exports passed\n";
echo "FP64 scalar arithmetic and half/subnormal exports passed\n";
?>
--EXPECT--
Typed scalar extraction and lossless integer exports passed
FP64 scalar arithmetic and half/subnormal exports passed
