--TEST--
Fast constructor, packed PHP export and binary transfers preserve types, layouts and validation
--INI--
memory_limit=512M
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
function check(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}
foreach (['float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'] as $dtype) {
    $input = [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]];
    $convert = static fn($x) => $dtype === 'bool' ? (bool)$x :
        (str_starts_with($dtype, 'float') ? (float)$x : $x);
    $expected = array_map(static fn($plane) => array_map(
        static fn($row) => array_map($convert, $row), $plane), $input);
    $tensor = new CudaArray($input, $dtype);
    check($tensor->toArray() === $expected, "$dtype constructor");
    check($tensor->toHost()->toArray() === $expected, "$dtype host");
    check(CudaArray::fromBuffer($tensor->toBuffer(), [2, 2, 3], $dtype)->toArray() === $expected, "$dtype binary");
    $flat = CudaArray::fromFlatArray(range(0, 11), [2, 2, 3], $dtype);
    check($flat->toArray() === $expected, "$dtype flat");
    $view = $tensor->slice('1:, :, ::2')->transpose();
    check(CudaArray::fromBuffer($view->toBuffer(), $view->getShape(), $dtype)->toArray() === $view->toArray(), "$dtype view");
    $rowBytes = $tensor->toBuffer();
    $elementSize = intdiv(strlen($rowBytes), 12);
    $largeView = CudaArray::fromBuffer(str_repeat($rowBytes, 8192), [8192, 12], $dtype)->transpose();
    $expectedBytes = '';
    for ($i = 0; $i < 12; $i++) $expectedBytes .= str_repeat(substr($rowBytes, $i * $elementSize, $elementSize), 8192);
    check($largeView->toBuffer() === $expectedBytes, "$dtype device pack");
    $scalar = $tensor->slice('1, 1, 2');
    check($scalar->toArray() === [$convert(11)], "$dtype scalar");
    check(CudaArray::fromBuffer($scalar->toBuffer(), [1], $dtype)->toArray() === [$convert(11)], "$dtype scalar bytes");
}
check((new CudaArray([9007199254740993, PHP_INT_MAX], 'int64'))->toArray() === [9007199254740993, PHP_INT_MAX], 'int64 precision');
check((new CudaArray([1.000000000000001], 'float64'))->toArray() === [1.000000000000001], 'float64 precision');
$mixed = new CudaArray([1, 2.5, true, false]);
check($mixed->toArray() === [1.0, 2.5, 1.0, 0.0], 'mixed numeric types');
$value = 3;
$row = [1, &$value];
check((new CudaArray([&$row, &$row], 'int32'))->toArray() === [[1, 3], [1, 3]], 'references');
$holes = [0 => 1, 1 => 99, 2 => 3];
unset($holes[1]);
check((new CudaArray($holes, 'int32'))->toArray() === [1, 3], 'packed holes');
check((new CudaArray(['a' => [1, 2], 'b' => [3, 4]], 'int32'))->toArray() === [[1, 2], [3, 4]], 'keys ignored');
foreach ([[], [[], []], [[[], []], [[], []]]] as $input) {
    $tensor = new CudaArray($input);
    check($tensor->toArray() === $input && $tensor->toBuffer() === '', 'empty');
    check($tensor->toHost()->toArray() === $input, 'empty host');
}
check(CudaArray::fromFlatArray([], [2, 0])->toArray() === [[], []], 'empty flat');
$tensor = new CudaArray([[1, 2], [3, 4]], 'int32');
$oldView = $tensor->slice('1');
$tensor->__construct([5, 6], 'int32');
check($tensor->toArray() === [5, 6] && $oldView->toArray() === [3, 4], 'reinitialization lifetime');
$bad = [[[1, 2], [3]], [[1], [2, 3]], [1, [2]], [[1], 2], [null], ['1'], [new stdClass()], [[], [1]]];
$deep = [1];
for ($i = 0; $i < 10; $i++) $deep = [$deep];
$bad[] = $deep;
$cycle = [];
$cycle[] = &$cycle;
$bad[] = $cycle;
foreach ($bad as $input) {
    try {
        new CudaArray($input);
        throw new RuntimeException('invalid input accepted');
    } catch (Cuda\InvalidArgumentException $error) {}
}
foreach ([[[1, 2], [3]], [[1, 2], []], [[[1, 2]], [2]], [[1], [-1]], [[1], [PHP_INT_MAX, 3]]] as [$input, $shape]) {
    try {
        CudaArray::fromFlatArray($input, $shape);
        throw new RuntimeException('invalid flat input accepted');
    } catch (Cuda\InvalidArgumentException $error) {}
}
try {
    $tensor->__construct([[1], [2, 3]], 'int32');
    throw new RuntimeException('invalid reinitialization accepted');
} catch (Cuda\InvalidArgumentException $error) {}
check($tensor->toArray() === [5, 6], 'failed constructor preserves tensor');
echo "types, validation, references and empty arrays checked\n";

// Exercise both sparse packing and dense strided packing, including exact 64-bit copies.
$large = CudaArray::fromFlatArray(range(0, 65535), [256, 256], 'int64');
$column = $large->slice(':, 7');
$expected = [];
for ($i = 0; $i < 256; $i++) $expected[] = $i * 256 + 7;
check($column->toArray() === $expected, 'sparse column');
check($column->toHost()->toArray() === $expected, 'sparse host');
check(CudaArray::fromBuffer($column->toBuffer(), [256], 'int64')->toArray() === $expected, 'sparse bytes');
$stepped = $large->slice('1::2, 3::2')->transpose();
$expected = [];
for ($j = 3; $j < 256; $j += 2) {
    $row = [];
    for ($i = 1; $i < 256; $i += 2) $row[] = $i * 256 + $j;
    $expected[] = $row;
}
check($stepped->toArray() === $expected, 'large strided');
check($stepped->toHost()->toArray() === $expected, 'large strided host');
check(CudaArray::fromBuffer($stepped->toBuffer(), $stepped->getShape(), 'int64')->toArray() === $expected, 'large strided bytes');
$bits = new CudaArray([PHP_INT_MAX, 9007199254740993], 'int64');
check($bits->toBuffer() === pack('q*', PHP_INT_MAX, 9007199254740993), 'binary precision');
$precise = CudaArray::full([256, 256], 1.000000000000001, 'float64');
check($precise->slice('::2, ::2')->toBuffer() === str_repeat(pack('d', 1.000000000000001), 128 * 128), 'packed float64 bits');
$boolean = CudaArray::fromBuffer(str_repeat("\0\1", 32768), [256, 256], 'bool')->transpose();
check($boolean->toBuffer() === str_repeat(str_repeat("\0", 256) . str_repeat("\1", 256), 128), 'packed bool bits');
$fused = Fusion::run(fn() => $large + 1);
check($fused->toBuffer() === (new CudaArray(range(1, 65536), 'int64'))->toBuffer(), 'Fusion export');
try {
    Fusion::compile(fn($x) => $x->toBuffer(), [$large]);
    throw new RuntimeException('capture read accepted');
} catch (Cuda\RuntimeException $error) {}
echo "strided packing, binary precision and Fusion checked\n";
$elements = 2 * 2049 * 2049;
$pattern = pack('V*', ...range(0, 1023));
$bytes = substr(str_repeat($pattern, (int)ceil($elements / 1024)), 0, $elements * 4);
$streamed = CudaArray::fromBuffer($bytes, [2, 2049, 2049], 'int32');
$output = $streamed->toArray();
check(count($output) === 2 && count($output[1]) === 2049 && count($output[1][2048]) === 2049, 'streamed nested shape');
foreach ([0, 8388607, 8388608, 8388609, $elements - 1] as $index) {
    $plane = intdiv($index, 2049 * 2049);
    $row = intdiv($index % (2049 * 2049), 2049);
    $column = $index % 2049;
    check($output[$plane][$row][$column] === $index % 1024, 'streamed nested boundaries');
}
unset($output, $streamed, $bytes);
$streamed = CudaArray::full([8388613], 2.5);
$output = $streamed->toArray();
check(count($output) === 8388613 && $output[8388608] === 2.5 && $output[8388612] === 2.5, 'streamed flat boundaries');
check((new CudaArray($output))->toBuffer() === $streamed->toBuffer(), 'streamed flat import');
unset($output, $streamed);
$row = array_fill(0, 1048580, 1.25);
$input = [$row, $row];
$streamed = new CudaArray($input, 'float64');
check($streamed->toBuffer() === str_repeat(pack('d', 1.25), 2 * 1048580), 'streamed nested import');
$input[1][1048579] = null;
try {
    new CudaArray($input, 'float64');
    throw new RuntimeException('invalid streamed import accepted');
} catch (Cuda\InvalidArgumentException $error) {}
unset($input, $row, $streamed);
echo "bounded staging windows checked\n";
?>
--EXPECT--
types, validation, references and empty arrays checked
strided packing, binary precision and Fusion checked
bounded staging windows checked
