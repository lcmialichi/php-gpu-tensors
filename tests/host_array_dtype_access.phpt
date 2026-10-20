--TEST--
CUDA-build HostArray scalar access and iteration preserve every storage dtype
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\HostArray;

function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}

$cases = [
    'float32' => [-2.5, 0.0, 1.5, 4.0],
    'float64' => [-2.5, 0.0, 1.5, 4.0],
    'float16' => [-2.5, 0.0, 1.5, 4.0],
    'bfloat16' => [-2.5, 0.0, 1.5, 4.0],
    'int8' => [-3, 0, 2, 7],
    'int16' => [-3, 0, 2, 7],
    'int32' => [-3, 0, 2, 7],
    'int64' => [-3, 0, 2, 7],
    'uint8' => [1, 0, 2, 7],
    'uint16' => [1, 0, 2, 7],
    'uint32' => [1, 0, 2, 7],
    'uint64' => [1, 0, 2, 7],
    'bool' => [true, false, true, false],
];

foreach ($cases as $dtype => $values) {
    $expected = array_chunk($values, 2);
    $constructed = new HostArray($expected, $dtype);
    $sources = [
        $constructed,
        HostArray::fromFlatArray($values, [2, 2], $dtype),
        HostArray::fromBuffer($constructed->toBuffer(), [2, 2], $dtype),
    ];
    foreach ($sources as $source) {
        check($source->dtype() === $dtype, "$dtype storage dtype");
        check($source->toArray() === $expected, "$dtype export");
        $rows = [];
        foreach ($source as $row) {
            $scalars = [];
            foreach ($row as $value) {
                $scalars[] = $value;
            }
            $rows[] = $scalars;
        }
        check($rows === $expected, "$dtype nested iteration");
        foreach ($expected as $r => $row) {
            foreach ($row as $c => $value) {
                check($source->at($r, $c) === $value, "$dtype at($r, $c)");
                check($source->get([$r, $c]) === $value, "$dtype get($r, $c)");
                check($source[$r][$c] === $value, "$dtype indexed access");
                check($source[$r]->at($c) === $value, "$dtype subarray at");
                check($source[$r]->get([$c]) === $value, "$dtype subarray get");
            }
        }
    }

    $flat = HostArray::fromFlatArray($values, [4], $dtype);
    $iterated = [];
    foreach ($flat as $value) {
        $iterated[] = $value;
    }
    check($iterated === $values, "$dtype flat iteration");
    foreach ($values as $index => $value) {
        check($flat[$index] === $value, "$dtype flat indexed access");
        check($flat->at($index) === $value, "$dtype flat at");
        check($flat->get([$index]) === $value, "$dtype flat get");
    }

    $empty = HostArray::fromBuffer('', [0], $dtype);
    $emptyValues = [];
    foreach ($empty as $value) {
        $emptyValues[] = $value;
    }
    check($emptyValues === [] && $empty->toArray() === [], "$dtype empty iteration");
    foreach ([fn() => $empty->at(0), fn() => $empty->get([0]), fn() => $empty[0]] as $read) {
        try {
            $read();
            throw new RuntimeException("$dtype accepted an empty-array index");
        } catch (PhpTensors\InvalidArgumentException $error) {
        }
    }
    echo "$dtype access and iteration\n";
}

foreach ([
    'float16' => [[0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0001], 2 ** -24],
    'bfloat16' => [[0x8000, 0x7f80, 0xff80, 0x7fc0, 0x0001], 2 ** -133],
] as $dtype => [$bits, $subnormal]) {
    $host = HostArray::fromBuffer(pack('S*', ...$bits), [5], $dtype);
    $iterated = [];
    foreach ($host as $value) {
        $iterated[] = $value;
    }
    $readers = [
        $host->toArray(),
        $iterated,
        array_map(fn($i) => $host->at($i), range(0, 4)),
        array_map(fn($i) => $host->get([$i]), range(0, 4)),
        array_map(fn($i) => $host[$i], range(0, 4)),
    ];
    foreach ($readers as $values) {
        check(fdiv(1.0, $values[0]) === -INF, "$dtype negative zero");
        check($values[1] === INF && $values[2] === -INF, "$dtype infinities");
        check(is_nan($values[3]), "$dtype NaN");
        check($values[4] === $subnormal, "$dtype smallest subnormal");
    }
    echo "$dtype special values\n";
}
?>
--EXPECT--
float32 access and iteration
float64 access and iteration
float16 access and iteration
bfloat16 access and iteration
int8 access and iteration
int16 access and iteration
int32 access and iteration
int64 access and iteration
uint8 access and iteration
uint16 access and iteration
uint32 access and iteration
uint64 access and iteration
bool access and iteration
float16 special values
bfloat16 special values
