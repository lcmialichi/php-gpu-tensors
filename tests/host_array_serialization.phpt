--TEST--
HostArray serialization validates metadata and restores existing objects transactionally
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
use PhpTensors\HostArray;

function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}

$host = new HostArray([[1, 2], [3, 4]], 'float64');
$payload = $host->__serialize();
$cudaBuild = array_key_exists('__contiguous_array_v1', $payload);
$version = $cudaBuild ? '__contiguous_array_v1' : '__host_array_v1';
$bytes = $cudaBuild ? 'data' : 'buffer';
$invalid = [
    [$version => '2'],
    [$version => 1],
    [$version => null],
    ['shape' => 'invalid'],
    ['shape' => [2, '2']],
    ['shape' => [2, -2]],
    ['shape' => [2 => 2, 3 => 2]],
    ['shape' => array_fill(0, 11, 1)],
    ['shape' => [2147483647, 2147483647, 2147483647]],
    ['shape' => [0, 2147483647, 2147483647, 2147483647]],
    ['dtype' => $cudaBuild ? -1 : 'unknown'],
    ['dtype' => $cudaBuild ? PHP_INT_MAX : 1],
    [$bytes => substr($payload[$bytes], 1)],
    [$bytes => $payload[$bytes] . "\0"],
    [$bytes => 42],
];
if ($cudaBuild) {
    $invalid = array_merge($invalid, [
        ['ndims' => -1], ['ndims' => PHP_INT_MAX], ['ndims' => '2'],
        ['total_elements' => -1], ['total_elements' => 3],
        ['element_size' => 1], ['element_size' => PHP_INT_MAX],
        ['offset' => 1], ['is_contiguous' => false],
        ['strides' => [1, 1]], ['strides' => [-1, 1]],
        ['strides' => [2, '1']], ['strides' => [2 => 2, 3 => 1]],
    ]);
}
foreach ($invalid as $index => $overrides) {
    try {
        $host->__unserialize(array_replace($payload, $overrides));
        throw new RuntimeException("Malformed host payload $index was accepted");
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
    check($host->__serialize() === $payload, "Failed host restore changed state ($index)");
    check($host->toArray() === [[1.0, 2.0], [3.0, 4.0]], "Host values lost ($index)");
}
foreach ([$version, 'shape', 'dtype', $bytes] as $key) {
    $bad = $payload;
    unset($bad[$key]);
    try {
        $host->__unserialize($bad);
        throw new RuntimeException("Missing $key was accepted");
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
    check($host->__serialize() === $payload, "Missing $key changed the host");
}
echo "Invalid host payloads preserve existing storage\n";

$dtypes = $cudaBuild
    ? ['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16',
       'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool']
    : ['float32', 'float64'];
foreach ($dtypes as $dtype) {
    $values = $dtype === 'bool' ? [[true, false], [false, true]] : [[1, 2], [3, 4]];
    $source = new HostArray($values, $dtype);
    $restored = unserialize(serialize($source));
    check($restored->toArray() === $source->toArray(), "$dtype host round trip");
    check($restored->toBuffer() === $source->toBuffer(), "$dtype host bytes");
    check($restored->dtype() === $dtype, "$dtype host dtype");
    for ($i = 0; $i < 5; $i++) {
        $host->__unserialize($source->__serialize());
        check($host->toArray() === $source->toArray(), "$dtype repeated replacement");
    }
    foreach ([[0], [2, 0]] as $shape) {
        $empty = HostArray::fromBuffer('', $shape, $dtype);
        $copy = unserialize(serialize($empty));
        check($copy->getShape() === $shape && $copy->toBuffer() === '', "$dtype empty restore");
    }
    if ($cudaBuild) {
        $row = $source[1];
        $copy = unserialize(serialize($row));
        check($copy->toArray() === $source->toArray()[1], "$dtype offset subarray values");
        check($copy->toBuffer() === $row->toBuffer(), "$dtype offset subarray bytes");
        check($copy->__serialize()['offset'] === 0, "$dtype canonical offset");
    }
}
echo "Host round trips, empty tensors and repeated replacement\n";
?>
--EXPECT--
Invalid host payloads preserve existing storage
Host round trips, empty tensors and repeated replacement
