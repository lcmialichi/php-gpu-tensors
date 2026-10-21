--TEST--
HostArray has one typed construction, indexing, iteration and persistence contract in every build
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
$dtypes = ['float32' => 4, 'float64' => 8, 'float16' => 2, 'bfloat16' => 2,
    'int8' => 1, 'int16' => 2, 'int32' => 4, 'int64' => 8,
    'uint8' => 1, 'uint16' => 2, 'uint32' => 4, 'uint64' => 8, 'bool' => 1];
foreach ($dtypes as $dtype => $bytes) {
    $scalar = $dtype === 'bool' ? true : (str_contains($dtype, 'float') ? 1.0 : 1);
    $expected = [[$scalar, $scalar], [$scalar, $scalar]];
    $host = new HostArray([[1, 1], [1, 1]], $dtype);
    check($host->toArray() === $expected, "$dtype values");
    check($host->dtype() === $dtype && $host->getElementSize() === $bytes, "$dtype storage");
    check(strlen($host->toBuffer()) === 4 * $bytes, "$dtype bytes");
    check($host->getShape() === [2, 2] && $host->getStrides() === [2, 1], "$dtype metadata");
    check($host->getSize() === 4 && $host->getNdims() === 2, "$dtype size");
    check(count($host) === 2 && $host->count() === 2, "$dtype count");
    check($host instanceof IteratorAggregate && $host instanceof Countable, "$dtype standard interfaces");
    check($host->get([1, 1]) === $scalar && $host->at(1, 1) === $scalar, "$dtype scalar access");
    check($host[1][1] === $scalar, "$dtype dimension handlers");
    $rows = iterator_to_array($host);
    check(count($rows) === 2 && $rows[1] instanceof HostArray, "$dtype iterable rows");
    check($rows[1]->toArray() === $expected[1], "$dtype iterable values");
    check(iterator_to_array(new IteratorIterator($host[0])) === $expected[0], "$dtype iterator interoperability");
    $first = $host->getIterator();
    $second = $host->getIterator();
    $first->rewind();
    $second->rewind();
    $first->next();
    check($first->key() === 1 && $second->key() === 0, "$dtype independent iterators");
    foreach ([clone $host, unserialize(serialize($host)),
              HostArray::fromBuffer($host->toBuffer(), [2, 2], $dtype),
              HostArray::fromFlatArray([1, 1, 1, 1], [2, 2], $dtype)] as $copy) {
        check($copy->toArray() === $expected && $copy->dtype() === $dtype, "$dtype round trip");
    }
    foreach ([[[], [0]], [[[], []], [2, 0]], [[[[], []]], [1, 2, 0]]] as [$values, $shape]) {
        $empty = new HostArray($values, $dtype);
        check($empty->getShape() === $shape && $empty->getSize() === 0, "$dtype empty shape");
        check($empty->toArray() === $values && $empty->toBuffer() === '', "$dtype empty contents");
        $iterated = iterator_to_array($empty);
        check(count($iterated) === $shape[0], "$dtype empty iteration");
        foreach ($iterated as $row) {
            check($row instanceof HostArray && $row->getSize() === 0, "$dtype empty row");
        }
        check(unserialize(serialize($empty))->toArray() === $values, "$dtype empty persistence");
    }
}
check((new HostArray([1, 2], null))->dtype() === 'float32', 'nullable constructor dtype');
check(HostArray::fromFlatArray([1, 2], [2], null)->dtype() === 'float32', 'nullable flat dtype');
check(HostArray::fromBuffer(pack('f*', 1, 2), [2], null)->dtype() === 'float32', 'nullable buffer dtype');
echo "13 native dtypes share construction, empties, indexing and standard iteration\n";

$host = new HostArray([[1, 2, 3], [4, 5, 6]], 'int64');
$row = $host[1];
$payload = $row->__serialize();
check($payload['offset'] === 0 && $payload['shape'] === [3], 'logical row metadata');
check(unserialize(serialize($row))->toArray() === [4, 5, 6], 'logical row persistence');
unset($host);
check($row->toArray() === [4, 5, 6], 'row retains storage');
$iterator = $row->getIterator();
unset($row);
check(iterator_to_array($iterator) === [4, 5, 6], 'iterator retains storage');
foreach ([[[], 1], [1, []], [[[]], []], [[1], [2, 3]], ['bad'], ['key' => 1]] as $bad) {
    try {
        new HostArray($bad);
        throw new RuntimeException('malformed construction accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {}
}
echo "logical subarray persistence and lifetime are build-independent\n";

$indices = new HostArray([[1, 2], [3, 4]]);
foreach ([
    fn() => $indices->get([0.5, 1]),
    fn() => $indices->get(['0', 1]),
    fn() => $indices->get(['row' => 0, 'column' => 1]),
    fn() => $indices->at(0.5, 1),
    fn() => $indices[0.5],
    fn() => $indices['0'],
    fn() => $indices[PHP_INT_MAX],
    fn() => $indices[-1],
] as $badIndex) {
    try {
        $badIndex();
        throw new RuntimeException('invalid index accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {}
}

$legacy = ['__host_array_v1' => '1', 'shape' => [2], 'dtype' => 'float64', 'buffer' => pack('d*', 3, 7)];
$target = new HostArray([99]);
$portable = ['__contiguous_array_v1' => '1', 'ndims' => 1, 'dtype' => 1,
    'offset' => 0, 'total_elements' => 2, 'element_size' => 8,
    'is_contiguous' => true, 'shape' => [2], 'strides' => [1], 'data' => pack('d*', 3, 7)];
$target->__unserialize($portable);
check($target->toArray() === [3.0, 7.0] && $target->__serialize() === $portable, 'shared cross-build wire fixture');
check(HostArray::fromBuffer(str_repeat("\xff", 8), [1], 'uint64')->toArray()
    === ['18446744073709551615'], 'uint64 output is exact above PHP_INT_MAX');
check((new HostArray([PHP_INT_MIN, PHP_INT_MAX], 'int64'))->toArray()
    === [PHP_INT_MIN, PHP_INT_MAX], 'int64 construction preserves integer endpoints');
$target->__unserialize($legacy);
check($target->toArray() === [3.0, 7.0], 'legacy CPU payload import');
check(isset($target->__serialize()['__contiguous_array_v1']), 'normalized portable format');
foreach ([
    array_replace($legacy, ['__host_array_v1' => '2']),
    array_replace($legacy, ['shape' => [-1]]),
    array_replace($legacy, ['dtype' => 'int32']),
    array_replace($legacy, ['buffer' => 'bad']),
] as $bad) {
    try {
        $target->__unserialize($bad);
        throw new RuntimeException('malformed legacy payload accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {}
    check($target->toArray() === [3.0, 7.0], 'failed legacy restore is transactional');
}
echo "legacy CPU payloads import transactionally into the shared format\n";
?>
--EXPECT--
13 native dtypes share construction, empties, indexing and standard iteration
logical subarray persistence and lifetime are build-independent
legacy CPU payloads import transactionally into the shared format
