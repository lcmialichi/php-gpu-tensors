--TEST--
HostArray clones own their data and repeated or failed initialization preserves storage
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
use PhpTensors\HostArray;
require __DIR__ . '/tensor_lifecycle.inc';

$values = array_fill(0, 8192, 3.0);
$host = new HostArray($values, 'float64');
$before = memory_get_usage();
for ($i = 0; $i < 100; $i++) {
    $host->__construct($values, 'float64');
}
check(memory_get_usage() - $before < 512 * 1024, 'Repeated construction leaked host storage');
check($host->getSize() === 8192 && $host->at(8191) === 3.0, 'Repeated construction lost data');
unset($host, $values);
echo "Repeated host construction has bounded memory\n";

$host = new HostArray([[1, 2], [3, 4]], 'float64');
$cudaBuild = array_key_exists('__contiguous_array_v1', $host->__serialize());
$dtypes = $cudaBuild
    ? ['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16',
       'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool']
    : ['float32', 'float64'];
foreach ($dtypes as $dtype) {
    $source = new HostArray([[1, 2], [3, 4]], $dtype);
    $expected = $source->toBuffer();
    $copy = clone $source;
    check($copy !== $source && $copy->getShape() === [2, 2], "$dtype clone shape");
    check($copy->dtype() === $dtype && $copy->toBuffer() === $expected, "$dtype clone data");
    $source->__construct([9.0], 'float32');
    check($copy->toBuffer() === $expected && $copy->getShape() === [2, 2],
        "$dtype clone shares source storage");
    unset($source);
    check($copy->toBuffer() === $expected, "$dtype clone lifetime");
    $second = clone $copy;
    $copy->__construct([8.0], 'float64');
    check($second->toBuffer() === $expected, "$dtype second clone lifetime");
    foreach ([[0], [2, 0]] as $shape) {
        $empty = clone HostArray::fromBuffer('', $shape, $dtype);
        check($empty->getShape() === $shape && $empty->toBuffer() === '', "$dtype empty clone");
        check($empty->dtype() === $dtype && $empty->toArray() === ($shape === [0] ? [] : [[], []]),
            "$dtype empty clone values");
    }
}
echo "Independent host clones preserve dtypes and empty shapes\n";

$host = new HostArray([[1, 2], [3, 4]], 'float64');
$expected = $host->__serialize();
foreach ([[[[1], [2, 3]]], [[new stdClass()]], [[1], 'invalid']] as $arguments) {
    try {
        $host->__construct(...$arguments);
        throw new RuntimeException('Invalid host reconstruction was accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
    check($host->__serialize() === $expected, 'Failed host reconstruction changed state');
}
echo "Failed host reconstruction preserves values and metadata\n";

$uninitialized = uninitializedTensor(HostArray::class);
foreach ([
    fn() => $uninitialized->toArray(), fn() => $uninitialized->toBuffer(),
    fn() => $uninitialized->__serialize(), fn() => $uninitialized->__debugInfo(),
    fn() => $uninitialized->getShape(), fn() => $uninitialized->getStrides(),
    fn() => $uninitialized->getSize(), fn() => $uninitialized->getNdims(),
    fn() => $uninitialized->getDtype(), fn() => $uninitialized->dtype(),
    fn() => $uninitialized->getElementSize(), fn() => $uninitialized->count(),
    fn() => $uninitialized->isPinned(), fn() => $uninitialized->toHost(),
    fn() => $uninitialized->toGpu(), fn() => $uninitialized->get([]),
    fn() => $uninitialized->at(0), fn() => clone $uninitialized,
    fn() => $uninitialized + 1, fn() => 1 + $uninitialized,
    fn() => $uninitialized->subtract(1), fn() => $host->add($uninitialized),
    fn() => HostArray::zerosLike($uninitialized), fn() => HostArray::onesLike($uninitialized),
] as $operation) {
    expectUninitialized($operation);
}
if ($cudaBuild) {
    expectUninitialized(fn() => $uninitialized[0]);
    expectUninitialized(fn() => count($uninitialized));
    expectUninitialized(function () use ($uninitialized) {
        foreach ($uninitialized as $value) {
            throw new RuntimeException('Uninitialized host iteration was accepted');
        }
    });
}
try {
    $uninitialized->__construct([[1], [2, 3]]);
    throw new RuntimeException('Invalid first host construction was accepted');
} catch (PhpTensors\InvalidArgumentException $error) {
}
expectUninitialized(fn() => $uninitialized->toArray());
$uninitialized->__construct([5, 6], 'float64');
check($uninitialized->toArray() === [5.0, 6.0], 'Construction after failure');
$restored = uninitializedTensor(HostArray::class);
$restored->__unserialize($host->__serialize());
check($restored->toArray() === $host->toArray(), 'Restore after failed initialization');
echo "Uninitialized host access throws and initialization can recover\n";

if ($cudaBuild) {
    $parent = new HostArray([[1, 2], [3, 4]], 'float64');
    $row = $parent[1];
    $copy = clone $row;
    check($copy->getShape() === [2] && $copy->toArray() === [3.0, 4.0], 'Offset row clone');
    $row->__construct([8, 9], 'float64');
    $parent->__construct([7], 'float32');
    unset($row, $parent);
    check($copy->toArray() === [3.0, 4.0], 'Offset row clone survives replacement');

    foreach (['construct', 'restore'] as $replacement) {
        foreach ([[1, 2, 3], [[1, 2], [3, 4]]] as $values) {
            $host = new HostArray($values, 'float64');
            $expected = $host->toArray();
            $seen = [];
            foreach ($host as $key => $value) {
                $seen[] = $value instanceof HostArray ? $value->toArray() : $value;
                if ($key === 0) {
                    if ($replacement === 'construct') {
                        $host->__construct([9.0], 'float64');
                    } else {
                        $host->__unserialize((new HostArray([9.0], 'float64'))->__serialize());
                    }
                }
            }
            check($seen === $expected, "$replacement invalidated an active host iterator");
            check($host->toArray() === [9.0], "$replacement did not replace host storage");
        }
    }
    $parent = new HostArray([[1, 2, 3], [4, 5, 6]], 'float64');
    $row = $parent[1];
    $seen = [];
    foreach ($row as $key => $value) {
        $seen[] = $value;
        if ($key === 0) {
            $parent->__construct([0]);
            $row->__construct([9]);
        }
    }
    check($seen === [4.0, 5.0, 6.0], 'Offset iterator lost its original storage');
}
echo "Host snapshots remain valid during storage replacement\n";
?>
--EXPECT--
Repeated host construction has bounded memory
Independent host clones preserve dtypes and empty shapes
Failed host reconstruction preserves values and metadata
Uninitialized host access throws and initialization can recover
Host snapshots remain valid during storage replacement
