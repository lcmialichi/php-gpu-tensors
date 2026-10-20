--TEST--
CudaArray clones and reconstruction have independent storage and recover from allocation failure
--INI--
phptensors.cuda.memory_pool_size=16M
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;
require __DIR__ . '/tensor_lifecycle.inc';

foreach (['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16',
          'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'] as $dtype) {
    $source = new CudaArray([[1, 2], [3, 4]], $dtype);
    foreach ([$source, $source->transpose(), $source[1]] as $view) {
        $copy = clone $view;
        check($copy->getShape() === $view->getShape() && $copy->dtype() === $dtype,
            "$dtype CUDA clone metadata");
        check($copy->toArray() === $view->toArray(), "$dtype strided CUDA clone values");
        $expected = $view->toBuffer();
        $copy->__construct([9], 'float32');
        check($view->toBuffer() === $expected, "$dtype CUDA clone shares storage");
    }
    $copy = clone $source;
    $expected = $copy->toBuffer();
    $source->__construct([7], 'float32');
    unset($source, $view);
    check($copy->toBuffer() === $expected, "$dtype CUDA clone lifetime");
    foreach ([[0], [2, 0]] as $shape) {
        $empty = clone CudaArray::fromBuffer('', $shape, $dtype);
        check($empty->getShape() === $shape && $empty->toBuffer() === '', "$dtype empty clone");
    }
}
$scalar = clone (new CudaArray([7]))->slice('0');
check($scalar->getShape() === [] && $scalar->item() === 7.0, 'Rank-zero CUDA clone');
$hostScalar = clone $scalar->toHost();
check($hostScalar->getShape() === [] && $hostScalar->toArray() === [7.0], 'Rank-zero host clone');
check($hostScalar->count() === 0 && count($hostScalar) === 0, 'Rank-zero host count');
try {
    $hostScalar[0];
    throw new RuntimeException('Rank-zero host indexing was accepted');
} catch (PhpTensors\InvalidArgumentException $error) {
}
$pinned = new PhpTensors\HostArray([[1, 2], [3, 4]], 'float64', true);
$pinnedCopy = clone $pinned[1];
unset($pinned);
check($pinnedCopy->isPinned() && $pinnedCopy->toArray() === [3.0, 4.0], 'Pinned subarray clone');
echo "CUDA and host clones preserve storage, views, dtypes and pinning\n";

$tensor = (new CudaArray([2.0, 3.0]))->requiresGrad();
($tensor * 2)->sum()->backward();
$expected = $tensor->__serialize();
foreach ([[[[1], [2, 3]]], [[new stdClass()]], [[1], 'invalid']] as $arguments) {
    try {
        $tensor->__construct(...$arguments);
        throw new RuntimeException('Invalid CUDA reconstruction was accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
    check($tensor->__serialize() === $expected, 'Failed CUDA reconstruction changed state');
    check($tensor->grad()->toArray() === [2.0, 2.0], 'Failed CUDA reconstruction lost gradients');
}
$copy = clone $tensor;
check($copy->grad() === null, 'Cloning must not copy accumulated gradients');
try {
    $copy->sum()->backward();
    throw new RuntimeException('Native cloning unexpectedly retained autograd');
} catch (PhpTensors\InvalidArgumentException $error) {
}
$view = $tensor->slice(':');
$tensor->__construct([[8.0]], 'float64');
check($tensor->getShape() === [1, 1] && $tensor->dtype() === 'float64', 'CUDA reconstruction metadata');
check($view->toArray() === [2.0, 3.0] && $copy->toArray() === [2.0, 3.0], 'Old CUDA storage lost');
check($tensor->grad() === null, 'Successful reconstruction retained old gradients');
echo "CUDA reconstruction is transactional and clone is an autograd snapshot\n";

$uninitialized = uninitializedTensor(CudaArray::class);
foreach ([
    fn() => $uninitialized->toArray(), fn() => $uninitialized->toBuffer(),
    fn() => $uninitialized->toHost(), fn() => $uninitialized->__serialize(),
    fn() => $uninitialized->__debugInfo(), fn() => $uninitialized->getShape(),
    fn() => $uninitialized->getStrides(), fn() => $uninitialized->getSize(),
    fn() => $uninitialized->getNdims(), fn() => $uninitialized->dtype(),
    fn() => $uninitialized->item(), fn() => $uninitialized[0],
    fn() => clone $uninitialized, fn() => $uninitialized + 1,
    fn() => 1 + $uninitialized, fn() => $tensor->add($uninitialized),
    fn() => CudaArray::zerosLike($uninitialized), fn() => CudaArray::onesLike($uninitialized),
    fn() => $uninitialized->requiresGrad(), fn() => $uninitialized->grad(),
    fn() => $uninitialized->copy(), fn() => $uninitialized->sum(),
] as $operation) {
    expectUninitialized($operation);
}
try {
    $uninitialized->__construct([[1], [2, 3]]);
    throw new RuntimeException('Invalid first CUDA construction was accepted');
} catch (PhpTensors\InvalidArgumentException $error) {
}
expectUninitialized(fn() => $uninitialized->toArray());
$uninitialized->__construct([5, 6], 'float64');
check($uninitialized->toArray() === [5.0, 6.0], 'CUDA construction after failure');
$restored = uninitializedTensor(CudaArray::class);
$restored->__unserialize($tensor->__serialize());
check($restored->toArray() === $tensor->toArray(), 'CUDA restore after failed initialization');
echo "Uninitialized CUDA access throws and initialization can recover\n";

$input = new CudaArray([1, 2]);
$deferred = null;
try {
    Fusion::run(function () use ($input, &$deferred) {
        $deferred = $input + 1;
        throw new RuntimeException('Abort capture');
    });
} catch (RuntimeException $error) {
    check($error->getMessage() === 'Abort capture', 'Unexpected capture failure');
}
try {
    $copy = clone $deferred;
    throw new RuntimeException('Cloned an aborted Fusion result');
} catch (PhpTensors\RuntimeException $error) {
    check(stripos($error->getMessage(), 'aborted Fusion') !== false, 'Incorrect clone failure');
}
unset($copy, $tensor, $view, $empty, $scalar, $hostScalar, $pinnedCopy,
    $uninitialized, $restored, $input, $deferred);

class TensorCloneMarker
{
    public static int $destroyed = 0;
    public function __destruct() { self::$destroyed++; }
}
$values = array_fill(0, 131072, 1.0);
$large = new CudaArray($values);
for ($i = 0; $i < 100; $i++) {
    $large->__construct($values);
}
check($large->getSize() === 131072 && $large[131071] === 1.0, 'Repeated CUDA construction lost data');
unset($values);
@$large->marker = new TensorCloneMarker();
$clones = [];
$failed = false;
$before = memory_get_usage();
for ($i = 0; $i < 128; $i++) {
    try {
        $clones[] = clone $large;
    } catch (PhpTensors\OutOfMemoryException $error) {
        $failed = true;
        break;
    }
}
check($failed && count($clones) > 0, 'The bounded CUDA pool did not exercise clone allocation failure');
check($large->marker instanceof TensorCloneMarker && TensorCloneMarker::$destroyed === 0,
    'Failed cloning damaged PHP member ownership');
unset($clones);
$copy = clone $large;
$copy[0] = 9;
check($large[0] === 1.0 && $copy[0] === 9.0, 'CUDA clone did not recover after pool exhaustion');
check($copy->marker === $large->marker, 'PHP clone member semantics changed');
unset($copy, $large);
check(TensorCloneMarker::$destroyed === 1, 'Clone members were leaked or destroyed twice');
check(memory_get_usage() - $before < 512 * 1024, 'Cloning leaked PHP/native metadata');
echo "Failed CUDA clones release storage and PHP members exactly once\n";
?>
--EXPECT--
CUDA and host clones preserve storage, views, dtypes and pinning
CUDA reconstruction is transactional and clone is an autograd snapshot
Uninitialized CUDA access throws and initialization can recover
Failed CUDA clones release storage and PHP members exactly once
