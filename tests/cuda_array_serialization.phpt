--TEST--
CudaArray serialization rejects malformed metadata without changing live tensors
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;

function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}
$tensor = new CudaArray([[1, 2], [3, 4]], 'float64');
$payload = $tensor->__serialize();
$invalid = [
    ['__cuda_array_v1' => '2'], ['__cuda_array_v1' => true],
    ['ndims' => -1], ['ndims' => PHP_INT_MAX], ['ndims' => '2'],
    ['shape' => [2, '2']], ['shape' => [2, -2]], ['shape' => [2 => 2, 3 => 2]],
    ['shape' => [2147483647, 2147483647, 2147483647], 'ndims' => 3],
    ['shape' => [0, 2147483647, 2147483647, 2147483647], 'ndims' => 4],
    ['dtype' => 'unknown'], ['dtype' => "float64\0invalid"],
    ['total_elements' => -1], ['total_elements' => 3],
    ['element_size' => 1], ['element_size' => PHP_INT_MAX],
    ['data' => substr($payload['data'], 1)], ['data' => $payload['data'] . "\0"],
];
foreach ($invalid as $index => $overrides) {
    try {
        $tensor->__unserialize(array_replace($payload, $overrides));
        throw new RuntimeException("Malformed CUDA payload $index was accepted");
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
    check($tensor->__serialize() === $payload, "Failed CUDA restore changed state ($index)");
    check($tensor->toArray() === [[1.0, 2.0], [3.0, 4.0]], "CUDA values lost ($index)");
}
foreach (['__cuda_array_v1', 'ndims', 'shape', 'dtype', 'total_elements', 'element_size', 'data'] as $key) {
    $bad = $payload;
    unset($bad[$key]);
    try {
        $tensor->__unserialize($bad);
        throw new RuntimeException("Missing CUDA $key was accepted");
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
    check($tensor->__serialize() === $payload, "Missing CUDA $key changed state");
}
echo "Invalid CUDA payloads preserve existing storage\n";

foreach (['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16',
          'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'] as $dtype) {
    $source = new CudaArray([[1, 2], [3, 4]], $dtype);
    $restored = unserialize(serialize($source));
    check($restored->toBuffer() === $source->toBuffer(), "$dtype CUDA round trip");
    check($restored->dtype() === $dtype && $restored->getShape() === [2, 2], "$dtype metadata");
    $tensor->__unserialize($source->__serialize());
    check($tensor->toBuffer() === $source->toBuffer(), "$dtype CUDA replacement");
    foreach ([[0], [2, 0]] as $shape) {
        $empty = CudaArray::fromBuffer('', $shape, $dtype);
        $copy = unserialize(serialize($empty));
        check($copy->getShape() === $shape && $copy->toBuffer() === '', "$dtype empty CUDA restore");
    }
}
$scalar = (new CudaArray([7.0]))->slice('0') + 1;
$restored = unserialize(serialize($scalar));
check($restored->getShape() === [] && $restored->item() === 8.0, 'CUDA scalar restore');
$scalarHost = $scalar->toHost();
$restoredHost = unserialize(serialize($scalarHost));
check($restoredHost->getShape() === [] &&
    $restoredHost->toBuffer() === $scalarHost->toBuffer(), 'Host scalar restore');
$tracked = (new CudaArray([2.0, 3.0]))->requiresGrad();
($tracked * 2)->sum()->backward();
$trackedPayload = $tracked->__serialize();
try {
    $tracked->__unserialize(array_replace($trackedPayload, ['__cuda_array_v1' => '2']));
    throw new RuntimeException('Tracked tensor accepted an invalid version');
} catch (PhpTensors\InvalidArgumentException $error) {
}
check($tracked->grad()->toArray() === [2.0, 2.0], 'Failed restore discarded existing gradients');
$snapshot = unserialize(serialize($tracked));
check($snapshot->grad() === null, 'Tensor snapshots must not restore gradients');
try {
    $snapshot->sum()->backward();
    throw new RuntimeException('Tensor snapshot unexpectedly retained gradient tracking');
} catch (PhpTensors\InvalidArgumentException $error) {
}
echo "All CUDA dtypes, empty tensors and rank-zero round trips\n";
?>
--EXPECT--
Invalid CUDA payloads preserve existing storage
All CUDA dtypes, empty tensors and rank-zero round trips
