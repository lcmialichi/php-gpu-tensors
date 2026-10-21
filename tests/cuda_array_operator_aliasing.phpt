--TEST--
CudaArray compound assignments preserve aliases, autograd and original allocation errors
--INI--
phptensors.cuda.memory_pool_size=16M
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
try {
    if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
} catch (PhpTensors\RuntimeException $error) {
    die('skip CUDA device query unavailable: ' . $error->getMessage());
}
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;
require __DIR__ . '/tensor_operator_aliasing.inc';
exerciseTensorOperators(CudaArray::class);
$matrix = new CudaArray([[1, 2], [3, 4]]);
$view = $matrix->transpose();
$view += 1;
check($view->toArray() === [[2.0, 4.0], [3.0, 5.0]], 'Strided GPU assignment');
check($matrix->toArray() === [[1.0, 2.0], [3.0, 4.0]], 'GPU assignment mutated shared storage');

$value = new CudaArray([2, 4]);
$old = $value;
$value **= 2;
check($value->toArray() === [4.0, 16.0] && $old->toArray() === [2.0, 4.0], 'Power assignment');
$leaf = (new CudaArray([2, 4]))->requiresGrad();
$result = $leaf;
$result *= $result;
$result += 1;
$result->sum()->backward();
check($leaf->grad()->toArray() === [4.0, 8.0], 'Compound assignment broke autograd');
check($leaf->toArray() === [2.0, 4.0], 'Compound assignment mutated the gradient leaf');

$value = new CudaArray([2, 4]);
$old = $value;
try {
    Fusion::run(function () use (&$value) {
        $value += 1;
        return $value;
    });
    throw new RuntimeException('Fusion capture accepted compound mutation');
} catch (PhpTensors\RuntimeException $error) {
}
check($value === $old && $value->toArray() === [2.0, 4.0], 'Rejected Fusion mutation changed input');
foreach ([
    function (&$value) { return $value++; },
    function (&$value) { return ++$value; },
    function (&$value) { return $value--; },
    function (&$value) { return --$value; },
] as $change) {
    try {
        Fusion::run(function () use (&$value, $change) {
            return $change($value);
        });
        throw new RuntimeException('Fusion capture accepted increment/decrement');
    } catch (PhpTensors\RuntimeException $error) {}
    check($value === $old && $value->toArray() === [2.0, 4.0], 'Rejected Fusion increment/decrement changed input');
}
echo "CUDA aliases, references and gradients survive functional rebinding\n";

$large = CudaArray::ones([131072]);
$before = memory_get_usage();
for ($i = 0; $i < 100; $i++) $large += 1;
check($large[131071] === 101.0, 'Repeated GPU assignment lost values');
check(memory_get_usage() - $before < 512 * 1024, 'GPU assignment leaked metadata');
$retained = [];
$full = false;
for ($i = 0; $i < 128; $i++) {
    try {
        $retained[] = clone $large;
    } catch (PhpTensors\OutOfMemoryException $error) {
        $full = true;
        break;
    }
}
check($full, 'CUDA pool exhaustion was not exercised');
$original = $large;
$reference =& $large;
try {
    $large += 1;
    throw new RuntimeException('Exhausted pool accepted another full allocation');
} catch (PhpTensors\OutOfMemoryException $error) {
}
try {
    $large->add(1);
    throw new RuntimeException('Exhausted pool accepted a method allocation');
} catch (PhpTensors\OutOfMemoryException $error) {
}
check($large === $original && $reference === $original, 'Allocation failure lost the operand');
foreach ([
    function (&$value) { return $value++; },
    function (&$value) { return ++$value; },
    function (&$value) { return $value--; },
    function (&$value) { return --$value; },
] as $change) {
    try {
        $change($large);
        throw new RuntimeException('Exhausted pool accepted increment/decrement');
    } catch (PhpTensors\OutOfMemoryException $error) {}
    check($large === $original && $reference === $original, 'Increment/decrement allocation failure lost the operand');
}
unset($retained);
$large += 1;
check($large[131071] === 102.0 && $original[131071] === 101.0, 'GPU assignment did not recover');
echo "CUDA compound assignment releases storage and preserves allocation exceptions\n";
?>
--EXPECT--
CUDA aliases, references and gradients survive functional rebinding
CUDA compound assignment releases storage and preserves allocation exceptions
