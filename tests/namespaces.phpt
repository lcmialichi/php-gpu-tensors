--TEST--
PhpTensors exposes a strict canonical namespace
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
$hasCuda = class_exists(PhpTensors\Fusion::class, false);
$cudaClasses = [
    'PhpTensors\\Fusion',
    'PhpTensors\\FusionGraph',
    'PhpTensors\\FusionExecution',
    'PhpTensors\\NN',
    'PhpTensors\\Compiler',
    'PhpTensors\\CompiledModule',
    'PhpTensors\\Optimizer',
];
foreach ($cudaClasses as $class) {
    if (class_exists($class, false) !== $hasCuda) {
        throw new RuntimeException("Unexpected class registration state for {$class}");
    }
}
var_dump(interface_exists(PhpTensors\Tensor::class, false));
var_dump(interface_exists(PhpTensors\OperatorOverloadable::class, false));
var_dump(is_subclass_of(PhpTensors\OperatorOverload::class, PhpTensors\OperatorOverloadable::class));
var_dump(class_exists(PhpTensors\HostArray::class, false));
var_dump(class_exists(PhpTensors\CudaArray::class, false));
var_dump(is_subclass_of(PhpTensors\CudaArray::class, PhpTensors\OperatorOverloadable::class));
var_dump((new ReflectionClass(PhpTensors\CudaArray::class))->getName() === PhpTensors\CudaArray::class);
var_dump((new ReflectionClass(PhpTensors\HostArray::class))->getName() === PhpTensors\HostArray::class);
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
