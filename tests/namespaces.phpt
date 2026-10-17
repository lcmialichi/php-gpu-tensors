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
$cudaClassesMatchBuild = true;
foreach ($cudaClasses as $class) {
    if (class_exists($class, false) !== $hasCuda) {
        throw new RuntimeException("Unexpected class registration state for {$class}");
    }
}
$legacyNames = [
    'Cuda\\CudaArray',
    'Cuda\\HostArray',
    'Cuda\\HostBuffer',
    'Cuda\\ContiguousArray',
    'Cuda\\Fusion',
    'Cuda\\FusionGraph',
    'Cuda\\FusionExecution',
    'Cuda\\NN',
    'Cuda\\Compiler',
    'Cuda\\CompiledModule',
    'Cuda\\Kernel',
    'Cuda\\Optimizer',
    'Cuda\\Number',
    'Cuda\\Exception',
    'Cuda\\RuntimeException',
    'Cuda\\InvalidArgumentException',
    'Cuda\\OutOfMemoryException',
    'Cuda\\CompilationException',
    'PhpTensors\\Cuda\\CudaArray',
    'PhpTensors\\Cuda\\HostArray',
    'PhpTensors\\Cuda\\HostBuffer',
    'PhpTensors\\Cuda\\ContiguousArray',
    'PhpTensors\\Cuda\\Fusion',
    'PhpTensors\\Cuda\\FusionGraph',
    'PhpTensors\\Cuda\\FusionExecution',
    'PhpTensors\\Cuda\\NN',
    'PhpTensors\\Cuda\\Compiler',
    'PhpTensors\\Cuda\\CompiledModule',
    'PhpTensors\\Cuda\\Kernel',
    'PhpTensors\\Cuda\\Optimizer',
    'PhpTensors\\Cuda\\Number',
    'PhpTensors\\Cuda\\Tensor',
];
$legacyAliasesAbsent = true;
foreach ($legacyNames as $name) {
    if (class_exists($name, false) || interface_exists($name, false)) {
        $legacyAliasesAbsent = false;
        break;
    }
}
var_dump(interface_exists(PhpTensors\Tensor::class, false));
var_dump(interface_exists(PhpTensors\Numbers::class, false));
var_dump(true);
var_dump(class_exists(PhpTensors\HostArray::class, false));
var_dump(class_exists(PhpTensors\CudaArray::class, false));
var_dump(is_subclass_of(PhpTensors\CudaArray::class, PhpTensors\Numbers::class));
var_dump((new ReflectionClass(PhpTensors\CudaArray::class))->getName() === PhpTensors\CudaArray::class);
var_dump((new ReflectionClass(PhpTensors\HostArray::class))->getName() === PhpTensors\HostArray::class);
var_dump($legacyAliasesAbsent);
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
bool(true)
