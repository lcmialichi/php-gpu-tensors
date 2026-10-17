--TEST--
PhpTensors canonical namespaces and legacy CUDA aliases
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
var_dump(interface_exists(PhpTensors\Tensor::class));
var_dump(interface_exists(PhpTensors\Numbers::class));
var_dump(class_exists(PhpTensors\Cuda\CudaArray::class));
var_dump(class_exists(PhpTensors\Fusion::class));
var_dump(class_exists(PhpTensors\Optimizer::class));
var_dump(class_exists(PhpTensors\HostArray::class));
var_dump(is_subclass_of(PhpTensors\Cuda\CudaArray::class, PhpTensors\Numbers::class));
var_dump((new ReflectionClass(Cuda\CudaArray::class))->getName() === PhpTensors\Cuda\CudaArray::class);
var_dump((new ReflectionClass(Cuda\Fusion::class))->getName() === PhpTensors\Fusion::class);
var_dump((new ReflectionClass(Cuda\Optimizer::class))->getName() === PhpTensors\Optimizer::class);
var_dump((new ReflectionClass(Cuda\HostArray::class))->getName() === PhpTensors\HostArray::class);
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
bool(true)
bool(true)
