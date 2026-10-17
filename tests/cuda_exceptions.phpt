--TEST--
CUDA exception hierarchy and runtime error types
--INI--
phptensors.cuda.memory_pool_size=1M
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
var_dump(is_subclass_of(PhpTensors\RuntimeException::class, PhpTensors\Exception::class));
var_dump(is_subclass_of(PhpTensors\OutOfMemoryException::class, PhpTensors\RuntimeException::class));
var_dump(is_subclass_of(PhpTensors\CompilationException::class, PhpTensors\RuntimeException::class));

try {
    new PhpTensors\CudaArray([1], 'invalid');
} catch (PhpTensors\InvalidArgumentException $error) {
    echo "invalid dtype\n";
}

try {
    (new PhpTensors\CudaArray([1.0]))->astype('int32');
} catch (PhpTensors\InvalidArgumentException $error) {
    echo "unsafe cast\n";
}

try {
    PhpTensors\CudaArray::ones([300000]);
} catch (PhpTensors\OutOfMemoryException $error) {
    echo "out of memory\n";
}
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
invalid dtype
unsafe cast
out of memory