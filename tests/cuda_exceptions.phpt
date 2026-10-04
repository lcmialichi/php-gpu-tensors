--TEST--
CUDA exception hierarchy and runtime error types
--INI--
cuda.memory_size=1M
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
var_dump(is_subclass_of(Cuda\RuntimeException::class, Cuda\Exception::class));
var_dump(is_subclass_of(Cuda\OutOfMemoryException::class, Cuda\RuntimeException::class));
var_dump(is_subclass_of(Cuda\CompilationException::class, Cuda\RuntimeException::class));

try {
    new Cuda\CudaArray([1], 'invalid');
} catch (Cuda\InvalidArgumentException $error) {
    echo "invalid dtype\n";
}

try {
    (new Cuda\CudaArray([1.0]))->astype('int32');
} catch (Cuda\InvalidArgumentException $error) {
    echo "unsafe cast\n";
}

try {
    Cuda\CudaArray::ones([300000]);
} catch (Cuda\OutOfMemoryException $error) {
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