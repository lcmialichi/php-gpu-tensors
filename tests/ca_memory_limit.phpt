--TEST--
CudaArray reports GPU memory budget exhaustion
--INI--
cuda.memory_size=1M
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
try {
    PhpTensors\CudaArray::ones([300000]);
    echo "unexpected success\n";
} catch (Throwable $error) {
    echo "allocation rejected\n";
}
?>
--EXPECTF--
Deprecated: PHP Startup: cuda.memory_size is deprecated; use phptensors.cuda.memory_pool_size instead. in Unknown on line 0
allocation rejected