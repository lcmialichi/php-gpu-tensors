--TEST--
CUDA Basic Test
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
var_dump(cuda_get_device_count() >= 0);
?>
--EXPECT--
bool(true)