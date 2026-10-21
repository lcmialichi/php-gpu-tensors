--TEST--
Abandoned kernel execution with failed completion disables CUDA safely during destruction
--SKIPIF--
<?php
if (!extension_loaded('phptensors') || !class_exists('PhpTensors\\NN', false)) die('skip CUDA-enabled build required');
try {
    if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
} catch (PhpTensors\RuntimeException $error) {
    die('skip CUDA device query unavailable: ' . $error->getMessage());
}
?>
--FILE--
<?php
$method = 'destroy';
require __DIR__ . '/fixtures/execution_failure.php';
?>
--EXPECT--
Unconfirmed completion disables the CUDA runtime
Clearing errors and resetting devices cannot revive the failed runtime
CPU tensors remain usable
Failed resources can be destroyed without recycling uncertain GPU storage
