--TEST--
CUDA device selection applies and rejects invalid devices
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
try {
    if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
} catch (Throwable $error) {
    die('skip CUDA runtime unavailable');
}
?>
--FILE--
<?php
$current_device = cuda_get_current_device();
$device_count = cuda_get_device_count();
$target_device = $device_count > 1 ? ($current_device + 1) % $device_count : $current_device;
var_dump(cuda_set_device($target_device));
var_dump(cuda_get_current_device() === $target_device);
try {
    cuda_set_device($device_count);
    echo "invalid device accepted\n";
} catch (PhpTensors\RuntimeException $error) {
    echo "invalid device rejected\n";
}
var_dump(cuda_get_current_device() === $target_device);
if ($target_device !== $current_device) {
    cuda_set_device($current_device);
}
?>
--EXPECT--
bool(true)
bool(true)
invalid device rejected
bool(true)