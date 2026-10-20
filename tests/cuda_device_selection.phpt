--TEST--
CUDA device selection preserves its owning device and rejects unsafe switches
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
$target_device = $current_device;
var_dump(cuda_set_device($target_device));
var_dump(cuda_get_current_device() === $target_device);
if ($device_count > 1) {
    try {
        cuda_set_device(($current_device + 1) % $device_count);
        throw new RuntimeException('another device accepted');
    } catch (PhpTensors\RuntimeException $error) {
        if (!str_contains($error->getMessage(), 'CUDA_VISIBLE_DEVICES')) throw $error;
    }
}
foreach ([-1, PHP_INT_MIN, PHP_INT_MAX, 4294967296] as $invalid) {
    foreach ([
        fn() => cuda_set_device($invalid),
        fn() => cuda_get_device_info($invalid),
        fn() => cuda_get_peer_access($invalid, 0),
        fn() => cuda_get_peer_access(0, $invalid),
    ] as $operation) {
        try {
            $operation();
            throw new RuntimeException('oversized or negative device accepted');
        } catch (PhpTensors\InvalidArgumentException $error) {
        }
    }
}
try {
    cuda_set_device($device_count);
    echo "invalid device accepted\n";
} catch (PhpTensors\RuntimeException $error) {
    echo "invalid device rejected\n";
}
var_dump(cuda_get_current_device() === $target_device);
?>
--EXPECT--
bool(true)
bool(true)
invalid device rejected
bool(true)