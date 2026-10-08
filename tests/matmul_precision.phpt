--TEST--
Matmul precision defaults to strict FP32 and supports opt-in TF32
--SKIPIF--
<?php
if (!extension_loaded('cuda') || cuda_get_device_count() < 1) die('skip CUDA GPU required');
?>
--FILE--
<?php
use Cuda\CudaArray as A;

function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}

check(cuda_get_backend_info()['precision'] === 'fp32-strict', 'strict FP32 must be the default');
try {
    cuda_set_matmul_precision('unknown');
    throw new RuntimeException('invalid precision accepted');
} catch (Cuda\InvalidArgumentException $error) {
}
check(cuda_get_backend_info()['precision'] === 'fp32-strict', 'invalid precision changed the mode');

$device = cuda_get_device_info();
$backend = cuda_get_backend_info();
if ($backend['cublas'] && $device['compute_capability_major'] >= 8) {
    $leftValues = array_map(static fn(int $i): float => sin($i * 0.013) + 0.000123, range(0, 65535));
    $rightValues = array_map(static fn(int $i): float => cos($i * 0.017) - 0.000321, range(0, 65535));
    $left = A::fromFlatArray($leftValues, [256, 256]);
    $right = A::fromFlatArray($rightValues, [256, 256]);

    $strict = $left->matmul($right)->toBuffer();
    cuda_set_matmul_precision('tf32');
    check(cuda_get_backend_info()['precision'] === 'tf32', 'TF32 mode was not selected');
    $fastInfo = cuda_get_backend_info();
    check($fastInfo['lastMatmul'] === ($fastInfo['cublasLt'] ? 'cublasLt' : 'cublas'),
        'TF32 did not use the cuBLAS backend');
    $fast = $left->matmul($right)->toBuffer();
    $strictValues = unpack('g*', $strict);
    $fastValues = unpack('g*', $fast);
    check(count($strictValues) === count($fastValues), 'precision modes returned different shapes');
    $maxDifference = 0.0;
    $maxMagnitude = 1.0;
    foreach ($strictValues as $index => $value) {
        $maxDifference = max($maxDifference, abs($value - $fastValues[$index]));
        $maxMagnitude = max($maxMagnitude, abs($value));
    }
    check($maxDifference > 0.0, 'TF32 unexpectedly produced strict FP32 results');
    check($maxDifference <= 0.02 * $maxMagnitude, 'TF32 error exceeded the test tolerance');

    cuda_set_matmul_precision('fp32-strict');
    check(cuda_get_backend_info()['precision'] === 'fp32-strict', 'strict FP32 mode was not restored');
    check($left->matmul($right)->toBuffer() === $strict, 'restored strict mode changed its result');
} else {
    try {
        cuda_set_matmul_precision('tf32');
        throw new RuntimeException('TF32 accepted on an unsupported backend/device');
    } catch (Cuda\RuntimeException $error) {
    }
}
echo "strict default and optional TF32 precision checks passed\n";
?>
--EXPECT--
strict default and optional TF32 precision checks passed
