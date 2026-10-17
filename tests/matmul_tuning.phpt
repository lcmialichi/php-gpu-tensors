--TEST--
Explicit cuBLASLt autotuning selects a cached zero-workspace matmul algorithm
--SKIPIF--
<?php
if (!extension_loaded('phptensors') || cuda_get_device_count() < 1) die('skip CUDA GPU required');
$backend = cuda_get_backend_info();
if (!$backend['cublas'] || !$backend['cublasLt']) die('skip cuBLASLt required');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;

function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}

$left = A::ones([256, 256]);
$right = A::ones([256, 256]);
$tuning = cuda_tune_matmul($left, $right);
check($tuning['backend'] === 'cublasLt', 'unexpected tuning backend');
check($tuning['precision'] === 'fp32-strict', 'tuning precision was not reported');
check($tuning['candidateCount'] > 0, 'no cuBLASLt candidates were measured');
check($tuning['selectedCandidate'] >= 0 && $tuning['selectedCandidate'] < $tuning['candidateCount'],
    'selected candidate index is out of range');
$selectedCount = 0;
foreach ($tuning['candidates'] as $index => $candidate) {
    check($candidate['workspaceBytes'] === 0, 'tuning selected a shared-workspace algorithm');
    check(is_finite($candidate['medianMs']) && $candidate['medianMs'] > 0,
        'invalid candidate timing');
    if ($candidate['selected']) {
        check($index === $tuning['selectedCandidate'], 'selected flag and index differ');
        $selectedCount++;
    }
}
check($selectedCount === 1, 'expected exactly one selected algorithm');

$product = $left->matmul($right)->toBuffer();
$values = unpack('g*', $product);
check(count($values) === 256 * 256, 'tuned matmul returned the wrong size');
check($values[1] === 256.0 && $values[256 * 256] === 256.0,
    'tuned matmul returned incorrect values');
check(cuda_get_backend_info()['lastMatmul'] === 'cublasLt',
    'subsequent matmul did not use the tuned algorithm');

try {
    cuda_tune_matmul(A::ones([32, 32]), A::ones([32, 32]));
    throw new RuntimeException('small matmul should not be eligible for autotuning');
} catch (PhpTensors\InvalidArgumentException $error) {
}
echo "explicit cuBLASLt autotuning checks passed\n";
?>
--EXPECT--
explicit cuBLASLt autotuning checks passed
