--TEST--
Large strict-FP32 GEMM dispatches to classic cuBLAS
--SKIPIF--
<?php
if (!extension_loaded('phptensors') || cuda_get_device_count() < 1) die('skip CUDA GPU required');
$backend = cuda_get_backend_info();
if (!$backend['cublas'] || !$backend['cublasLt']) die('skip cuBLAS and cuBLASLt required');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;

$left = A::ones([4096, 4096]);
$right = A::ones([4096, 4096]);
$product = $left->matmul($right);
$backend = cuda_get_backend_info();
if ($backend['lastMatmul'] !== 'cublas') {
    throw new RuntimeException('large strict-FP32 GEMM must use classic cuBLAS');
}
echo "large strict-FP32 GEMM dispatch passed\n";
?>
--EXPECT--
large strict-FP32 GEMM dispatch passed
