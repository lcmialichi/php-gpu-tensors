<?php

declare(strict_types=1);

use PhpTensors\CudaArray;

/**
 * Mathematical Primitives & Reductions
 * * Demonstrates built-in GPU functions that operate element-wise 
 * or reduce a tensor to a single value (aggregations).
 */

$data = CudaArray::rand([5000, 5000], 0, 100);

// --- Element-wise Parallel Functions ---
$logarithm = $data->log();
$squareRoot = $data->sqrt();
$sineWave   = $data->sin();

// --- Reduction Operations ---
// These are optimized tree-reductions on the GPU hardware
$totalSum = $data->sum();
$maxValue = $data->max();
$meanValue = $totalSum / $data->getSize();

// --- Linear Algebra ---
$matrixA = CudaArray::rand([128, 256]);
$matrixB = CudaArray::rand([256, 128]);

// High-performance Matrix Multiplication
$product = $matrixA->matmul($matrixB);