<?php

declare(strict_types=1);

use PhpTensors\CudaArray;

/**
 * Shape & Dimensionality Control
 * * GPU memory layout can be reinterpreted without moving data (Reshape)
 * or expanded logically to match other tensors (Broadcasting).
 */

// 1. Reshaping and Transposition
$tensor = CudaArray::rand([12], 0, 1);
$matrix = $tensor->reshape([3, 4]);
$flipped = $matrix->transpose([1, 0]); // Swap axes (4x3)

/**
 * 2. Automatic Broadcasting
 * * Adding a 1D vector to a 2D matrix. 
 * The vector is logically "stretched" to match the matrix dimensions.
 */
$bias   = new CudaArray([1.0, 2.0, 3.0]);
$points = CudaArray::full([3, 3], 10.0);

$normalized = $points + $bias;

var_dump($normalized->toArray());