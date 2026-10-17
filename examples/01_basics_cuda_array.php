<?php

declare(strict_types=1);

use PhpTensors\CudaArray;

/**
 * Basic Tensor Lifecycle
 * * This example shows how to initialize data directly in GPU memory (VRAM)
 * and perform arithmetic using native PHP operator overloading.
 */

// 1. Memory Allocation (GPU-side)
$ones = CudaArray::ones([1024, 1024]);
$rand = CudaArray::rand([1024, 1024], 0, 1);

/**
 * 2. Operator Overloading
 * * These operations are offloaded to CUDA cores. 
 * The data remains in VRAM during the entire calculation.
 */
$transformed = ($ones * 5.5 + $rand) / 2.0;
$result      = $transformed ** 2;

// 3. Synchronization (GPU -> CPU)
// The toArray() call is the only point where data is moved back to System RAM.
$output = $result->toArray();

var_dump($output[0][0]);