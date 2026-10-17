<?php

declare(strict_types=1);

use PhpTensors\Compiler;
use PhpTensors\CudaArray;

/**
 * Asynchronous Kernel Execution
 * * Demonstrates how to launch GPU tasks without blocking the main PHP thread.
 * This allows for massive parallelism between the CPU and GPU.
 */

// --- 1. JIT Compilation ---

$src =  <<<'CUDA'
extern "C" __global__ void heavy_math(float *data, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) {
        return;
    }

    int idx = row * cols + col;
    float val = data[idx];
    for (int i = 0; i < 100; i++) {
        val = sinf(val) * cosf(val);
    }

    data[idx] = val;
}
CUDA;

$compiler = new Compiler(source: $src);
$headers = ['#include <math.h>'];
$parameters = [
    ['name' => 'data', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'rows', 'dtype' => 'int32'],
    ['name' => 'cols', 'dtype' => 'int32'],
];

$compiler->kernel('heavy_math', $parameters, $headers);
$module = $compiler->compile();

// --- 2. Data & Configuration ---

$size = 2_000_000;
$data = CudaArray::rand([32, $size], 0, 1);
[$rows, $cols] = $data->getShape();

$block = [32, 8, 1];
$config = [
    'block' => [256, 1, 1],
    'grid' => [(int) ceil($cols / $block[0]), (int) ceil($rows / $block[1]), 1]
];

// --- 3. Asynchronous Launch ---

/**
 * runAsync() returns an operation ID immediately.
 * The PHP engine does not wait for the GPU to finish.
 */
$opId = $module->launchAsync('heavy_math', args: [$data, $rows, $cols], config: $config);
// --- 4. Concurrent CPU Processing ---

/**
 * While the GPU is crunching numbers, PHP is free to perform other tasks.
 * This is perfect for I/O bound operations or preparing other datasets.
 */
echo "GPU is processing heavy math in the background...\n";

while (!$module->isFinished($opId)) {
    // Perform some CPU work here
    usleep(1000); // Simulate other logic
    echo "PHP is still free to run other code...\n";

    // Optional: Check status
    $status = $module->getAsyncStatus($opId);
}
// --- 5. Final Synchronization ---

/**
 * Ensures all pending GPU operations are complete before moving forward.
 * Required before calling toArray() to ensure data integrity.
 */
$module->sync();
$result = $data->toArray();
echo "Computation finished successfully.\n";