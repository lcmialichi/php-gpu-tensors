<?php

declare(strict_types=1);

use PhpTensors\Compiler;
use PhpTensors\CudaArray;

/**
 * Custom CUDA Kernels (JIT)
 */

$source = <<<'CUDA'
extern "C" __global__ void v_scale(float *data, int factor, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}
CUDA;

$compiler = new Compiler(source: $source);
$compiler->kernel('v_scale', [
    ['name' => 'data', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'factor', 'dtype' => 'int32'],
    ['name' => 'n', 'dtype' => 'int32'],
]);


// JIT: NVRTC compiles CUDA source -> PTX -> GPU Module
$module = $compiler->compile();
$module->initialize();

// --- Data & Execution ---

$size = 1_000_000;
$gpuData = CudaArray::ones([$size]);

// Define Parallelism Geometry (Grid/Block)
$launchConfig = [
    'block' => [256, 1, 1],
    'grid' => [(int) ceil($size / 256), 1, 1]
];

// Launch the custom kernel
$module->launch('v_scale', args: [$gpuData, 10, $size], config: $launchConfig);

var_dump($gpuData->toArray()[0]); // Expected: 10.0