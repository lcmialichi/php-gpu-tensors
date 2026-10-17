<?php

declare(strict_types=1);

namespace App\Cuda;

use PhpTensors\Compiler;
use PhpTensors\CudaArray;
use PhpTensors\CompiledModule;

/**
 * High-Performance Vector Scaling using JIT Compilation and Caching.
 * * This script checks for a pre-compiled binary on disk to bypass the 
 * NVRTC compilation overhead.
 */

$cachePath = __DIR__ . '/cache/v_scale.pcu';

if (file_exists($cachePath)) {
    /** @var CompiledModule $module */
    echo "LOG: Loading pre-compiled module from disk...\n";
    $module = unserialize(file_get_contents($cachePath));
} else {
    echo "LOG: Cache miss. Initiating NVRTC compilation...\n";

    $src = <<<'CUDA'
extern "C" __global__ void v_scale(float *data, int factor, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}
CUDA;

    $compiler = new Compiler(source: $src);
    $parameters = [
        ['name' => 'data', 'type' => 'array', 'dtype' => 'float32'],
        ['name' => 'factor', 'dtype' => 'int32'],
        ['name' => 'n', 'dtype' => 'int32'],
    ];

    $compiler->kernel('v_scale', $parameters);

    // Generate the GPU Module
    $module = $compiler->compile();

    // Persist the serialized module to disk for future requests
    if (!is_dir(dirname($cachePath))) {
        mkdir(dirname($cachePath), 0755, true);
    }

    file_put_contents($cachePath, serialize($module));
    echo "LOG: Module successfully compiled and cached.\n";
}

/**
 * OPTIONAL: Explicit Initialization
 * * Calling $module->initialize() proactively uploads the PTX binary to the 
 * GPU VRAM. If omitted, the extension will automatically perform 
 * "Lazy Initialization" during the first kernel launch.
 * * Explicit calls are recommended for catching CUDA context errors early.
 * 
 *  @var CompiledModule $module
 */
$module->initialize();

// --- Data Preparation & Execution ---

$elementCount = 1_000_000;
$gpuBuffer = CudaArray::ones([$elementCount]);

// Launching the kernel. If not initialized yet, lazy-loading happens here.
$module->launch(
    'v_scale',
    args: [$gpuBuffer, 10, $elementCount],
    config: $module->autoGrid('v_scale', $gpuBuffer)
);

// Synchronize and verify results
$result = $gpuBuffer->toHost();
echo "Execution Complete. First Element: " . $result[0] . " (Expected: 10)\n";
