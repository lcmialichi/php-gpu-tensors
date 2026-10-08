# PHP GPU Tensors Examples

This directory contains functional implementations of the extension's API. The examples are organized by complexity, moving from high-level tensor abstractions to low-level JIT kernel compilation.

## Directory Overview

| File | Feature Demonstrated | Key Concepts |
| :--- | :--- | :--- |
| `01_basics_cuda_array.php` | Memory Management | VRAM allocation, Operator overloading. |
| `02_math_and_reductions.php` | Data Aggregation | Parallel math functions, tensor reductions (sum/mean/min/max/product). |
| `03_advanced_manipulation.php` | Tensor Geometry | Reshaping, Transposition, Broadcasting. |
| `04_custom_jit_kernels.php` | JIT Compilation | CUDA source strings, typed parameters, Grid/Block config. |
| `05_jit_async_execution.php` | Concurrency | Non-blocking execution, Op polling, Stream sync. |
| `06_serialize_compiled_module.php` | JIT Serialization | PHP Serialize CompiledModule object example |
| `07_fusion_graph.php` | Fusion and CUDA Graph | Scoped capture, compiled replay, private streams, async result and optional timing comparison |
| `08_gpu_classifier.php` | Neural Network Training | End-to-end MLP classifier, stored/replayed Fusion plans, AdamW/SGD optimizers, dataset orchestration (MNIST/CSV). |

`10_kernel_benchmark.php` measures square, skinny, MLP/classifier, transposed,
strided, batched, and broadcast matmul shapes alongside large reductions. It
checks outputs and records the dispatched backend per matmul case; use
`--precision=tf32` to measure the opt-in Tensor Core mode.

Example 08 keeps the entrypoint small. Model settings live in
`Support/ModelConfiguration.php`; network math, optimizer state, data handling,
and orchestration are documented functions under `Support/`. It compiles and
retains `Cuda\FusionGraph` plans by batch shape, then calls `run()` for each
step rather than recompiling in the training loop. `--profile` reports fused
kernel counts and native boundaries for the captured plans.

Forward, backward, and optimizer elementwise expressions are captured together.
`matmul()` and reductions remain native boundaries, with the supported
elementwise work fused around them; the example's self-tests compare eager and
compiled training steps and inspect the plan boundary counts.

## Execution Requirements

1. **NVIDIA Driver** & **CUDA Toolkit** installed.
2. **PHP GPU Tensors** built and loaded in PHP (module name: `cuda`).
3. An active NVIDIA GPU visible to the system.

To run any example:
```bash
php 01_basics_cuda_array.php
```

## Technical Notes
### Memory Lifecycle
Data in a ``CudaArray`` stays in GPU VRAM. The ``toArray()`` method is the explicit trigger for a Device-to-Host (D2H) memory transfer. Minimize these calls to maintain performance.

## JIT Process
Custom kernels defined in PHP classes undergo the following pipeline:
- Source registration: Passing CUDA C/C++ source strings and typed parameter metadata.
- Compilation: Generation of PTX (Parallel Thread Execution) code.
- Loading: Injection of the binary module into the current CUDA context

## Execution Geometry
For custom kernels (``04`` and ``05``), execution is defined by:
- Block: Number of threads per multiprocessor.
- Grid: Number of blocks launched. The total parallel threads = grid * block.