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
| `08_gpu_classifier.php` | MNIST Training | One standalone file: data, MLP, autograd, AdamW, retained CUDA Graphs and held-out evaluation. |

`10_kernel_benchmark.php` measures square, skinny, MLP/classifier, transposed,
strided, batched, and broadcast matmul shapes alongside large reductions. It
checks outputs and records the dispatched backend per matmul case. It also
compares eager, Fusion, and CUDA Graph replay for elementwise chains and
reductions around a native boundary. It also measures repeated same-shape
allocation/release workloads to expose memory-pool reuse latency. See the
[benchmark guide](../docs/guide.md#kernel-benchmark) for command-line options; use `--precision=tf32` to measure the
opt-in Tensor Core mode. Use `--tune-matmul` to benchmark and cache eligible
cuBLASLt algorithms before timing the matrix workloads.

### Train MNIST: one standalone file

Run `php examples/08_gpu_classifier.php` from the repository root.
No configuration or command-line options are needed. The example contains
every step, including the data loader, and does not include `Support/` files:

1. Download and normalize MNIST; reserve 5,000 training images for validation.
2. Upload batches, with small random shifts for training only.
3. Predict with a fixed `784 -> 512 -> 256 -> 10` ReLU MLP.
4. Compute stable cross-entropy, call `backward()`, clip gradients and update
   weights with `Optimizer::adamW()`.
5. Keep the best validation weights and measure accuracy on the untouched test set.

The named functions (`predict`, `crossEntropy`, `trainBatch`, `evaluate`) are
defined in the same file, directly above the training loop. There is no
`eval()` or callback generation. `trainBatch()` expresses the actual math.
`trainingInputs()` and `compileTrainingPlan()` show the small adapter needed
because Fusion accepts tensors, not nested PHP optimizer-state arrays:
six parameters, six first moments, six second moments and two beta powers.
The compiled callback uses normal PHP variadic arguments.

`Fusion::compile(..., cudaGraph: true)` captures the step once per batch shape:
128 rows and the final partial batch of 88. Each subsequent `run()` binds the
current weights, AdamW state and learning rate and replays the retained CUDA
Graph; it does not invoke the PHP model/backward/optimizer callback again.
The four seeded training permutations stay on the GPU.
See [`07_fusion_graph.php`](07_fusion_graph.php) for more replay examples.

MNIST downloads automatically into `examples/training/data/mnist`; subsequent
runs reuse those files. The example deliberately has no automatic model saving,
profiling or inference modes. The advanced CSV/Fashion-MNIST, persistence and diagnostic
helpers remain in [`Support/ClassifierRunner.php`](Support/ClassifierRunner.php);
they are not part of this example's execution path.

The training recipe retains ReLU, He/Xavier initialization, AdamW at
0.001 with weight decay 0.01, gradient clipping at 5, two warmup epochs followed
by cosine decay, and four seeded permutations with shifts up to two pixels.
The short Box-Muller initializer deliberately preserves the original seeded
Gaussian weights; switching random-number generators changes the trained result.
There are 55,000 training, 5,000 validation and 10,000 test samples.
Validation loss selects the best weights, with early stopping after eight
epochs without improvement and a maximum of 40 epochs at batch size 128.
The header reports the total number of trainable parameters (weights and biases).
Each epoch reports its elapsed time in seconds, including validation and any
first-epoch plan compilation, but excluding initial dataset preparation.
Changing the model or training code can change accuracy.

The [training regression](../tests/classifier_training.phpt) compares the
standalone step's parameters and AdamW state against the previous numerical
implementation, including a partial batch, without downloading MNIST.

#### Measured training result

On PHP 8.1 NTS, CUDA 12.6.3 and an MX570 A (4 GB, driver 560.94, Docker WSL 2),
the old runner and this standalone example both completed 40 epochs:

| Result | Old retained-plan runner | Standalone CUDA Graph example |
| --- | --- | --- |
| Best epoch | 37 | 37 |
| Validation accuracy | 99.12% | 99.12% |
| Test accuracy (10,000 images) | 98.95% | 98.95% |
| Whole-run time | 38.0 s | 34.97 s |

All 40 epoch losses and validation accuracies matched exactly in that comparison.
The explicit ReLU, clipping and averaged-gradient expressions preserve the old
FP32 training path; replacing apparently equivalent expressions was not
numerically interchangeable in the full run.
The first standalone revision used `Fusion::run()` per batch and took 181.4 s.
Profiling 300 warmed steps found about 10.14 ms per scoped step with 300 PTX
cache hits and zero misses: cached PTX did not avoid rebuilding/loading the plan
and preparing execution buffers. Retained replay measured about 1.84 ms/step;
CUDA Graph replay measured about 1.52 ms/step in a separate 500-step comparison.
The final example retained exactly two plans and performed 17,200 CUDA Graph
launches over 40 epochs. The full run improved from 181.4 s to 34.97 s
(about 5.2x faster) without changing any epoch's losses or validation accuracy.
Running the actual CLI entrypoint separately also succeeded in 33.19 s, with
the same final accuracies and exit status 0.
These are local observations, not a dedicated benchmark or a universal speedup.

The core `CudaArray` API also includes shape views (`squeeze()`,
`unsqueeze()`, `broadcastTo()`), elementwise bounds (`maximum()`, `minimum()`,
`clamp()`), logical and statistical reductions (`all()`, `any()`, `var()`,
`std()`), and `item()` for reading one value as a PHP scalar. `gather()` and
`scatterAdd()` provide indexed reads and additive updates; they currently run
eagerly rather than inside Fusion. See the [usage guide](../docs/guide.md) for their dtype,
broadcasting, and indexing requirements.

## Execution Requirements

1. **NVIDIA Driver** & **CUDA Toolkit** installed.
2. **PHP GPU Tensors** built and loaded in PHP (module name: `phptensors`).
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