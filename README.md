# PHP GPU Tensors

<p align="center">
    <img src="art/php-gpu-tensors.jpg" alt="PHP GPU Tensors: high-performance computing with PHP and NVIDIA GPUs" width="350">
</p>

<p align="center">
    <strong>GPU tensors, runtime-compiled CUDA kernels and fused tensor expressions for PHP.<br>No Python runtime required.</strong>
</p>

<p align="center">
    <a href="https://packagist.org/packages/lcmialichi/php-gpu-tensors"><img alt="Packagist" src="https://img.shields.io/packagist/v/lcmialichi/php-gpu-tensors?include_prereleases&label=packagist"></a>
    <img alt="PHP 8.1 to 8.5" src="https://img.shields.io/badge/PHP-8.1%20%E2%80%93%208.5-777BB4">
    <img alt="NVIDIA CUDA" src="https://img.shields.io/badge/NVIDIA-CUDA-76B900">
    <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/github/license/lcmialichi/php-gpu-tensors"></a>
</p>

PHP GPU Tensors is a native extension for numerical workloads on NVIDIA GPUs.
It provides GPU-resident arrays, tensor operations, custom CUDA kernels,
optional expression fusion, reverse-mode automatic differentiation and
experimental machine-learning building blocks.

**Status:** beta · Linux · PHP 8.1-8.5 · NTS and ZTS · NVIDIA GPU required.
The documented package baseline is `0.1.0-beta.5`; this source checkout may
contain newer features.

This is a low-level GPU library, not a replacement for a complete machine
learning framework. It has no CPU fallback. GPU execution requires a compatible
NVIDIA driver and a visible GPU.

**Contents:** [Quick start](#quick-start) · [Results](#results) ·
[Install](#install) · [`CudaArray`](#gpu-tensors-with-cudaarray) ·
[Autograd and optimizers](#autograd-and-optimizers) ·
[Custom kernels](#custom-cuda-kernels) · [Fusion](#optional-kernel-fusion) ·
[Streams and CUDA Graphs](#streams-asynchronous-execution-and-cuda-graphs) ·
[Matmul and reductions](#accelerated-kernels-and-backend-diagnostics) ·
[CNN inference](#cnn-inference-with-optional-cudnn) ·
[Training example](#training-example) ·
[API and limits](#api-and-limits) · [Validation](#validation-status) ·
[Contributing](#contribute) · [Benchmarks](#benchmarks)

## Quick start

Tensor operations run on the GPU by default. Fusion is opt-in; use it when a
compatible expression is repeated and worth compiling.

```php
use Cuda\CudaArray;
use Cuda\Fusion;

$a = CudaArray::ones([4]);
$b = CudaArray::full([4], 2.0);
$c = CudaArray::full([4], 3.0);

// Eager execution: separate operation calls.
$eager = $a + $b * $c;

// Fusion: capture once, then replay a compiled plan.
$plan = Fusion::compile(fn($x, $y, $z) => $x + $y * $z, inputs: [$a, $b, $c]);
$fused = $plan->run($a, $b, $c);

print_r($fused->toArray()); // [7, 7, 7, 7]
```

`CudaArray` keeps its data on the GPU. Calls such as `toArray()`, `toBuffer()`,
`toHost()` and `item()` explicitly transfer results to the host.

## Results

The following Fusion/eager comparison was measured on PHP 8.3 NTS with a
GeForce MX570 A (4 GB, compute capability 8.6), driver 12.6 and CUDA runtime
12.3. It is a small-model end-to-end training example, not peak GPU throughput
or a general-purpose benchmark.

| Mode | Time per step | Patches per second | 1,280 steps |
| --- | ---: | ---: | ---: |
| Eager | 4.08 ms | 125,444 | 5.22 s |
| Fusion replay | 0.66 ms | 776,251 | 0.84 s |

Both modes produced the same reported accuracy (77.29%), ROC AUC (0.853) and
confusion matrix. The workload was a small MLP with 256 hidden units and
handcrafted PatchCamelyon features; the result illustrates reduced per-step
overhead for that workload and should not be generalized to other GPUs or
models. See [benchmarks](#benchmarks) for repeatable measurements.

## Install

### Requirements

| Requirement | Details |
| --- | --- |
| Operating system | Linux |
| PHP | 8.1-8.5 development headers, `phpize` and `php-config`; NTS or ZTS |
| Build toolchain | CUDA Toolkit with NVRTC, C/C++ compiler, `make` and `autoconf` |
| Runtime | NVIDIA driver providing `libcuda.so.1` and a CUDA-capable GPU |

The CUDA Toolkit is required to build. The host NVIDIA driver is required at
runtime.

### Build from source

```bash
git clone https://github.com/lcmialichi/php-gpu-tensors.git
cd php-gpu-tensors
./compile.sh
./run-tests.sh --require-gpu
php -n -d extension=./cuda_build-8.3/modules/cuda.so examples/01_basics_cuda_array.php
```

The module is written to `cuda_build-<PHP major.minor>/modules/cuda.so`;
replace `8.3` with the PHP version selected by `php-config`. To install the
extension and its INI configuration, use `./compile.sh --install`.

Choose a PHP ABI explicitly if multiple versions are installed:

```bash
PHP_BIN=php8.3 PHPIZE=phpize8.3 PHP_CONFIG=php-config8.3 ./compile.sh
PHP_BIN=php8.3 PHP_CONFIG=php-config8.3 ./run-tests.sh --require-gpu
```

Build options:

- Set `CUDA_HOME` if the Toolkit is not installed at `/usr/local/cuda`.
- Set `CUDA_ARCH=sm_86` when cross-building for a specific GPU architecture.
- Set `CUDA_USE_CUBLAS=no` to disable cuBLAS and use the built-in matrix
  kernels. Set `CUDA_USE_CUBLASLT=no` to disable cuBLASLt while retaining
  cuBLAS. Both are enabled by default when headers and libraries are available.
- Set `CUDA_USE_CUDNN=auto` (default), `yes` (require cuDNN), or `no`.
  `CUDA_CUDNN_ROOT` selects a cuDNN prefix; runtime libraries must also be
  visible to the dynamic loader.

### Install with PIE

```bash
pie install lcmialichi/php-gpu-tensors:0.1.0-beta.5
```

PIE builds the extension for the selected PHP installation; it does not install
the CUDA Toolkit or NVIDIA driver. Those must be available on the system.

### Build and test with Docker

With the NVIDIA Container Toolkit and a working host driver:

The development image defaults to CUDA 12.6.3 with cuDNN on Ubuntu 22.04, and
the container requires cuDNN during configuration. Use `--build` to ensure the
image matches the configured CUDA/cuDNN version:

```bash
docker compose run --build --rm php_cuda_dev bash -lc './compile.sh && ./run-tests.sh --require-gpu'
```

Set `CUDA_IMAGE` to select another CUDA base image. Set `CUDA_USE_CUDNN=no` only
when intentionally building without cuDNN.

### Run tests

```bash
./run-tests.sh                 # CPU-side C tests and PHP tests
./run-tests.sh --cpu-only      # Host-side tests; does not validate CUDA
./run-tests.sh --require-gpu   # Fail if no GPU is visible
```

`--require-gpu` prevents GPU tests skipped due to an unavailable GPU from being
mistaken for successful GPU validation.

## GPU tensors with `CudaArray`

`CudaArray` stores tensor data on the GPU. The constructor accepts rectangular
nested PHP arrays; `fromFlatArray()` accepts flat values plus an explicit shape.
The default dtype is `float32`. Storage, transfer and elementwise arithmetic
also support `float16` and `bfloat16`; low-precision elementwise operations
compute in `float32` and round to the promoted output dtype.

```php
use Cuda\CudaArray;

$input = new CudaArray([[1, 2], [3, 4]], 'float32');
$weights = CudaArray::ones([2, 2]);
$output = $input->add($weights)->multiply(2);
$columnMeans = $input->mean(0);

print_r($output->toArray()); // [[4, 6], [8, 10]]
echo $output->dtype();       // float32
```

### Creation and data transfer

- `new CudaArray($nested, $dtype)` validates and uploads rectangular numeric
  arrays. Ragged arrays, nonnumeric values and excessive nesting are rejected.
- `fromFlatArray($values, $shape, $dtype)` checks that value count matches shape.
- `fromBuffer($bytes, $shape, $dtype)` uploads packed row-major bytes.
- `fromFile($path, $shape, $dtype)` reads raw tensor bytes.
- `fromNpy($path)` imports C-order, little-endian numeric and boolean `.npy`
  arrays. Fortran-order and big-endian files are not supported.
- `zeros()`, `ones()`, `full()`, `eye()` and `identity()` create GPU tensors;
  `arange()` and `linspace()` generate numeric sequences directly on the GPU.
  Integer-only `arange()` calls default to `int64`; floating-point sequences
  default to `float64`, and `linspace()` includes its endpoint by default.
  `zerosLike()`, `onesLike()` and `fullLike()` use a prototype's shape and
  preserve its dtype unless one is specified.
- `rand()` generates uniform values in the requested range and `randn()`
  generates standard-normal values. Calls without a seed generate fresh
  values; passing an integer seed makes the result reproducible.
- `toArray()` copies values into nested PHP arrays; `toBuffer()` returns packed
  bytes without expanding every value into a PHP scalar.
- `toHost()` returns a CPU `ContiguousArray` (`Cuda\HostArray` is an alias).
  `ContiguousArray` supports packed buffers and optional pinned host memory.
- `item()` transfers one value from a tensor containing exactly one element.

`float16` and `bfloat16` values can be created from PHP numeric arrays, packed
buffers, fill/like factories, `eye()` and `randn()`. They can be exported with
`toArray()`, `toBuffer()` and `toHost()`, and safely widened to `float32` or
`float64` with `astype()`.

For large inputs and outputs, packed data avoids much of the per-value PHP
conversion cost. Keep intermediate tensors on the GPU when possible.

### Operations

`CudaArray` supports:

- **Arithmetic:** `add()`, `subtract()`, `multiply()`, `divide()`, `power()`,
  scalar arithmetic, and PHP arithmetic operators.
- **Elementwise math:** `sqrt()`, `exp()`, `expm1()`, `exp2()`, `log()`,
  `log1p()`, `log2()`, `log10()`, `sin()`, `cos()`, `tan()`, `abs()`, `neg()`,
  `asin()`, `acos()`, `atan()`, `sinh()`, `cosh()`, `tanh()`, `floor()`,
  `ceil()` and `round()`. `expm1()` and `log1p()` use numerically stable CUDA
  functions near zero.
- **Normalization:** `softmax()` computes stable probabilities along an axis
  (last axis by default), preserving `float32`/`float64`, widening low-
  precision inputs to `float32`, and promoting integer and boolean inputs to
  `float64`. `layerNorm()` normalizes across an explicit axis list; optional
  `scale` and `bias` shapes follow that list's order.
- **Comparisons and selection:** `gt()`, `lt()`, `eq()`, `ne()`, `ge()`,
  `le()`, `where()`, `maximum()`, `minimum()` and `clamp()`.
- **Matrix products:** `matmul()`, including supported batched and broadcast
  layouts. Matching `float32`, `float16` and `bfloat16` inputs are supported;
  low-precision products accumulate in `float32` and return the input dtype.
- **Reductions:** `sum()`, `mean()`, `min()`, `max()`, `prod()`, `argMin()`,
  `argMax()`, `all()`, `any()`, `var()` and `std()`. Reductions can operate
  globally or on an axis where supported.
- **Dtype conversion:** `astype()` for safe conversions.
- **Inspection:** `getShape()`, `getSize()`, `getStrides()`, `getNdims()` and
  `dtype()`.

Tensor arithmetic follows broadcasting rules and dtype promotion. `mean()`
returns `float32` for `float32` input and `float64` for `float64`, integer and
boolean input. `all()` and `any()` return boolean tensors. `where()` broadcasts
its condition and value tensors, treats nonzero condition values as true, and
requires the two value tensors to have the same dtype. `var()` and `std()` take
an optional axis and correction; correction defaults to zero. `layerNorm()`
uses population variance and requires a non-empty list of distinct axes and a
finite positive epsilon. Optional floating-point `scale` and `bias` tensors
must match the normalized dimensions in axis-list order. It preserves
`float32`/`float64`, promotes integer and boolean input to `float64`, and uses
`float64` when either affine parameter is `float64`.

Low-precision `matmul()` uses cuBLAS for eligible 2D workloads and a tiled CUDA
kernel otherwise, including batched layouts. `float16` and `bfloat16` matmul
outputs retain the operand dtype after `float32` accumulation. Elementwise
arithmetic, unary math and comparisons compute in `float32`, then round to the
promoted output dtype. Low-precision inputs can also serve as `where()`
conditions, but low-precision `where()` values, reductions and indexed
operations remain unsupported and report an explicit error. The global
`cuda_set_matmul_precision()` setting applies to `float32` matmul only.
`requiresGrad()` supports low-precision matmul and elementwise arithmetic
paths; returned leaf gradients use `float32`, while low-precision operation
outputs and explicit backward seeds retain their dtype. Fusion supports
low-precision elementwise expressions and the matmul backward path; widening
FP16/BF16 casts to `float32` use a native boundary, which is not compatible
with asynchronous replay or CUDA Graph capture.
`softmax()` and `layerNorm()` widen low-precision inputs to `float32` and
return `float32` results.

### Shape operations, views and indexing

`reshape()`, `flatten()`, `transpose()`, `swapaxes()`, `moveaxis()`,
`squeeze()`, `unsqueeze()` and `broadcastTo()` manipulate tensor shape.
`slice()` provides Python-style positive-step slicing. Views can share storage
with their parent; use `copy()` for independent contiguous storage. Reshape
requires a contiguous input. `diagonal()`, `repeat()`, `tile()`, `flip()` and
`roll()` materialize contiguous GPU results and support float32/float64
autograd. Fusion captures their index mappings directly in generated kernels,
including expressions that combine transforms with elementwise operations.
Backward propagation inside Fusion uses a native GPU scatter-add boundary.
`repeat()` currently accepts one scalar repeat count rather than per-element
counts.

```php
$row = new CudaArray([[1.0, 2.0, 3.0]]);
$batch = $row->broadcastTo([2, 3]);
$columns = $batch->slice(':, 1:3');
$flat = $columns->flatten();
```

`slice()` uses exclusive range stops, supports negative indices and bounds, and
accepts a comma-separated expression or per-axis selectors. Integer selectors
remove an axis; ranges retain it. `null`, `':'`, omitted selectors and a
zero-argument `slice()` select the full extent. Steps must be positive.
Ellipsis, new axes, index lists and executable expressions are not supported.
The older `__invoke()` and `[]` selection syntax remains available and retains
its inclusive range behavior.

`concat()` joins tensors along an axis. `gather($indices, $axis)` reads values
using an `int32` index tensor. `scatterAdd($indices, $updates, $axis)` returns a
new tensor with updates added at indexed positions; duplicate indices
accumulate. Gather supports tensor dtypes; `scatterAdd()` currently supports
`float32` and `float64`. Both are eager-only and reject Fusion capture.

Empty dimensions are supported for many operations and transfers. For example,
sum and product over an empty axis return 0 and 1, respectively; mean returns
NaN. Min/max and arg reductions reject empty reduced axes because they have no
identity or index. A fully indexed scalar has shape `[]` and contains one
element.

### Serialization

`CudaArray` supports PHP serialization. Empty materialized tensors can also be
serialized. Serialization transfers tensor contents, so it should not be used
as a substitute for keeping frequently reused data on the GPU.

## Autograd and optimizers

Reverse-mode automatic differentiation is opt-in. Mark floating-point leaf
tensors with `requiresGrad()`, compute a loss, and call `backward()`:

```php
$input = CudaArray::fromFlatArray([1, 2, 3, 4], [2, 2])->requiresGrad();
$weight = CudaArray::fromFlatArray([1, 0, 0, 1], [2, 2])->requiresGrad();
$loss = $input->matmul($weight)->mean();
$loss->backward();

$gradient = $weight->grad();
$weight->zeroGrad();
```

Gradients accumulate until `zeroGrad()` is called. `detach()` returns a
shared-storage view disconnected from gradient history. Non-scalar outputs
require an explicit seed with matching shape and dtype. Graphs are released
after a successful backward pass unless `retainGraph: true` is passed. If a
tensor saved for backward is modified in place, backward rejects the stale
graph.

Supported gradient rules include arithmetic and broadcasting, batched matmul,
sum/mean/product and min/max reductions, `where()`, floating-point casts,
reshape, transpose, slicing, concatenation and common unary math. Ties in
min/max reductions share the gradient. Comparisons and `where()` conditions do
not receive gradients; unsupported rules fail explicitly.

Backward can run inside a Fusion capture. Mark inputs with `requiresGrad()`
before compiling, then return gradients or use them for functional updates.

`Cuda\Optimizer` provides functional SGD and AdamW. `step()` returns new
parameters and state rather than mutating its inputs, so optimizer math can be
captured by Fusion. State is explicit: SGD keeps velocity tensors; AdamW keeps
first and second moments plus beta powers. Parameters, gradients and state
preserve compatible shapes and dtypes. A `weightDecayMask` can disable decay
for selected parameters.

## Custom CUDA kernels

`Cuda\Compiler` uses NVRTC to compile CUDA C++ source to PTX. Register a kernel
and its argument metadata, compile it, initialize the resulting
`Cuda\CompiledModule`, then launch with a grid and block configuration:

```php
use Cuda\Compiler;

$source = <<<'CUDA'
extern "C" __global__ void scale(float *data, int factor, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) data[index] *= factor;
}
CUDA;

$compiler = new Compiler(source: $source);
$compiler->kernel('scale', [
    ['name' => 'data', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'factor', 'dtype' => 'int32'],
    ['name' => 'count', 'dtype' => 'int32'],
]);
$module = $compiler->compile();
$module->initialize();
$module->launch('scale', args: [$data, 3, 512],
    config: ['block' => [256, 1, 1], 'grid' => [2, 1, 1]]);
```

`launch()` is synchronous. `launchAsync()` and `launchAsyncBatch()` submit
asynchronous work; use `sync()`, `wait()` or completion queries to collect it.
Keep arrays alive until their operations finish. Compiled modules expose kernel
metadata, PTX, statistics and pending-operation controls, and support
serialization and saving compiled artifacts. See the
[synchronous JIT example](examples/04_custom_jit_kernels.php),
[async example](examples/05_jit_async_execution.php) and
[serialization example](examples/06_serialize_compiled_module.php).

## Optional kernel fusion

Fusion is opt-in; eager execution remains the default. Capture a callback once
and either run it immediately or compile a reusable plan:

```php
use Cuda\Fusion;

$result = Fusion::run(fn() => $a + $b * $c);

$graph = Fusion::compile(
    fn($x, $y, $z) => $x + $y * $z,
    inputs: [$a, $b, $c]
);
$result = $graph->run($a, $b, $c);
print_r($graph->getStats());
print_r($graph->getPlan());
echo $graph->getSource();
```

The callback runs during capture/compilation, not on each compiled replay.
Compatible input shapes, dtypes and strides are required; input values and
pointers may change between replays. Captured closure tensors stay alive with
the graph, so use callback parameters for replaceable inputs.

Fusion can combine elementwise arithmetic with implicit broadcasting, scalar
operations, unary math, comparisons, `where()`, safe `astype()` conversions,
and reshape, transpose, slice and explicit `broadcastTo()` views. It can build
generated kernels around native boundaries:

- **Native boundaries:** reductions, `matmul()` (including matching
  `float32`, `float16` and `bfloat16` inputs), powers, `concat()`, `gather()`
  and `scatterAdd()` use existing kernels. Indexed operations validate indices
  synchronously; `concat()` and indexed operations are not compatible with
  asynchronous replay or CUDA Graph capture.
- **Not captured:** cuDNN inference and operations requiring CPU reads are not
  captured. Other unsupported operations materialize their inputs during
  scoped `Fusion::run()`; they are rejected when they would require reading
  placeholder data during `Fusion::compile()`.
- **Inspect and cache:** `getStats()`, `getPlan()` and `getSource()` describe
  generated work. PTX is cached per PHP request/thread; use
  `Fusion::getCacheStats()` and `Fusion::clearCache()` to inspect or clear it.
  The cache uses LRU eviction at 16 entries or 16 MiB.

The planner can split large expressions into multiple kernels, deduplicate
eligible repeated nodes and avoid materializing intermediates when operations
can be fused. One callback is not guaranteed to become one kernel. The
[Fusion example](examples/07_fusion_graph.php) covers capture, replay,
diagnostics and benchmarking.

## Streams, asynchronous execution and CUDA Graphs

Compatible compiled plans use private nonblocking CUDA streams. A plan can be
submitted with `runAsync()` and returns a `Cuda\FusionExecution`:

```php
$graph = Fusion::compile(
    fn($x, $y) => ($x + $y)->sqrt(),
    inputs: [$a, $b],
    cudaGraph: true
);
$pending = $graph->runAsync($a, $b);
$finished = $pending->isFinished();
$result = $pending->wait();
```

Streams and asynchronous replay support generated kernels, matmul and reduction
boundaries. CUDA Graph replay is optional and supports compatible plans; plans
with native boundaries recapture the stream sequence on replay and may have
extra host overhead. Benchmark them against ordinary stream replay for the
target workload. Power boundaries currently require synchronous execution.

Stream plans allow concurrent asynchronous executions. A CUDA Graph executable
allows only one outstanding replay; collect it with `wait()` before replaying
again. Inputs and captured tensors must remain alive until completion is
collected. Mutations, custom kernel launches and device reset are blocked while
an execution is outstanding. Destroying a pending execution synchronizes before
releasing resources.

`getStats()` reports the backend (`cuda-graph`, `stream` or `native`) and
replay/resource counters. Check `cudaGraphCompatible` and
`cudaGraphIncompatibility` before choosing a replay mode.
`setProfiling(true)` enables synchronous replay phase timings such as bind,
execute and collect time; these are end-to-end phases, not isolated CUDA kernel
timings. See the
[stream and CUDA Graph example](examples/07_fusion_graph.php).

## Accelerated kernels and backend diagnostics

The extension uses a hybrid execution path:

- Custom CUDA kernels and Fusion for elementwise work.
- CUB for eligible large reductions; coalesced kernels are available for
  non-last-axis reductions.
- cuBLAS/cuBLASLt for compatible matrix products, with built-in kernels for
  small products and fallback layouts.
- Optional cuDNN for CNN inference.

Reduction scratch is cached by device and stream. Arg reductions retain first
index tie behavior and ignore NaNs. Floating-point reduction order can differ
between parallel algorithms, changing low-order bits.

### Matrix multiplication precision and tuning

Matmul uses strict FP32 by default. On supported compute capability 8.0+
hardware with cuBLAS, TF32 Tensor Core math can be explicitly enabled:

```php
cuda_set_matmul_precision('tf32');
$result = $left->matmul($right);
cuda_set_matmul_precision('fp32-strict');
```

Precision policy is per request/thread and resets to strict FP32 at request
shutdown. It affects eligible cuBLAS/cuBLASLt GEMMs, not tensor dtypes, dot
products or built-in fallback kernels. Set precision before compiling a Fusion
plan containing `matmul()`; the plan captures the selected mode.

For repeatedly used compatible large 2D `float32` shapes, `cuda_tune_matmul()`
benchmarks zero-workspace cuBLASLt candidates and caches the fastest matching
algorithm for the request. Tuning synchronizes the device and runs trial GEMMs;
it is an explicit setup step, not part of normal `matmul()` execution.
It currently requires cuBLASLt, 2D `float32` operands, a BLAS-compatible layout
and at least 10 million multiply-accumulate operations.

```php
$tuning = cuda_tune_matmul($left, $right);
print_r($tuning['candidates']);
$product = $left->matmul($right);
```

`cuda_get_backend_info()` reports compiled capabilities, the last matmul
backend, selected precision, BLAS status and per-request submission counters.
Counters track submissions rather than completed GPU execution or CUDA Graph
replays.

### Kernel benchmark

Run the benchmark with the extension loaded:

```bash
php -n -d memory_limit=-1 -d extension=./cuda_build-8.3/modules/cuda.so \
  examples/10_kernel_benchmark.php \
  --binary=./cuda_build-8.3/modules/cuda.so --output=kernel-results.json
```

It checks numerical results and measures API wall-clock latency, including
allocation and final GPU synchronization. Cases cover square, skinny, MLP,
transposed, strided, batched and broadcast matmul; reductions; eager/Fusion/CUDA
Graph execution; and repeated same-shape allocation. It is not an isolated
CUDA-event kernel benchmark. Adjust `--warmups`, `--samples` and
`--pool-iterations`; use `--precision=tf32` or `--tune-matmul` to test those
explicit matmul options.

## CNN inference with optional cuDNN

`Cuda\NN` provides static inference methods when the extension is built with
CUDA. Inputs are initialized, contiguous `float32` tensors in NCHW format;
filters use OIHW. cuDNN is preferred when available and supports the operation;
otherwise the extension uses native CUDA kernels. There is no CPU fallback.
NN operations do not require a backend preflight. They do not silently cast or
pack views, participate in autograd or participate in Fusion capture.

```php
use Cuda\NN;

$features = NN::conv2d($input, $weights, $bias, padding: [1, 1]);
$pooled = NN::pool2d($features, window: [2, 2], stride: [2, 2]);
$probabilities = NN::softmax($pooled);
```

- `conv2d()` supports stride, padding, dilation, groups and optional per-output
  channel bias. It performs cross-correlation.
- `pool2d()` supports deterministic max pooling and average pooling that
  excludes padded cells.
- `softmax()` computes channel softmax.

Both backends preserve the same FP32 inference semantics. The cuDNN
convolution path uses deterministic FMA algorithms and a reusable bounded
workspace; native CUDA kernels cover convolution, pooling and softmax when
cuDNN is absent or reports a recoverable unsupported configuration. Calls
complete on the default stream. Backpropagation, mixed precision, asynchronous
CNN plans and convolution/activation graph fusion are not currently included.

## Training example

[`examples/08_gpu_classifier.php`](examples/08_gpu_classifier.php) trains an
MLP on MNIST, Fashion-MNIST or a custom CSV dataset. It demonstrates PHP data
preparation, packed GPU batches, autograd, SGD/AdamW and Fusion replay without
requiring custom CUDA source:

```bash
php -n -d extension=./cuda_build-8.3/modules/cuda.so \
  examples/08_gpu_classifier.php --dataset=mnist --epochs=40 \
  --batch-size=128 --optimizer=adam
```

The example's `$config` array controls dataset, hidden layers, epochs, batch
size, optimizer and learning rate; command-line options can override it.
`--profile` reports Fusion plan and resource data, `--predict-index=N` runs
inference for one sample, `--no-save` disables saving and `--load` evaluates a
saved model. Compare validation metrics as well as throughput when changing
batch size or learning rate.

## API and limits

| API | Purpose |
| --- | --- |
| `Cuda\CudaArray` | GPU storage, arithmetic, reductions, views, indexing, data transfer and opt-in autograd |
| `Cuda\HostArray` / `Cuda\ContiguousArray` | CPU storage, packed buffers, optional pinned memory and `toGpu()` |
| `Cuda\Optimizer` | Functional SGD and AdamW updates with explicit state |
| `Cuda\Fusion` / `Cuda\FusionGraph` | Capture, compiled replay, PTX cache and plan diagnostics |
| `Cuda\FusionExecution` | Pending asynchronous Fusion execution and result collection |
| `Cuda\Compiler` / `Cuda\CompiledModule` | NVRTC compilation, kernel launch, async operations and PTX |
| `Cuda\NN` | CUDA CNN inference with optional cuDNN acceleration |
| `cuda_*()` functions | Device management, memory, synchronization, backend diagnostics and matmul settings |
| `Cuda\Exception` | Base class for runtime, argument, allocation and compilation errors |

Annotated signatures are in [class stubs](stubs/cuda.stub.php) and
[device-function stubs](stubs/cuda_methods.stub.php). Runnable examples are
listed in the [examples guide](examples/README.md).

Current limits:

- Linux and NVIDIA GPUs only; there is no CPU fallback.
- Autograd supports a documented subset of operations and floating-point
  tensors; unsupported gradient rules fail explicitly.
- `astype()` allows safe conversions; autograd tracks floating-point casts.
- Fusion matmul supports matching `float32`, `float16` and `bfloat16` operands;
  low-precision products accumulate in `float32` and retain the input dtype.
  Low-precision elementwise expressions compute in `float32` and round at each
  node to preserve eager semantics. Widening FP16/BF16 casts to `float32` use a
  native boundary and are not compatible with asynchronous replay or CUDA
  Graph capture; integer-output scalar powers remain native boundaries.
- `gather()` and `scatterAdd()` support synchronous Fusion execution but not
  asynchronous replay or CUDA Graph capture. `scatterAdd()` supports
  `float32` and `float64`.
- cuDNN is optional. Its CNN inference API rejects Fusion capture and
  gradient-tracked inputs; it does not participate in autograd.

## Validation status

Build and CPU tests do not substitute for GPU runtime tests. The source has been
validated on the following configurations:

| PHP / mode | GPU and configuration | Validation |
| --- | --- | --- |
| 8.1, Docker development image | GeForce MX570 A | Current source: 45 tests passed; optional cuDNN test skipped; MLP smoke test completed |
| 8.3 NTS | GeForce MX570 A, cuDNN 8.9.7 | Beta.5: all 44 GPU PHPT tests passed, including CNN inference |
| 8.3 and 8.5 NTS | GeForce MX570 A, without cuDNN | Beta.5: 43 GPU PHPT tests passed per runtime; optional cuDNN test skipped |
| 8.3 NTS | GeForce MX570 A, without cuBLAS/cuDNN | Beta.5: 43 GPU PHPT tests passed; optional cuDNN test skipped |
| 8.5 NTS/ZTS and 8.1 ZTS | RTX A2000 | Earlier tensor/JIT validation; not the latest Fusion changes |
| PHP 8.1-8.5, NTS/ZTS | CI without GPU | Build, CPU and API matrix; GPU execution is not covered |

Reports for other PHP, driver, CUDA or GPU versions are welcome. Include those
versions and the result of `./run-tests.sh --require-gpu` in an
[issue](https://github.com/lcmialichi/php-gpu-tensors/issues).

## Contribute

Contributions can be code, tests, documentation, examples or reports from other
PHP/CUDA/GPU combinations. See [open issues](https://github.com/lcmialichi/php-gpu-tensors/issues),
[bug reports](https://github.com/lcmialichi/php-gpu-tensors/issues/new?template=bug_report.yml),
[feature requests](https://github.com/lcmialichi/php-gpu-tensors/issues/new?template=feature_request.yml),
the [contribution guide](CONTRIBUTING.md) and [roadmap](ROADMAP.md).
Documentation, examples and host-side tests can be worked on without an NVIDIA
GPU.

## Benchmarks

The benchmark suite is maintained in the separate
[PHP GPU Tensors Benchmarks repository](https://github.com/lcmialichi/php-gpu-tensors-benchmarks).
It contains focused matmul, import and Fusion runs, JSON/HTML reports and
published workload reports, including [PHP 8.5 NTS vs ZTS results](https://github.com/lcmialichi/php-gpu-tensors-benchmarks/blob/main/published-reports/php85-nts-vs-zts/README.md).

## License

Licensed under the [MIT License](LICENSE).
