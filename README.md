# PHP Tensors

<p align="center">
    <img src="art/php-gpu-tensors.jpg" alt="PHP Tensors: tensor computing with PHP" width="350">
</p>

<p align="center">
    <strong>Tensor operations, runtime-compiled CUDA kernels and fused expressions for PHP.<br>No Python runtime required.</strong>
</p>

<p align="center">
    <a href="https://packagist.org/packages/lcmialichi/php-gpu-tensors"><img alt="Packagist" src="https://img.shields.io/packagist/v/lcmialichi/php-gpu-tensors?include_prereleases&label=packagist"></a>
    <img alt="PHP 8.1 to 8.5" src="https://img.shields.io/badge/PHP-8.1%20%E2%80%93%208.5-777BB4">
    <img alt="NVIDIA CUDA" src="https://img.shields.io/badge/NVIDIA-CUDA-76B900">
    <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/github/license/lcmialichi/php-gpu-tensors"></a>
</p>

PHP Tensors is a native PHP extension with an explicit CPU tensor and an
optional CUDA implementation. It provides GPU-resident arrays, tensor
operations, custom CUDA kernels, optional expression fusion, reverse-mode
automatic differentiation and experimental machine-learning building blocks.

**Status:** beta · Linux · PHP 8.1-8.5 · NTS and ZTS · CPU-only builds supported; CUDA is optional.
The documented package baseline is `0.1.0-beta.5`; this source checkout may
contain newer features.

This is a low-level tensor library, not a replacement for a complete machine
learning framework. CUDA operations require the CUDA Toolkit at build time and
a compatible NVIDIA driver at runtime; CPU-only builds need no CUDA installation.
Custom backends may implement the public PHP contracts.

**Contents:** [Quick start](#quick-start) · [Results](#results) ·
[Install](#install) · [`CudaArray`](#gpu-tensors-with-cudaarray) ·
[Backends](#backend-contracts-and-custom-backends) ·
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
use PhpTensors\CudaArray;
use PhpTensors\Fusion;

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

Measured on PHP 8.3 NTS with a GeForce MX570 A (4 GB, compute capability 8.6),
driver 12.6 and CUDA runtime 12.3, this small MLP training example compares
eager execution with Fusion replay:

| Mode | Time per step | Patches per second | 1,280 steps |
| --- | ---: | ---: | ---: |
| Eager | 4.08 ms | 125,444 | 5.22 s |
| Fusion replay | 0.66 ms | 776,251 | 0.84 s |

Both modes produced 77.29% accuracy, 0.853 ROC AUC and the same confusion
matrix. Results are specific to this 256-unit MLP and handcrafted
PatchCamelyon features; they are not peak-throughput or cross-GPU claims.
See [benchmarks](#benchmarks) for repeatable measurements.

## Install

### Requirements

| Requirement | Details |
| --- | --- |
| Operating system | Linux |
| PHP | 8.1-8.5 development headers, `phpize` and `php-config`; NTS or ZTS |
| Build toolchain | C/C++ compiler, `make`, `autoconf` and PHP development headers |
| Optional CUDA backend | CUDA Toolkit with NVRTC and development libraries |
| Optional GPU runtime | NVIDIA driver providing `libcuda.so.1` and a CUDA-capable GPU |

`compile.sh` builds CUDA when it detects a Toolkit; otherwise it builds the
CPU-only extension. CUDA runtime use additionally requires a visible GPU. Both
build modes support PHP NTS and ZTS.

### Build from source

```bash
git clone https://github.com/lcmialichi/php-gpu-tensors.git
cd php-gpu-tensors
./compile.sh --cpu-only
./run-tests.sh --cpu-only
php -n -d extension=./phptensors_build-8.3/modules/phptensors.so -r \
  '$a = new PhpTensors\HostArray([[1, 2], [3, 4]]); print_r($a->multiply(2)->toArray());'
```

Pass `--cpu-only` to `compile.sh` to force a CPU build even when CUDA is
installed. The default build mode is selected independently of runtime GPU
availability: a CUDA build can compile without a GPU, but CUDA operations
require a working NVIDIA driver and device.
Omit `--cpu-only` to select CUDA automatically when a Toolkit is detected; use
`./run-tests.sh --require-gpu` for the CUDA runtime suite.

The module is written to `phptensors_build-<PHP major.minor>/modules/phptensors.so`.
Replace `8.3` with the version selected by `php-config`; add `--install` to
install the extension and its INI file.

Choose a PHP ABI explicitly if multiple versions are installed:

```bash
PHP_BIN=php8.3 PHPIZE=phpize8.3 PHP_CONFIG=php-config8.3 ./compile.sh
PHP_BIN=php8.3 PHP_CONFIG=php-config8.3 ./run-tests.sh --require-gpu
```

Build options:

- `CUDA_HOME` selects the Toolkit prefix; `CUDA_ARCH` selects a target such as
  `sm_86`.
- `./compile.sh --cpu-only` skips CUDA detection and builds the CPU-only extension.
  `./configure --without-cuda` also selects a CPU-only build.
- `CUDA_USE_CUBLAS=no` disables cuBLAS; `CUDA_USE_CUBLASLT=no` disables cuBLASLt.
  Both default to enabled when available.
- `CUDA_USE_CUDNN=auto|yes|no` controls optional cuDNN detection;
  `CUDA_CUDNN_ROOT` selects its prefix. Runtime libraries must be loader-visible.
- The CUDA memory pool defaults to `3G`. Configure it with
  `phptensors.cuda.memory_pool_size`; the old `cuda.memory_size` key is accepted
  temporarily and emits a deprecation notice when set.

### Install with PIE

```bash
pie install lcmialichi/php-gpu-tensors:0.1.0-beta.5
```

PIE builds for the selected PHP installation. CUDA is optional; provide
`--with-cuda=/path/to/toolkit` when building the CUDA backend. A CPU-only
installation does not need the Toolkit or an NVIDIA driver.

### Build and test with Docker

With NVIDIA Container Toolkit and a working host driver, build and test in the
CUDA 12.6.3/cuDNN development container:

```bash
docker compose run --build --rm php_cuda_dev bash -lc './compile.sh && ./run-tests.sh --require-gpu'
```

Set `CUDA_IMAGE` to choose another base image. Use `CUDA_USE_CUDNN=no` only
when deliberately building without cuDNN.

### Run tests

```bash
./run-tests.sh                 # CPU-side C tests and PHP tests
./run-tests.sh --cpu-only      # CPU-only backend PHPT; does not validate CUDA
./run-tests.sh --require-gpu   # Fail if no GPU is visible
```

Use `--require-gpu` in CI to fail instead of silently skipping GPU tests when no
device is available.

## GPU tensors with `CudaArray`

`CudaArray` stores tensor data on the GPU. The constructor accepts rectangular
nested PHP arrays; `fromFlatArray()` accepts flat values plus an explicit shape.
The default dtype is `float32`. Storage, transfer and elementwise arithmetic
also support `float16` and `bfloat16`; low-precision elementwise operations
compute in `float32` and round to the promoted output dtype.

```php
use PhpTensors\CudaArray;

$input = new CudaArray([[1, 2], [3, 4]], 'float32');
$weights = CudaArray::ones([2, 2]);
$output = $input->add($weights)->multiply(2);
$columnMeans = $input->mean(0);

print_r($output->toArray()); // [[4, 6], [8, 10]]
echo $output->dtype();       // float32
```

### Creation and data transfer

- **Import:** `new CudaArray($nested, $dtype)`, `fromFlatArray()`,
  `fromBuffer()`, `fromFile()` and `fromNpy()`. Nested values must be
  rectangular and numeric. `.npy` import supports C-order, little-endian
  numeric and boolean arrays, but not Fortran-order or big-endian files.
- **Factories:** `zeros()`, `ones()`, `full()`, `eye()`, `identity()`,
  `arange()`, `linspace()`, `zerosLike()`, `onesLike()` and `fullLike()`.
  Integer-only `arange()` defaults to `int64`; floating sequences default to
  `float64`; `linspace()` includes the endpoint. Like-factories preserve the
  prototype dtype unless overridden.
- **Random:** `rand()` generates uniform values and `randn()` standard-normal
  values. Calls without a seed are fresh; an integer seed is reproducible.
- **Export:** `toArray()` creates nested PHP arrays; `toBuffer()` returns
  packed bytes; `toHost()` returns a `PhpTensors\HostArray`.
  `item()` transfers a single-element tensor's value.

`float16` and `bfloat16` values can be created from PHP numeric arrays, packed
buffers, fill/like factories, `eye()` and `randn()`. They can be exported with
`toArray()`, `toBuffer()` and `toHost()`, or safely widened with `astype()`.
Packed transfers avoid per-value PHP conversion; keep intermediates on GPU.

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
- **Reductions/scans:** `sum()`, `mean()`, `min()`, `max()`, `prod()`,
  `argMin()`, `argMax()`, `all()`, `any()`, `var()`, `std()`, `cumsum()` and
  `cumprod()`. Cumulative methods accept an axis; null flattens first.
- **Dtype conversion:** `astype()` for safe conversions.
- **Inspection:** `getShape()`, `getSize()`, `getStrides()`, `getNdims()` and
  `dtype()`.

Arithmetic uses broadcasting and dtype promotion. `mean()` returns `float32`
for `float32` input and `float64` for `float64`, integer or boolean input;
`all()` and `any()` return booleans. `where()` broadcasts its condition and
values, treats nonzero conditions as true and requires matching value dtypes.
`var()` and `std()` accept an optional axis and correction (default `0`).
`layerNorm()` uses population variance, distinct non-empty axes and a finite
positive epsilon; optional `scale` and `bias` match normalized dimensions in
axis-list order. It preserves `float32`/`float64`, promotes integer/boolean to
`float64`, and uses `float64` if either affine parameter is `float64`.
`cumsum()` and `cumprod()` use CUB scans, support strided inputs and promote
small integers and booleans to `int64`; `uint64` stays unsigned. FP16/BF16
reductions and cumulative scans promote to `float32`. Autograd supports
floating-point `cumsum()` and zero-safe `cumprod()` in eager execution and
synchronous Fusion.

Low-precision matmul accumulates in `float32` and returns the operand dtype,
using cuBLAS for eligible 2D workloads and tiled kernels otherwise. Elementwise
math and comparisons compute in `float32` and round to the promoted dtype.
Low-precision `where()` conditions and values are supported. `gather()` also
supports FP16/BF16. Value reductions accumulate in and return `float32`;
arg-reductions return `int32`, while `all()`/`any()` return booleans. Cumulative
scans also promote to `float32`. `scatterAdd()` promotes low-precision inputs
and updates to `float32`, returning FP32 results and gradients.
`cuda_set_matmul_precision()` only affects `float32` matmul. Low-precision
matmul and elementwise autograd are supported; leaf gradients use `float32`,
while operation outputs and explicit backward seeds retain their dtype.
`softmax()` and `layerNorm()` return `float32` for low-precision inputs.

### Shape operations, views and indexing

Shape methods include `reshape()`, `flatten()`, `transpose()`, `swapaxes()`,
`moveaxis()`, `squeeze()`, `unsqueeze()`, `broadcastTo()` and Python-style
positive-step `slice()`. Views may share storage; `copy()` creates independent
contiguous storage. `reshape()` requires contiguous input. `diagonal()`,
scalar-count `repeat()`, `tile()`, `flip()` and `roll()` materialize contiguous
results and support float32/float64 autograd. Fusion inlines their index
mappings, including chained transforms; its backward uses a native GPU
scatter-add boundary.
`repeat()` accepts one scalar count or a list of non-negative per-element
counts (one per flattened input element when axis is null, otherwise one per
element along that axis). Counts must fit the supported output dimension.
Per-element counts also support float32/float64 autograd in eager execution
and synchronous Fusion.

```php
$row = new CudaArray([[1.0, 2.0, 3.0]]);
$batch = $row->broadcastTo([2, 3]);
$columns = $batch->slice(':, 1:3');
$flat = $columns->flatten();
```

`slice()` uses exclusive stops, supports negative indices/bounds and accepts a
comma-separated expression or per-axis selectors. Integer selectors remove
axes; ranges retain them. `null`, `':'`, omitted selectors and zero-argument
`slice()` select the full extent. Steps must be positive. Ellipsis, new axes,
index lists and executable expressions are unsupported. Legacy `__invoke()`
and `[]` selection remains available with inclusive range stops.

`concat()` joins tensors along an axis. `gather($indices, $axis)` reads values
using an `int32` index tensor; `scatterAdd($indices, $updates, $axis)` returns
a new tensor and accumulates duplicate indices. `gather()` supports all tensor
dtypes. `scatterAdd()` supports `float32`/`float64`; FP16/BF16 values are
promoted to `float32`. Both run in synchronous Fusion, but not async replay or
CUDA Graph capture.

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

Gradients accumulate until `zeroGrad()`. `detach()` returns a shared-storage
view without gradient history. Non-scalar outputs need a matching seed.
Backward releases the graph by default; pass `retainGraph: true` to keep it.
In-place changes to saved tensors make backward fail rather than use stale data.

Gradient rules cover arithmetic/broadcasting, batched matmul, sum/mean/product
and min/max reductions, `where()`, floating-point casts, shape/view operations,
concatenation and common unary math. Min/max ties share gradients. Comparisons
and `where()` conditions have no gradients; unsupported rules fail explicitly.
Backward can run in Fusion; mark inputs with `requiresGrad()` before compiling.

`PhpTensors\Optimizer` implements functional SGD and AdamW: `step()` returns updated
parameters and explicit state, suitable for Fusion capture. SGD state contains
velocity; AdamW state contains first/second moments and beta powers.
`weightDecayMask` can exclude selected parameters from decay.

## Custom CUDA kernels

`PhpTensors\Compiler` compiles CUDA C++ to PTX with NVRTC. Register the kernel and
argument metadata, compile and initialize the module, then launch it:

```php
use PhpTensors\Compiler;

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

`launch()` is synchronous; `launchAsync()` and `launchAsyncBatch()` submit work
for collection with `sync()`, `wait()` or completion queries. Keep arrays alive
until completion. Modules expose metadata, PTX, statistics and pending-operation
controls, and can be serialized or saved. See the
[synchronous JIT example](examples/04_custom_jit_kernels.php),
[async example](examples/05_jit_async_execution.php) and
[serialization example](examples/06_serialize_compiled_module.php).

## Optional kernel fusion

Fusion is opt-in; eager execution remains the default. Capture a callback once
and either run it immediately or compile a reusable plan:

```php
use PhpTensors\Fusion;

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

The callback runs during capture, not on each replay. Input shapes, dtypes and
strides must remain compatible, but values and pointers may change. Closure
tensors stay alive with the graph; use callback parameters for replaceable
inputs.

Fusion combines elementwise arithmetic, implicit broadcasting, scalars, unary
math, comparisons, `where()`, safe casts and reshape/transpose/slice/
`broadcastTo()` views. Scalar `repeat()` plus `diagonal()`, `tile()`, `flip()`
and `roll()` are inlined as index mappings. Generated kernels can surround
native boundaries:

- **Native boundaries:** reductions, cumulative scans, per-element `repeat()`,
  `matmul()` (matching `float32`, `float16` or `bfloat16` inputs), powers,
  `concat()`, `gather()` and `scatterAdd()` use existing kernels. Contiguous
  flattened and last-axis scans support async replay and CUDA Graph capture;
  strided or other-axis scans remain synchronous. Indexed operations validate
  indices synchronously; per-element `repeat()`, `concat()` and indexed
  operations do not support async replay or CUDA Graph capture.
- `softmax()` and `layerNorm()` are capturable, but are currently decomposed
  into elementwise segments and separate native reduction boundaries rather
  than a single kernel.
- **Unsupported capture:** cuDNN and operations requiring CPU reads are not
  captured. Other unsupported operations may materialize inputs in scoped
  `Fusion::run()`, but fail in `Fusion::compile()` if they need placeholder data.
- **Diagnostics/cache:** `getStats()`, `getPlan()` and `getSource()` describe
  generated work. Per-request/thread PTX cache uses LRU eviction (16 entries
  or 16 MiB); inspect or clear it with `getCacheStats()` / `clearCache()`.

The planner can split large expressions, deduplicate eligible repeated nodes
and avoid intermediates; one callback is not necessarily one kernel. The
[Fusion example](examples/07_fusion_graph.php) covers capture, replay,
diagnostics and benchmarking.

## Streams, asynchronous execution and CUDA Graphs

Compatible compiled plans use private nonblocking CUDA streams. A plan can be
submitted with `runAsync()` and returns a `PhpTensors\FusionExecution`:

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

Async replay supports generated kernels, matmul and reduction boundaries.
Compatible CUDA Graph plans recapture sequences containing native boundaries,
which may add host overhead; power boundaries currently require synchronous
execution. Stream plans allow concurrent executions, but a graph executable
allows only one outstanding replay. Collect it with `wait()` before replaying.
Keep inputs alive until collection; mutations, custom launches and device reset
are blocked while work is pending. Destroying a pending execution synchronizes.

`getStats()` reports the backend and replay/resource counters;
`cudaGraphCompatible` and `cudaGraphIncompatibility` explain graph eligibility.
`setProfiling(true)` reports end-to-end bind/execute/collect timings, not
isolated kernel time. See the
[stream and CUDA Graph example](examples/07_fusion_graph.php).

## Accelerated kernels and backend diagnostics

Execution uses a hybrid backend:

- Custom CUDA kernels and Fusion for elementwise work.
- CUB for eligible large reductions; coalesced kernels are available for
  non-last-axis reductions.
- cuBLAS/cuBLASLt for compatible matrix products, with built-in kernels for
  small products and fallback layouts.
- Optional cuDNN for CNN inference.

Reduction scratch is cached per device and stream. Arg reductions choose the
first tied index and ignore NaNs; parallel reduction order may change low bits.

### Matrix multiplication precision and tuning

Matmul uses strict FP32 by default. On supported compute capability 8.0+
hardware with cuBLAS, TF32 Tensor Core math can be explicitly enabled:

```php
cuda_set_matmul_precision('tf32');
$result = $left->matmul($right);
cuda_set_matmul_precision('fp32-strict');
```

The per-request/thread policy resets to strict FP32 at shutdown. It affects
eligible cuBLAS/cuBLASLt GEMMs, not tensor dtypes, dot products or built-in
fallback kernels. Set it before compiling Fusion plans with `matmul()`; plans
capture the selected mode.

For repeatedly used large 2D `float32` shapes, `cuda_tune_matmul()` benchmarks
zero-workspace cuBLASLt candidates and caches the fastest match for the request.
This explicit setup step synchronizes and runs trial GEMMs. It requires
cuBLASLt, BLAS-compatible layouts and at least 10 million multiply-accumulates.

```php
$tuning = cuda_tune_matmul($left, $right);
print_r($tuning['candidates']);
$product = $left->matmul($right);
```

`cuda_get_backend_info()` reports compiled capabilities, matmul backend,
precision, BLAS status and per-request submission counters. Counters measure
submissions, not completed work or CUDA Graph replays.

### Kernel benchmark

Run the benchmark with the extension loaded:

```bash
php -n -d memory_limit=-1 -d extension=./phptensors_build-8.3/modules/phptensors.so \
  examples/10_kernel_benchmark.php \
  --binary=./phptensors_build-8.3/modules/phptensors.so --output=kernel-results.json
```

The benchmark checks results and measures API wall-clock latency, including
allocation and final synchronization—not isolated kernel time. It covers
square/skinny/MLP, transposed/strided/batched/broadcast matmul, reductions,
eager/Fusion/CUDA Graph and repeated allocation. Configure `--warmups`,
`--samples` and `--pool-iterations`; `--precision=tf32` and `--tune-matmul`
exercise the explicit matmul options.

## CNN inference with optional cuDNN

`PhpTensors\NN` provides static inference methods for initialized, contiguous
`float32` tensors (NCHW inputs, OIHW filters). It prefers cuDNN when available
and uses native CUDA kernels otherwise; there is no CPU fallback or backend
preflight. Inputs are not silently cast or packed. NN does not support
autograd or Fusion capture.

```php
use PhpTensors\NN;

$features = NN::conv2d($input, $weights, $bias, padding: [1, 1]);
$pooled = NN::pool2d($features, window: [2, 2], stride: [2, 2]);
$probabilities = NN::softmax($pooled);
```

- `conv2d()` supports stride, padding, dilation, groups and optional per-output
  channel bias. It performs cross-correlation.
- `pool2d()` supports deterministic max pooling and average pooling that
  excludes padded cells.
- `softmax()` computes channel softmax.

Both paths preserve FP32 inference semantics. cuDNN convolution uses
deterministic FMA algorithms and a bounded reusable workspace; native kernels
cover convolution, pooling and softmax when cuDNN is unavailable or cannot
handle a supported configuration. Calls use the default stream. Backprop,
mixed precision, async CNN plans and convolution/activation fusion are not
included.

## Training example

[`examples/08_gpu_classifier.php`](examples/08_gpu_classifier.php) trains an
MLP on MNIST, Fashion-MNIST or a custom CSV dataset. It demonstrates PHP data
preparation, packed GPU batches, autograd, SGD/AdamW and Fusion replay without
requiring custom CUDA source:

```bash
php -n -d extension=./phptensors_build-8.3/modules/phptensors.so \
  examples/08_gpu_classifier.php --dataset=mnist --epochs=40 \
  --batch-size=128 --optimizer=adam
```

The `$config` array or command-line options select the dataset, hidden layers,
epochs, batch size, optimizer and learning rate. `--profile` reports Fusion
stats; `--predict-index=N` predicts one sample; `--no-save` disables saving;
`--load` evaluates a saved model.

## API and limits

| API | Purpose |
| --- | --- |
| `PhpTensors\Tensor` | Minimal backend-independent tensor metadata and transfer contract |
| `PhpTensors\Numbers` / `PhpTensors\Number` | Operator-overload capability marker and base class for PHP-defined numeric objects |
| `PhpTensors\BackendInterface` | Optional contract for user-created backend implementations |
| `PhpTensors\CudaArray` | GPU storage, arithmetic, reductions, views, indexing, data transfer and opt-in autograd |
| `PhpTensors\HostArray` | Host storage, CPU arithmetic and explicit transfer to/from `CudaArray` |
| `PhpTensors\Optimizer` | Functional SGD and AdamW updates with explicit state |
| `PhpTensors\Fusion` / `PhpTensors\FusionGraph` | Capture, compiled replay, PTX cache and plan diagnostics |
| `PhpTensors\FusionExecution` | Pending asynchronous Fusion execution and result collection |
| `PhpTensors\Compiler` / `PhpTensors\CompiledModule` | NVRTC compilation, kernel launch, async operations and PTX |
| `PhpTensors\NN` | CUDA CNN inference with optional cuDNN acceleration |
| `cuda_*()` functions | Device management, memory, synchronization, backend diagnostics and matmul settings |
| `PhpTensors\Exception` and subclasses | Runtime, argument, allocation and compilation errors |

Annotated signatures are in [class stubs](stubs/cuda.stub.php) and
[device-function stubs](stubs/cuda_methods.stub.php). Runnable examples are
listed in the [examples guide](examples/README.md).

### Backend contracts and custom backends

`Tensor` describes the common metadata and transfer surface. Users choose the
implementation by constructing its concrete class:

```php
use PhpTensors\HostArray;
use PhpTensors\CudaArray;

$cpu = new HostArray([1, 2, 3]);
$gpu = new CudaArray([1, 2, 3]);
```

`PhpTensors\CudaArray` and `PhpTensors\HostArray` are the public tensor classes.
All public classes use the `PhpTensors` namespace.
Mixed-backend arithmetic does not transfer data implicitly: use
`HostArray::toGpu()` or `CudaArray::toHost()` to request a transfer explicitly.
Host arithmetic currently supports same-shape `float32`/`float64` tensors and
scalars. CUDA builds also preserve other dtypes when transferring through
`HostArray`.

`BackendInterface` is an optional userland contract, not a registry or an
automatic dispatcher. An application can instantiate its backend directly and
call its `createTensor()` / `execute()` methods. Tensor operations and custom
operator behavior remain the responsibility of the tensor implementation.

`Numbers` is a marker for objects that support arithmetic overloading.
`Number` is an abstract base class that installs PHP's operator handler for
user-defined numeric classes; an interface alone cannot install that Zend
handler. A custom tensor can implement `Tensor` and extend `Number` to define
its own PHP arithmetic operators. `CudaArray` implements `Numbers` and keeps
its native operator handler.

CPU-only builds provide `HostArray` without requiring CUDA libraries or a GPU.
It supports float32/float64 storage, nested-array and packed-buffer
import/export, same-shape binary arithmetic, and scalar arithmetic. In CUDA
builds the same `HostArray` class is used for host-resident tensor storage and
`CudaArray::toHost()` results; it can also use pinned storage for transfers.
Methods that do not yet have a CPU implementation are present on `HostArray`
and throw `PhpTensors\RuntimeException` explicitly. Attempts to instantiate
or call `CudaArray` in a CPU-only build fail explicitly.

The CUDA backend requires Linux/NVIDIA hardware for execution. Autograd supports
documented operation/dtype subsets; `cumsum()` and `cumprod()` gradients are
supported. `cumprod()` backward uses a synchronous native boundary in Fusion.
Unsupported gradients and unsafe casts fail explicitly. Fusion
matmul requires matching `float32`, `float16` or `bfloat16`
inputs; low-precision elementwise nodes round at each step to match eager
semantics. FP16/BF16 reductions and scans widen to `float32`; `scatterAdd()`
does likewise and returns FP32. Casts are inlined; integer-output scalar
powers, `concat()`, `gather()` and `scatterAdd()` use native boundaries and are
not compatible with CUDA Graph capture. Powers, `concat()`, `gather()` and
`scatterAdd()` also require synchronous replay. Contiguous flattened and
last-axis scans support async replay and CUDA Graph capture, including FP16/BF16
inputs promoted to `float32`; scans over other axes or strided inputs remain
synchronous. cuDNN is optional; CNN inference cannot use Fusion or autograd.

## Validation status

Build and CPU tests do not substitute for GPU runtime tests. The source has been
validated on the following configurations:

| PHP / mode | GPU and configuration | Validation |
| --- | --- | --- |
| 8.1, Docker development image | GPU-enabled CUDA 12.6.3 environment | Current source: build succeeded; API snapshot matched; 67/67 PHPT tests passed |
| 8.3 NTS | GeForce MX570 A, cuDNN 8.9.7 | Beta.5: all 44 GPU PHPT tests passed, including CNN inference |
| 8.3 and 8.5 NTS | GeForce MX570 A, without cuDNN | Beta.5: 43 GPU PHPT tests passed per runtime; optional cuDNN test skipped |
| 8.3 NTS | GeForce MX570 A, without cuBLAS/cuDNN | Beta.5: 43 GPU PHPT tests passed; optional cuDNN test skipped |
| 8.5 NTS/ZTS and 8.1 ZTS | RTX A2000 | Earlier tensor/JIT validation; not the latest Fusion changes |
| PHP 8.1-8.5, NTS/ZTS | CI without CUDA/GPU | CPU-only build, backend PHPT and API matrix; GPU execution is not covered |

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
