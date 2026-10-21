# PHP Tensors guide

[Project overview and quick start](../README.md)

Detailed usage, API contracts, installation options and validation evidence.
Shell commands below run from the repository root.

PHP Tensors is a native PHP extension for tensor operations on the CPU or
optional NVIDIA GPUs. It includes custom CUDA kernels, Fusion and reverse-mode
autograd.

**Status:** beta · Linux · PHP 8.1-8.5 · NTS/ZTS · CPU-only builds supported.
The package baseline is `0.1.0-beta.5`; this checkout may include newer APIs.
CUDA builds require the Toolkit; running CUDA operations also needs a compatible
NVIDIA driver and GPU. This is a low-level library, not a complete ML framework.

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

Use `CudaArray` for GPU tensors. Fusion is opt-in and useful for expressions
replayed many times.

```php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;

$a = CudaArray::ones([4]);
$b = CudaArray::full([4], 2.0);
$c = CudaArray::full([4], 3.0);

$eager = $a + $b * $c;

$plan = Fusion::compile(fn($x, $y, $z) => $x + $y * $z, inputs: [$a, $b, $c]);
$fused = $plan->run($a, $b, $c);

print_r($fused->toArray()); // [7, 7, 7, 7]
```

`CudaArray` keeps data on the GPU. `toArray()`, `toBuffer()`, `toHost()` and
`item()` explicitly read data back to the host.

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
git clone https://github.com/lcmialichi/php-cuda-ext.git
cd php-cuda-ext
./compile.sh --cpu-only
./run-tests.sh --cpu-only
php -n -d extension=./phptensors_build-8.3/modules/phptensors.so -r \
  '$a = new PhpTensors\HostArray([[1, 2], [3, 4]]); print_r($a->multiply(2)->toArray());'
```

Without `--cpu-only`, the build uses CUDA when a Toolkit is detected. CUDA
runtime operations still require a GPU. The extension is written to
`phptensors_build-<PHP version>/modules/phptensors.so`; replace `8.3` below
with the version selected by `php-config`. Add `--install` to install it.

```bash
PHP_BIN=php8.3 PHPIZE=phpize8.3 PHP_CONFIG=php-config8.3 ./compile.sh
```

For advanced configuration, see `./configure --help` in the build directory.

### Install with PIE

```bash
pie install lcmialichi/php-gpu-tensors:0.1.0-beta.5
```

PIE builds for the selected PHP installation. CUDA is optional.

### Docker and tests

GPU tests in Docker require NVIDIA Container Toolkit and a working host driver:

```bash
docker compose run --build --rm php_cuda_dev bash -lc './compile.sh && ./run-tests.sh --require-gpu'
```

```bash
./run-tests.sh               # Run CPU-side and available PHP tests
./run-tests.sh --cpu-only    # Test the CPU-only backend
./run-tests.sh --require-gpu # Require a visible GPU
./run-tests.sh --cuda-api-only # CUDA metadata and parser tests, without GPU execution
```

All runner modes check the frozen stub snapshot, generated arginfo freshness and
isolated native/stub reflection for the selected build profile. The CPU-only CUDA
availability guard has an internal [test fixture](../tests/fixtures/cpu_cuda_array.stub.php),
outside the public stubs and the `PhpTensors` namespace. It does not add a public
class or claim the full CUDA tensor API. The CUDA API-only mode still needs a loadable
CUDA extension and its shared libraries. CI uses an SDK linker stub exclusively
for metadata and host-side parser checks, never as GPU execution evidence.

The [PHP stubs](../stubs/cuda.stub.php) are the authoritative signature source.
After intentional signature changes, regenerate native metadata and review the
frozen API update:

```bash
php -n tests/generate_arginfo.php
php -n tests/check_api_surface.php --update
```

Do not edit [generated arginfo](../src/api_arginfo.h) manually. Native comparison
covers interfaces, inheritance, visibility/final/static/abstract flags, required
counts, parameter names/types/defaults/references/variadics and return types.
Its diagnostics identify the exact field; a stub snapshot alone is not proof of
native API compatibility.

Signature corrections expose existing native final restrictions on `Compiler`,
`CompiledModule` and `CudaArray`, and the existing integer/boolean scalar support
in CUDA comparisons and power. Named CUDA device arguments now use the documented
`deviceId`, not the legacy native `device_id`. `launchAsyncBatch` requires an array.
Nullable compiler kernel arrays mean no parameters/extra headers; a null module
operation ID means all operations. CUDA factories accept null for the default
FP32 dtype, and concatenation accepts null as the default axis zero.

To include the Linux PHP-FPM worker/repeated-request regression, point the
runner at a PHP-FPM binary with the same PHP version, ABI and build mode:

```bash
TEST_PHP_FPM_EXECUTABLE=/usr/sbin/php-fpm8.1 ./run-tests.sh --require-gpu
```

Without that variable, the FPM regression is explicitly skipped. The test uses
a private Unix socket and temporary configuration; no system FPM service is changed.

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

Create tensors from nested arrays or with `fromFlatArray()`, `fromBuffer()`,
`fromFile()` and `fromNpy()`. Factories include `zeros()`, `ones()`, `full()`,
`eye()`, `arange()`, `linspace()`, `rand()` and `randn()`.
`toArray()` returns nested PHP arrays, `toBuffer()` returns packed bytes,
`toHost()` copies to a `HostArray`, and `item()` reads a single value.

`full()` and `fullLike()` accept numeric or boolean fills. Integer fills and
PHP-array imports reject NaN, infinity, overflow and negative unsigned values;
in-range fractions truncate toward zero. Boolean conversion uses nonzero = true.
`CudaArray` and `HostArray` storage factories support all 13 dtypes.
`HostArray` uses the same implementation in CPU-only and CUDA builds; host
arithmetic supports `float16`, `bfloat16`, `float32` and `float64`.

`astype()` uses dtype-based rules: integer narrowing, signed-to-unsigned and
float-to-integer casts are rejected. Integer-to-float casts may lose precision.
FP16/BF16 can widen to FP32/FP64 in eager execution and Fusion.

`arange()` excludes `stop`; integer arguments preserve exact integer values.
`linspace()` includes `stop` by default and preserves integer endpoints when
an integer dtype is requested. Fractional integer results truncate toward zero.
Generated values must fit the destination dtype.

`rand()` requires finite, ordered bounds representable by the destination dtype,
including for empty shapes. Boolean bounds must be 0 or 1. Integer random values
stay within bounds, but use a 24-bit random fraction, not an unbiased full-width
integer sampler.

Scalar reads preserve PHP types: floats, integers and booleans. Integers outside
PHP's `int` range export as exact decimal strings, consistently across indexing,
`item()`, `toArray()` and `HostArray` reads in either build. Use `toBuffer()` and
`fromBuffer()` for lossless binary round trips. FP64 tensors remain FP64 when
dividing by a scalar or raising to a negative scalar power.

`serialize()` snapshots tensor values, shape and dtype, not gradients. Host
subarrays are packed in logical order; copy a `CudaArray` view before serializing
it. Invalid restore data leaves existing storage unchanged. Payloads use native
byte order and build-specific formats, not a portable checkpoint format.
`CompiledModule` uses versioned parameter fields instead of C structs: recompile
legacy module caches and regenerate unversioned CPU payloads. Restore modules
into new objects; initialized or active modules cannot be overwritten.

### Operations

- **Math:** arithmetic operators and methods, elementwise functions,
  comparisons, `where()`, `maximum()`, `minimum()` and `clamp()`.
- **Linear algebra:** `matmul()` supports batched tensors and broadcasting.
- **Reductions:** `sum()`, `mean()`, `min()`, `max()`, `prod()`, `argMin()`,
  `argMax()`, `all()`, `any()`, `var()`, `std()`, `cumsum()` and `cumprod()`.
- **Neural network helpers:** `softmax()` and `layerNorm()`.
- **Indexing:** `concat()`, `gather()` and `scatterAdd()`.
- **Shape and dtype:** views, transforms, `astype()` and tensor metadata.

`eq()` and `ne()` use exact equality in eager and Fusion, without an implicit
epsilon. Results are 0/1 masks in the promoted dtype; NaN is unequal to every
value, and signed zeros compare equal.

Arithmetic broadcasts compatible shapes and promotes dtypes. Mixed signed and
unsigned tensors use the smallest exact common integer dtype: `int8 + uint8`
produces `int16`. Signed integers, including PHP `int` scalars, cannot be mixed
with `uint64`; use explicitly typed unsigned operands or convert to `float64`
when rounding is acceptable. Ordinary integer scalar arithmetic retains the
tensor dtype and rejects out-of-range scalar values. Division and power between
tensors use FP64 for a 64-bit common integer dtype. Scalar operations producing
floating results also use FP64 with 64-bit integer tensors.

FP16/BF16
operations use FP32 intermediates; reductions and `scatterAdd()` may return
FP32. Unsupported dtype combinations or gradients raise exceptions. See the
[API stubs](../stubs/cuda.stub.php) for method signatures and options.

### Shape operations, views and indexing

Shape methods include `reshape()`, `flatten()`, `transpose()`, `swapaxes()`,
`moveaxis()`, `squeeze()`, `unsqueeze()`, `broadcastTo()` and `slice()`.
Views may share storage; `copy()` creates independent storage. `reshape()`
requires contiguous input. `repeat()`, `tile()`, `diagonal()`, `flip()` and
`roll()` create transformed tensors.

```php
$row = new CudaArray([[1.0, 2.0, 3.0]]);
$batch = $row->broadcastTo([2, 3]);
$columns = $batch->slice(':, 1:3');
$flat = $columns->flatten();
```

`slice()` uses exclusive stops, supports negative indices and positive steps,
and accepts per-axis selectors. Integer selectors remove axes; ranges retain
them. Ellipsis and index lists are not supported.

`gather()` uses an `int32` index tensor. `scatterAdd()` returns a new tensor and
adds updates at the requested indices. Empty dimensions are supported by many
operations; some reductions, such as min/max, require non-empty input.

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

Gradients accumulate until `zeroGrad()`. Use `detach()` to stop gradient
tracking. Non-scalar outputs need a matching seed; pass `retainGraph: true` to
reuse the graph. Supported gradients depend on operation and dtype; unsupported
cases raise exceptions. Inputs must enable gradients before Fusion capture.
Autograd supports first-order derivatives; `retainGraph` reuses the forward graph,
not a higher-order derivative graph.

`Fusion::run()` preserves the history for backward inside or after the callback.
With `Fusion::compile()`, capture backward inside the callback and return its
gradients; tracked outputs without captured backward are rejected. For a
forward-only graph, detach outputs explicitly. Replay cannot enable gradients
that were absent from the example inputs, and replay values have no new dynamic
backward history.

Discrete reductions (`argMax()`, `argMin()`, `all()`, `any()`) require an explicit
`detach()` on tracked inputs, including during Fusion capture:
`$input->detach()->argMax()`. Reduction axes must be integers or `null`; `null`
reduces all elements.

`where()` differentiates its value branches, not its condition. Using a tensor
only as a condition does not consume its gradient history. The saved mask must
remain unchanged until backward. Branch routing uses genuine zeros, rather
than multiplying non-finite gradient seeds by zero.

FP16/BF16 concat, slice and copy preserve gradient paths in eager and Fusion;
their gradients use FP32, including repeated inputs and empty tensors. Accumulated
gradients use the input dtype for FP32/FP64 and FP32 for FP16/BF16, even when
forward operations promote to a wider dtype.

`maximum()` and `minimum()` split gradients equally at exact ties, including
scalar bounds. `clamp()` composes these operations: its input gradient is `0.5`
at either boundary, or `0.25` when both bounds coincide.

Singular derivatives follow IEEE NaN/Inf rules. Logarithm gradients are NaN
outside the real domain; `std()` uses zero input gradients for constant groups.
`abs()` uses zero at zero, and `floor()`/`ceil()`/`round()` have zero gradients.
Constant power partials (`x ** 0`, `1 ** x`, and `0 ** x` for positive `x`)
also use genuine zeros.

Arithmetic, safe floating casts, reductions/scans, indexing, views, transforms and
tensor composites preserve floating gradient paths. Comparisons reject backward;
integer/bool tracking and FP64 matmul are rejected explicitly. `copy()` preserves
history; native `clone`, serialization and host/byte exports are value snapshots.

`Optimizer` provides functional SGD and AdamW. Each `step()` returns updated
parameters and explicit state; `weightDecayMask` can exclude parameters.

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

`launch()` runs synchronously. Use `launchAsync()` or `launchAsyncBatch()` for
asynchronous work, and keep input arrays alive until it completes. See the
[JIT examples](../examples/README.md).

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

The callback runs during capture, not on replay. Input shapes, dtypes and
strides must remain compatible. Fusion combines supported elementwise work and
views; reductions, matmul and some indexing operations use native kernels.
Unsupported operations may work in `Fusion::run()` but fail in
`Fusion::compile()`. The planner can generate multiple kernels, so one
expression does not always mean one launch.

Use `getStats()`, `getPlan()` and `getSource()` to inspect a plan. See the
[Fusion example](../examples/07_fusion_graph.php) for capture and replay.

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
$result = $pending->wait();
```

Async plans return a `FusionExecution`; call `wait()` to collect results and
keep inputs alive until then. A CUDA Graph plan allows only one outstanding
replay. Some operations are not graph-compatible; inspect
`cudaGraphCompatible` and `cudaGraphIncompatibility` in `getStats()`.
Profiling reports end-to-end timings, not isolated kernel time.

## Accelerated kernels and backend diagnostics

The CUDA backend uses custom kernels and Fusion for elementwise work, CUB for
reductions, cuBLAS/cuBLASLt for compatible matrix products and optional cuDNN
for CNN inference. Floating-point reduction order may affect low bits.

### Matrix multiplication precision and tuning

Matmul uses strict FP32 by default. On supported compute capability 8.0+
hardware with cuBLAS, TF32 Tensor Core math can be explicitly enabled:

```php
cuda_set_matmul_precision('tf32');
$result = $left->matmul($right);
cuda_set_matmul_precision('fp32-strict');
```

This setting affects eligible cuBLAS/cuBLASLt FP32 matmuls only. Set it before
compiling Fusion plans that use matmul.

For repeated large 2D FP32 matmuls, `cuda_tune_matmul()` benchmarks and caches
compatible cuBLASLt choices. It synchronizes and runs trial operations.

```php
$tuning = cuda_tune_matmul($left, $right);
print_r($tuning['candidates']);
$product = $left->matmul($right);
```

`cuda_get_backend_info()` reports available capabilities and backend counters.

### Kernel benchmark

Run the benchmark with the extension loaded:

```bash
php -n -d memory_limit=-1 -d extension=./phptensors_build-8.3/modules/phptensors.so \
  examples/10_kernel_benchmark.php \
  --binary=./phptensors_build-8.3/modules/phptensors.so --output=kernel-results.json
```

The benchmark validates results and measures end-to-end latency, including
allocation and synchronization. Options control warmups, samples and matmul
precision/tuning.

## CNN inference with optional cuDNN

`PhpTensors\NN` provides CUDA inference for contiguous FP32 tensors (NCHW input,
OIHW filters). It uses cuDNN when available and native kernels otherwise.
There is no CPU fallback; NN does not support autograd or Fusion capture.

```php
use PhpTensors\NN;

$features = NN::conv2d($input, $weights, $bias, padding: [1, 1]);
$pooled = NN::pool2d($features, window: [2, 2], stride: [2, 2]);
$probabilities = NN::softmax($pooled);
```

`conv2d()` supports stride, padding, dilation, groups and optional bias;
`pool2d()` supports max and average pooling; `softmax()` normalizes channels.

## Training example

[`examples/08_gpu_classifier.php`](../examples/08_gpu_classifier.php) trains an
MLP on MNIST, Fashion-MNIST or a custom CSV dataset using autograd and SGD/AdamW.

```bash
php -n -d extension=./phptensors_build-8.3/modules/phptensors.so \
  examples/08_gpu_classifier.php --dataset=mnist --epochs=40 \
  --batch-size=128 --optimizer=adam
```

Use `--profile` for Fusion stats, `--predict-index=N` for a sample prediction,
`--no-save` to disable saving, or `--load` to evaluate a saved model.

## API and limits

| API | Purpose |
| --- | --- |
| `PhpTensors\Tensor` | Minimal backend-independent tensor metadata and transfer contract |
| `PhpTensors\OperatorOverloadable` / `PhpTensors\OperatorOverload` | Operator-overload marker and userland base class |
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

See [API stubs](../stubs/cuda.stub.php) for signatures and the
[examples guide](../examples/README.md) for runnable code.

CUDA builds own one device and its primary context per PHP process. Set
`CUDA_VISIBLE_DEVICES` before starting PHP to choose a GPU; its visible device
ID is usually `0`. `cuda_set_device()` accepts only the owning device, and
`cuda_device_reset()` throws rather than invalidating tensors, modules or graphs.
Start a new PHP process to reset CUDA. Compiled modules, eager operations and
Fusion share the same primary context.
CUDA initialization is lazy: loading the extension and using ordinary `HostArray`
storage do not initialize CUDA. CLI and PHP-FPM with isolated workers are
supported. Fork workers **before the first CUDA API call**, including device
queries, compiler construction and pinned host allocation. Do not use CUDA in
FPM master/preload code. Each worker must create its own tensors, modules and
graphs after forking.

Once a process has accessed CUDA, its children cannot reuse that runtime:
CUDA access throws `PhpTensors\RuntimeException` before touching inherited
contexts or locks. Use a fresh PHP process/`exec` instead. Destruction in a
rejected child does not synchronize or free the parent's CUDA resources;
inherited native resources are abandoned until that child exits.

The first CUDA caller also owns the process's CUDA thread for its entire
lifetime. Other threads are rejected, even after the original thread exits.
Concurrent CUDA calls from PHP/ZTS threads are **not supported**; use isolated
processes. Internal GPU streams may still execute concurrently. Native pthread
tests verify rejection, not concurrent PHP/ZTS GPU execution.

The pool is allocated on the first operation needing a CUDA context and remains
bounded and reusable across requests. Configure its positive byte count, or a
`K`/`M`/`G` suffix, before starting PHP, for example:

```ini
phptensors.cuda.memory_pool_size=256M
```

Invalid/overflowing sizes throw on the first pool allocation, without preventing
CPU-only work. The first pool configuration is fixed for the worker's lifetime;
`ini_set()` emits a warning, returns `false` and leaves the value unchanged.
Request shutdown completes pending Fusion work, clears request-local caches and
backend workspaces/counters, and restores strict FP32 matmul precision without
resetting the CUDA device or destroying the process pool.

Shapes have at most 10 axes, with dimensions from `0` to `INT_MAX`. Factories
check element counts, strides and byte sizes before allocating; empty axes are
valid but do not bypass overflow checks. `getSize()` returns a full PHP integer.
Flattening and global arg reductions reject sizes above `INT_MAX`, because they
require a single representable axis. Oversized indices and axes throw instead
of being truncated.

### Backend contracts and custom backends

Choose a backend by constructing its tensor class:

```php
use PhpTensors\HostArray;
use PhpTensors\CudaArray;

$cpu = new HostArray([1, 2, 3]);
$gpu = new CudaArray([1, 2, 3]);
```

`HostArray` stores data on the CPU; `CudaArray` stores it on the GPU. Operations
do not move data implicitly. Use `toGpu()` and `toHost()` for synchronous copies
that preserve shape and dtype.

`clone $tensor` creates independent storage on the same backend and preserves
host pinning. It copies values, not autograd history; use `$gpu->copy()` for a
differentiable copy. Repeating `__construct()` replaces storage only on success;
failure leaves the tensor intact. Uninitialized storage access throws, and
active host iterators retain their original storage during replacement.

`HostArray` supports CPU `+`, `-`, `*` and `/` for all four floating storage dtypes
and numeric scalars, including scalar-left operators and NumPy-style trailing-axis
broadcasting. Tensor promotion follows CUDA: mixing float32/float64 produces
float64, mixing FP16/BF16 produces float32, and division of non-FP64 inputs produces
float32. Scalars follow the CUDA scalar-promotion rules and are rounded to the
result dtype before arithmetic. Compound
assignments remain functional: aliases and input views are not modified.
Division by zero now follows CUDA's IEEE NaN/Inf behavior, including scalar-left
division. This intentionally replaces the earlier CPU-only zero-divisor exception.
CPU-only builds need no CUDA installation; `toGpu()` and `CudaArray` operations
throw if CUDA is unavailable. `HostArray` stores all 13 dtypes at their native
width in either build: FP32 uses four bytes per element, FP64 eight, and
FP16/BF16 two. Pinned allocation remains an explicit CUDA-only capability.
Scalar access (`get()`, `at()`, `$host[$i]`) and iteration support these storage
dtypes, including FP16/BF16; low-precision values are returned as PHP floats.
Indices must be integers, not coercible strings or fractional values.
Empty nested arrays preserve their shape: `[]` has shape `[0]`, while
`[[], []]` has shape `[2, 0]`.

`copy()` allocates independent, dense row-major storage containing the logical
elements of a tensor or view. `astype($dtype)` uses the same dtype compatibility
policy as CUDA: unsafe/narrowing casts are rejected, and integer-to-float
conversions follow ordinary floating-point rounding rather than guaranteeing
lossless representation. Changing dtype materializes dense storage; casting to
the existing dtype returns a new object sharing the original storage/layout.
Both methods support all 13 storage dtypes and preserve pinning.

`broadcastTo($shape)` returns a zero-copy view, using zero strides for expanded
dimensions. Views retain their source storage even after the source object is
released. Empty axes follow the CUDA rules: a dimension of one can broadcast to
zero, but zero cannot broadcast to a nonempty extent. Indexing, iteration,
`copy()`, `astype()`, serialization and GPU upload consume logical view contents:

```php
$column = new HostArray([[10], [20]], 'float32');
$row = new HostArray([1, 2, 3], 'float64');
print_r(($column + $row)->toArray()); // [[11, 12, 13], [21, 22, 23]], float64
$view = $row->broadcastTo([2, 3]);    // shape [2, 3], strides [0, 1]
$dense = $view->copy();               // independent storage, strides [3, 1]
```

The minimum CPU numerical core also includes:

| Operations | Contract |
| --- | --- |
| `reshape($shape)` | Zero-copy dense view with explicit nonnegative dimensions and unchanged element count. A non-contiguous input requires `copy()` first, like CUDA. |
| `transpose($axis = null)` | Retained view with an explicit axis permutation; null reverses the axes. |
| `slice(...$selectors)` | The parser/layout rules are shared with CUDA. Integers remove an axis and support negative indices; null selects the whole axis; `[start, stop, step]` and `"start:stop:step"` use an exclusive stop, clamp bounds and require a positive step. A single string may separate axes with commas. Selecting every axis by integer produces a rank-zero view. |
| `item()` | Reads exactly one logical element, including rank-zero views. Large unsigned integers retain exact decimal-string export. |
| `gt`, `lt`, `eq`, `ne`, `ge`, `le` | Broadcasting and scalar operands, with CUDA promotion rules. Results are numeric zero/one masks, **not necessarily bool tensors**. Ambiguous int64/uint64 promotion is rejected. |
| `HostArray::where($condition, $x, $y)` | Broadcasts all inputs, treats numeric nonzero/NaN conditions as true, and copies the selected bytes exactly. `x` and `y` must have matching dtype; all 13 storage dtypes are supported. |
| Floating unary math | `sqrt`, `exp`, `expm1`, `exp2`, `log`, `log1p`, `log2`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `abs`, `neg`, `floor`, `ceil`, `round`. All four floating dtypes preserve storage dtype; FP16/BF16 compute in FP32. Cast integer inputs explicitly. |
| `sum`, `mean`, `min`, `max` | All storage dtypes; null axis reduces globally to shape `[1]`, while positive/negative axes remove that axis (rank-one reductions also return `[1]`). FP16/BF16 produce FP32; integer/bool mean produces FP64. Integer sum wraps modulo its dtype width and bool sum is logical OR. |
| `matmul($other)` | Matching FP32/FP16/BF16 inputs, rank >= 2, broadcast batch axes and arbitrary input strides. Accumulation is FP32. FP64 and vector inputs are rejected, matching the current CUDA matmul API. No BLAS/SIMD dependency is added. |

Empty reductions use sum zero and mean NaN; min/max reject empty groups when the
output is nonempty. Floating sum/mean preserve NaN/Inf; CPU min/max ignore NaNs
and return their +/-infinity identity if no ordered value remains. Floating
reductions are not bitwise reproducible between sequential CPU accumulation and
parallel GPU trees. `round()` uses halfway-away-from-zero; `abs()` preserves
negative zero, following the existing CUDA unary contract.

```php
$input = new HostArray([[1, 2, 3], [4, 5, 6]], 'float32');
$sample = $input->slice(':, 0:3:2'); // [[1, 3], [4, 6]], retained strided view
$features = $input->matmul($input->transpose())->tanh();
$selected = HostArray::where($features->gt(0), $features, HostArray::zerosLike($features));
print_r($selected->sum(axis: -1)->toArray());
```

`HostArray` implements `IteratorAggregate` and `Countable`. `count($host)` and
`$host->count()` return the first dimension; `getSize()` returns the total
element count. One-dimensional iteration yields PHP scalars, while higher ranks
yield zero-copy `HostArray` rows that retain their backing storage:

```php
$host = new HostArray([[1, 2], [3, 4]], 'int32');
$rows = iterator_to_array($host);
print_r($rows[1]->toArray()); // [3, 4]
print_r(iterator_to_array($host->getIterator())[0]->toArray()); // [1, 2]
```

Host serialization uses `__contiguous_array_v1` in both builds, packing logical
row-major contents with offset zero rather than the parent allocation. Subarray
snapshots, empty shapes and dtype bytes round-trip across CPU/CUDA build profiles
on machines with the same native byte order. Bytes are native-endian (little-endian
on the validated Linux x86-64 platforms), not a cross-endian interchange format.
The old CPU-only `__host_array_v1` FP32/FP64 format remains readable and is
normalized on the next export. Pinning and autograd state are not serialized;
restoration produces ordinary host storage. CPU autograd/Fusion, integer/bool
arithmetic operators, the legacy `__invoke()` slicing API and numerical methods
beyond the operations documented above remain unsupported and throw explicitly.
The minimum core is implemented; its new real-GPU parity matrix remains a release
validation gate until a GPU-enabled Docker runtime is available.

To measure host allocation footprint, buffer import/export and arithmetic:

```bash
php -n -d extension=phptensors_build-8.4/modules/phptensors.so tests/benchmark_host_storage.php
```

The benchmark reports actual Zend-managed allocation including metadata and
seven-run median timings. It requires normal Zend allocation accounting
(`USE_ZEND_ALLOC=0` is not suitable for footprint measurement). These are host
buffer-copy measurements by default. In a CUDA build, append `--gpu` to also
verify byte-exact GPU round trips and measure synchronous upload/download,
including result allocation; uploads explicitly synchronize before timing ends.

Custom backends can implement `BackendInterface`, but it does not automatically
dispatch tensor operations. To overload PHP arithmetic, extend `OperatorOverload`.
Also implement `Tensor` when the object represents a tensor; it is not required
for operator dispatch itself. `OperatorOverloadable` is only a marker: implementing
it or defining hook-named methods alone does not install operator handlers.
The base class already implements
`OperatorOverloadable` and routes operators to `__add`, `__sub`, `__mul`,
`__div`, `__pow` and `__mod`. Override only the hooks you support; the defaults
throw `PhpTensors\RuntimeException`. Use `mixed` parameters and a `mixed` or
narrower return type. PHP routes `++` through `__add($value, 1)` and `--` through
`__sub($value, 1)`; the unused `__inc`/`__dec` hooks have been removed.
The example below supports vector addition and multiplication; other operators
use the default exception:

```php
use PhpTensors\OperatorOverload;
use PhpTensors\Tensor;

final class VectorTensor extends OperatorOverload implements Tensor
{
    public function __construct(private array $values)
    {
        if (!array_is_list($values) ||
            array_filter($values, static fn($value) => !is_int($value) && !is_float($value))) {
            throw new \InvalidArgumentException('Expected a flat list of numbers');
        }
    }

    public function getBackend(): string { return 'example'; }
    public function getShape(): array { return [count($this->values)]; }
    public function getSize(): int { return count($this->values); }
    public function dtype(): string { return 'float64'; }
    public function toArray(): array { return $this->values; }

    private static function apply(mixed $left, mixed $right, string $operator): self
    {
        $leftValues = $left instanceof self ? $left->values : null;
        $rightValues = $right instanceof self ? $right->values : null;
        if (($leftValues === null && !is_int($left) && !is_float($left)) ||
            ($rightValues === null && !is_int($right) && !is_float($right))) {
            throw new \InvalidArgumentException('Expected vectors or numeric scalars');
        }
        if ($leftValues === null && $rightValues === null) {
            throw new \InvalidArgumentException('At least one operand must be a vector');
        }
        if ($leftValues !== null && $rightValues !== null &&
            count($leftValues) !== count($rightValues)) {
            throw new \InvalidArgumentException('Vector lengths must match');
        }

        $length = count($leftValues ?? $rightValues);
        $result = [];
        for ($i = 0; $i < $length; $i++) {
            $a = $leftValues === null ? $left : $leftValues[$i];
            $b = $rightValues === null ? $right : $rightValues[$i];
            $result[] = match ($operator) {
                '+' => $a + $b,
                '*' => $a * $b,
            };
        }
        return new self($result);
    }

    public function __add(mixed $left, mixed $right): mixed { return self::apply($left, $right, '+'); }
    public function __mul(mixed $left, mixed $right): mixed { return self::apply($left, $right, '*'); }
}

$vector = new VectorTensor([1, 2, 3]);
$sum = $vector + new VectorTensor([10, 20, 30]);
$scaled = 2 * $vector; // Reversed scalar operands are passed in their original order.

print_r($sum->toArray()); // [11, 22, 33]
print_r($scaled->toArray()); // [2, 4, 6]
```

The operator methods receive both operands in their original order, including
expressions such as `2 * $vector`. Native `HostArray` and `CudaArray` implement
`OperatorOverloadable` directly and use native handlers.

| PHP expression | Hook call on the selected object |
| --- | --- |
| `$a + $b`, `$a += $b` | `__add($a, $b)` |
| `$a - $b`, `$a -= $b` | `__sub($a, $b)` |
| `$a * $b`, `$a *= $b` | `__mul($a, $b)` |
| `$a / $b`, `$a /= $b` | `__div($a, $b)` |
| `$a ** $b`, `$a **= $b` | `__pow($a, $b)` |
| `$a % $b`, `$a %= $b` | `__mod($a, $b)` |
| `+$a`, `-$a` | `__mul($a, 1)`, `__mul($a, -1)` |
| `++$a`, `$a++` | `__add($a, 1)` |
| `--$a`, `$a--` | `__sub($a, 1)` |

With two `OperatorOverload` instances, the left object's hook is selected, even
when they belong to different subclasses. With a scalar on the left, the right
object receives both operands without swapping them. A missing hook or an
exception is not an invitation to retry the other object's hook. There is no
reverse-hook naming convention or `NotImplemented` sentinel; custom hooks define
their own accepted operand types. Mixing native tensors and custom objects does
not automatically convert backends or bypass the selected handler's validation.

Bitwise operators (`&`, `|`, `^`, `~`) and shifts (`<<`, `>>`) are not overloaded;
unsupported uses raise PHP `TypeError` without invoking arithmetic hooks.
Concatenation uses ordinary PHP string conversion (`__toString()` when provided);
comparison, identity and truthiness retain ordinary PHP object semantics, not
elementwise tensor semantics. Prefix increment/decrement return the new result;
postfix forms return the previous value.

Hooks can return any value, including null/false/scalars/arrays; the bridge does
not interpret falsy values as failure. Returned references are copied as values.
Callback exceptions propagate unchanged and failed operations preserve the
destination binding, but cannot undo mutations or other side effects performed
inside user code. Return fresh values without mutating operands to preserve
functional aliasing; an incompatible result for a typed property raises PHP
`TypeError`. Cloning a subclass retains its arithmetic handlers and ordinary PHP
member/`__clone()` behavior, including exceptions raised by the clone hook.

The [protocol regression](../tests/operator_overload_protocol.phpt) and
[ownership regression](../tests/operator_overload_aliasing.phpt) cover these rules.
API-02 validation passed four targeted host/backend tests on each PHP 8.4/8.5 ZTS
build and all five host tests in the PHP 8.3 CUDA API-only gate. The three operator
regressions also passed with system malloc in all three configurations. The
separate CUDA/Fusion/OOM regression still requires a functioning GPU runtime.

Compound assignments compute a new value and rebind the variable; they do not
modify the original tensor's storage. PHP references (`&`) follow the rebinding,
while other object aliases keep the previous value:

```php
use PhpTensors\HostArray;

$a = new HostArray([1, 2, 3]);
$old = $a;
$reference =& $a;
$a += 2; // $a and $reference: [3, 4, 5]; $old: [1, 2, 3]
```

Custom hooks should return a value without mutating their operands. A failed
computation preserves the binding and its original exception; mutations made
inside user hooks cannot be rolled back. Compound GPU assignments remain
unsupported inside Fusion capture.

```php
use PhpTensors\CudaArray;
use PhpTensors\HostArray;

$cpu = new HostArray([[1, 2], [3, 4]], 'float32');
$cpuResult = $cpu * 2;                   // Runs on CPU
$gpuResult = $cpuResult->toGpu() + 1;    // Copy, then run on GPU
$hostResult = $gpuResult->toHost();      // Copy back to CPU
```

CUDA autograd and Fusion support documented subsets of operations and dtypes.
Unsupported combinations fail explicitly; see the API stubs and Fusion
diagnostics for details.

Built-in CUDA offset/grid arithmetic preserves 64-bit values. Unsupported grid
sizes fail explicitly; device and memory-pool limits still apply.

## Validation status

The CPU-02 minimum numerical core implementation is complete. Its real-CUDA
parity and affected GPU/Fusion regression checks remain an open release-validation
gate, tracked separately as CPU-02-GPU in [TODO.md](../TODO.md).

Build and CPU tests do not substitute for GPU runtime tests. The source has been
validated on the following configurations:

| PHP / mode | GPU and configuration | Validation |
| --- | --- | --- |
| 8.1, Docker development image | GPU-enabled CUDA 12.6.3 environment | Previous CPU-02 foundation slice only: build, shared native shape check and 19 targeted PHPTs passed, including all 169 CPU/CUDA cast decisions, broadcast arithmetic, transfers, lifecycle and Fusion regressions; three numerical regressions passed with system malloc and two host tests with no visible GPU. Earlier milestone: full FPM-enabled suite passed 97 PHPTs with one expected CPU-only skip and seven native checks |
| 8.1 NTS CLI and PHP-FPM | GPU-enabled CUDA 12.6.3 environment | Pre-CUDA worker fork/recycling, post-CUDA fork rejection and safe inherited-object destruction passed; 24 GPU and eight CPU-only requests across four recycled FPM workers passed with request-state reset, pending shutdown and bounded warm memory |
| 8.4 and 8.5 ZTS | CPU-only Docker builds | Builds, snapshot, generated arginfo and native/stub reflection of all 12 classes/interfaces pass; 17/17 selected PHPTs per version, including the complete operator protocol. The new API regression also passes with system malloc on PHP 8.4. Earlier CPU-02 validation passed eight host/numerical malloc regressions per version and those eight on a separate PHP 8.4 UBSan build without undefined-behavior reports |
| 8.3 NTS | CUDA 12.3.2 build, no GPU/driver | Full CUDA build and native/stub reflection of all 19 classes/interfaces pass. SDK linker stub permits host-only loading: 15 shared/API tests pass, including two API/parser regressions that also pass with system malloc; both real-GPU parity matrices explicitly skip. The earlier native shape check passed. This is not GPU execution evidence: Docker GPU startup cannot load `libnvidia-ml.so.1` |
| 8.3 NTS | GeForce MX570 A, cuDNN 8.9.7 | Beta.5: all 44 GPU PHPT tests passed, including CNN inference |
| 8.3 and 8.5 NTS | GeForce MX570 A, without cuDNN | Beta.5: 43 GPU PHPT tests passed per runtime; optional cuDNN test skipped |
| 8.3 NTS | GeForce MX570 A, without cuBLAS/cuDNN | Beta.5: 43 GPU PHPT tests passed; optional cuDNN test skipped |
| 8.5 NTS/ZTS and 8.1 ZTS | RTX A2000 | Earlier tensor/JIT validation; not the latest Fusion changes |
| PHP 8.1-8.5, NTS/ZTS | CI without CUDA/GPU | CPU-only build, backend PHPT and API matrix; GPU execution is not covered |

A CUDA-profile run of [runtime shape boundaries](../tests/tensor_shape_boundaries.phpt)
reaches device memory-pool initialization and fails without the driver; its CPU
runs pass. That CUDA runtime path remains pending with CPU-02-GPU, not covered by
the successful metadata-only checks. The new CUDA CI job is configured for PHP
8.1 / CUDA 12.6.3; its entry point was validated locally on PHP 8.3 / CUDA 12.3.2,
not by running the remote workflow.

Reports for other PHP, driver, CUDA or GPU versions are welcome. Include those
versions and the result of `./run-tests.sh --require-gpu` in an
[issue](https://github.com/lcmialichi/php-cuda-ext/issues).

## Contribute

Contributions can be code, tests, documentation, examples or reports from other
PHP/CUDA/GPU combinations. See [open issues](https://github.com/lcmialichi/php-cuda-ext/issues),
[bug reports](https://github.com/lcmialichi/php-cuda-ext/issues/new?template=bug_report.yml),
[feature requests](https://github.com/lcmialichi/php-cuda-ext/issues/new?template=feature_request.yml),
the [contribution guide](../CONTRIBUTING.md) and [roadmap](../ROADMAP.md).
Documentation, examples and host-side tests can be worked on without an NVIDIA
GPU.

## Benchmarks

The benchmark suite is maintained in the separate
[PHP GPU Tensors Benchmarks repository](https://github.com/lcmialichi/php-gpu-tensors-benchmarks).
It contains focused matmul, import and Fusion runs, JSON/HTML reports and
published workload reports, including [PHP 8.5 NTS vs ZTS results](https://github.com/lcmialichi/php-gpu-tensors-benchmarks/blob/main/published-reports/php85-nts-vs-zts/README.md).

## License

Licensed under the [MIT License](../LICENSE).
