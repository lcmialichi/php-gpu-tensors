# PHP GPU Tensors

<p align="center">
    <img src="art/php-gpu-tensors.jpg" alt="PHP GPU Tensors: high-performance computing with PHP and NVIDIA GPUs" width="350">
</p>

Native PHP extension for GPU tensors and NVIDIA CUDA-accelerated numerical
workloads. Build tensor operations and machine-learning data pipelines in PHP,
move data explicitly between host and GPU, and compile custom CUDA C++ kernels
at runtime with NVRTC. Optionally fuse tensor expressions and replay compiled
plans, including asynchronous execution and CUDA Graph for compatible plans.
No Python runtime required.

**Status:** `0.1.0-beta.4`, supporting PHP 8.1 through 8.5 on Linux in NTS and
ZTS modes. Tensor operations, Python-style slicing, JIT compilation and optional
Fusion execution are part of the extension's public API. The core API has a
frozen `0.1.0` baseline; eager execution remains the default, and Fusion is
explicitly opt-in. All ten PHP version/thread-mode combinations passed builds
and CPU/API checks for this release. The current implementation passed 41 GPU tests
on PHP 8.3 NTS with an NVIDIA GeForce MX570 A, including gradient/lifetime
regressions, plus an Optdigits training check. Earlier tensor/JIT validation covered
PHP 8.5 NTS/ZTS and PHP 8.1 ZTS on an RTX A2000; it does not validate the latest
Fusion changes on those builds. Build/CPU checks do not substitute for GPU
runtime validation on each PHP version and thread mode.

## Start here

You need a CUDA-capable NVIDIA GPU, a compatible host driver, the CUDA Toolkit
(including NVRTC), PHP development headers (`phpize`, `php-config`), a C/C++
toolchain, `make`, and `autoconf`. The extension builds on Linux. Building
requires the toolkit; running requires the host driver's `libcuda.so.1`.

```bash
git clone https://github.com/lcmialichi/php-gpu-tensors.git
cd php-gpu-tensors
./compile.sh
./run-tests.sh --require-gpu
php -n -d extension=./cuda_build-8.3/modules/cuda.so examples/01_basics_cuda_array.php
```

The build stays in `cuda_build-<PHP major.minor>/modules/cuda.so`; replace
`8.3` above with the PHP version used by `php-config`. To install the extension
and its INI configuration instead, run `./compile.sh --install` with permission
to write to your PHP extension/INI directories. To choose another PHP ABI:

```bash
PHP_BIN=php8.3 PHPIZE=phpize8.3 PHP_CONFIG=php-config8.3 ./compile.sh
PHP_BIN=php8.3 PHP_CONFIG=php-config8.3 ./run-tests.sh --require-gpu
```

### Install with PIE 🥧

The extension is published as
[`lcmialichi/php-gpu-tensors`](https://packagist.org/packages/lcmialichi/php-gpu-tensors).
Release `0.1.0-beta.4` adds optional Fusion replay, reduced execution overhead
and Python-style slicing. CI builds PHP 8.1-8.5 in both NTS and ZTS modes;
GPU validation of these latest features is on PHP 8.3 NTS. Use this release or
newer for the Fusion APIs and training example described here; older releases
may not include them.

PIE builds the native extension for the selected PHP installation; it does not
install an NVIDIA driver or CUDA Toolkit. Those must already be available on
the system. GPU execution additionally requires a compatible NVIDIA driver
and visible GPU.

Set `CUDA_HOME` if the toolkit is not at `/usr/local/cuda`, or `CUDA_ARCH=sm_86`
when cross-building. cuBLAS is used when available for compatible larger
matrix products; `CUDA_USE_CUBLAS=no ./compile.sh` builds with the extension's
built-in CUDA matrix kernels instead.

Docker users with the NVIDIA Container Toolkit and a working host driver can
build and test in the development image:

```bash
docker compose run --rm php_cuda_dev bash -lc './compile.sh && ./run-tests.sh --require-gpu'
```

`./run-tests.sh` runs CPU-side C tests and the PHP test suite. Use
`--cpu-only` to run only the host-side C tests in build environments without a
GPU; this does not validate CUDA execution. `--require-gpu` fails immediately
when no GPU is visible, instead of treating skipped GPU tests as success.

## GPU Tensors in PHP

```php
use Cuda\CudaArray;

$input = new CudaArray([[1, 2], [3, 4]], 'float32');
$weights = CudaArray::ones([2, 2]);
$output = $input->add($weights)->multiply(2);
$column_means = $input->mean(0);

print_r($output->toArray()); // [[4, 6], [8, 10]]
echo $output->dtype();       // float32
```

`CudaArray` holds GPU storage; operations return GPU tensors. `toArray()`
transfers to the CPU and expands all values into PHP arrays. Operations include
arithmetic, broadcasting, comparisons, `matmul()` (including batches), shape
views, and `sum()`, `mean()`, `min()`, `max()`, `prod()`, `argMax()`, and
`argMin()` with an optional axis. PHP arithmetic operators also dispatch to
tensor methods. `mean()` reduces all values when called without an axis, or
reduces one dimension when given an axis. It returns `float32` for `float32`
input and `float64` for `float64`, integer, and boolean input.

## Python-style slicing

`CudaArray::slice()` accepts a comma-separated expression, or one selector per
axis. Ranges have an exclusive stop, omitted bounds default to the axis limits,
and negative indices/bounds count from the end. Range bounds are clipped to the
axis size; individual indices outside the axis throw `InvalidArgumentException`.

```php
$batch = $tensor->slice('10:20, :');
$columns = $tensor->slice(':, 2:8:2');
$lastRows = $tensor->slice('-10:');
$row = $tensor->slice(3);       // Removes the first axis.
$oneRow = $tensor->slice('3:4'); // Keeps that axis with size 1.

// Dynamic equivalents: [start, stop] or [start, stop, step].
$batch = $tensor->slice([$start, $stop], null);
$columns = $tensor->slice(null, [2, 8, 2]);
```

Integers remove axes; `null`, `':'`, omitted axes and `slice()` with no arguments
select the full extent. A fully indexed tensor has shape `[]` and size 1;
`toArray()` and `toHost()->toArray()` represent it as a one-element array.
Steps must be positive integers no larger than `INT_MAX`. Zero/negative steps,
ellipsis, new axes, index lists and executable expressions are not supported.
Expressions are parsed as integers/ranges, never evaluated as PHP.

Outside Fusion, slices are zero-copy views with shared storage: writes through a
view affect its parent, and the parent remains alive while the view is retained.
Host transfers pack strided views into contiguous host storage. Row assignments
involving strided tensors stage the source on the host before writing, preserving
overlapping-source correctness; this is not a GPU-only bulk scatter operation.
Fusion captures
slice index transformations without materializing during capture; returned
compiled/scoped outputs use the existing independent-output storage contract.

Empty ranges are supported, preserving the remaining shape: `[0, 4]` converts to
`[]`, whereas `[3, 0]` converts to `[[], [], []]`. Elementwise operations, casts,
transfers, reshape/transpose and matmul handle zero elements. Sum/product over an
empty axis return 0/1; mean returns NaN. Min/max and arg reductions throw when an
empty reduced axis would produce values, because no identity/index exists.
An empty non-reduced output remains empty. Packed-buffer imports accept zero
dimensions; materialized empty results support serialization.

The legacy `__invoke()` and `[]` selection syntax remain unchanged, including
their inclusive ranges. Do not interpret their bounds as the new exclusive
`slice()` bounds.

## Optional kernel fusion

Eager execution remains the default. `Fusion::run()` captures tensor operations
inside a callback and returns materialized `CudaArray` outputs:

```php
use Cuda\Fusion;

$result = Fusion::run(fn() => $tensorA + $tensorB * $tensorC);
$eager = Fusion::run(fn() => $tensorA + $tensorB * $tensorC, enabled: false);
```

For repeated execution, compile a specialized plan once:

```php
$graph = Fusion::compile(
    fn($a, $b, $c) => $a + $b * $c,
    inputs: [$tensorA, $tensorB, $tensorC]
);
$result = $graph->run($tensorA, $tensorB, $tensorC);
$other = $graph->run($otherA, $otherB, $otherC);
print_r($graph->getStats());
print_r($graph->getPlan());
echo $graph->getSource();
```

Compilation invokes the callback once with metadata-only placeholders.
Replay does not invoke PHP callback code again. Inputs must match the example
shapes, dtypes and strides, and execution must use the compilation device and
CUDA context. Input values and pointers can change between executions; keep
the device/context alive until the graph is released. Tensors captured by a
closure are retained by the graph; use callback parameters for replaceable inputs.

Addition, subtraction, multiplication, division, unary operations and comparison methods fuse
into elementwise kernels, with broadcasting, strided/view inputs, scalar
operands and dtype promotion. Each node converts to its own result dtype,
preserving intermediate rounding/narrowing. Generated kernels use the eager
backend's fast-math settings but disable cross-node FMA contraction. `where()`,
safe explicit `astype()` conversions and reshape/transpose/slice index transformations
also fuse. Safe casts work in eager execution too, using the same generated
conversion kernel; unsafe narrowing retains the existing rejection policy.

Reductions, `matmul()` (currently float32) and powers are execution boundaries
using existing kernels. All generated elementwise kernels in a plan are compiled
together through the existing `Compiler` NVRTC infrastructure, then executed in
dependency order around these boundaries. Expressions split at a weighted
cost budget of 32 (math functions, index transforms and float64 have higher
cost). Shared expensive expressions can be materialized instead of recomputed.
Pure repeated binary/unary/cast nodes are deduplicated, and unreachable nodes
are not included in compiled plans. Capture is limited to 512 nodes.

Callbacks can return tensors or nested arrays of tensors, preserving array keys.
Up to four adjacent independent outputs with the same shape can share a kernel.
Other outputs use separate kernels; fusion does not promise one kernel for an
entire callback. `getPlan()` reports step kinds, output counts and the reason
for each materialization, including zero-copy `view` steps for layouts consumed
by compatible native operations. `getStats()` exposes planned fused kernels, boundary
steps, intermediate buffer count, scratch reuse and successful replay count. For
`$a + $b * $c`, the plan has one fused kernel and no intermediate data buffers.

During `run()`, CPU reads, legacy slicing (`__invoke()` and `[]`), serialization and operations not captured
by the planner materialize their required inputs and continue eager execution;
later elementwise operations can form a new segment. During `compile()`, reads
of placeholder data are rejected rather than specializing on example values.
Shape, stride and dtype queries do not execute kernels. Nested capture, Fiber
switching, tensor mutation (including compound assignments), custom kernel
launches and device changes/reset are prohibited during capture. Exceptions
restore eager execution; escaped tensors from an aborted capture cannot be read.
`run()` remains synchronous. Generated PTX is cached per PHP request/thread,
with LRU eviction at 16 entries or 16 MiB. Keys include generated source
(operations, constants, dtypes and layouts), compute capability and CUDA
driver/runtime versions, not input pointers. `Fusion::getCacheStats()` reports
hits, misses, compilations and evictions. `Fusion::clearCache()` drops PTX
without invalidating existing graphs. `compile()` also avoids repeated planning
and module loading during replay.

### Streams, asynchronous execution and CUDA Graph

Plans containing generated kernels, matmul and reductions run on a private
nonblocking stream. Reduction descriptors are kernel parameters rather than
shared global state, so concurrent replays cannot overwrite each other's shapes.
Synchronous compiled replay keeps a reusable stream, readiness event and scratch
workspace; async replays use separate scratch storage. Slots are planned by last consumer,
including native consumers and aliased views. Returned outputs always have
independent storage across replays.
Scratch remains allocated until the graph is released and counts toward the
configured GPU memory budget.

```php
$graph = Fusion::compile(
    fn($a, $b, $c) => ($a + $b * $c)->sqrt(),
    [$tensorA, $tensorB, $tensorC],
    cudaGraph: true
);
$pending = $graph->runAsync($tensorA, $tensorB, $tensorC);
$finished = $pending->isFinished(); // Query without waiting.
$result = $pending->wait();        // Synchronize and retrieve outputs.
```

`cudaGraph: true` opts into a CUDA Graph executable for compatible plans.
Kernel parameters are updated for new input/output pointers before each launch.
`getStats()['backend']` is `cuda-graph`, `stream` or `native`.
CUDA Graph currently supports generated-kernel plans only. Matmul/reduction
plans can use streams and `runAsync()`, but requesting CUDA Graph does not
enable a graph executable for them. Check `cudaGraphCompatible` and
`cudaGraphIncompatibility` to distinguish graph support from `asyncCompatible`.
Power boundaries still require the synchronous native executor, report the
reason in `incompatibility`, and reject `runAsync()`. CUDA failures are reported
rather than hidden behind fallback.

Stream plans allow concurrent `runAsync()` calls. A CUDA Graph executable
allows only one outstanding replay: call `wait()` (or release the execution)
before replaying that graph again. A completion query alone does not release
the execution's retained resources. Inputs and closure-captured tensors remain
alive until completion is collected. While an execution is outstanding, tensor
mutation, custom kernel launches and device changes/reset are blocked.
Inputs must be ready before submission; independent custom async producer
streams still require their existing synchronization contract.

Repeated `wait()` calls return the same outputs. Destroying a pending
`FusionExecution` synchronizes before releasing storage. Execution submission
preallocates tensors and can incur allocation synchronization;
asynchronous kernel submission does not imply a zero-blocking PHP call.
Device shape/stride metadata is allocated lazily, only for kernels that need
it. Generated kernels, reductions, unaries and matmul use compiled or host
descriptors instead of uploading metadata for every result tensor.

For optional synchronous replay phase timing:

```php
$graph->setProfiling(true); // Enable timing and reset phase totals.
$result = $graph->run($tensorA, $tensorB, $tensorC);
print_r($graph->getStats());
$graph->setProfiling(false); // Disable timing and reset phase totals.
```

`profiledExecutions`, `bindTimeNs`, `executeTimeNs` and `collectTimeNs` accumulate
successful synchronous `run()` calls only. Execution timing includes preparation,
submission and waiting; it is not isolated GPU kernel time. Profiling is off by
default. `tensorAllocations` counts replay-created data tensors, not pool cache
misses or CUDA allocation calls. `workspaceBuffers` counts retained scratch slots;
`bufferReuses` and `synchronizations` are cumulative replay counters.

Run the focused comparison of eager, cached scoped, stream replay and CUDA
Graph replay with:

```sh
php -n -d extension=./cuda_build-8.3/modules/cuda.so \
  examples/07_fusion_graph.php --benchmark --elements=65536 --iterations=100
```

The example validates output bytes, reports cold/cache compilation costs and
end-to-end timings, and does not assume CUDA Graph is faster for every workload.

### Real training with Fusion

[`fused.php`](fused.php) trains a 64-64-10 ReLU classifier on UCI Optdigits
without custom CUDA source or environment switches. Packed float32 batches are
uploaded once with `fromBuffer()` and kept on the GPU, including their transpose
views. Forward, stable softmax cross-entropy, backward and clipped SGD are
compiled once per batch shape and replayed with new parameter tensors.
Matmul/reduction boundaries and generated elementwise kernels share the private
stream, with one final synchronization per successful training replay.
Loss is transferred only on reporting epochs, and inference transfers only
predicted class indices.

```sh
php -n -d extension=./cuda_build-8.3/modules/cuda.so fused.php \
  --epochs=200 --batch-size=256 --learning-rate=0.05 --no-save
```

Defaults are 1,000 epochs, batch size 256 and learning rate 0.05. Every default
run trains from deterministic initial weights and checks at least 80% test
accuracy. Omit `--no-save` to save parameters after successful evaluation;
`--load-model` explicitly evaluates the compatible saved model instead of
training. Dataset and model files are ignored by Git. The script reports
one-time uploads, compilation, fused/native step counts and training throughput.
It also reports throughput by epoch block. Add `--profile` to print per-plan
timing, allocation, scratch and synchronization counters after training.
[`fusion_training.phpt`](tests/fusion_training.phpt) checks a complete training
step against eager execution, CPU loss and finite-difference gradients, including
large-logit stability and retained outputs across replays.

## PHP GPU Computing for Machine Learning

Use this PHP CUDA extension to build GPU-accelerated numerical steps into PHP
applications: tensor arithmetic, matrix multiplication, broadcasting,
reductions such as `mean()`, and custom CUDA kernels. These primitives can
support machine-learning data preparation and inference workloads while the
data remains in NVIDIA GPU memory. This is a low-level GPU computing library,
not a complete machine-learning framework. The training example implements
backpropagation explicitly; automatic differentiation and Python
interoperability are not provided.

For data already in packed row-major bytes, avoid creating individual PHP
scalars. `fromFile()` reads raw bytes, whereas `fromNpy()` parses NumPy's
`.npy` format:

```php
use Cuda\CudaArray;
use Cuda\HostArray;

$bytes = pack('g*', 1, 2, 3, 4); // little-endian float32
$host = HostArray::fromBuffer($bytes, [2, 2]);
$gpu = $host->toGpu();
$result = CudaArray::where(
    CudaArray::fromBuffer(pack('C*', 1, 0), [2], 'bool'),
    $gpu,
    CudaArray::zeros([2, 2])
);
$cpuCopy = $result->toHost();
$raw = $cpuCopy->toBuffer();
```

`HostArray` is an alias of `Cuda\ContiguousArray`, a contiguous CPU tensor.
Pass `pinned: true` to its constructor or `fromBuffer()` for page-locked host
storage when repeated transfers justify the extra host memory. `where()`
broadcasts its three inputs; its mask treats nonzero values as true, and its
two value tensors must have the same dtype. `.npy` imports support C-order
little-endian numeric and boolean arrays; Fortran order and big-endian data
are rejected.

## Custom kernels

Register the CUDA kernel's argument metadata, compile to PTX, then launch
with explicit grid and block dimensions:

```php
use Cuda\Compiler;
use Cuda\CudaArray;

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

$data = CudaArray::ones([512]);
$module->launch('scale',
    args: [$data, 3, 512],
    config: ['block' => [256, 1, 1], 'grid' => [2, 1, 1]]
);
print_r(array_slice($data->toArray(), 0, 4)); // [3, 3, 3, 3]
```

`launch()` synchronizes; `launchAsync()` returns an operation ID for `sync()`
or `wait()`. Keep tensors alive until asynchronous work finishes. See
[the JIT examples](examples/04_custom_jit_kernels.php) and
[asynchronous execution](examples/05_jit_async_execution.php).

## API and limits

| API | Purpose |
| --- | --- |
| `Cuda\CudaArray` | GPU allocation, tensor math, reductions, views, imports and `where()` |
| `Cuda\HostArray` / `Cuda\ContiguousArray` | CPU storage, packed buffers, optional pinned memory and `toGpu()` |
| `Cuda\Fusion` / `Cuda\FusionGraph` | Optional expression capture, compiled replay, PTX cache and plan diagnostics |
| `Cuda\FusionExecution` | Pending compatible execution, completion query and synchronized result collection |
| `Cuda\Compiler` / `Cuda\CompiledModule` | NVRTC compilation, cached PTX, synchronous and asynchronous kernels |
| `cuda_get_device_count()` and other `cuda_*` functions | Device selection, properties, memory and synchronization |
| `Cuda\Exception` | Base class for runtime, argument, allocation and compilation errors |

The annotated signatures are in [class stubs](stubs/cuda.stub.php) and
[device function stubs](stubs/cuda_methods.stub.php); runnable examples live
in [examples](examples/README.md). `astype()` supports safe dtype conversions.
GPU data has no CPU fallback. The core API baseline is frozen, and the current
release is beta. Kernel fusion is an optional execution mode: existing code
continues to use eager execution unless capture is explicitly enabled.
Matmul/reduction plans support async replay,
but not CUDA Graph yet; power boundaries still require synchronous execution.

## Contribute

Contributions can be code, tests, documentation, runnable examples, or reports
from another PHP/CUDA/GPU combination. Browse
[open issues](https://github.com/lcmialichi/php-gpu-tensors/issues),
[report a bug](https://github.com/lcmialichi/php-gpu-tensors/issues/new?template=bug_report.yml),
or [propose a feature](https://github.com/lcmialichi/php-gpu-tensors/issues/new?template=feature_request.yml).
Start with the [contribution guide](CONTRIBUTING.md); documentation, examples,
and host-side tests are possible without an NVIDIA GPU.

## Benchmarks

The benchmark suite is maintained in the separate
[PHP GPU Tensors Benchmarks repository](https://github.com/lcmialichi/php-gpu-tensors-benchmarks).
It includes focused `--matmul`, `--import` and `--fusion` runs, JSON/HTML reports,
and a [beta.4 full-suite report](https://github.com/lcmialichi/php-gpu-tensors-benchmarks/blob/main/published-reports/beta4-php83-mx570/README.md)
covering 398 cases on PHP 8.3 NTS and an MX570 A, including Fusion replay and
slicing. Earlier results include a
[published PHP 8.5 NTS vs ZTS comparison](https://github.com/lcmialichi/php-gpu-tensors-benchmarks/blob/main/published-reports/php85-nts-vs-zts/README.md),
as well as a [full-suite benchmark report](https://github.com/lcmialichi/php-gpu-tensors-benchmarks/blob/main/published-reports/php85-nts-vs-zts-full/README.md)
covering 368 cases across five workload groups with downloadable raw reports.

For possible directions, see the [roadmap](ROADMAP.md). A feature does not
need to be listed there to be worth discussing.

Licensed under the [MIT License](LICENSE).