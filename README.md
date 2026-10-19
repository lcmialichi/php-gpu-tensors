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
git clone https://github.com/lcmialichi/php-gpu-tensors.git
cd php-gpu-tensors
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
```

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

### Operations

- **Math:** arithmetic operators and methods, elementwise functions,
  comparisons, `where()`, `maximum()`, `minimum()` and `clamp()`.
- **Linear algebra:** `matmul()` supports batched tensors and broadcasting.
- **Reductions:** `sum()`, `mean()`, `min()`, `max()`, `prod()`, `argMin()`,
  `argMax()`, `all()`, `any()`, `var()`, `std()`, `cumsum()` and `cumprod()`.
- **Neural network helpers:** `softmax()` and `layerNorm()`.
- **Indexing:** `concat()`, `gather()` and `scatterAdd()`.
- **Shape and dtype:** views, transforms, `astype()` and tensor metadata.

Arithmetic broadcasts compatible shapes and promotes dtypes. FP16/BF16
operations use FP32 intermediates; reductions and `scatterAdd()` may return
FP32. Unsupported dtype combinations or gradients raise exceptions. See the
[API stubs](stubs/cuda.stub.php) for method signatures and options.

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
[JIT examples](examples/README.md).

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
[Fusion example](examples/07_fusion_graph.php) for capture and replay.

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

[`examples/08_gpu_classifier.php`](examples/08_gpu_classifier.php) trains an
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

See [API stubs](stubs/cuda.stub.php) for signatures and the
[examples guide](examples/README.md) for runnable code.

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

`HostArray` supports CPU `+`, `-`, `*` and `/` for same-shape, same-dtype
float32/float64 tensors and numeric scalars. Broadcasting is not supported.
CPU-only builds need no CUDA installation; `toGpu()` and `CudaArray` operations
throw if CUDA is unavailable. In CUDA builds, `HostArray` also stores
`CudaArray::toHost()` results and supports the CUDA storage dtypes.

Custom backends can implement `BackendInterface`, but it does not automatically
dispatch tensor operations. To overload PHP operators, extend
`OperatorOverload` and implement `Tensor`. The base class already implements
`OperatorOverloadable` and routes operators to `__add`, `__sub`, `__mul`,
`__div`, `__pow`, `__mod`, `__inc` and `__dec`. Implement each method for your
tensor; throw an exception for operations you do not support. The example below
implements vector arithmetic with another vector or a scalar:

```php
use PhpTensors\OperatorOverload;
use PhpTensors\OperatorOverloadable;
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
            if (($operator === '/' || $operator === '%') && $b == 0) {
                throw new \DivisionByZeroError();
            }
            $result[] = match ($operator) {
                '+' => $a + $b,
                '-' => $a - $b,
                '*' => $a * $b,
                '/' => $a / $b,
                '**' => $a ** $b,
                '%' => fmod($a, $b),
            };
        }
        return new self($result);
    }

    public function __add(mixed $left, mixed $right): mixed { return self::apply($left, $right, '+'); }
    public function __sub(mixed $left, mixed $right): mixed { return self::apply($left, $right, '-'); }
    public function __mul(mixed $left, mixed $right): mixed { return self::apply($left, $right, '*'); }
    public function __div(mixed $left, mixed $right): mixed { return self::apply($left, $right, '/'); }
    public function __pow(mixed $left, mixed $right): mixed { return self::apply($left, $right, '**'); }
    public function __mod(mixed $left, mixed $right): mixed { return self::apply($left, $right, '%'); }
    public function __inc(): mixed { return self::apply($this, 1, '+'); }
    public function __dec(): mixed { return self::apply($this, 1, '-'); }
}

$vector = new VectorTensor([1, 2, 3]);
$sum = $vector + new VectorTensor([10, 20, 30]);
$scaled = 2 * $vector; // Reversed scalar operands are passed in their original order.

print_r($sum->toArray()); // [11, 22, 33]
var_dump($scaled instanceof OperatorOverloadable); // true
```

The operator methods receive both operands in their original order, including
expressions such as `2 * $vector`. Native `HostArray` and `CudaArray` implement
`OperatorOverloadable` directly and use native handlers.

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
