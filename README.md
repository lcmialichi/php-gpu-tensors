# PHP Tensors: native CPU and CUDA tensors for PHP

<p align="center">
    <img src="art/php-gpu-tensors.jpg" alt="PHP Tensors: native numerical computing, CUDA kernels and kernel fusion for PHP" width="350">
</p>

<p align="center">
    <strong>Write tensor math in PHP. Run it on the CPU or an NVIDIA GPU.<br>Fuse repeated expressions without leaving PHP. No Python runtime required.</strong>
</p>

<p align="center">
    <a href="https://github.com/lcmialichi/php-gpu-tensors/actions/workflows/ci.yml"><img alt="Build and CPU/API tests" src="https://github.com/lcmialichi/php-gpu-tensors/actions/workflows/ci.yml/badge.svg"></a>
    <a href="https://packagist.org/packages/lcmialichi/php-gpu-tensors"><img alt="Latest package version, including beta releases" src="https://img.shields.io/packagist/v/lcmialichi/php-gpu-tensors?include_prereleases&label=package"></a>
    <img alt="PHP 8.1 through 8.5, NTS and ZTS builds" src="https://img.shields.io/badge/PHP-8.1--8.5-777BB4">
    <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/github/license/lcmialichi/php-gpu-tensors"></a>
</p>

PHP Tensors is a **native PHP extension** for numerical computing, GPU tensor
operations and building machine-learning experiments. Start with CPU arrays,
move data explicitly to CUDA, or compile reusable computation plans with Fusion.
It is a low-level tensor library, **not a complete ML framework**.

**Beta | Linux | PHP 8.1-8.5 | NTS/ZTS builds | CUDA optional**

[Get started](#get-started) · [CPU tensors](#cpu-tensors) ·
[GPU and Fusion](#gpu-tensors-and-optional-fusion) ·
[Capabilities](#what-can-it-do) · [Examples](#learn-by-example) ·
[API map](#api-map-for-developers-and-ai-tools) ·
[Full guide](docs/guide.md) · [Contribute](#help-build-it)

## Why PHP Tensors?

- **Familiar expressions:** write `$a + $b * $c`, with broadcasting and typed storage.
- **A useful CPU backend:** views, floating-point arithmetic, comparisons,
  selection, reductions and batched matrix multiplication without CUDA.
- **Opt-in GPU acceleration:** keep tensors on the device and choose when to transfer.
- **Optional kernel fusion:** capture a callback, compile once and replay it;
  supported elementwise work is fused around native operation boundaries.
- **Tools beyond array math:** reverse-mode autograd, functional SGD/AdamW,
  runtime-compiled CUDA C++ kernels and optional cuDNN CNN inference.
- **Inspectable execution:** examine generated source, plan boundaries and
  backend diagnostics instead of relying on hidden device migration.

## Get started

### Build without a GPU

You need Linux, PHP development headers, `phpize`, `php-config`, a C/C++
compiler, `make` and `autoconf`. No CUDA installation is needed for this path.

```bash
git clone https://github.com/lcmialichi/php-gpu-tensors.git
cd php-gpu-tensors
./compile.sh --cpu-only
./run-tests.sh --cpu-only

PHP_VERSION=$(php-config --version | cut -d. -f1,2)
EXTENSION="$PWD/phptensors_build-${PHP_VERSION}/modules/phptensors.so"
php -n -d extension="$EXTENSION" -r 'var_dump(extension_loaded("phptensors"));'
```

The last command should print `bool(true)`. Load that module when running the
examples below, for example `php -n -d extension="$EXTENSION" your-script.php`.
The extension name is `phptensors`; its PHP namespace is `PhpTensors`.

### Enable the CUDA backend

Install a CUDA Toolkit with NVRTC and development libraries, then run
`./compile.sh` without `--cpu-only`. GPU execution additionally requires a
compatible NVIDIA driver and a visible CUDA-capable GPU.

```bash
./compile.sh
./run-tests.sh --require-gpu
```

The build auto-detects the Toolkit and otherwise selects CPU-only mode.
Native Windows and macOS builds are not supported.

For PIE installation, PHP binary selection, Docker and PHP-FPM setup, see the
[installation guide](docs/guide.md#install). The published package baseline is
`lcmialichi/php-gpu-tensors:0.1.0-beta.5`; this checkout contains newer work.

## CPU tensors

```php
use PhpTensors\HostArray;

$a = new HostArray([[1, 2, 3], [4, 5, 6]], 'float32');
$bias = new HostArray([10, 20, 30], 'float32');

$result = $a + $bias; // Broadcast the row across both rows.
print_r($result->toArray()); // [[11, 22, 33], [14, 25, 36]]

$features = $a->matmul($a->transpose())->tanh();
print_r($features->getShape()); // [2, 2]
```

`HostArray` stores all 13 supported dtypes at their native widths. Arithmetic
supports the four floating dtypes; other operations have their own dtype
contracts. [See the CPU operation table](docs/guide.md#backend-contracts-and-custom-backends).

## GPU tensors and optional Fusion

```php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;

$a = CudaArray::ones([4]);
$b = CudaArray::full([1], 2.0); // Broadcasts to [4].
$c = CudaArray::full([4], 3.0);

$eager = $a + $b * $c; // Normal GPU execution; no capture required.

$plan = Fusion::compile(
    fn($x, $y, $z) => $x + $y * $z,
    inputs: [$a, $b, $c]
);
$fused = $plan->run($a, $b, $c); // Reuse the plan with compatible inputs.

print_r($fused->toArray()); // [7, 7, 7, 7]; explicitly downloads to PHP.
```

Eager execution is the default. For a scoped, one-shot capture, use
`Fusion::run(fn() => $a + $b * $c)`. Compiled plans also support
[async replay and compatible CUDA Graphs](docs/guide.md#streams-asynchronous-execution-and-cuda-graphs).
The callback runs during capture, not on each replay. Not every operation is
fusible, and a plan may contain multiple kernels and native boundaries.

**Data placement is explicit:** `HostArray::toGpu()` uploads,
`CudaArray::toHost()` downloads, and `toArray()` exports PHP values. CPU
operations do not silently migrate to the GPU, or vice versa.

## What can it do?

| Capability | CPU `HostArray` | CUDA backend |
| --- | --- | --- |
| Typed storage, imports/exports, copy and safe casts | 13 dtypes | 13 dtypes |
| Floating arithmetic and broadcasting | FP16, BF16, FP32, FP64 | Supported; operation-dependent dtype rules |
| Retained views, comparisons and `where` | Supported core | Supported, with a wider transform/indexing API |
| Reductions | `sum`, `mean`, `min`, `max` | Also scans, statistics and discrete reductions |
| Batched matrix multiplication | Matching FP16/BF16/FP32, rank >= 2 | Matching FP16/BF16/FP32; accelerated compatible paths |
| Reverse-mode autograd and SGD/AdamW | Not supported | Supported operation/dtype subsets; first-order gradients |
| Fusion, async replay and CUDA Graphs | Not supported | Opt-in; compatibility depends on the plan |
| Custom CUDA C++ kernels compiled with NVRTC | Not supported | Supported |
| CNN inference | Not supported | Contiguous FP32; optional cuDNN, native CUDA alternative |

This is an overview, not a promise that every dtype works with every operation.
See the [API and runtime limits](docs/guide.md#api-and-limits),
[autograd contracts](docs/guide.md#autograd-and-optimizers) and
[1.0 readiness checklist](TODO.md) before relying on a beta API.
CUDA uses one owning device and thread per PHP process; concurrent CUDA calls
from PHP/ZTS threads are not supported. PHP-FPM workers must fork before CUDA
initialization. Shapes have at most 10 axes. See the guide for lifecycle rules.

## Learn by example

| Start here | Learn |
| --- | --- |
| [Tensor basics](examples/01_basics_cuda_array.php) | Allocation, arithmetic and data transfer |
| [Math and reductions](examples/02_math_and_reductions.php) | Numerical operations and aggregation |
| [Views and broadcasting](examples/03_advanced_manipulation.php) | Shape and layout manipulation |
| [Custom kernels](examples/04_custom_jit_kernels.php) | Compile CUDA C++ from PHP |
| [Async launches](examples/05_jit_async_execution.php) | Submit work, poll and synchronize |
| [Fusion and CUDA Graphs](examples/07_fusion_graph.php) | Capture, inspect and replay plans |
| [Train an MLP](examples/08_gpu_classifier.php) | MNIST in one readable loop: autograd, AdamW and Fusion replay |

The runnable examples above require the CUDA backend and a working GPU.
See the [examples guide](examples/README.md) for setup and the
[full usage guide](docs/guide.md) for CPU and custom-backend examples.

### A measured Fusion workload

An earlier 256-unit MLP experiment on a **GeForce MX570 A (4 GB), PHP 8.3 NTS,
CUDA runtime 12.3** measured **4.08 ms/step eager versus 0.66 ms/step with
Fusion replay**, with the same 77.29% accuracy and 0.853 ROC AUC.
This used handcrafted PatchCamelyon features, not an end-to-end image CNN.
It is one workload, not a universal speedup or a benchmark of this exact checkout.

Use the [kernel benchmark](docs/guide.md#kernel-benchmark) to measure your
workload, or explore the separate
[benchmark reports](https://github.com/lcmialichi/php-gpu-tensors-benchmarks).
For data movement, use the [transfer-cost benchmark](docs/guide.md#transfer-cost-benchmark):
it separates host staging from GPU copies and reports latency, throughput and
explicit memory/allocation scope.

## API map for developers and AI tools

The authoritative signatures live in [stubs/cuda.stub.php](stubs/cuda.stub.php).
Prefer those over inferring methods from examples. The CPU-only build exposes
a CUDA-unavailable guard, not the full CUDA API.

| Entry point | Purpose |
| --- | --- |
| `PhpTensors\HostArray`, `PhpTensors\CudaArray` | Explicit CPU/GPU storage and operations |
| `PhpTensors\Fusion`, `FusionGraph`, `FusionExecution` | Capture, compiled plans and pending results |
| `PhpTensors\Compiler`, `CompiledModule` | NVRTC compilation and kernel launches |
| `PhpTensors\Optimizer`, `NN` | Functional optimizers and CUDA CNN inference |
| `PhpTensors\Tensor`, `BackendInterface` | Metadata/transfer and optional custom-backend contracts |
| `PhpTensors\OperatorOverload` | Base class for arithmetic hooks in userland PHP |
| `cuda_*()` functions | Devices, synchronization, memory and diagnostics |

`OperatorOverloadable` is a marker; it does not install operator dispatch.
Custom arithmetic objects extend `OperatorOverload`. See the
[working custom tensor example and operator contract](docs/guide.md#backend-contracts-and-custom-backends).
Native metadata is generated from the stubs and checked against loaded-extension
reflection. [Development and API checks](docs/guide.md#docker-and-tests)
explain how to keep those contracts aligned.

## Validation and release status

CPU build/tests run in CI for PHP 8.1-8.5, NTS and ZTS. A separate CUDA-profile
job builds and checks metadata and host-side behavior **without executing GPU work**.
Current local CPU validation includes PHP 8.4/8.5 ZTS; earlier milestones also
ran real-GPU suites.

Local Docker GPU passthrough now works under WSL 2. CPU/CUDA numerical parity,
transfer correctness and healthy lifetime/ordering checks pass on the MX570 A.
**Release validation is not all green:** the selected run has 16 passed,
five failed fault-injection tests and zero skips; GPU memory/race acceptance
remains open. Historical SDK-stub runs are host-only evidence, not GPU coverage.
See the
[validation history](docs/guide.md#validation-status) and open gates in
[TODO.md](TODO.md) for the scope of each run.

## Help build it

Found a use case? Try an [example](examples/README.md), share your PHP/GPU
configuration, or [open an issue](https://github.com/lcmialichi/php-gpu-tensors/issues).
Tests, documentation, CPU-only work and reproducible benchmarks are valuable
contributions; owning an NVIDIA GPU is not required to contribute.

Read the [contribution guide](CONTRIBUTING.md), follow the
[release checklist](TODO.md), and explore the [longer-term roadmap](ROADMAP.md).
If the project is useful to you, a GitHub star helps others discover it.

## License

[MIT](LICENSE).
