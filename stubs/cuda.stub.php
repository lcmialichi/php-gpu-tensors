<?php

namespace PhpTensors;

class Exception extends \Exception {}
class InvalidArgumentException extends Exception {}
class RuntimeException extends Exception {}
class UnexpectedValueException extends Exception {}
class OutOfMemoryException extends RuntimeException {}
class CompilationException extends RuntimeException {}

interface Numbers {}

interface Tensor
{
    public function getBackend(): string;
    public function getShape(): array;
    public function getSize(): int;
    public function dtype(): string;
    public function toArray(): array;
}

interface BackendInterface
{
    public function getName(): string;
    public function isAvailable(): bool;
    public function supports(string $operation, array $context = []): bool;
    public function createTensor(array $values, array $shape, ?string $dtype = 'float32'): Tensor;
    public function execute(string $operation, array $inputs, array $arguments = []): Tensor;
}

namespace PhpTensors;

/**
 * Synchronous CUDA inference for contiguous FP32 tensors. Uses cuDNN when
 * available and native CUDA kernels otherwise; never falls back to CPU.
 */
final class NN
{
    private function __construct() {}
    /** Cross-correlation, floor output dimensions, optional bias per output channel. */
    public static function conv2d(CudaArray $input, CudaArray $weights, ?CudaArray $bias = null,
        array $stride = [1, 1], array $padding = [0, 0], array $dilation = [1, 1], int $groups = 1): CudaArray {}
    /** Max pooling or average pooling excluding padded cells; floor output dimensions. */
    public static function pool2d(CudaArray $input, array $window, array $stride = [2, 2],
        array $padding = [0, 0], string $mode = 'max'): CudaArray {}
    /** Stable softmax across channels independently at each N,H,W coordinate. */
    public static function softmax(CudaArray $input): CudaArray {}
}

namespace PhpTensors;

/** Optional scoped elementwise fusion; eager operators remain the default. */
final class Fusion
{
    private function __construct() {}
    /** Execute once and return materialized tensor outputs; false uses eager execution. */
    public static function run(callable $callback, bool $enabled = true): mixed {}
    /** Trace with metadata-only inputs and compile all fused segments together. */
    public static function compile(callable $callback, array $inputs, bool $cudaGraph = false): FusionGraph {}
    /** Request-local bounded PTX cache counters. */
    public static function getCacheStats(): array {}
    /** Clear cached PTX without invalidating existing compiled graphs. */
    public static function clearCache(): void {}
}

/** Reusable synchronous plan specialized for input shapes, dtypes and strides. */
final class FusionGraph
{
    private function __construct() {}
    /** Execute with new tensor values without invoking the capture callback again. */
    public function run(CudaArray ...$inputs): mixed {}
    /** Queue a compatible plan on a private stream and retain its inputs until wait(). */
    public function runAsync(CudaArray ...$inputs): FusionExecution {}
    /** Planned kernels, execution boundaries, buffers and replay count. */
    public function getStats(): array {}
    /** Generated CUDA source for all fused segments. */
    public function getSource(): string {}
    /** Execution steps and the reason for each materialization boundary. */
    public function getPlan(): array {}
    /** Enable/reset optional phase timing for synchronous run(); counters remain available when disabled. */
    public function setProfiling(bool $enabled): void {}
}

/** Pending fusion result; destruction waits before releasing tensors. */
final class FusionExecution
{
    private function __construct() {}
    /** Synchronize and return materialized outputs; repeated waits return the same result. */
    public function wait(): mixed {}
    /** Query the completion event without synchronizing. */
    public function isFinished(): bool {}
}

namespace PhpTensors;

/** Compile CUDA C++ source into PTX with NVRTC. */
class Compiler
{
    /** Configure source and target architecture (e.g. sm_86); null selects the device. */
    public function __construct(string $source, ?string $target = null, int $optimization = 2, bool $debug = false, bool $fast_math = true) {}
    /**
     * Register a kernel entry point.
     * @param list<array{name:string,dtype:string,type?:string}>|null $parameters
     */
    public function kernel(string $name, ?array $parameters = [], ?array $headers = []): static {}
    /** Add a CUDA header line. */
    public function header(string $code): static {}
    /** Compile or reuse cached PTX. @throws CompilationException */
    public function compile(bool $optimize = true, bool $debug = false): CompiledModule {}
    /** Registered entry point metadata. @return array<string,array<string,mixed>> */
    public function getKernels(): array {}
    /** Cache counters and compiler configuration. @return array<string,mixed> */
    public function getCacheStats(): array {}
    /** Discard cached PTX. */
    public function clearCache(): bool {}
}

/** PTX module supporting synchronous and queued stream launches. */
class CompiledModule
{
    /** Load PTX into a CUDA context. @throws RuntimeException */
    public function initialize(): bool {}
    /** Choose grid and block sizes for an element count or tensor. */
    public function autoGrid(string $name, int|CudaArray $elements): array {}
    /** Execute a kernel and synchronize. @throws RuntimeException */
    public function launch(string $name, ?array $config = [], ?array $args = []): bool {}
    /** Queue a kernel and return its operation ID. @throws RuntimeException */
    public function launchAsync(string $name, ?array $config = [], ?array $args = []): int {}
    /** Queue a batch and return statuses after synchronization. @throws RuntimeException */
    public function launchAsyncBatch(?array $operations): bool|array {}
    /** Synchronize one operation ID or all outstanding operations. */
    public function sync(?int $op_id = null): bool {}
    /** Query completion without blocking. */
    public function isFinished(?int $op_id = null): bool {}
    /** Wait for one or all operations; timeout is in milliseconds. */
    public function wait(?int $op_id = null, int $timeout_ms = -1): bool {}
    /** Check whether an entry point was registered. */
    public function hasKernel(string $name): bool {}
    /** List registered entry point names. @return list<string> */
    public function getKernels(): array {}
    /** Return loaded PTX, or null before compilation. */
    public function getPtx(): ?string {}
    /** Persist PTX to a file. @throws RuntimeException */
    public function save(string $filename): bool {}
    /** Serialize PTX and entry point metadata. */
    public function __serialize(): array {}
    /** Restore a serialized PTX module. */
    public function __unserialize(array $data): void {}
    /** Query all statuses or one by ID; null means no matching operation. */
    public function getAsyncStatus(?int $op_id = null): ?array {}
    /** Execution counters, timing, and stream statistics. */
    public function getStats(): array {}
    /** Return active operation metadata. */
    public function getPendingOperations(): array {}
    /** Cancel an operation and release its resources. */
    public function cancelOperation(int $op_id): bool {}
    /** Remove completed operation records and return their count. */
    public function cleanup(): int {}
}

/** Opaque CUDA kernel metadata registered by a Compiler. */
final class Kernel {}

namespace PhpTensors;

/**
 * Host-resident tensor; unsupported operations throw RuntimeException.
 */
final class HostArray implements Tensor, Numbers
{
    /** Copy a rectangular numeric PHP array into CPU storage. */
    public function __construct(array $values, ?string $dtype = 'float32', bool $pinned = false) {}
    /** Create from row-major flat values and an explicit shape. */
    public static function fromFlatArray(array $values, array $shape, ?string $dtype = 'float32'): self {}
    /** Create a zero-filled host tensor. */
    public static function zeros(array $shape, ?string $dtype = 'float32'): self {}
    /** Create a one-filled host tensor. */
    public static function ones(array $shape, ?string $dtype = 'float32'): self {}
    /** Create a constant-filled host tensor. */
    public static function full(array $shape, float|int $value, ?string $dtype = 'float32'): self {}
    /** Create a zero-filled tensor matching a prototype's shape and dtype. */
    public static function zerosLike(self $prototype, ?string $dtype = null): self {}
    /** Create a one-filled tensor matching a prototype's shape and dtype. */
    public static function onesLike(self $prototype, ?string $dtype = null): self {}
    /** Import row-major packed bytes. */
    public static function fromBuffer(string $bytes, array $shape, ?string $dtype = 'float32', bool $pinned = false): self {}
    /** Whether storage is page-locked; always false for CPU tensors. */
    public function isPinned(): bool {}
    /** Backend identifier for this tensor. */
    public function getBackend(): string {}
    /** Dtype name. */
    public function dtype(): string {}
    /**
     * Read one element by multidimensional coordinates.
     * @param list<int> $indices
     */
    public function get(array $indices): mixed {}
    /** Read one element with variadic coordinates. */
    public function at(int ...$indices): mixed {}
    /** Shape in row-major dimension order. @return list<int> */
    public function getShape(): array {}
    /** Element strides in row-major order. @return list<int> */
    public function getStrides(): array {}
    /** Materialize nested PHP arrays. */
    public function toArray(): array {}
    /** Number of dimensions. */
    public function getNdims(): int {}
    /** Total number of elements. */
    public function getSize(): int {}
    /** Element dtype name, e.g. float32, float16 or bfloat16. */
    public function getDtype(): string {}
    /** Bytes per element. */
    public function getElementSize(): int {}
    /** Length of the first dimension. */
    public function count(): int {}
    /** Add another same-shaped CPU tensor or scalar. */
    public function add(HostArray|int|float|bool $other): self {}
    /** Subtract another same-shaped CPU tensor or scalar. */
    public function subtract(HostArray|int|float|bool $other): self {}
    /** Multiply by another same-shaped CPU tensor or scalar. */
    public function multiply(HostArray|int|float|bool $other): self {}
    /** Divide by another same-shaped CPU tensor or non-zero scalar. */
    public function divide(HostArray|int|float|bool $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function fromFile(string $path, array $shape, ?string $dtype = 'float32'): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function fromNpy(string $path): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function where(self $condition, self $x, self $y): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function __invoke(int|array|null ...$slices): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function slice(int|string|array|null ...$selectors): self {}
    /** Shape, dtype, and element count for debugging. */
    public function __debugInfo(): array {}
    /** Unsupported on the CPU backend until implemented. */
    public function concat(array $tensors, ?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function maximum(self|int|float|bool $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function minimum(self|int|float|bool $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function clamp(float|int|null $min = null, float|int|null $max = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function matmul(self $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function transpose(?array $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function copy(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function swapaxes(int $axis1, int $axis2): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function moveaxis(int|array $source, int|array $destination): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function diagonal(int $offset = 0, int $axis1 = 0, int $axis2 = 1): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function repeat(int|array $repeats, ?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function tile(int|array $reps): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function flip(int|array|null $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function roll(int|array $shift, int|array|null $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function power(self|float|int $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function sqrt(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function exp(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function expm1(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function exp2(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function log(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function log1p(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function log2(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function log10(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function sin(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function cos(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function tan(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function asin(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function acos(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function atan(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function sinh(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function cosh(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function tanh(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function abs(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function neg(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function floor(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function ceil(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function round(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function gt(float|self $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function lt(float|self $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function eq(float|self $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function ne(float|self $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function ge(float|self $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function le(float|self $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function sum(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function mean(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function max(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function min(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function prod(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function cumsum(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function cumprod(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function argMax(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function argMin(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function all(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function any(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function var(?int $axis = null, int $correction = 0): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function std(?int $axis = null, int $correction = 0): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function softmax(int $axis = -1): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function layerNorm(array $axes, float $epsilon = 0.00001, ?self $scale = null, ?self $bias = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function gather(self $indices, int $axis = 0): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function scatterAdd(self $indices, self $updates, int $axis = 0): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function reshape(array $shape): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function squeeze(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function unsqueeze(int $axis): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function broadcastTo(array $shape): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function flatten(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function fullLike(self $prototype, float|int $value, ?string $dtype = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function eye(int $rows, ?int $columns = null, int $k = 0, ?string $dtype = 'float32'): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function identity(int $size, ?string $dtype = 'float32'): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function arange(float|int $start, float|int|null $stop = null, float|int $step = 1, ?string $dtype = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function linspace(float|int $start, float|int $stop, int $num = 50, bool $endpoint = true, ?string $dtype = 'float64'): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function rand(array $shape, float|int $min = 0, float|int $max = 1, ?string $dtype = 'float32', ?int $seed = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function randn(array $shape, ?string $dtype = 'float32', ?int $seed = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function astype(string $dtype): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function requiresGrad(bool $requiresGrad = true): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function backward(?self $gradient = null, bool $retainGraph = false): void {}
    /** Unsupported on the CPU backend until implemented. */
    public function grad(): ?self {}
    /** Unsupported on the CPU backend until implemented. */
    public function zeroGrad(): void {}
    /** Unsupported on the CPU backend until implemented. */
    public function detach(): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function item(): int|float|bool {}
    /** Copy CPU storage into a GPU tensor; requires a CUDA-enabled build. */
    public function toGpu(): CudaArray {}
    /** Return this already host-resident tensor. */
    public function toHost(): self {}
    /** Export contiguous row-major bytes without PHP scalar expansion. */
    public function toBuffer(): string {}
    /** Serialize shape, dtype, and raw CPU storage. */
    public function __serialize(): array {}
    /** Restore a serialized host tensor. */
    public function __unserialize(array $data): void {}
}

namespace PhpTensors;

/** Contiguous host-memory buffer for explicit copies to CUDA. */
/** GPU-resident n-dimensional tensor; [] uses native dimension handlers. */
class CudaArray implements \PhpTensors\Tensor, \PhpTensors\Numbers
{
    /** Import row-major little-endian packed bytes matching shape and dtype. */
    public static function fromBuffer(string $bytes, array $shape, ?string $dtype = 'float32'): CudaArray {}
    /**
     * Copy a flat numeric PHP array directly into the requested row-major shape.
     * Supports float16 and bfloat16 storage.
     */
    public static function fromFlatArray(array $values, array $shape, ?string $dtype = 'float32'): CudaArray {}
    /** Read raw packed bytes (not .npy) into GPU storage. */
    public static function fromFile(string $path, array $shape, ?string $dtype = 'float32'): CudaArray {}
    /** Import a C-order little-endian NumPy .npy file. */
    public static function fromNpy(string $path): CudaArray {}
    /** Select x or y by nonzero mask with broadcasting; x/y must share dtype. */
    public static function where(CudaArray $condition, CudaArray $x, CudaArray $y): CudaArray {}
    /** Copy a rectangular numeric PHP array onto the GPU; keys are ignored in iteration order. */
    public function __construct(array $data, ?string $dtype = 'float32') {}
    /** Serialize contiguous GPU storage; views cannot be serialized directly. */
    public function __serialize(): array {}
    /** Restore a serialized GPU tensor. */
    public function __unserialize(array $data): void {}
    /** Slice a tensor (integer, null for full dimension, or [start,end] range). */
    public function __invoke(int|null|array ...$slices): CudaArray {}
    /** Python-style, stop-exclusive slicing; integers remove axes. Returns a shared view outside Fusion. */
    public function slice(int|string|array|null ...$selectors): CudaArray {}
    /** Return shape, dtype, and count for debugging. */
    public function __debugInfo(): array {}
    /**
     * Concatenate tensors along an axis. Supported in synchronous Fusion execution,
     * but not asynchronous replay or CUDA Graph capture.
     * @param list<CudaArray> $tensors
     */
    public function concat(array $tensors, ?int $axis = null): CudaArray {}
    /** Elementwise multiply with scalar or broadcast-compatible tensor. */
    public function multiply(CudaArray|float|int|bool $other): CudaArray {}
    /** Elementwise divide with scalar or broadcast-compatible tensor. */
    public function divide(CudaArray|float|int|bool $other): CudaArray {}
    /** Elementwise add with scalar or broadcast-compatible tensor. */
    public function add(CudaArray|float|int|bool $other): CudaArray {}
    /** Elementwise subtract with scalar or broadcast-compatible tensor. */
    public function subtract(CudaArray|float|int|bool $other): CudaArray {}
    /** Elementwise maximum with broadcasting. Ties route gradients to the left operand. */
    public function maximum(CudaArray|float|int|bool $other): CudaArray {}
    /** Elementwise minimum with broadcasting. Ties route gradients to the left operand. */
    public function minimum(CudaArray|float|int|bool $other): CudaArray {}
    /** Clamp values to optional inclusive scalar bounds. */
    public function clamp(float|int|null $min = null, float|int|null $max = null): CudaArray {}
    /**
     * Matrix multiply (2D or batched). Matching float32, float16 and bfloat16
     * operands are supported; low-precision products accumulate in float32
     * and retain the operand dtype. FP16/BF16 gradients are returned as float32.
     */
    public function matmul(CudaArray $other): CudaArray {}
    /** Permute axes; omitting axis reverses their order. */
    public function transpose(?array $axis = null): CudaArray {}
    /** Copy logical elements to independent contiguous storage. */
    public function copy(): CudaArray {}
    /** Swap two axes and return a view. */
    public function swapaxes(int $axis1, int $axis2): CudaArray {}
    /**
     * Move one or more axes to new positions and return a view.
     * @param int|list<int> $source
     * @param int|list<int> $destination
     */
    public function moveaxis(int|array $source, int|array $destination): CudaArray {}
    /**
     * Extract a diagonal into a tensor; supported by Fusion kernels.
     */
    public function diagonal(int $offset = 0, int $axis1 = 0, int $axis2 = 1): CudaArray {}
    /**
     * Repeat each element a fixed or per-element number of times.
     * Null axis flattens first; per-element counts are synchronous Fusion boundaries.
     * @param int|list<int> $repeats
     */
    public function repeat(int|array $repeats, ?int $axis = null): CudaArray {}
    /**
     * Repeat the tensor by scalar or per-axis repetition counts.
     * Supported by Fusion kernels.
     * @param int|list<int> $reps
     */
    public function tile(int|array $reps): CudaArray {}
    /**
     * Reverse selected axes, or all axes by default.
     * Supported by Fusion kernels.
     * @param int|list<int>|null $axis
     */
    public function flip(int|array|null $axis = null): CudaArray {}
    /**
     * Circularly shift values; null axis rolls the flattened logical tensor.
     * Supported by Fusion kernels.
     * @param int|list<int> $shift
     * @param int|list<int>|null $axis
     */
    public function roll(int|array $shift, int|array|null $axis = null): CudaArray {}
    /** Elementwise exponentiation. */
    public function power(CudaArray|float|int $other): CudaArray {}
    /** Elementwise square root. */
    public function sqrt(): CudaArray {}
    /** Elementwise exponential. */
    public function exp(): CudaArray {}
    /** Elementwise exp(x) - 1, evaluated with improved precision near zero. */
    public function expm1(): CudaArray {}
    /** Elementwise base-2 exponential. */
    public function exp2(): CudaArray {}
    /** Elementwise natural logarithm. */
    public function log(): CudaArray {}
    /** Elementwise log(1 + x), evaluated with improved precision near zero. */
    public function log1p(): CudaArray {}
    /** Elementwise base-2 logarithm. */
    public function log2(): CudaArray {}
    /** Elementwise base-10 logarithm. */
    public function log10(): CudaArray {}
    /** Elementwise sine. */
    public function sin(): CudaArray {}
    /** Elementwise cosine. */
    public function cos(): CudaArray {}
    /** Elementwise tangent. */
    public function tan(): CudaArray {}
    /** Elementwise inverse sine. */
    public function asin(): CudaArray {}
    /** Elementwise inverse cosine. */
    public function acos(): CudaArray {}
    /** Elementwise inverse tangent. */
    public function atan(): CudaArray {}
    /** Elementwise hyperbolic sine. */
    public function sinh(): CudaArray {}
    /** Elementwise hyperbolic cosine. */
    public function cosh(): CudaArray {}
    /** Elementwise hyperbolic tangent. */
    public function tanh(): CudaArray {}
    /** Elementwise absolute value. */
    public function abs(): CudaArray {}
    /** Elementwise negation. */
    public function neg(): CudaArray {}
    /** Elementwise floor. */
    public function floor(): CudaArray {}
    /** Elementwise ceil. */
    public function ceil(): CudaArray {}
    /** Elementwise rounding. */
    public function round(): CudaArray {}
    /** Elementwise greater-than comparison. */
    public function gt(float|CudaArray $other): CudaArray {}
    /** Elementwise less-than comparison. */
    public function lt(float|CudaArray $other): CudaArray {}
    /** Elementwise equality comparison. */
    public function eq(float|CudaArray $other): CudaArray {}
    /** Elementwise inequality comparison. */
    public function ne(float|CudaArray $other): CudaArray {}
    /** Elementwise greater-than-or-equal comparison. */
    public function ge(float|CudaArray $other): CudaArray {}
    /** Elementwise less-than-or-equal comparison. */
    public function le(float|CudaArray $other): CudaArray {}
    /** Sum across an axis; omit axis to reduce all elements. */
    public function sum(?int $axis = null): CudaArray {}
    /** Mean across an axis; omit axis to reduce all elements. */
    public function mean(?int $axis = null): CudaArray {}
    /** Maximum across an axis; omit axis to reduce all elements. */
    public function max(?int $axis = null): CudaArray {}
    /** Minimum across an axis; omit axis to reduce all elements. */
    public function min(?int $axis = null): CudaArray {}
    /** Product across an axis; omit axis to reduce all elements. */
    public function prod(?int $axis = null): CudaArray {}
    /** Cumulative sum; null axis flattens the input. Integer inputs widen to int64. */
    public function cumsum(?int $axis = null): CudaArray {}
    /** Cumulative product; null axis flattens the input. Integer inputs widen to int64. */
    public function cumprod(?int $axis = null): CudaArray {}
    /** Logical AND over nonzero values; returns a bool tensor. */
    public function all(?int $axis = null): CudaArray {}
    /** Logical OR over nonzero values; returns a bool tensor. */
    public function any(?int $axis = null): CudaArray {}
    /** Population variance by default; correction adjusts the denominator. */
    public function var(?int $axis = null, int $correction = 0): CudaArray {}
    /** Square root of var(). */
    public function std(?int $axis = null, int $correction = 0): CudaArray {}
    /** Numerically stable softmax; low-precision input is widened to float32. */
    public function softmax(int $axis = -1): CudaArray {}
    /**
     * Normalize across explicit axes, promoting integer input to float64.
     * Scale and bias shapes follow the axes in the same order.
     * @param list<int> $axes
     */
    public function layerNorm(array $axes, float $epsilon = 0.00001, ?CudaArray $scale = null, ?CudaArray $bias = null): CudaArray {}
    /** Index of maximum across an axis. */
    public function argMax(?int $axis = null): CudaArray {}
    /** Index of minimum across an axis. */
    public function argMin(?int $axis = null): CudaArray {}
    /** Tensor shape in row-major dimension order. @return list<int> */
    public function getShape(): array {}
    /** Element strides, including for noncontiguous views. @return list<int> */
    public function getStrides(): array {}
    /** Number of dimensions. */
    public function getNdims(): int {}
    /** Number of tensor elements. */
    public function getSize(): int {}
    /** Materialize nested PHP arrays on the CPU. */
    public function toArray(): array {}
    /** Copy into contiguous CPU storage without PHP scalar expansion. */
    public function toHost(): \PhpTensors\HostArray {}
    /** Download logical row-major bytes directly, without PHP scalar expansion. */
    public function toBuffer(): string {}
    /** Create a view with the same element count. */
    public function reshape(array $shape): CudaArray {}
    /** Remove singleton dimensions; an optional axis must have size one. */
    public function squeeze(?int $axis = null): CudaArray {}
    /** Insert a singleton dimension at axis. */
    public function unsqueeze(int $axis): CudaArray {}
    /** Broadcast singleton axes or add leading axes without copying; captured by synchronous Fusion. */
    public function broadcastTo(array $shape): CudaArray {}
    /** Create a one-dimensional view. */
    public function flatten(): CudaArray {}
    /** Allocate a zero-filled GPU tensor. */
    public static function zeros(array $shape, ?string $dtype = 'float32'): CudaArray {}
    /** Allocate a one-filled GPU tensor. */
    public static function ones(array $shape, ?string $dtype = 'float32'): CudaArray {}
    /** Allocate a constant-filled GPU tensor. */
    public static function full(array $shape, float|int $value, ?string $dtype = 'float32'): CudaArray {}
    /** Allocate a zero-filled tensor with a prototype's shape and, by default, dtype. */
    public static function zerosLike(CudaArray $prototype, ?string $dtype = null): CudaArray {}
    /** Allocate a one-filled tensor with a prototype's shape and, by default, dtype. */
    public static function onesLike(CudaArray $prototype, ?string $dtype = null): CudaArray {}
    /** Allocate a constant-filled tensor with a prototype's shape and, by default, dtype. */
    public static function fullLike(CudaArray $prototype, float|int $value, ?string $dtype = null): CudaArray {}
    /** Create a rectangular identity matrix with an optional diagonal offset. */
    public static function eye(int $rows, ?int $columns = null, int $k = 0, ?string $dtype = 'float32'): CudaArray {}
    /** Create a square identity matrix. */
    public static function identity(int $size, ?string $dtype = 'float32'): CudaArray {}
    /**
     * Create evenly spaced values in the half-open interval [start, stop).
     * With one argument, it is interpreted as stop. Integer arguments default to int64;
     * any floating-point argument defaults to float64.
     */
    public static function arange(float|int $start, float|int|null $stop = null, float|int $step = 1, ?string $dtype = null): CudaArray {}
    /**
     * Create num evenly spaced values between start and stop.
     * The default output dtype is float64.
     */
    public static function linspace(float|int $start, float|int $stop, int $num = 50, bool $endpoint = true, ?string $dtype = 'float64'): CudaArray {}
    /**
     * Generate uniformly distributed GPU values in [min, max).
     * Omitted seeds produce new values; a seed makes the result reproducible.
     */
    public static function rand(array $shape, float|int $min = 0, float|int $max = 1, ?string $dtype = 'float32', ?int $seed = null): CudaArray {}
    /**
     * Generate standard-normal GPU values.
     * Omitted seeds produce new values; a seed makes the result reproducible.
     */
    public static function randn(array $shape, ?string $dtype = 'float32', ?int $seed = null): CudaArray {}
    /**
     * Gather values using same-rank int32 indices along an axis.
     * Synchronous Fusion execution is supported; async and CUDA Graph capture are not.
     */
    public function gather(CudaArray $indices, int $axis = 0): CudaArray {}
    /**
     * Add updates at indexed positions and return a new tensor. Duplicate indices accumulate.
     * Currently supports float32/float64. Synchronous Fusion execution is
     * supported; async and CUDA Graph capture are not.
     */
    public function scatterAdd(CudaArray $indices, CudaArray $updates, int $axis = 0): CudaArray {}
    /**
     * Identical dtype reuses storage; safe conversions allocate converted storage.
     * @throws InvalidArgumentException For an unsafe conversion.
     * @throws RuntimeException If compilation or GPU execution fails.
     */
    public function astype(string $dtype): CudaArray {}
    /** Element dtype name, e.g. float32, float16 or bfloat16. */
    public function dtype(): string {}
    /** Backend that owns this tensor's storage. */
    public function getBackend(): string {}
    /** Enable or disable reverse-mode gradient tracking on a leaf tensor. */
    /** FP16/BF16 tracking is currently supported for matmul and shape views only; gradients use float32. */
    public function requiresGrad(bool $requiresGrad = true): CudaArray {}
    /** Compute gradients; non-scalar tensors require a seed. Graphs are freed unless retained. */
    public function backward(?CudaArray $gradient = null, bool $retainGraph = false): void {}
    /** Return the accumulated gradient of a leaf tensor, or null before backward(). */
    public function grad(): ?CudaArray {}
    /** Clear the accumulated gradient. */
    public function zeroGrad(): void {}
    /** Return a view that shares storage but is disconnected from gradient history. */
    public function detach(): CudaArray {}
    /** Return the sole tensor element as a PHP scalar. */
    public function item(): int|float|bool {}
}

/**
 * Functional SGD and AdamW updates. Optimizer state is explicit so steps can
 * be captured and replayed by Fusion without storing hidden tensor history.
 */
namespace PhpTensors;

final class Optimizer
{
    private function __construct() {}
    public static function adamW(float $learningRate = 0.001, float $beta1 = 0.9,
        float $beta2 = 0.999, float $epsilon = 1.0e-8, float $weightDecay = 0.01): self {}
    public static function sgd(float $learningRate = 0.01, float $momentum = 0.0,
        float $weightDecay = 0.0): self {}
    /** Create zeroed moment/velocity tensors for this ordered parameter list. */
    public function initState(array $parameters): array {}
    /** Clear accumulated gradients on the supplied leaf parameters. */
    public function zeroGrad(array $parameters): void {}
    /**
     * Apply one functional optimizer step.
     * @param list<CudaArray> $parameters
     * @param list<CudaArray> $gradients
     * @param array<string,mixed> $state State returned by initState() or step().
     * @param list<bool>|null $weightDecayMask Defaults to decaying every parameter.
     * @return array{parameters:list<CudaArray>,state:array<string,mixed>}
     */
    public function step(array $parameters, array $gradients, array $state,
        ?CudaArray $learningRate = null, ?array $weightDecayMask = null): array {}
}

/** Abstract base dispatching PHP arithmetic operators to magic methods. */
abstract class Number implements Numbers
{
    /** Handle addition. */
    abstract public function __add(mixed $left, mixed $right): mixed;
    /** Handle subtraction. */
    abstract public function __sub(mixed $left, mixed $right): mixed;
    /** Handle multiplication. */
    abstract public function __mul(mixed $left, mixed $right): mixed;
    /** Handle division. */
    abstract public function __div(mixed $left, mixed $right): mixed;
    /** Handle exponentiation. */
    abstract public function __pow(mixed $left, mixed $right): mixed;
    /** Handle remainder. */
    abstract public function __mod(mixed $left, mixed $right): mixed;
    /** Return the incremented value. */
    abstract public function __inc(): mixed;
    /** Return the decremented value. */
    abstract public function __dec(): mixed;
}
