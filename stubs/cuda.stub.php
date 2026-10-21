<?php

namespace PhpTensors;

class Exception extends \Exception {}
class InvalidArgumentException extends Exception {}
class RuntimeException extends Exception {}
class UnexpectedValueException extends Exception {}
class OutOfMemoryException extends RuntimeException {}
class CompilationException extends RuntimeException {}

/** Marker only; userland arithmetic dispatch requires extending OperatorOverload. */
interface OperatorOverloadable {}

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
    /** Compile segments together. Tracked outputs require backward inside the callback or explicit detach. */
    public static function compile(callable $callback, array $inputs, bool $cudaGraph = false): FusionGraph {}
    /** Request-local bounded PTX cache counters. */
    public static function getCacheStats(): array {}
    /** Clear cached PTX without invalidating existing compiled graphs. */
    public static function clearCache(): void {}
}

/** Plan specialized for shapes/dtypes/strides and captured gradient inputs; replay does not build dynamic autograd history. */
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

/**
 * Compile CUDA C++ source into PTX with NVRTC.
 * Construct only after worker fork; CUDA belongs to the first calling process/thread.
 */
final class Compiler
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
final class CompiledModule
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
    public function launchAsyncBatch(array $operations): bool|array {}
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

namespace PhpTensors;

/**
 * Host-resident tensor; unsupported operations throw RuntimeException.
 */
final class HostArray implements Tensor, OperatorOverloadable, \IteratorAggregate, \Countable
{
    /** Iterate scalars in one dimension or zero-copy HostArray rows in higher ranks. */
    public function getIterator(): \Traversable {}
    /** Copy a rectangular numeric PHP array into CPU storage. */
    public function __construct(array $values, ?string $dtype = 'float32', bool $pinned = false) {}
    /** Create from row-major flat values and an explicit shape. */
    public static function fromFlatArray(array $values, array $shape, ?string $dtype = 'float32'): self {}
    /** Create a zero-filled host tensor. */
    public static function zeros(array $shape, ?string $dtype = 'float32'): self {}
    /** Create a one-filled host tensor. */
    public static function ones(array $shape, ?string $dtype = 'float32'): self {}
    /** Create a constant-filled host tensor. */
    public static function full(array $shape, float|int|bool $value, ?string $dtype = 'float32'): self {}
    /** Create a zero-filled tensor matching a prototype's shape and dtype. */
    public static function zerosLike(self $prototype, ?string $dtype = null): self {}
    /** Create a one-filled tensor matching a prototype's shape and dtype. */
    public static function onesLike(self $prototype, ?string $dtype = null): self {}
    /** Create a constant-filled tensor matching a prototype's shape and dtype. */
    public static function fullLike(self $prototype, float|int|bool $value, ?string $dtype = null): self {}
    /** Import row-major packed bytes. */
    public static function fromBuffer(string $bytes, array $shape, ?string $dtype = 'float32', bool $pinned = false): self {}
    /** Whether storage is page-locked; always false in CPU-only builds. */
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
    /** Add a broadcast-compatible floating CPU tensor or scalar. */
    public function add(HostArray|int|float|bool $other): self {}
    /** Subtract a broadcast-compatible floating CPU tensor or scalar. */
    public function subtract(HostArray|int|float|bool $other): self {}
    /** Multiply by a broadcast-compatible floating CPU tensor or scalar. */
    public function multiply(HostArray|int|float|bool $other): self {}
    /** Divide floating CPU tensors/scalars with broadcasting and IEEE NaN/Inf. */
    public function divide(HostArray|int|float|bool $other): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function fromFile(string $path, array $shape, ?string $dtype = 'float32'): self {}
    /** Unsupported on the CPU backend until implemented. */
    public static function fromNpy(string $path): self {}
    /** Select exact x/y storage using numeric truth and broadcasting; x/y dtypes must match. */
    public static function where(self $condition, self $x, self $y): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function __invoke(int|array|null ...$slices): self {}
    /** Retained view; exclusive-stop ranges, positive steps and negative indices match CUDA slice(). */
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
    /** Batched matrix multiplication for matching float32/float16/bfloat16 tensors of rank >= 2. */
    public function matmul(self $other): self {}
    /** Zero-copy axis permutation; null reverses all axes. */
    public function transpose(?array $axis = null): self {}
    /** Independent dense copy of logical contents, preserving dtype and pinning. */
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
    public function power(self|float|int|bool $other): self {}
    /** Floating-point unary math; all four floating storage dtypes are supported. */
    public function sqrt(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function exp(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function expm1(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function exp2(): self {}
    /** Floating-point unary math, preserving IEEE singularities and dtype. */
    public function log(): self {}
    /** Floating-point unary math, preserving IEEE singularities and dtype. */
    public function log1p(): self {}
    /** Floating-point unary math, preserving IEEE singularities and dtype. */
    public function log2(): self {}
    /** Floating-point unary math, preserving IEEE singularities and dtype. */
    public function log10(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function sin(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function cos(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function tan(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function asin(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function acos(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function atan(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function sinh(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function cosh(): self {}
    /** Floating-point unary math, preserving dtype. */
    public function tanh(): self {}
    /** Floating-point absolute value; follows CUDA's signed-zero convention. */
    public function abs(): self {}
    /** Negate floating-point values. */
    public function neg(): self {}
    /** Round floating-point values toward negative infinity. */
    public function floor(): self {}
    /** Round floating-point values toward positive infinity. */
    public function ceil(): self {}
    /** Round floating-point values; halfway values go away from zero. */
    public function round(): self {}
    /** Broadcast comparison returning numeric zero/one masks in the promoted dtype. */
    public function gt(self|int|float|bool $other): self {}
    /** Broadcast comparison returning numeric zero/one masks in the promoted dtype. */
    public function lt(self|int|float|bool $other): self {}
    /** Broadcast comparison returning numeric zero/one masks in the promoted dtype. */
    public function eq(self|int|float|bool $other): self {}
    /** Broadcast comparison returning numeric zero/one masks in the promoted dtype. */
    public function ne(self|int|float|bool $other): self {}
    /** Broadcast comparison returning numeric zero/one masks in the promoted dtype. */
    public function ge(self|int|float|bool $other): self {}
    /** Broadcast comparison returning numeric zero/one masks in the promoted dtype. */
    public function le(self|int|float|bool $other): self {}
    /** Global/axis sum; low precision accumulates in FP32, integer output wraps at its dtype width. */
    public function sum(?int $axis = null): self {}
    /** Global/axis mean; integer/bool inputs produce FP64, low precision produces FP32. */
    public function mean(?int $axis = null): self {}
    /** Global/axis maximum; empty reduction groups are rejected. */
    public function max(?int $axis = null): self {}
    /** Global/axis minimum; empty reduction groups are rejected. */
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
    /** Retained dense view with explicit nonnegative dimensions; non-contiguous inputs need copy(). */
    public function reshape(array $shape): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function squeeze(?int $axis = null): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function unsqueeze(int $axis): self {}
    /** Retained zero-stride view with CUDA-compatible trailing-axis broadcasting. */
    public function broadcastTo(array $shape): self {}
    /** Unsupported on the CPU backend until implemented. */
    public function flatten(): self {}
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
    /** CUDA-compatible safe dtype cast; matching dtype shares storage, otherwise materializes dense storage. */
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
    /** Return the sole logical element; large unsigned integers use exact decimal strings. */
    public function item(): int|float|bool|string {}
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
/**
 * GPU-resident n-dimensional tensor; [] uses native dimension handlers.
 * Lazily initializes CUDA storage in the worker; post-CUDA fork and other threads are rejected.
 */
final class CudaArray implements \PhpTensors\Tensor, \PhpTensors\OperatorOverloadable
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
    /** Select x or y by nonzero mask with broadcasting; x/y must share dtype. Gradients ignore the condition. */
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
    /** Elementwise maximum with broadcasting. Exact ties split gradients equally, including scalar bounds. */
    public function maximum(CudaArray|float|int|bool $other): CudaArray {}
    /** Elementwise minimum with broadcasting. Exact ties split gradients equally, including scalar bounds. */
    public function minimum(CudaArray|float|int|bool $other): CudaArray {}
    /** Compose maximum/minimum bounds: input derivative 0.5 at either boundary, 0.25 where equal bounds coincide. */
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
    /** Elementwise exponentiation. A scalar zero exponent has a zero input gradient. */
    public function power(CudaArray|float|int|bool $other): CudaArray {}
    /** Elementwise square root. */
    public function sqrt(): CudaArray {}
    /** Elementwise exponential. */
    public function exp(): CudaArray {}
    /** Elementwise exp(x) - 1, evaluated with improved precision near zero. */
    public function expm1(): CudaArray {}
    /** Elementwise base-2 exponential. */
    public function exp2(): CudaArray {}
    /** Elementwise natural logarithm; backward is NaN outside the real domain and follows IEEE at zero. */
    public function log(): CudaArray {}
    /** Precise log(1 + x); backward is NaN below -1 and follows IEEE at -1. */
    public function log1p(): CudaArray {}
    /** Elementwise base-2 logarithm; backward is NaN below zero and follows IEEE at zero. */
    public function log2(): CudaArray {}
    /** Elementwise base-10 logarithm; backward is NaN below zero and follows IEEE at zero. */
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
    public function gt(CudaArray|int|float|bool $other): CudaArray {}
    /** Elementwise less-than comparison. */
    public function lt(CudaArray|int|float|bool $other): CudaArray {}
    /** Elementwise equality comparison. */
    public function eq(CudaArray|int|float|bool $other): CudaArray {}
    /** Elementwise inequality comparison. */
    public function ne(CudaArray|int|float|bool $other): CudaArray {}
    /** Elementwise greater-than-or-equal comparison. */
    public function ge(CudaArray|int|float|bool $other): CudaArray {}
    /** Elementwise less-than-or-equal comparison. */
    public function le(CudaArray|int|float|bool $other): CudaArray {}
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
    /** Logical AND over nonzero values; returns bool. Tracked inputs require detach() first. */
    public function all(?int $axis = null): CudaArray {}
    /** Logical OR over nonzero values; returns bool. Tracked inputs require detach() first. */
    public function any(?int $axis = null): CudaArray {}
    /** Population variance by default; correction adjusts the denominator. */
    public function var(?int $axis = null, int $correction = 0): CudaArray {}
    /** Square root of var(); constant groups have zero input gradients. */
    public function std(?int $axis = null, int $correction = 0): CudaArray {}
    /** Numerically stable softmax; low-precision input is widened to float32. */
    public function softmax(int $axis = -1): CudaArray {}
    /**
     * Normalize across explicit axes, promoting integer input to float64.
     * Scale and bias shapes follow the axes in the same order.
     * @param list<int> $axes
     */
    public function layerNorm(array $axes, float $epsilon = 0.00001, ?CudaArray $scale = null, ?CudaArray $bias = null): CudaArray {}
    /** Index of maximum across an axis. Tracked inputs require detach() first. */
    public function argMax(?int $axis = null): CudaArray {}
    /** Index of minimum across an axis. Tracked inputs require detach() first. */
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
    public static function full(array $shape, float|int|bool $value, ?string $dtype = 'float32'): CudaArray {}
    /** Allocate a zero-filled tensor with a prototype's shape and, by default, dtype. */
    public static function zerosLike(CudaArray $prototype, ?string $dtype = null): CudaArray {}
    /** Allocate a one-filled tensor with a prototype's shape and, by default, dtype. */
    public static function onesLike(CudaArray $prototype, ?string $dtype = null): CudaArray {}
    /** Allocate a constant-filled tensor with a prototype's shape and, by default, dtype. */
    public static function fullLike(CudaArray $prototype, float|int|bool $value, ?string $dtype = null): CudaArray {}
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
    /** Supported FP16/BF16 gradient paths accumulate in float32; unsupported derivatives raise exceptions. */
    public function requiresGrad(bool $requiresGrad = true): CudaArray {}
    /** Compute gradients; non-scalar tensors require a seed. Graphs are freed unless retained. */
    public function backward(?CudaArray $gradient = null, bool $retainGraph = false): void {}
    /** Return the accumulated gradient of a leaf tensor, or null before backward(). */
    public function grad(): ?CudaArray {}
    /** Clear the accumulated gradient. */
    public function zeroGrad(): void {}
    /** Return a view that shares storage but is disconnected from gradient history. */
    public function detach(): CudaArray {}
    /** Return the sole element; integers outside PHP's int range use decimal strings. */
    public function item(): int|float|bool|string {}
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

/**
 * Abstract base that marks a class as operator-overloadable and dispatches PHP
 * arithmetic operators to the corresponding optional hooks. Unimplemented hooks
 * throw RuntimeException. ++ and -- use __add and __sub with a right operand of 1.
 * Unary + and - use __mul with a right operand of 1 and -1 respectively.
 * Hooks receive original operand order; the left subclass handles mixed classes.
 * Return fresh values to preserve aliases; userland side effects are not rolled back.
 */
abstract class OperatorOverload implements OperatorOverloadable
{
    /** Handle addition and increment. Return a value without mutating operands. */
    public function __add(mixed $left, mixed $right): mixed {}
    /** Handle subtraction. */
    public function __sub(mixed $left, mixed $right): mixed {}
    /** Handle multiplication. */
    public function __mul(mixed $left, mixed $right): mixed {}
    /** Handle division. */
    public function __div(mixed $left, mixed $right): mixed {}
    /** Handle exponentiation. */
    public function __pow(mixed $left, mixed $right): mixed {}
    /** Handle remainder. */
    public function __mod(mixed $left, mixed $right): mixed {}
}
