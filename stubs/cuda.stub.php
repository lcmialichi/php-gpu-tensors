<?php

namespace Cuda;

/** Optional synchronous cuDNN inference. NCHW inputs, OIHW filters, contiguous FP32; no Fusion capture. */
final class NN
{
    private function __construct() {}
    public static function isAvailable(): bool {}
    /** Cross-correlation, floor output dimensions, optional bias per output channel. */
    public static function conv2d(CudaArray $input, CudaArray $weights, ?CudaArray $bias = null,
        array $stride = [1, 1], array $padding = [0, 0], array $dilation = [1, 1], int $groups = 1): CudaArray {}
    /** Max pooling or average pooling excluding padded cells; floor output dimensions. */
    public static function pool2d(CudaArray $input, array $window, array $stride = [2, 2],
        array $padding = [0, 0], string $mode = 'max'): CudaArray {}
    /** Stable softmax across channels independently at each N,H,W coordinate. */
    public static function softmax(CudaArray $input): CudaArray {}
}

/** Base error from the CUDA extension. */
class Exception extends \Exception {}
/** Device or kernel operation failed. */
class RuntimeException extends Exception {}
/** Invalid shape, dtype, axis, or other caller input. */
class InvalidArgumentException extends Exception {}
/** Device or host tensor allocation failed. */
class OutOfMemoryException extends RuntimeException {}
/** NVRTC compilation failed. */
class CompilationException extends RuntimeException {}

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

/** Contiguous CPU tensor, also available as Cuda\HostArray; [] uses native handlers. */
class ContiguousArray
{
    /** Copy a rectangular numeric PHP array into CPU storage. */
    public function __construct(array $values, ?string $dtype = 'float32', bool $pinned = false) {}
    /** Import row-major packed bytes. */
    public static function fromBuffer(string $bytes, array $shape, ?string $dtype = 'float32', bool $pinned = false): self {}
    /** Whether storage is page-locked with cudaMallocHost. */
    public function isPinned(): bool {}
    /**
     * Read one element by multidimensional coordinates.
     * @param list<int> $indices
     */
    public function get(array $indices): mixed {}
    /** Read one element with variadic coordinates. */
    public function at(int ...$indices): mixed {}
    /** Shape in row-major dimension order. @return list<int> */
    public function getShape(): array {}
    /** Materialize nested PHP arrays. */
    public function toArray(): array {}
    /** Number of dimensions. */
    public function getNdims(): int {}
    /** Total number of elements. */
    public function getSize(): int {}
    /** Element dtype name, e.g. float32. */
    public function getDtype(): string {}
    /** Bytes per element. */
    public function getElementSize(): int {}
    /** Length of the first dimension. */
    public function count(): int {}
    /** Copy CPU storage into a GPU tensor. */
    public function toGpu(): CudaArray {}
    /** Export contiguous row-major bytes without PHP scalar expansion. */
    public function toBuffer(): string {}
    /** Serialize shape, dtype, and raw CPU storage. */
    public function __serialize(): array {}
    /** Restore a serialized host tensor. */
    public function __unserialize(array $data): void {}
}

/** Public name for ContiguousArray, retained for compatibility. */
class_alias(ContiguousArray::class, 'Cuda\\HostArray');

/** GPU-resident n-dimensional tensor; [] uses native dimension handlers. */
class CudaArray
{
    /** Import row-major little-endian packed bytes matching shape and dtype. */
    public static function fromBuffer(string $bytes, array $shape, ?string $dtype = 'float32'): CudaArray {}
    /** Copy a flat numeric PHP array directly into the requested row-major shape. */
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
     * Concatenate tensors along an axis.
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
    /** Matrix multiply (2D or batched); cuBLAS is optional when available. */
    public function matmul(CudaArray $other): CudaArray {}
    /** Permute axes; omitting axis reverses their order. */
    public function transpose(?array $axis = null): CudaArray {}
    /** Elementwise exponentiation. */
    public function power(CudaArray|float|int $other): CudaArray {}
    /** Elementwise square root. */
    public function sqrt(): CudaArray {}
    /** Elementwise exponential. */
    public function exp(): CudaArray {}
    /** Elementwise natural logarithm. */
    public function log(): CudaArray {}
    /** Elementwise sine. */
    public function sin(): CudaArray {}
    /** Elementwise cosine. */
    public function cos(): CudaArray {}
    /** Elementwise tangent. */
    public function tan(): CudaArray {}
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
    /** Logical AND over nonzero values; returns a bool tensor. */
    public function all(?int $axis = null): CudaArray {}
    /** Logical OR over nonzero values; returns a bool tensor. */
    public function any(?int $axis = null): CudaArray {}
    /** Population variance by default; correction adjusts the denominator. */
    public function var(?int $axis = null, int $correction = 0): CudaArray {}
    /** Square root of var(). */
    public function std(?int $axis = null, int $correction = 0): CudaArray {}
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
    public function toHost(): ContiguousArray {}
    /** Download logical row-major bytes directly, without PHP scalar expansion. */
    public function toBuffer(): string {}
    /** Create a view with the same element count. */
    public function reshape(array $shape): CudaArray {}
    /** Remove singleton dimensions; an optional axis must have size one. */
    public function squeeze(?int $axis = null): CudaArray {}
    /** Insert a singleton dimension at axis. */
    public function unsqueeze(int $axis): CudaArray {}
    /** Broadcast as a shared-storage view to the requested shape. */
    public function broadcastTo(array $shape): CudaArray {}
    /** Create a one-dimensional view. */
    public function flatten(): CudaArray {}
    /** Allocate a zero-filled GPU tensor. */
    public static function zeros(array $shape, ?string $dtype = 'float32'): CudaArray {}
    /** Allocate a one-filled GPU tensor. */
    public static function ones(array $shape, ?string $dtype = 'float32'): CudaArray {}
    /** Allocate a constant-filled GPU tensor. */
    public static function full(array $shape, float|int $value, ?string $dtype = 'float32'): CudaArray {}
    /** Generate uniform random GPU values. */
    public static function rand(array $shape, float|int $min = 0, float|int $max = 1, ?string $dtype = 'float32'): CudaArray {}
    /** Gather values using same-rank int32 indices along an axis. Not supported in Fusion capture. */
    public function gather(CudaArray $indices, int $axis = 0): CudaArray {}
    /**
     * Add updates at indexed positions and return a new tensor. Duplicate indices accumulate.
     * Currently supports float32/float64 and is not supported in Fusion capture.
     */
    public function scatterAdd(CudaArray $indices, CudaArray $updates, int $axis = 0): CudaArray {}
    /**
     * Identical dtype reuses storage; safe conversions allocate converted storage.
     * @throws InvalidArgumentException For an unsafe conversion.
     * @throws RuntimeException If compilation or GPU execution fails.
     */
    public function astype(string $dtype): CudaArray {}
    /** Element dtype name, e.g. float32. */
    public function dtype(): string {}
    /** Enable or disable reverse-mode gradient tracking on a leaf tensor. */
    public function requiresGrad(bool $requiresGrad = true): CudaArray {}
    /** Compute gradients; non-scalar tensors require an explicit gradient seed. */
    public function backward(?CudaArray $gradient = null): void {}
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
abstract class Number
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