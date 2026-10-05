<?php

namespace Cuda;

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
    /**
     * Identical dtype reuses storage; safe conversions allocate converted storage.
     * @throws InvalidArgumentException For an unsafe conversion.
     * @throws RuntimeException If compilation or GPU execution fails.
     */
    public function astype(string $dtype): CudaArray {}
    /** Element dtype name, e.g. float32. */
    public function dtype(): string {}
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