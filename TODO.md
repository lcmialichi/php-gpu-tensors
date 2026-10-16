# Extension Improvement Roadmap

Prioritize correctness, useful tensor operations, supported dtypes and a
consistent public API across the PHP extension.

## 1. Improve existing reductions

- [x] Add a low-temporary implementation path for `var()` and `std()`.
- [x] Handle non-contiguous views without unnecessary full-tensor copies.
- [x] Preserve correction modes, dtype behavior, empty dimensions, NaNs and
  autograd semantics.
- [x] Add coverage for contiguous and strided inputs, global and axis
  reductions, correction values and gradient propagation.

## 2. Expand numerical operations and dtype support

- [x] Add general tensor `softmax()` with an axis argument and numerically
  stable behavior.
- [x] Add general tensor `layerNorm()` with explicit normalized axes,
  epsilon and optional scale/bias tensors.
- [x] Extend matrix operations to FP16 and BF16 where the CUDA backend supports
  them; document accumulation and output dtype rules.
- [x] Add tests for dtype conversions, overflow/underflow and numerical
  tolerances.
- [x] Add eager and Fusion elementwise arithmetic, unary math and comparison
  support for FP16/BF16 with FP32 computation and per-operation output rounding.

## 3. Complete autograd and Fusion coverage

- [x] Add Fusion concat capture, native execution and autograd coverage for
  strided inputs and supported dtypes.
- [x] Audit differentiable `CudaArray` and NN operations against implemented
  backward rules.
- [x] Audit tensor operations against Fusion capture, code generation and
  execution support.
- [ ] Implement missing autograd and Fusion support for useful operations,
  preserving explicit errors where semantics are undefined.
- [x] Add autograd for FP16/BF16 matmul with documented gradient dtype and
  broadcasted-batch behavior.
- [ ] Test eager/Fusion result parity and gradients across supported dtypes,
  shapes and layouts.

## 4. Reduce eager dispatch and temporary tensors

- [ ] Extend Fusion capture to common elementwise expressions that currently
  require separate native calls.
- [x] Fuse floating-output power operations with eager-compatible integer-
  exponent behavior; retain a native boundary for integer-output scalar power.
- [x] Capture implicit broadcasting and explicit `broadcastTo()` views,
  including transposed sources and backward reduction through broadcast axes.
- [ ] Review composite operations for avoidable intermediates while preserving
  eager evaluation semantics.
- [ ] Keep eager and fused results consistent across supported dtypes, shapes
  and layouts.

## 5. Improve data movement and execution

- [ ] Preserve packed-buffer paths so large transfers do not require PHP scalar
  expansion.
- [ ] Define explicit lifetime and synchronization rules for asynchronous
  host/device transfers.
- [ ] Add stream-aware transfer support without weakening current synchronous
  behavior or error reporting.

## 6. Improve indexing and autograd coverage

- [x] Complete the audit of `gather()` and `scatterAdd()` validation and
  gradients across eager and Fusion execution.
- [x] Add synchronous Fusion support for `gather()` and `scatterAdd()`; keep
  asynchronous execution and CUDA Graph capture explicitly unsupported while
  index validation synchronizes with the host.
- [x] Reduce synchronization in Fusion `gather()` and `scatterAdd()` validation
  by recording invalid indices in their execution kernels and reporting them at
  the single final synchronization; eager calls retain immediate validation.
- [ ] Add Fusion support for useful indexed expressions where semantics allow.
- [ ] Expand gradients through non-contiguous views and indexed expressions.

## 7. Keep the public API consistent

- [x] Add automatic native CUDA fallback for `Cuda\NN` when cuDNN is absent or
  reports a recoverable unsupported operation; never fall back to CPU.
- [x] Remove backend preflight from the `Cuda\NN` inference API.
- [ ] Keep README, PHP stubs, runtime arginfo and tests aligned for
  `CudaArray`, `ContiguousArray`/`HostArray`, `Compiler`, `CompiledModule`,
  Fusion, optimizers, NN and global `cuda_*` functions.
- [x] Add CuPy-like matrix and tensor factories: `eye()`, `identity()`,
  `zerosLike()`, `onesLike()`, `fullLike()` and standard-normal `randn()`;
  random constructors use fresh default seeds and support reproducible seeds.
- [x] Add GPU-generated `arange()` and `linspace()` sequence factories with
  endpoint, step, dtype inference and integer precision handling.
- [x] Add `copy()`, `swapaxes()` and `moveaxis()` with Fusion and autograd
  support for copies and axis permutations.
- [x] Add `diagonal()`, `repeat()`, `tile()`, `flip()` and `roll()` with
  device-side transformations and autograd gradient mappings.
- [x] Capture `diagonal()`, `repeat()`, `tile()`, `flip()` and `roll()` as
  index mappings in generated Fusion kernels, including chained expressions.
- [ ] Support per-element repeat counts in `repeat()`.
- [x] Add eager, Fusion and autograd support for `expm1()`, `exp2()`, `log1p()`,
  `log2()`, `log10()`, `asin()`, `acos()`, `atan()`, `sinh()`, `cosh()` and
  `tanh()`.
- [ ] Document method limits, serialization restrictions, asynchronous
  lifetime rules and typed error behavior.
- [ ] Retain CPU/API and PHP NTS/ZTS compatibility checks.

## Next implementation

Continue auditing common CuPy mathematical and cumulative methods, starting
with cumulative sum/product support; preserve eager/Fusion parity and keep
README, PHP stubs, runtime arginfo and tests aligned.
