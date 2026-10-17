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
- [x] Add eager and Fusion `where()` support for low-precision conditions and
  values, including FP32 gradient accumulation.
- [x] Add eager and Fusion FP16/BF16 reductions using FP32 accumulation and
  outputs, with FP32 gradient propagation.

## 3. Complete autograd and Fusion coverage

- [x] Add Fusion concat capture, native execution and autograd coverage for
  strided inputs and supported dtypes.
- [x] Audit differentiable `CudaArray` and NN operations against implemented
  backward rules.
- [x] Audit tensor operations against Fusion capture, code generation and
  execution support.
- [x] Add FP16/BF16 cumulative scans with FP32 accumulation/output and
  FP32 gradient propagation.
- [x] Add autograd for FP16/BF16 matmul with documented gradient dtype and
  broadcasted-batch behavior.
- [x] Test eager/Fusion parity and gradients for audited dtype, shape and
  layout combinations; continue extending coverage with new support.

## 4. Reduce eager dispatch and temporary tensors

- [x] Capture common elementwise expressions such as `clamp()` in one Fusion
  kernel instead of separate eager calls.
- [x] Fuse floating-output power operations with eager-compatible integer-
  exponent behavior; retain a native boundary for integer-output scalar power.
- [x] Capture implicit broadcasting and explicit `broadcastTo()` views,
  including transposed sources and backward reduction through broadcast axes.
- [ ] Evaluate single-kernel reduction-based composites such as `softmax()` and
  `layerNorm()` to reduce intermediates while preserving eager semantics.
- [x] Keep eager and fused results consistent across supported dtypes, shapes
  and layouts with targeted parity and gradient tests.

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
- [x] Add Fusion support for indexed expressions including chained `gather()`,
  `scatterAdd()` and slice/view mappings where semantics allow.
- [x] Expand gradients through non-contiguous views and indexed expressions,
  including gather through transposed inputs in eager and Fusion execution.
- [x] Support FP16/BF16 `gather()` in eager and Fusion, with FP32 gradients.
- [x] Support FP16/BF16 `scatterAdd()` by promoting accumulation and output to
  FP32 in eager and Fusion, with FP32 gradient propagation.

## 7. Keep the public API consistent

- [x] Add automatic native CUDA fallback for `PhpTensors\NN` when cuDNN is absent or
  reports a recoverable unsupported operation; never fall back to CPU.
- [x] Remove backend preflight from the `PhpTensors\NN` inference API.
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
- [x] Support per-element repeat counts in `repeat()` with eager execution and
  a synchronous native Fusion boundary.
- [x] Add float32/float64 autograd support for per-element `repeat()` counts
  in eager execution and synchronous Fusion.
- [x] Add eager, Fusion and autograd support for `expm1()`, `exp2()`, `log1p()`,
  `log2()`, `log10()`, `asin()`, `acos()`, `atan()`, `sinh()`, `cosh()` and
  `tanh()`.
- [x] Add CUB-based `cumsum()` and `cumprod()` with axis/flattened semantics,
  dtype promotion, strided-input support and synchronous Fusion boundaries.
- [x] Add autograd support for `cumsum()` in eager and Fusion execution.
- [x] Add zero-safe autograd support for `cumprod()` in eager and synchronous
  Fusion execution.
- [x] Support asynchronous replay and CUDA Graph capture for cumulative scans
  over contiguous flattened or last-axis inputs; retain synchronous execution
  for layouts that require temporary packing.
- [ ] Document method limits, serialization restrictions, asynchronous
  lifetime rules and typed error behavior.
- [ ] Retain CPU/API and PHP NTS/ZTS compatibility checks.

## Next implementation

Keep README, PHP stubs, runtime arginfo and tests aligned as capabilities are
added.
