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
- [ ] Reduce synchronization in `gather()` and `scatterAdd()` validation while
  retaining precise invalid-index errors.
- [ ] Add Fusion support for useful indexed expressions where semantics allow.
- [ ] Expand gradients through non-contiguous views and indexed expressions.

## 7. Keep the public API consistent

- [x] Add automatic native CUDA fallback for `Cuda\NN` when cuDNN is absent or
  reports a recoverable unsupported operation; never fall back to CPU.
- [x] Remove backend preflight from the `Cuda\NN` inference API.
- [ ] Keep README, PHP stubs, runtime arginfo and tests aligned for
  `CudaArray`, `ContiguousArray`/`HostArray`, `Compiler`, `CompiledModule`,
  Fusion, optimizers, NN and global `cuda_*` functions.
- [ ] Document method limits, serialization restrictions, asynchronous
  lifetime rules and typed error behavior.
- [ ] Retain CPU/API and PHP NTS/ZTS compatibility checks.

## Next implementation

Continue closing Fusion's documented limits, prioritizing low-precision
elementwise expressions, then async indexed execution and broader eager/Fusion
parity. Preserve explicit errors where a backend boundary remains intentional.
