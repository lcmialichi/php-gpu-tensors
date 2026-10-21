# PhpTensors 1.0 readiness checklist

This replaces the previous improvement checklist. It is a release plan, not a list
of everything a tensor library could eventually implement.

Audit baseline: commit `3077a36`. Scope: native CPU/CUDA storage and operations,
transfers, operators, public contracts, Fusion/JIT, autograd, optimizers, NN,
tests, build/CI, packaging, examples and documentation.

## How to use this plan

- **P0:** correctness or lifetime blockers. Fix before a release candidate.
- **P1:** required work for the declared 1.0 scope. Complete before API freeze,
  or explicitly defer the feature and remove any unsupported promise.
- **P2:** subsequent improvements; not a reason to postpone a dependable core.
- **Evidence:** a source-backed observation, not necessarily a reproduced failure.
  Reproduced observations are identified separately below.
- Close an item only with its regression tests and acceptance evidence, not
  merely because the method exists.

### Baseline to preserve

Already implemented, although not uniformly available across backends/dtypes:

- Explicit `HostArray` / `CudaArray` construction and synchronous transfers.
- Buffer imports/exports and layout-aware GPU downloads.
- Native arithmetic/operators and custom PHP `OperatorOverload` subclasses.
- CUDA broadcasting, reductions/scans, shape/index transforms, FP16/BF16 paths,
  softmax/layer normalization, and accelerated matrix multiplication.
- Optional Fusion capture, multi-kernel plans, compilation/cache, replay,
  asynchronous execution, workspace reuse and CUDA Graph execution.
- Reverse-mode gradients for many tensor operations; functional SGD/AdamW.
- FP32 CNN inference with optional cuDNN and native CUDA fallback.

Do not turn these into duplicate implementation tasks. Complete their contracts,
coverage and unsupported cases instead.

## 1. Freeze the scope before freezing the API

- [ ] **SCOPE-01 — Publish a capability matrix.**
  Cover CPU-only HostArray, CUDA-build HostArray, eager CudaArray, Fusion run,
  compiled replay, async replay and CUDA Graphs. For every operation record
  dtype/rank/layout restrictions, output dtype/shape, gradients and synchronization.
  **Accept:** every public method is classified as implemented, explicitly
  unsupported or experimental; tests exercise each advertised cell.

- [ ] **SCOPE-02 — Define what "CPU fallback" means.**
  Recommended 1.0: explicit placement/transfers and a useful CPU numerical core,
  not implicit device migration or automatic per-operation backend switching.
  Decide whether custom backends are minimal extension points or a portable
  computation protocol; define operation names and `supports()` context.
  **Accept:** a real custom backend example computes and rejects unsupported
  operations correctly; no BackendManager/registry is needed.

- [ ] **SCOPE-03 — Record stable versus experimental boundaries.**
  Decide CLI/FPM support, single-device versus multi-device behavior, serialization
  compatibility, supported PHP/Toolkit/GPU targets, and whether NN remains
  inference-only. Resolve scalar/rank-zero, axis, equality, casting and aliasing
  semantics before 1.0.
  **Accept:** explicit decisions and compatibility/deprecation rules; experimental
  APIs are identified before users rely on a stable 1.0 promise.

## 2. P0 — Storage, lifetime and input correctness

- [x] **MEM-01 — Make host access safe for every constructible dtype.**
  Fixed: scalar access now uses one checked dtype-reader table including FP16/BF16;
  iteration caches a typed reader instead of silently returning undefined values
  ([implementation](src/contiguous_array_ce.c)).
  **Accept:** all 13 dtypes pass `get`, `at`, scalar/subarray iteration and export
  equivalence tests; invalid dtype metadata never indexes a function table.
  **Verified:** the [regression](tests/host_array_dtype_access.phpt) failed before
  the fix on FP16 iteration and passes after a fresh CUDA build. All 13 dtypes,
  offset subarrays, empty storage, and half/bfloat signed zero, infinities, NaNs
  and smallest subnormals are covered. PHP 8.1 / CUDA 12.6: 4/4 selected PHPTs pass,
  zero skips (`host_array_dtype_access`, `host_array_cuda`, `ca_transfer`,
  `ca_transfer_fast`).

- [x] **MEM-02 — Make deserialization validated and transactional.**
  Fixed: CPU/CUDA host and CUDA tensor payloads validate exact versions, types,
  dimensions and byte lengths before replacing storage
  ([shared host](src/contiguous_array_ce.c), [host storage](src/host_tensor.c)).
  Host snapshots pack logical subarray contents and use canonical contiguous
  metadata. CompiledModule persists explicit versioned parameter fields, rejects
  malformed entries and legacy blobs, and bounds argument lists at 128
  ([module](src/module_ce.c)). Restoring initialized/active modules is rejected;
  metadata-only replacements are transactional.
  **Accept:** version/type/rank/dimension/dtype/byte-width/length validation with
  checked products before allocation; bounded kernel metadata; unknown versions
  fail explicitly. Invalid restore on an existing object preserves old data.
  Version module metadata independently of C ABI; test corrupt/truncated payloads.
  **Verified:** fresh CPU PHP 8.4/8.5 ZTS builds each pass 4/4 selected PHPTs;
  fresh PHP 8.1 / CUDA 12.6 build passes 9/9, zero skips.
  Regressions cover [host snapshots](tests/host_array_serialization.phpt),
  [CUDA snapshots](tests/cuda_array_serialization.phpt) and
  [module persistence](tests/compiled_module_serialization.phpt), including
  repeated replacement, malformed metadata, empty/scalar tensors, all advertised
  storage dtypes, actual restored kernel launches and pending-operation guards.
  No allocation-failure injection or sanitizer verification was performed here;
  those remain part of QA-03.

- [x] **MEM-03 — Define clone, uninitialized-object and reconstruction behavior.**
  Fixed: both HostArray implementations clone logical values into independent
  storage, preserving dtype, shape and host pinning. CPU construction prepares the
  replacement before releasing the previous allocation
  ([shared host](src/contiguous_array_ce.c), [host storage](src/host_tensor.c)).
  Storage access, arithmetic and like-factories reject uninitialized host objects.
  Active host iterators retain a shared-storage snapshot so reconstruction or
  restoration cannot invalidate their cached pointers. CUDA cloning preserves
  the original allocation exception and returns an object for VM-owned cleanup,
  rather than returning NULL or destroying PHP members twice
  ([CUDA handlers](src/cuda_array/cuda_array_ce.c)).
  **Accept:** clones are independent copies or explicitly rejected; bypassed
  constructors and failed initialization throw rather than dereference storage;
  repeated construction/destruction has bounded memory and preserves old data
  on failure.
  **Verified:** the [host lifecycle regression](tests/host_array_lifecycle.phpt)
  first failed on the CPU reconstruction leak. Fresh PHP 8.4/8.5 ZTS CPU builds
  each pass 5/5 selected PHPTs; the fresh PHP 8.1 / CUDA 12.6 build passes all 12
  selected regressions, zero skips. The
  [CUDA lifecycle regression](tests/cuda_array_lifecycle.phpt) covers all storage
  dtypes, empty/scalar/strided clones, pinned host subarrays, failed reconstruction
  preserving gradients, aborted Fusion results, and clone failure/recovery with a
  bounded 16 MB CUDA pool. PHP member destructors run exactly once. CUDA lifecycle
  and module persistence also pass with `USE_ZEND_ALLOC=0` (2/2). General host
  allocation-failure injection and sanitizer coverage remain part of QA-03.

- [x] **MEM-04 — Make operator result aliasing explicit and leak-free.**
  Fixed: operators compute into a temporary value and use a shared
  [result commit helper](src/operator_result.h). Failed computations preserve
  the binding and original exception; successful assignments replace the value
  before releasing the previous operand, including when its destructor executes
  user code. Callback references are copied as values rather than escaping into
  result bindings. All four native/custom handlers use the same ownership contract
  ([operator bridge](src/operator_overload_ce.c),
  [shared host storage](src/host_tensor.c),
  [CUDA-build host](src/contiguous_array_ce.c),
  [CudaArray](src/cuda_array/cuda_array_ce.c)).
  OperatorOverload now provides six typed optional hooks with explicit unsupported
  errors, rather than requiring every operation. Removed unused __inc/__dec hooks:
  PHP dispatches increment/decrement through addition/subtraction with 1.
  Updated stubs, the API baseline and the README example together.
  **Accept:** `+=`, `-=`, `*=`, `/=`, self-assignment, PHP references, surviving
  object aliases, and throwing overload callbacks preserve defined values and
  release replaced objects exactly once. Include destructor/allocation checks,
  not only arithmetic output checks.
  **Verified:** the [host regression](tests/host_array_operator_aliasing.phpt)
  first failed because a replaced operand remained alive through its WeakReference.
  CPU PHP 8.4/8.5 ZTS builds each pass 7/7 selected PHPTs; PHP 8.1 / CUDA 12.6
  passes 12/12, zero skips, with matching API snapshots. Regressions cover
  [custom hooks](tests/operator_overload_aliasing.phpt) and
  [CUDA operators](tests/cuda_array_operator_aliasing.phpt), including typed
  properties/references, identity/falsy/by-reference results, pre/post increment,
  exceptions from callbacks/destructors, bounded repeated assignment, exact
  destructor counts, shared views and gradients. Exhausting a 16 MB CUDA pool
  preserves OutOfMemoryException in both operator and method forms and recovers
  after release. Fusion capture still rejects compound mutation. Additional
  USE_ZEND_ALLOC=0 runs pass 2/2 CPU and 3/3 CUDA tests; sanitizers remain QA-03.

- [x] **MEM-05 — Establish device/context ownership and safe reset.**
  The process-wide pool records its owning device/context, rejects allocation in
  another context and temporarily restores ownership for cleanup. GPU tensor
  creation/access, pinned allocation and Fusion capture check ownership.
  `cuda_set_device()` is idempotent only on the owning device; negative/oversized
  IDs are rejected before narrowing. `cuda_device_reset()` rejects reset without
  touching live resources. Shutdown frees extension resources without resetting
  the device. CompiledModule retains/releases the pool's primary context instead
  of creating/destroying private contexts or silently switching context
  ([tensor](src/tensor.h), [pool](src/backends/cuda/memory_pool.c),
  [device API](src/cuda.c), [module](src/module_ce.c),
  [Fusion context checks](src/fusion_execute.c)).
  Validation: [ownership regression](tests/cuda_context_ownership.phpt) exercises
  ten two-module lifecycles, pending async module destruction, native Fusion
  replay, views, transfers, backend handles and allocation after rejected reset.
  [Pool unit tests](tests/memory_pool_test.c) mock foreign devices/contexts and
  verify rejected allocations plus cleanup context restoration. CUDA regressions
  pass 14/14; CPU PHP 8.4/8.5 ZTS pass 7/7 each. Additional system-malloc CUDA
  runs pass 2/2. Only one real GPU was used; multi-device support is not claimed.
  Process/thread/request ownership remains MEM-06, not covered by this item.

- [x] **MEM-06 — Validate process/thread/request lifecycle.**
  CUDA/pool startup is now lazy, after worker fork. CLI and PHP-FPM use one CUDA
  owner process/thread; queries also establish ownership. Post-CUDA fork and
  other threads throw before CUDA or inherited mutex access. Inherited module,
  tensor, pinned storage and Fusion destruction skips native CUDA cleanup in a
  rejected child; start a fresh process/exec rather than reuse its runtime.
  Ordinary CPU storage and request-local cache cleanup do not need CUDA ownership.
  Pool sizes reject empty, negative, malformed and overflowing values; runtime
  `ini_set()` cannot change the first worker pool configuration.
  Validation: [native pool tests](tests/memory_pool_test.c) exercise pre-init fork,
  post-init fork with a deliberately inherited locked pool mutex, foreign pthread
  rejection and owner-thread exit without ownership transfer. The
  [GPU process regression](tests/cuda_process_lifecycle.phpt) exercises recycled
  workers, query ownership, inherited API/object/pinned/async cleanup, parent
  continuity and a fork inside Fusion capture. The
  [real FPM regression](tests/cuda_fpm_lifecycle.phpt) passes 24 GPU and eight
  CPU-only requests across four recycled workers, checking precision/counters,
  request-local caches, pending work and bounded warm host/GPU memory.
  [Pool configuration regression](tests/cuda_lazy_pool_configuration.phpt) passes
  checked sizes and immutable runtime INI. Full GPU suite with FPM enabled:
  95 passed, one expected CPU-only skip, seven native checks; CPU PHP 8.4/8.5 ZTS:
  8/8 each. Eight lifecycle regressions also pass with `USE_ZEND_ALLOC=0`,
  including real FPM; three CPU-storage regressions pass in the CUDA build with
  no visible GPU. Concurrent CUDA through PHP/ZTS threads remains explicitly unsupported;
  rejection is tested with native pthreads, not claimed as concurrent GPU support.

- [x] **MEM-07 — Audit shape/index/stride arithmetic end to end.**
  Progress: [shared shape metrics](src/tensor_shape.h) validate reverse stride
  products, logical size and bytes before CPU/GPU/host allocation, reshape and
  Fusion metadata creation. Empty axes cannot hide overflowing strides.
  Broadcast strides remain `size_t` throughout the host/kernel interface.
  `getSize()` preserves PHP integer width. Scalar writes check rank before
  accessing strides; reshape, transpose, indexing, concat and reduction validate
  before narrowing. Flatten/global arg reductions reject unrepresentable axes
  in eager and Fusion. Python-style slice bounds still clamp without truncation.
  Native view construction checks offset/span arithmetic. Launch helpers check
  dimensions before conversion and return invalid configurations for CUDA to
  report, instead of wrapping or dividing by zero.
  Built-in linear indices and grid steps widen before multiplication; matmul
  preserves `size_t` strides/offsets and uses wide K-tile counters. Reduction
  iteration and pooling window counts no longer overflow signed 32-bit counters.
  Generic, arg, moments and column reductions reject unrepresentable grids
  before narrowing instead of silently executing only part of the output.
  Validation: [factory boundaries](tests/tensor_shape_boundaries.phpt),
  [CUDA index/axis/virtual-size boundaries](tests/cuda_shape_boundaries.phpt),
  [native shape tests](tests/tensor_shape_test.c) and
  [launch boundary tests](tests/launch_config_test.cpp), without huge buffers.
  The [GPU index/grid probe](tests/kernel_index_test.cu) checks arithmetic above
  2^31/2^32 and links the production reduction object: four oversized-grid paths
  must fail without modifying sentinel buffers, followed by a valid reduction.
  The original object reproduced false success for 2^32+1 output blocks; the
  corrected object passes. These are bounded arithmetic/launch tests, not
  execution over physically allocated multi-billion-element tensors.
  Native views now validate their complete strided footprint against root backing
  storage before pointer construction/refcount changes, including nested views,
  scalar views, empty/null storage and virtual zero-stride broadcasts. Removed
  the unused internal dtype-reinterpret view builder, which changed element
  sizes after validation. Reduction/transform/slice products reuse checked shape
  metrics; empty Fusion slices do not accumulate irrelevant logical offsets or
  overflowing singleton strides. Empty concat gradient views normalize offsets.
  Full CUDA suite: 83 passed, one expected CPU-only skip, no failures; six
  native checks passed. The native shape/view test also passes AddressSanitizer
  and UBSan; five selected P0 PHPTs pass with system malloc.
  CPU PHP 8.4/8.5 ZTS: rebuilt, API snapshot matched, 8/8 per version.
  Changed C implementations compile with implicit function declarations treated
  as errors. The alternate, unlinked CUDA kernel unit also compiles.
  **Accept:** validate before narrowing/dereferencing; checked shape/byte/offset
  products; consistent rank/axis/negative-index policy. Test scalars, empty axes,
  maximum rank, singleton/broadcast strides, oversized PHP integers and overflow
  rejection without allocating enormous buffers.

## 3. P0 — Numerical behavior must not depend on execution mode

- [x] **NUM-01 — Unify equality/comparison semantics across eager and Fusion.**
  Implemented exact `==`/`!=` in eager functors, matching Fusion code generation.
  Numeric 0/1 masks retain the existing promoted dtype. No implicit tolerance:
  NaNs compare unequal, equal infinities compare equal and signed zeros compare
  equal. The README documents this intentional pre-1.0 behavior change.
  [Comparison regressions](tests/comparison_semantics.phpt) cover all 13 storage
  dtypes, near-equal floats, non-finite values, wide integers, mixed dtypes,
  scalar operands, broadcasting, strided views and compiled graph replay.
  The original eager kernel reproduced the near-equal mismatch; corrected eager
  and Fusion pass, including system malloc and the full CUDA suite.
  **Accept:** one documented equality policy. Prefer separate explicit
  approximate-comparison semantics if needed; test near-equal values, NaNs,
  infinities, signed zero, integers and mixed dtypes across execution modes.

- [x] **NUM-02 — Preserve precision and scalar types at public boundaries.**
  [Shared scalar decoding](src/tensor_scalar.h) preserves FP64, typed PHP integer
  and boolean reads, FP16/BF16 values and subnormals. CUDA indexing and `item()`
  share the decoder. Integer exports outside PHP's signed range use exact
  decimal strings, consistently across CUDA/HostArray exports, host access and
  iteration; binary round trips preserve the original unsigned storage.
  FP64 scalar division and negative scalar powers remain FP64 in both operand
  orders and execution modes. Stubs, runtime item return metadata, API snapshot,
  README and existing slice type expectations were updated together.
  [Scalar regressions](tests/scalar_precision.phpt) cover all 13 dtypes, 2^24+1,
  PHP integer limits, uint64 max, offsets, strides, non-finite half values,
  subnormals, signed zero and FP64 scalar arithmetic. Full CUDA and system
  malloc suites pass; CPU PHP 8.4/8.5 ZTS builds and selected tests remain green.
  **Accept:** no accidental FP64 downcast; typed integer/bool scalar extraction;
  explicit uint64 values beyond PHP integer range; half/bfloat scalar export.
  Test `2**24 + 1`, PHP integer boundaries, subnormals and scalar-left operations.

- [x] **NUM-03 — Complete dtype factories and conversion policy.**
  **Fixed:** scalar conversion and fill/uniform-range dispatch cover all 13
  dtypes, including bool input. Integer fills, PHP-array imports and scalar
  assignment share checked conversion; failed reconstruction preserves storage.
  HostArray implements `fullLike()` in CUDA and CPU-only builds.
  Random bounds are checked before allocation, including empty shapes; boolean
  ranges respect constant endpoints. Integer interpolation avoids FP32 boundary
  corruption and floating interpolation avoids overflowing the bound difference.
  Eager/Fusion casting agrees for all 13 x 13 pairs, including FP16/BF16 to FP64.
  Narrowing and signed-to-unsigned casts remain rejected; permitted integer-to-
  float casts can round. Integer random interpolation uses a 24-bit fraction,
  not an unbiased full-width integer sampler.
  Integer `arange()` no longer passes uint64 values through double; integer
  `linspace()` interpolates exact PHP integer endpoints across the full int64
  span without overflowing intermediate products. Fractional integer values
  truncate toward zero consistently with fills. Floating interpolation keeps
  constant endpoints exact and avoids overflowing intermediate differences.
  Mixed signed/unsigned operations and comparisons use the smallest exact common
  integer dtype. Signed + uint64, including PHP integer scalars, is rejected
  before launch/capture, not implicitly converted to FP64. Ordinary integer
  scalar arithmetic checks its operand against the retained tensor dtype.
  Floating results use FP64 for promoted 64-bit integer inputs, including scalar
  division and arithmetic with floating scalars.
  **Verified:** [factory regressions](tests/dtype_factories.phpt), the
  [native scalar matrix](tests/scalar_cast_test.c) under ASan/UBSan and
  float-cast-overflow checks, and production GPU kernel probes at exact signed
  and unsigned 64-bit random endpoints. [Range regressions](tests/range_precision.phpt)
  cover all 13 dtypes, eye/normal factories and 1,458 rational interpolation
  cases; [integer promotion](tests/integer_promotion.phpt) covers all 64 pairs,
  eager/Fusion arithmetic/comparisons, scalar failures, empty inputs and replay.
  Full CUDA suite: 86 passed, one expected
  CPU-only skip; seven native checks passed. Rebuilt CPU PHP 8.4/8.5 ZTS:
  API snapshots matched and 8/8 selected PHPTs passed per version. Four targeted
  factory/lifecycle regressions and both new range/promotion regressions also
  passed with `USE_ZEND_ALLOC=0`.
  **Accept:** exhaustive factory/cast tests for every advertised dtype; define
  safe versus lossy casts, narrowing/overflow, bool behavior and signed/unsigned
  promotion. Unsupported combinations throw before launch; no silent zero fills
  or uninitialized successful outputs.

- [x] **NUM-04 — Never silently sever a differentiable path.**
  Completed: floating autograd nodes now retain FP16/BF16 edges regardless of node
  kind; concat, slices and floating casts no longer silently lose those edges.
  Unsupported floating derivatives retain an explicit unsupported node and fail
  during backward instead of appearing detached. Empty concat gradient views
  respect backing-storage bounds.
  [Gradient path regressions](tests/autograd_low_precision_paths.phpt) reproduce
  the original missing-gradient error and now verify eager/Fusion concat,
  stepped slices, copies, repeated-parent accumulation, conditional branches,
  explicit unsupported comparison derivatives and empty gradients. FP16/BF16
  gradients use FP32. Full CUDA and system malloc validation pass.
  Discrete reductions (`argMax`, `argMin`, `all`, `any`) now require explicit
  `detach()` on tracked inputs before eager launch or Fusion recording, including
  global, axis, strided and empty paths. Explicit null axes consistently mean a
  global reduction; all public reductions reject non-integer axes.
  `where()` retains mask mutation checks but does not traverse or consume
  condition-only gradient histories. Conditions also used as value branches
  still participate in differentiation. True zero tensors replace multiplication
  by zero for inactive branches and zero derivatives, avoiding NaN/Inf seed
  contamination in where, bounds selection and rounding gradients.
  [Discrete-output regressions](tests/autograd_discrete_outputs.phpt) and
  [selection regressions](tests/autograd_selection_paths.phpt) cover four floating
  dtypes, eager/Fusion, capture/replay, independent condition histories,
  broadcasts, non-finite seeds and saved-mask mutation checks.
  `maximum`/`minimum` now split exact ties equally, including scalar constants,
  broadcasting and repeated parents. Scalar predicates use the forward result
  dtype instead of rounding promoted bounds back to the input dtype. `clamp`
  retains composition semantics: input derivative 0.5 at either boundary, 0.25
  where equal bounds coincide. Max/min reduction backward also uses true zeros
  for inactive elements with non-finite seeds. Gradient accumulation preserves
  FP32/FP64 input dtype and uses FP32 for FP16/BF16, including promoted forward
  operations and mixed-precision contributions.
  [Extrema regressions](tests/autograd_extrema_boundaries.phpt) cover four floating
  dtypes, eager/Fusion, replay, strided inputs, adjacent representable values,
  signed zeros, infinite ties, non-finite seeds and empty tensors.
  Scalar power backward now constructs correctly typed constants and widens
  FP16/BF16 intermediates to FP32; exponent zero and inverse scalar base one
  produce genuine zero derivatives even with non-finite seeds. `exp2`, `log2`
  and `log10` preserve FP64 derivative constants instead of rounding them to
  FP32. A malformed low-precision scalar previously triggered undefined behavior
  and a native Fusion abort; corrected derivative constants avoid that path,
  and unsupported codegen scalar types now raise an explicit exception with
  source-buffer cleanup instead of relying on a release-disabled assertion.
  [Numerical regressions](tests/autograd_numerical.phpt) cover scalar powers in
  all four floating dtypes, boolean/integer/floating scalar constants, non-finite
  seeds, repeated compiled replay and first-order-only backward behavior.
  The [complete path matrix](tests/autograd_operation_matrix.phpt) audits public
  tensor arithmetic, all 22 unary operations, reductions/scans, casts, copies,
  views, index transforms, repeated-index gather/scatter, concat, matmul and
  var/std/softmax/layerNorm composites in eager and Fusion for all four floating
  input dtypes. It covers all 16 mixed floating pairs for the seven binary
  operations, strided and empty paths, all 13 cast targets, integer/bool tracking
  rejection, unsupported comparison derivatives and FP64 matmul rejection.
  Global cumsum backward now reverses the flattened sequence before restoring
  the input shape; low-precision abs backward uses valid FP32 scalar constants.
  [Singularity and capture regressions](tests/autograd_singularities.phpt) define
  NaN outside real logarithm domains and IEEE boundary derivatives; constant
  standard-deviation groups have genuine zero gradients, including corrected
  reductions and non-finite seeds. Binary power partials route true zeros for
  zero exponents, unit bases and zero bases with positive exponents, without
  hiding the other partial's undefined derivative.
  Compiled Fusion rejects tracked outputs without backward captured for that
  output, including nested/partially differentiated outputs. Replay rejects
  gradient-enabled inputs absent from capture, in sync/async/CUDA Graph paths;
  explicit detach remains supported. Captured retained backward and gradients
  after Fusion::run remain supported. Consumed expressions are not reused by
  capture CSE when new autograd history is required. Codegen failure propagates
  before module loading, with normal scope/plan cleanup instead of a native abort.
  Full CUDA suite: 92 passed, one expected CPU-only skip; seven native checks.
  The API snapshot remains unchanged. All eight autograd regressions (basic,
  low-precision, discrete outputs, selection, extrema, numerical, operation matrix
  and singularities) also pass with
  `USE_ZEND_ALLOC=0`.
  Intentional history boundaries remain explicit: detach, native clone, host/
  byte export and serialization are snapshots; factories do not differentiate
  their shape prototypes. NN is inference-only and rejects tracked inputs.
  First-order-only support and singularity/selection conventions are documented.
  Extending finite-difference accuracy proofs beyond the current numerical
  fixtures remains the separate P1 task GRAD-01, not a missing edge in this audit.
  **Accept:** implement the derivative or reject the unsupported tracked operation
  explicitly. Audit the full operation/dtype matrix, including casts, copies,
  transforms and composites; no forward success that silently drops required
  gradient edges.

## 4. P1 — A useful, consistent CPU backend and extension API

- [x] **CPU-01 — Remove accidental build-dependent HostArray behavior.**
  Both profiles now compile the same [HostArray class](src/contiguous_array_ce.c),
  [typed storage](src/host_tensor.c), scalar conversion, shape validation and
  host export code. The duplicate double-backed CPU implementation was removed.
  All 13 dtypes support identical construction, explicit/nested empties, metadata,
  scalar/subarray indexing, clone and persistence. Nullable default dtypes agree;
  invalid indices cannot be silently truncated/coerced. `IteratorAggregate`,
  `Countable` and `getIterator()` provide standard iteration interoperability.
  Pinned allocation and GPU transfers remain explicitly CUDA-only capabilities.
  Validation: the [shared matrix](tests/host_array_contract.phpt) passes in CUDA
  PHP 8.1 and CPU PHP 8.4/8.5 ZTS, including the same versioned binary fixture,
  exact int64/uint64 boundaries, retained iterators and transactional legacy reads.
  CPU suites pass 15/15 per version; eight host regressions per CPU version also
  pass with system malloc. The CUDA build passes the matrix with no visible GPU.

- [x] **CPU-02 — Implement the declared minimum CPU numerical core.**
  **Implementation closed; real-GPU acceptance remains tracked separately below.**
  The declared minimum is implemented: copy/cast, broadcasting, retained
  reshape/transpose/slice views, scalar `item()`, comparisons/where, 22 floating
  unary functions, sum/mean/min/max and batched FP32/FP16/BF16 matmul.
  Slice parsing/layout and broadcast/matmul shape selection are shared with CUDA.
  Arithmetic supports all four floating dtypes and now uses the user-approved IEEE
  zero-divisor behavior instead of the legacy CPU exception. Integer reductions
  preserve dtype-width wrap and exact 64-bit extrema; selection copies storage bytes.
  Views support logical exports, retained iterators, rank-zero/empty results and
  explicit `copy()` before reshaping non-contiguous layouts.
  Validation: [cast matrix](tests/host_array_copy_cast.phpt),
  [broadcast/operators](tests/host_array_broadcast.phpt),
  [views](tests/host_array_views.phpt), [math](tests/host_array_math.phpt) and
  [reductions](tests/host_array_reductions.phpt). CPU PHP 8.4/8.5 ZTS pass 15/15
  each, plus eight host regressions each with system malloc. A separate PHP 8.4
  UBSan build passes all eight without undefined-behavior reports. CUDA 12.3.2 /
  PHP 8.3 compiles and passes ten shared host tests with the SDK linker stub
  (host-only execution), plus the native shape check. The three new core host
  regressions also pass with system malloc in that CUDA build.
  Remaining methods beyond the documented minimum still throw explicitly
  ([unsupported methods](src/host_array_unsupported.h)).
  **Accept:** the selected minimum is implemented and covered by the CPU suites;
  unsupported operations fail explicitly. CPU autograd/Fusion remain deferred.
  Closing implementation does not waive the CPU-02-GPU acceptance gate.

- [x] **CPU-02-GPU — Validate the completed CPU core against real CUDA execution.**
  The expanded [CPU/CUDA parity matrix](tests/host_array_core_cuda.phpt) covers all
  dtype pairs for six comparisons, views, reductions, unary math, IEEE division
  and batched matmul. Both it and the
  [foundation matrix](tests/host_array_numeric_cuda.phpt) now pass on PHP 8.1 NTS,
  CUDA 12.6.3, MX570 A (driver 560.94) under Docker WSL 2. The seven targeted
  parity/view/broadcast/operator/shape PHPTs pass without skips.
  The earlier Libkrun `libnvidia-ml.so.1` failure is historical, not a current
  hardware limitation. These are extension results, beyond the native CUDA probe.
  Both matrices still skip explicitly when CUDA device discovery fails; neither
  skip is a pass. The targeted acceptance set also includes
  [view manipulation](tests/array_manipulation_views.phpt),
  [Fusion views](tests/fusion_views.phpt), [broadcast/casts](tests/fusion_broadcast_cast.phpt),
  [shape boundaries](tests/tensor_shape_boundaries.phpt) and
  [CUDA operators](tests/cuda_array_operator_aliasing.phpt).
  **Accept:** each selected operation matches the documented CUDA contract within
  dtype-specific tolerances, including scalar-left operators and non-contiguous
  layouts. The seven-PHPT result covers the shared slice parser, CUDA shape
  boundaries and preserved Fusion/OOM exceptions from all four increment/decrement
  forms. This closes the numerical parity gate, not EXEC-02's separate fault and
  GPU memory/race acceptance. Real-GPU results remain separate from SDK-stub runs.

- [x] **CPU-03 — Use dtype-native CPU storage and measure it.**
  Shared storage allocates `elements * dtype_size`, not a double array with
  logical FP32 values. The [allocation regression](tests/host_array_storage.phpt)
  checks all 13 physical widths against actual Zend-managed bytes with bounded
  metadata overhead. The [benchmark](tests/benchmark_host_storage.php) reports
  allocation, seven-run medians for buffer import/export and addition, and
  optional byte-exact synchronous GPU transfers (`--gpu`).
  Measured CPU PHP 8.4 ZTS, 1,048,576 elements: FP32 4,194,712 allocated bytes
  (4 MiB payload + 408 metadata), FP64 8,389,016 (8 MiB + 408).
  FP32/FP64 buffer import medians were 1.741/3.125 ms, export 1.941/3.171 ms,
  addition 3.178/4.858 ms. CUDA PHP 8.1 NTS on the validated GPU measured
  synchronous FP32/FP64 upload 0.742/1.475 ms and download 1.833/3.333 ms,
  including result allocation; round-trip bytes matched. These local measurements
  are reproducible evidence, not portable speed guarantees. No SIMD/BLAS was added.

- [x] **API-01 — Compare native API reflection with stubs.**
  The [native checker](tests/check_native_api.php) loads the real extension and
  reflects stubs in a separate `php -n` process. Explicit CPU/CUDA profiles compare
  all exposed classes/interfaces and functions, inheritance, flags, required
  counts, parameter names/types/defaults/references/variadics and return types.
  CPU's CUDA availability guard has a separate internal
  [test fixture](tests/fixtures/cpu_cuda_array.stub.php), outside public stubs and
  the `PhpTensors` namespace, rather than exposing a synthetic public class or
  pretending the complete CUDA API exists in that build.
  [Generated arginfo](src/api_arginfo.h) now comes from the stubs through the
  [generator](tests/generate_arginfo.php); all native registrations use it.
  The frozen snapshot remains a separate public-change check, not a replacement
  for native reflection. All runner modes check both contracts and generation
  freshness. CI includes a CUDA build/metadata job with SDK-only host execution;
  release drafting depends on that job as well as the CPU matrix.
  Corrected native types, returns, required counts and named arguments. Stubs now
  expose the already-native final classes and integer/boolean scalar unions;
  batches require arrays. Nullable CUDA dtypes, compiler kernel arrays,
  concatenation axis and module operation IDs also agree with their parsers.
  Validation: CPU PHP 8.4/8.5 ZTS match all 12 classes/interfaces and pass 16/16
  selected PHPTs each. CUDA PHP 8.3 matches all 19 classes/interfaces and passes
  both [signature/named-call](tests/native_api_contract.phpt) and
  [driver-free parser](tests/native_api_cuda_parsers.phpt) regressions.
  A deliberately renamed parameter in a rebuilt native module was rejected with
  exit 1 and its exact reflection path; canonical metadata and binary were restored.
  This is API evidence, not real-GPU numerical acceptance; CPU-02-GPU stays open.
  **Accept:** isolated native/stub reflection for CPU and CUDA profiles, covering
  interfaces, flags, types, defaults and named arguments. A deliberately changed
  native signature must fail CI. Prefer one authoritative signature definition.

- [x] **API-02 — Test the complete custom operator protocol.**
  Added a [complete protocol matrix](tests/operator_overload_protocol.phpt) for
  six binary/compound operators, both scalar orders, distinct subclasses, native
  host/custom boundaries, exact callback exceptions and lack of retry after
  rejection. Marker-only objects do not gain executable handlers. Unsupported
  bitwise/shift opcodes, standard string conversion/comparison/truthiness, unary
  signs, pre/post increment/decrement, cloning and arbitrary hook side effects
  have explicit coverage and README contracts. Existing
  [ownership tests](tests/operator_overload_aliasing.phpt) retain falsy/identity/
  reference results, typed properties, destructor exceptions and leak checks.
  The new matrix exposed two defects: increment/decrement replaced pending hook
  exceptions with PHP TypeError, and the standard clone lost custom handlers.
  The [shared result helper](src/operator_result.h) now marks exceptions as
  handled without committing an invalid result; CUDA failure exits use it too.
  A [custom clone handler](src/operator_overload_ce.c) preserves arithmetic
  dispatch and normal PHP member/user-clone semantics; failed clones are released.
  [Host increment errors](tests/host_array_operator_aliasing.phpt) preserve the
  original exception and binding in both profiles. The expanded
  [CUDA counterpart](tests/cuda_array_operator_aliasing.phpt) covers Fusion/OOM
  cases but explicitly skips without a driver; real-GPU execution remains in
  CPU-02-GPU, not counted as passed here.
  Validation: four targeted PHPTs pass on each PHP 8.4/8.5 ZTS CPU build; the
  CUDA PHP 8.3 API-only runner passes all five host tests. Three operator
  regressions pass with system malloc in each of those three configurations.
  Generated arginfo, frozen API and native/stub contracts remain unchanged.
  Both CI runner selections include the new protocol regression.
  **Accept:** document marker interface versus executable base; cover all supported
  operators, operand order, mixed classes, unsupported opcodes, exception propagation,
  pre/post increment/decrement and aliasing. Inheriting `OperatorOverload` already
  supplies `OperatorOverloadable`; users must not repeat it.

- [x] **API-03 — Make serialization follow logical tensor contents.**
  Shared [host serialization](src/contiguous_array_ce.c) exports logical row-major
  contents, offset zero and canonical strides, packing non-contiguous layouts.
  Indexed rows retain their storage but serialize only their own values, including
  empty shapes. Both profiles use `__contiguous_array_v1`; legacy CPU-only
  `__host_array_v1` FP32/FP64 payloads remain readable transactionally.
  [Host persistence](tests/host_array_serialization.phpt) and the
  [cross-profile contract](tests/host_array_contract.phpt) cover dtype bytes,
  subarray offsets, empty tensors, fixed wire fixtures and malformed replacement.
  Existing [CUDA persistence](tests/cuda_array_serialization.phpt) covers logical
  GPU views. README specifies native byte order (validated little-endian),
  same-endian CPU/CUDA portability and snapshot semantics: no gradient state or
  pinned allocation is restored. Full GPU/FPM suite: 97 passed, one expected
  CPU-only skip, seven native checks. Seven host/transfer/lifecycle regressions
  also pass with system malloc.

## 5. P1 — Transfers, execution and performance guarantees

- [ ] **EXEC-01 — Validate CPU/GPU transfer correctness and cost separately.**
  Preserve existing direct contiguous paths and device packing for strided views.
  **Progress:** a shared [raw-byte fixture](tests/fixtures/transfer_contract.php)
  covers all 13 dtypes, including IEEE signed zeros, infinities, NaN payloads,
  subnormals and exact 64-bit integer patterns. The
  [host staging regression](tests/host_array_transfer_layout.phpt) checks nine
  dense/offset/strided/broadcast/scalar/empty/singleton layouts, retained views,
  independent copies and large dense/sparse packing. It passes on PHP 8.4/8.5 ZTS
  CPU and PHP 8.3 CUDA-profile host code, including system malloc in all three.
  The selected CPU runner passes 19/19 per version; CUDA API-only passes 7/7.
  The [GPU transfer matrix](tests/cuda_transfer_contract.phpt) exercises those
  layouts from pinned/unpinned sources, independent upload/download storage and
  device-packed dense/sparse exports. It and both existing transfer regressions
  now pass on PHP 8.1 NTS / CUDA 12.6.3 / MX570 A under WSL 2: three passed,
  zero skipped. They retain explicit skips for unavailable devices.
  The [cost benchmark](tests/benchmark_transfers.php) now separates host
  import/export, upload, download and device-buffer export, with all 13 dtypes,
  three default sizes and five layouts. Each case has a fresh worker, byte checks,
  first-call resolution flags, calibrated warm batches, raw samples, median/p95,
  logical bytes/s, whole-case RSS/Zend peaks and explicit allocation scope.
  The [report contract](tests/transfer_benchmark_contract.phpt) validates statistics,
  process isolation, CLI errors and bounded staging for a 4 MiB transpose.
  Local PHP 8.4 ZTS CPU measurement completed all 195 default cases and 234
  operation measurements (one extra warmup, three samples), all byte-exact.
  The 4 MiB FP32 transpose used 8,394,488 extra Zend peak bytes, below the
  13,631,488-byte staging budget. GPU requests explicitly fail on CPU-only builds
  or unavailable device queries without emitting a success-shaped report.
  Warm samples include allocation/release; p95 describes batch means, not
  individual-call tail latency. Device memory snapshots are not a peak counter.
  GPU measurements and full peak-memory/performance acceptance remain pending.
  **Accept:** byte-exact supported-dtype round trips for contiguous tensors,
  offset/strided/broadcast views, empties and pinned/unpinned host storage;
  defined ownership/copy independence and exception behavior.
  Measure allocation-inclusive and steady-state latency, bytes/s and peak memory
  for small/medium/large tensors; packed buffers must not expand into PHP scalars.

- [ ] **EXEC-02 — Formalize async resource and ordering contracts.**
  **Progress:** CompiledModule now retains exact native tensor allocations during
  argument preparation and through stream completion/cleanup, rather than relying
  only on PHP object references. This covers reconstruction from scalar-conversion
  callbacks, duplicate arguments, retained views, synchronous launches, async
  operation records and earlier work in a batch. Conversion exceptions stop the
  failing submission; previous batch work is synchronized before lease release.
  Async records are detached before PHP argument destructors can reenter cleanup.
  The batch's temporary status array is now released before returning its existing
  aggregate boolean result.
  The [native lease unit](tests/module_tensor_lease_test.c) passes with PHP 8.4/8.5
  CPU headers and the PHP 8.3 CUDA profile, including an ASan/UBSan run on PHP 8.4.
  CPU runners pass 19/19 per version; the rebuilt CUDA module passes 7/7 host/API
  tests. Public signatures/snapshot are unchanged.
  The expanded [GPU regression](tests/cuda_async_lifetime.phpt) covers callback
  reconstruction, aborted scalar conversion and reentrant argument cleanup.
  It and the [mixed-ordering regression](tests/cuda_async_ordering.phpt) now pass
  on PHP 8.1 NTS / CUDA 12.6.3 / MX570 A under WSL 2 without skips.
  Healthy overlap/workspace memory/race checks and fault-injection acceptance
  remain open. No device-wide synchronization was added to the allocator.
  **Completion/ordering increment:** fixed a use-after-free in global `wait()`,
  added shared owner-context checks for async queries/synchronization, and stopped
  treating failed driver checks as completion. Public checks, timeout maintenance
  and pool-pressure reclamation keep active records/leases after failure.
  Query methods no longer synchronize expired pending streams, and `wait()` polling
  does not invoke that blocking maintenance path. Pending flags are refreshed after
  completion/cleanup, including reentrant argument destruction. Global `sync()` and
  `wait()` release records previously completed by ID.
  Streams are no longer borrowed from still-owned records merely because a query
  succeeds; pool pressure completes/reclaims the owning record before reuse.
  The [stream-status unit](tests/module_stream_status_test.c) passes 24 simulated
  driver/context outcomes in CPU 8.4/8.5 and CUDA 8.3 gates, plus host ASan/UBSan.
  CPU suites remain 19/19 per version; CUDA build and 7/7 host/API tests pass.
  GPU regressions now also cover completion flags, repeated global waits and
  pressure beyond 100 submitted operations; these assertions passed in the
  current real-GPU lifetime run.
  **Fail-closed implementation:** unconfirmed CompiledModule/Fusion completion,
  including sync/batch/destruction and warmup/value cleanup, now disables CUDA
  process-wide and requests worker restart. The allocator retains device storage
  instead of recycling/freeing it; poisoned-runtime teardown abandons CUDA handles,
  pinned storage and library workspaces to process-exit cleanup. The original PHP
  exception survives cleanup failures, with an explicit secondary log.
  Ordinary argument errors and healthy allocation reuse keep their existing paths.
  The pool and production runtime-reporting/owner helpers pass driverless native
  tests and host ASan/UBSan; both now run in the CUDA API-only gate.
  Added isolated GPU trap regressions for sync, async, batch, query and destruction,
  plus explicit custom/Fusion/custom ordering and released PHP inputs. All seven
  GPU regressions (including lifetime) skipped in the earlier driverless run.
  CUDA build + 7/7 host/API and CPU 19/19 per version passed in that run.
  Docker now exposes the GPU through WSL 2; the earlier SKIPs are historical,
  not a claim that this machine lacks a GPU.
  **Current real-GPU result:** lifetime and mixed ordering pass, but all five
  isolated fault-injection PHPTs fail (zero skips). Sync, async, batch and query
  reach the restart-error check, then fail the fixture's `runtime remains disabled`
  assertion; destruction does not observe the expected failure at `unset()`.
  These failures need investigation, not weaker expectations or device skips.
  The complete selected run is 16 passed / 5 failed / 0 skipped out of 21 PHPTs;
  see [commands and scope](docs/guide.md#validation-status).
  **Final acceptance OPEN:** resolve the fault-injection failures and collect
  healthy overlap/workspace GPU memory/race evidence. Keep EXEC-02 open
  until that evidence exists; host sanitizers and SDK stubs do not substitute for it.
  Evidence: the general allocator recycles freed blocks immediately without stream/
  event ownership ([pool](src/backends/cuda/memory_pool.c)); Fusion and CompiledModule
  implement different retention/synchronization mechanisms.
  **Accept:** prove every allocation remains live until its last consuming stream
  completes, including temporary workspaces and failure cleanup. Test overlapping
  graphs/custom kernels, released PHP inputs, mutations, reads, destruction and
  abandoned executions under real GPU memory/race checking. Do not add mandatory
  device-wide synchronization to every allocation as a performance shortcut.

- [ ] **EXEC-03 — Harden Fusion replay contracts and failure recovery.**
  Preserve existing shape/dtype/stride/context checks, bounded cache and graph
  pending-execution guards.
  **Accept:** differential tests for eager/run/compiled/async/CUDA Graph outputs,
  layout changes, captured constants, multiple outputs, output independence between
  replays, node limits, cache eviction, indexed errors, failed compilation and OOM.
  Every unsupported native boundary is diagnosed in the plan; failures restore
  usable eager execution and release temporary resources.

- [ ] **EXEC-04 — Define reproducibility and precision modes.**
  Evidence: generated Fusion compilation enables fast math while disabling FMA
  contraction ([compiler](src/compiler_ce.c)); matmul separately selects strict
  FP32/TF32. Seeded random factories already exist.
  **Accept:** documented numerical tolerances and determinism boundaries for
  reductions, scatter atomics, TF32/FP16/BF16, RNG and Fusion math. Cache keys include
  all semantic options. Test NaNs/infinities/subnormals and seed replay; do not
  require bitwise equality where a declared backend algorithm cannot guarantee it.

- [ ] **EXEC-05 — Make custom JIT launch/persistence behavior reliable.**
  **Accept:** validate launch dimensions against the actual device, argument
  metadata/ranges, tensor layout assumptions and initialization state. Context
  selection follows MEM-05; retained async arguments remain valid until completion.
  Document that arbitrary custom kernels own their bounds checks and gradient/
  mutation obligations. Define timeout/cancellation as waiting/cleanup, not an
  unsupported promise to interrupt an executing CUDA kernel.

- [ ] **PERF-01 — Establish reproducible performance budgets.**
  **Accept:** benchmark imports, transfers, CPU arithmetic, eager dispatch, cold JIT,
  warm Fusion replay, reductions and small/large/batched matmul separately.
  Reports include commit/tag, exact command, PHP NTS/ZTS, hardware, Toolkit/driver,
  build flags, shapes/dtypes/layouts, warmup, raw samples, median/p95 and memory.
  Freeze numeric regression thresholds from measured baselines before RC; suggested
  starting budget is at most 10% median slowdown on a dedicated stable runner,
  with noise analysis and explicit approval for justified trade-offs.

## 6. P1 — Numerical coverage and practical workflows

- [ ] **MATH-01 — Close fundamental linear algebra and reduction API gaps.**
  Evidence: matmul currently requires rank >= 2 and matching FP32/FP16/BF16;
  FP64 matmul/vector products are not available
  ([dispatch](src/cuda_array/ca_private.c)). Basic reductions accept one axis and
  do not expose `keepdims` ([public API](src/cuda_array/cuda_array_ce.c)).
  **Accept:** decide and implement the 1.0 subset: recommended vector dot/matvec,
  FP64 matmul, multi-axis reductions and `keepdims`. Test batched broadcasting,
  empty dimensions, non-contiguous layouts and gradients. Defer wider solvers
  explicitly rather than promising general linear algebra.

- [ ] **GRAD-01 — Add numerical gradient validation, not only fixed examples.**
  Progress: [numerical regressions](tests/autograd_numerical.phpt) compare 42
  small FP64 objectives against a five-point finite-difference stencil in both
  eager and Fusion. Coverage includes 22 unary operations, seven broadcast
  binary operations, scalar and inverse arithmetic, repeated parents,
  transpose/slice/flip/repeat, zero-safe product reductions and scans, variance,
  standard deviation, weighted softmax and affine layerNorm gradients.
  The stencil uses step `1e-4 * max(1, abs(x))`; comparisons require absolute
  error <= `2e-10 + 1e-9 * abs(reference)` and verify gradient shape, dtype and
  element count. Scalar-power analytical checks use absolute/relative budgets
  of 2e-6 (FP32), 2e-12 (FP64), 4e-3 (FP16) and 2e-2 (BF16), including compiled
  replay. These are regression thresholds for these fixtures, not global
  accuracy guarantees. Backward is documented and tested as first-order only;
  retaining a forward graph does not build a higher-order derivative graph.
  Full CUDA and system-allocator checks pass. NUM-04 separately closes the public
  path-preservation/rejection matrix and singularity contracts using analytical
  regressions. Remaining here: extend finite-difference accuracy validation to
  every advertised derivative, more indexing/casting/layout combinations and
  dtype-specific numerical budgets. This numerical baseline alone does not
  establish universal accuracy.
  **Accept:** finite-difference checks on small FP64 tensors for every advertised
  differentiable operation; cover broadcasting, views, repeated indices, zero-safe
  products/scans, ties and composite softmax/layerNorm/variance paths. Establish
  per-dtype tolerances, eager/Fusion agreement and tests that unsupported paths fail.
  Define first-order-only versus higher-order support.

- [ ] **GRAD-02 — Complete training lifecycle and checkpoint contracts.**
  **Accept:** an explicit inference/no-gradient scope or equivalent documented
  mechanism; verify gradient accumulation, zeroing, detach aliasing, saved-tensor
  mutation detection, retain/release behavior, and failure cleanup. Optimizer state
  restore preserves step/bias-correction and parameter mapping. FP16/BF16 training
  must either support master weights/loss scaling/finite-gradient handling or be
  clearly restricted; current optimizers accept FP32/FP64
  ([optimizer](src/optimizer.c)).

- [ ] **NN-01 — Finish the inference contract without implying CNN training.**
  Evidence: NN explicitly requires contiguous untracked FP32 tensors and rejects
  Fusion capture ([NN API](src/nn.c)).
  **Accept:** differential cuDNN/native-fallback tests for convolution groups,
  dilation, padding, pooling modes, softmax and nonfinite inputs; documented layout
  and memory limits. CNN backward/Fusion integration is optional unless training
  support is deliberately added to the 1.0 scope.

## 7. P1 — Tests, CI and release engineering

- [ ] **QA-01 — Require real CUDA evidence before publishing.**
  Evidence: the release workflow depends on CPU-only compatibility checks
  ([CI](.github/workflows/ci.yml)).
  **Accept:** CPU NTS/ZTS matrix plus CUDA compilation and real-GPU tests for the
  exact release commit, accelerated libraries enabled and native fallback paths.
  No runner available means a draft release, not an unvalidated public release.
  Store commands, pass/fail/skip counts and hardware/toolchain identity.

- [ ] **QA-02 — Repair and run native host test targets.**
  Evidence: the CPU runner exits before C tests; the root Makefile searches for
  C++ tests while the native tests are C
  ([runner](run-tests.sh), [Makefile](Makefile)).
  **Accept:** documented working CPU-PHPT and native-C targets; all three existing
  C tests execute in CI without a GPU. GPU tests cannot silently become green by
  skipping every CUDA assertion.

- [ ] **QA-03 — Add lifecycle, differential and generated-input testing.**
  **Accept:** host ASan/UBSan or equivalent, GPU Compute Sanitizer runs, leak/peak
  memory checks and bounded stress tests for aliasing, malformed imports,
  serialization, OOM, repeated replay and shutdown. Use deterministic generated
  shapes/layouts/dtypes against independent numerical references. Assertions must
  fail for intentionally wrong values; fixed success output is not a test.

- [ ] **PKG-01 — Make package metadata and installation reproducible.**
  Evidence: strict Composer validation fails because lockfile PHP constraints are
  stale; PIE checks inspect metadata rather than installing the package
  ([manifest](composer.json), [lockfile](composer.lock),
  [package check](tests/check_pie_package.php)).
  **Accept:** strict validation succeeds; install the release source archive with
  PIE in clean CPU/CUDA environments, load the extension and compute a result.
  Verify version/name/configure metadata and absence of generated artifacts.

- [ ] **PKG-02 — Make documented build/install commands actually succeed.**
  Evidence: scripts are tracked without executable mode although recipes invoke
  them directly; install paths can warn and return success without enabling PHP
  ([build script](compile.sh), [README](README.md)).
  **Accept:** executable scripts or consistent `bash` recipes; requested install
  failure returns failure/explicit partial status. Fresh-clone CPU and Docker
  instructions run verbatim; verify loading for the selected PHP/SAPI.

- [ ] **PKG-03 — Bound supported toolchains and headless builds.**
  **Accept:** publish and test PHP, Toolkit/driver, compute-capability, Linux
  architecture and optional-library combinations. Headless compilation has an
  explicit target; unsupported NVCC architecture selections fail clearly.
  Do not advertise Windows/macOS/ARM or arbitrary Docker PHP overrides without
  validation ([configuration](config.m4), [Dockerfile](Dockerfile)).

## 8. P1 — Examples and documentation must be trustworthy

- [ ] **DOC-01 — Repair executable JIT examples and remove obsolete caches.**
  Evidence: the async example computes a grid for `[32,8,1]` but launches
  `[256,1,1]`, processing only part of its rows
  ([async example](examples/05_jit_async_execution.php)).
  The serialization example loads a tracked cache with the removed
  `Cuda\CompiledModule` class
  ([serialization example](examples/06_serialize_compiled_module.php)).
  **Accept:** bounded cold/warm example smoke tests check every output; generated
  caches stay out of distribution and invalid cache formats trigger recompilation.

- [ ] **DOC-02 — Fix training validation and saved preprocessing.**
  Evidence: relative-tolerance comparisons accept finite/infinite and opposite-
  infinity mismatches; CSV scaling/class mapping is not preserved with model state
  ([helpers](examples/Support/functions.php),
  [training runner](examples/Support/ClassifierRunner.php)).
  **Accept:** explicit nonfinite comparisons; saved normalization and label mapping;
  fixed raw-row predictions survive save/load and dataset order/split changes.
  Incompatible model/dataset mappings fail explicitly.

- [ ] **DOC-03 — Align all user-facing support and capability information.**
  **Accept:** concise README with links to the operation matrix, transfer/alias/
  synchronization rules and troubleshooting; CPU contribution instructions without
  CUDA prerequisites; correct extension diagnostics and issue forms; executable
  custom tensor example. Reconcile outdated support claims and scheduling in
  [ROADMAP](ROADMAP.md), [contributing guide](CONTRIBUTING.md) and
  [examples guide](examples/README.md). Link historical performance claims to
  immutable reports instead of calling old results "current source".

## 9. P2 — Follow-up work, not automatic 1.0 blockers

- [ ] Add stream-aware asynchronous host/device transfers with pinned-memory
  ownership and completion handles, after EXEC-01/02.
- [ ] Optimize reduction composites (single-/few-kernel softmax/layerNorm),
  CPU SIMD/BLAS and shape-aware dispatch using PERF-01 evidence.
- [ ] Consider reusable RNG generator/state APIs; seeded factories already exist.
- [ ] Extend indexing with masked selection/assignment, range writes, sorting,
  argsort/top-k and unique; define shape/gradient semantics before implementation.
- [ ] Add norms/einsum and selected decompositions/solvers according to real demand.
- [ ] Add portable tensor export/checkpoint formats beyond raw buffers and current
  NPY import; consider external-buffer/DLPack interoperability only with a clear
  lifetime protocol.
- [ ] Explore CNN backward, mixed-precision optimizer workflows, higher-order
  gradients and activation checkpointing after first-order correctness.
- [ ] Add model-only offline inference without loading a training dataset.
- [ ] Treat sparse/complex/quantized tensors, additional devices/platforms,
  multi-GPU/distributed execution and persistent cross-request graph caches as
  separately scoped projects.

## Suggested execution order

| Milestone | Work | Exit condition |
|---|---|---|
| A. Safety baseline | SCOPE-01/03, MEM-01 through MEM-07, NUM-01 through NUM-04, API-01, QA-02/03 | Regressions reproduce the observed defects and pass after fixes; no unresolved P0 |
| B. Useful CPU core | SCOPE-02, CPU-01/02/03, API-02/03, MATH-01 | Agreed CPU/CUDA matrix and public signatures pass in both builds |
| C. Execution/training contracts | EXEC-01 through EXEC-05, GRAD-01/02, NN-01 | Differential, lifetime and training/checkpoint evidence for the supported modes |
| D. Release candidate | PERF-01, QA-01, PKG-01/02/03, DOC-01/02/03 | Clean installation, real-GPU validation, trustworthy examples and performance report |

Dependencies: storage/device semantics precede new async features; dtype/shape
semantics precede expanding CPU/Fusion/gradient coverage; native API conformance
precedes API freeze; CI and clean-package installation precede public release.
Independent test and packaging work can proceed in parallel.

## Audit evidence and limits

Source review found the issues above; this document does not claim every item has
been reproduced or that every path in the project has been dynamically validated.

Checks performed during the original audit:

- `php -n tests\check_api_surface.php`: passed, but checks stubs only.
- `composer validate --strict --no-check-publish --no-interaction`: failed
  (exit 2, stale lockfile).
- Package metadata check, lint of all 14 example PHP files, and
  `docker compose config --quiet`: passed.
- Targeted GPU probes on the existing PHP 8.1 / CUDA 12.6 development build:
  eager/Fusion `eq` disagreement; FP64 scalar division becoming FP32; indexed
  `16777217.0` becoming `16777216` while `item()` preserves it; int16/uint16
  `full()` rejection; FP16 `item()` rejection; lost FP16/BF16 concat tracking.
  Offset-view scalar-add probes passed for FP32/FP16/BF16.
- The probes used an existing local extension binary, not a clean rebuild of
  the audited commit. Source inspection supports the listed observations;
  promote each into a fresh-build regression before closing its item.
- The original audit did not execute a full PHP/GPU suite, fresh CUDA build,
  multi-GPU, FPM concurrency, sanitizer run or performance benchmark.
  Implementation verification is recorded in completed items above; unrelated
  earlier successful runs do not close the remaining release gates.

## Final release gate

- [ ] No open P0; all in-scope P1 items have regression/acceptance evidence.
- [ ] Deferred features have explicit support boundaries, not silent failures.
- [ ] CPU/CUDA runtime/stub contracts are frozen and versioned.
- [ ] Exact release source archive installs and loads in supported environments.
- [ ] CPU and real-GPU results, skip counts, sanitizer evidence and performance
  report are attached to the release candidate.
- [ ] Examples run from a clean checkout without legacy classes or bundled caches.
- [ ] Changelog, limitations, upgrade notes, version metadata and release/tag
  provenance agree; no generated caches/build products/secrets in the archive.
- [ ] Release candidate soak testing is complete; publishing is a deliberate step
  after validation, not merely the result of creating a tag.
