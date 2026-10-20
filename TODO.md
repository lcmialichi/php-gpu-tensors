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
  ([CUDA-build host](src/contiguous_array_ce.c), [CPU host](src/cpu_host_array.c)).
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
  ([CPU host](src/cpu_host_array.c), [CUDA-build host](src/contiguous_array_ce.c)).
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
  [CPU host](src/cpu_host_array.c),
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

- [ ] **MEM-06 — Validate process/thread/request lifecycle.**
  Evidence: MINIT initializes CUDA/pool storage before any request; the pool uses
  shared static state, while Fusion and backend workspaces have different lifetimes
  ([startup/shutdown](src/cuda.c), [pool](src/backends/cuda/memory_pool.c)).
  This is a deployment risk to investigate, not a reproduced FPM failure.
  **Accept:** lazy initialization after fork where needed, or an explicit supported
  CLI-only boundary. Exercise worker startup/recycling, repeated requests, ZTS
  concurrency and shutdown with pending work; no inherited invalid contexts,
  cross-request state leakage, deadlocks or unbounded allocation growth.

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

- [ ] **NUM-04 — Never silently sever a differentiable path.**
  Progress: floating autograd nodes now retain FP16/BF16 edges regardless of node
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
  Full CUDA suite: 88 passed, one expected CPU-only skip; seven native checks.
  The API snapshot remains unchanged. Both new regressions and the existing
  low-precision path regression also pass with `USE_ZEND_ALLOC=0`.
  **Remaining:** complete the operation/dtype derivative matrix, including
  boundary/tie/singularity behavior across transforms and composites. The
  verified paths above alone do not establish that full contract.
  **Accept:** implement the derivative or reject the unsupported tracked operation
  explicitly. Audit the full operation/dtype matrix, including casts, copies,
  transforms and composites; no forward success that silently drops required
  gradient edges.

## 4. P1 — A useful, consistent CPU backend and extension API

- [ ] **CPU-01 — Remove accidental build-dependent HostArray behavior.**
  Evidence: CPU-only supports FP32/FP64 with no dimension handlers/iteration;
  CUDA-build host has broader dtypes/indexing but rejects some empty construction
  cases. Serialization formats differ
  ([CPU host](src/cpu_host_array.c), [CUDA-build host](src/contiguous_array_ce.c)).
  CUDA-build HostArray supports `foreach` but is not marked `Traversable`;
  `iterator_to_array()` rejects it. Include standard iterator interoperability
  when unifying the CPU/CUDA-build host contract.
  **Accept:** one shared behavioral test matrix for construction, empties,
  metadata, indexing, iteration, errors and persistence across build profiles.
  Share storage/validation helpers where practical instead of maintaining two
  independent numerical contracts.

- [ ] **CPU-02 — Implement the declared minimum CPU numerical core.**
  Current arithmetic supports same-shape/same-dtype operations; most other methods
  are throwing placeholders ([unsupported methods](src/host_array_unsupported.h)).
  Recommended minimum: copy/cast, broadcasting, reshape/transpose/slice views,
  comparisons/where, basic unary functions, sum/mean/min/max and matrix multiply.
  **Accept:** each selected operation matches the documented CUDA contract within
  dtype-specific tolerances, including scalar-left operators and non-contiguous
  layouts. Explicitly defer CPU autograd/Fusion rather than imply full parity.

- [ ] **CPU-03 — Use dtype-native CPU storage and measure it.**
  Evidence: CPU-only FP32 is backed by `double *`, converting during buffer
  import/export while reporting four-byte elements ([CPU host](src/cpu_host_array.c)).
  **Accept:** real contiguous FP32/FP64 storage, or a clearly separated logical/
  physical-size contract. Record actual allocation footprint and transfer/
  arithmetic throughput; add SIMD/BLAS only after correctness and profiling.

- [ ] **API-01 — Compare native API reflection with stubs.**
  Evidence: the current "frozen API" check reflects loaded PHP stubs rather than
  native declarations ([checker](tests/check_api_surface.php)).
  Existing arginfo/parser differences include nullability, required argument counts,
  arithmetic unions and return types.
  **Accept:** isolated native/stub reflection for CPU and CUDA profiles, covering
  interfaces, flags, types, defaults and named arguments. A deliberately changed
  native signature must fail CI. Prefer one authoritative signature definition.

- [ ] **API-02 — Test the complete custom operator protocol.**
  **Accept:** document marker interface versus executable base; cover all supported
  operators, operand order, mixed classes, unsupported opcodes, exception propagation,
  pre/post increment/decrement and aliasing. Inheriting `OperatorOverload` already
  supplies `OperatorOverloadable`; users must not repeat it.

- [ ] **API-03 — Make serialization follow logical tensor contents.**
  Evidence: host subarray indexing advances a cached pointer, but serialization
  copies the parent tensor's starting bytes
  ([CUDA-build host](src/contiguous_array_ce.c)).
  **Accept:** indexed subarrays and strided views round-trip logical row-major
  values, or unsupported views fail explicitly and have a documented copy path.
  Specify format version, byte order, backend portability and gradient-state rules.

## 5. P1 — Transfers, execution and performance guarantees

- [ ] **EXEC-01 — Validate CPU/GPU transfer correctness and cost separately.**
  Preserve existing direct contiguous paths and device packing for strided views.
  **Accept:** byte-exact supported-dtype round trips for contiguous tensors,
  offset/strided/broadcast views, empties and pinned/unpinned host storage;
  defined ownership/copy independence and exception behavior.
  Measure allocation-inclusive and steady-state latency, bytes/s and peak memory
  for small/medium/large tensors; packed buffers must not expand into PHP scalars.

- [ ] **EXEC-02 — Formalize async resource and ordering contracts.**
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
