# PHP GPU Tensors Roadmap

This document describes possible directions, not promised deadlines.
Suggestions and concrete proposals are welcome via GitHub issues; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Vision

Make PHP a first-class language for GPU-accelerated computing, enabling web
applications, data pipelines, and scientific research to run heavy numerical
workloads without leaving the PHP ecosystem.

**Guiding principles:**

1. **No Python dependency** or bindings to external frameworks
2. **Native PHP syntax and semantics** — no strange DSLs
3. **Explicit control** over GPU execution (no implicit transfers)
4. **Transparency and performance** over "magical" abstractions
5. **Build fundamental primitives**, not a black box

## Short Term (next 3 months)

Focus: **stability and foundation**

### API & Stability
- [x] Freeze the public PHP API at v0.1.0 (signatures documented in `stubs/`)
- [x] Define Semantic Versioning (SemVer) policy
- [ ] Clearly mark experimental APIs with `@internal` or `_experimental` suffix

### Quality & Testing
- [ ] Reach ≥ 70% test coverage for core operations
- [ ] Add regression tests for known bugs
- [x] Set up CI that builds the extension (even without a GPU)
- [ ] Add memory leak tests (valgrind/ASan)

### Documentation
- [ ] Translate README and CONTRIBUTING to other languages (if contributors step up)
- [ ] Create `docs/getting-started.md` with a full tutorial
- [ ] Document architectural decisions (ADRs) under `docs/adr/`
- [ ] Add examples for every supported operation

### Community
- [ ] Create 10+ issues labeled `good first issue`
- [ ] Enable GitHub Discussions
- [ ] Create a Discord/Matrix server
- [ ] Publish the first technical blog post

## Mid Term (3 to 12 months)

Focus: **expanding capabilities**

### Operations & Types
- [ ] Extend cuBLAS coverage beyond compatible `float32` matrix products and batched GEMMs
- [ ] More activation functions: `tanh`, `sigmoid`, `relu`, `softmax`
- [ ] Linear algebra ops: `dot`, `norm`, basic `einsum`
- [ ] Support `complex64` / `complex128`
- [ ] Slicing and advanced indexing

### Interoperability
- [ ] Multi-GPU support
- [ ] Explicit CUDA streams for parallelism
- [ ] Integration with PHP's `ffi` for advanced use cases
- [ ] Export `.npy` and explore safetensors (C-order `.npy` import already exists)

### Tooling
- [ ] Built-in profiler (kernel time, memory usage)
- [ ] `gpu-tensors` CLI for common tasks
- [ ] Debug mode with shape/bounds checking

### Performance
- [ ] Benchmark and improve existing pinned host transfers and stream usage
- [x] Opt-in elementwise kernel fusion with scoped capture and compiled replay
- [x] CUDA Graph backend for replaying compatible multi-kernel fusion plans
- [x] Fused unaries, selection, safe casts, layout transforms and grouped outputs
- [x] Bounded PTX cache and private-stream async fusion execution
- [x] Adapt native reduction/matmul boundaries to private streams
- [x] Reuse mixed-plan scratch and avoid repeated device metadata transfers
- [ ] Adapt power boundaries to private streams and native boundaries to CUDA Graph
- [ ] Public, comparable benchmark suite

## Long Term (1+ year)

Focus: **ecosystem**

### Standard Library
- [ ] `php-gpu-tensors/nn` — neural network layers (linear, conv, pooling)
- [ ] `php-gpu-tensors/optim` — optimizers (SGD, Adam, RMSprop)
- [ ] `php-gpu-tensors/data` — data pipeline, augmentation
- [ ] `php-gpu-tensors/io` — common dataset loaders

### Integrations
- [ ] Windows support (at least via WSL2) and macOS (via eGPU if viable)
- [x] Optional cuDNN FP32 CNN inference: convolution, pooling and channel softmax
- [x] Parallel CUB global reductions and layout/alignment-cached cuBLASLt GEMM
- [ ] cuDNN training/backward, mixed precision and CNN graph fusion
- [ ] Optional cuSPARSE bindings
- [ ] Plugins for PHP frameworks (Laravel, Symfony) for ML tasks

### Education
- [ ] Extensive book/docs: "GPU Computing in PHP"
- [ ] Courses and workshops
- [ ] Academic papers using the project

### Sustainability
- [ ] Seek funding (GitHub Sponsors, NLnet, Sovereign Tech Fund)
- [ ] Establish governance with multiple maintainers
- [ ] Annual community events

## How to Participate

Any item here can become an issue. If you want to work on something:

1. Open an issue referencing the roadmap item
2. Describe the expected behavior and test strategy
3. Read [CONTRIBUTING.md](CONTRIBUTING.md)

## Out of Scope (for now)

To stay focused, we do **not** plan to:

- Reimplement PyTorch or TensorFlow in PHP
- Support non-NVIDIA GPUs (AMD, Intel) — the focus is CUDA
- Compete with Python on raw ML performance in production
- Provide "magical" abstractions that hide the GPU

The goal is to offer **solid foundations** for the community to build on.