# Contributing to PHP GPU Tensors

Thanks for considering a contribution. The extension is in beta: small,
reproducible changes with tests and clear API behavior are especially useful.
Documentation, examples, bug reports, and results from other GPUs and PHP
versions count as contributions too.

## Start a conversation

Search existing GitHub issues before opening a new one. For a bug, include
your PHP version, CUDA Toolkit version, NVIDIA driver/GPU, build command,
minimal PHP reproducer, actual behavior, and expected behavior. Paste text
logs (remove private paths and secrets) rather than screenshots of errors.

For new public APIs or changes to ownership, synchronization, or dtypes,
describe the desired behavior in an issue before implementing it. The
[roadmap](ROADMAP.md) suggests directions, not a prerequisite for proposals.
Issues with a narrowly scoped reproducer or a missing regression test are
good places to begin.

## First contribution

Documentation fixes, examples, API-stub updates, and host-side tests are useful
contributions that do not need access to an NVIDIA GPU. For host-side tests,
you still need the CUDA Toolkit and PHP development headers to build the
extension, but no visible GPU is required:

```bash
bash ./compile.sh
bash ./run-tests.sh --cpu-only
```

For CUDA behavior changes, run `bash ./run-tests.sh --require-gpu` on an NVIDIA
system when possible. If you cannot, include the CPU-only results and state
which GPU tests were not run. Check the open issues for a task that fits your
environment; ask before starting a larger API or ownership change.

## API and versioning

The PHP API is frozen at `0.1.0`, as declared by the signatures in
[`stubs/cuda.stub.php`](stubs/cuda.stub.php) and
[`stubs/cuda_methods.stub.php`](stubs/cuda_methods.stub.php). The freeze covers
public names, signatures, parameter defaults, return types, aliases, and
documented behavior; internal C/CUDA symbols are not public API. New releases
may add backward-compatible functionality, but must not remove or change the
existing contract. Any unavoidable breaking change before `1.0.0` requires
maintainer approval, migration notes, and a minor-version bump. From `1.0.0`,
breaking changes require a major-version bump. Deprecate public APIs for at
least one release before removal when practical. CI checks the reflected API
against [`tests/api_surface.json`](tests/api_surface.json); update that
baseline with `php tests/check_api_surface.php --update` only for an approved
API change.

This beta is not a claim of production readiness or broad hardware support.
Release tags use the `vMAJOR.MINOR.PATCH` format, with prerelease suffixes for betas.
CI publishes a release only after the build and CPU-side tests pass; releases contain source archives,
not prebuilt CUDA binaries.

## Build and test

Linux, a compatible NVIDIA driver and CUDA Toolkit, and PHP development tools
are needed for GPU tests. Build without installing the extension system-wide:

```bash
./compile.sh
./run-tests.sh --require-gpu
```

Use `PHP_BIN`, `PHPIZE`, and `PHP_CONFIG` for non-default PHP installations;
see the [README](README.md) for versioned builds and Docker. `./run-tests.sh`
runs CPU-side C tests before the PHPT suite. With no accessible NVIDIA GPU,
you can still build and run CPU-side tests, but GPU PHPTs will skip; mention
this limitation in the pull request. Do not treat skipped tests as GPU coverage.

If a test fails, include the command, relevant test name and diff, and whether
the failure reproduces without your changes. GPU benchmarks are optional and
should identify hardware, software versions, tensor shapes, and whether data
transfers were timed.

## Submit a change

1. Keep the patch focused; avoid generated build files and benchmark reports.
2. Add or update a `tests/*.phpt` regression for behavior changes. Use the
   existing CPU-side tests for shape, format, or allocator logic when possible.
3. Update the PHP signatures and docblocks in `stubs/` and any affected
   README or example when the public API changes. Check PHP syntax with
   `php -n -l stubs/cuda.stub.php` (the `-n` avoids loading the extension
   while parsing its declarations).
4. Run the relevant tests and report your environment, results, and remaining
   coverage gaps in the pull request. Explain why the chosen behavior is safe
   for PHP reference counting and asynchronous CUDA execution.

Reviews look for predictable error handling (`Cuda\Exception` subclasses),
correct ownership of device/host memory, reproducible tests, and compatibility
with supported PHP builds. Please be patient with iteration: CUDA hardware
and toolchain combinations are varied, and this is not yet a stable release.

By contributing, you agree that your changes are distributed under the
repository's [MIT License](LICENSE).