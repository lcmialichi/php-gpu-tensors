#!/bin/bash
set -e

EXT_NAME="phptensors"
PHP_CONFIG=${PHP_CONFIG:-php-config}
PHP_BIN=${PHP_BIN:-php}
PHP_VERSION=$($PHP_CONFIG --version | cut -d. -f1,2)
BUILD_DIR=${BUILD_DIR:-"./${EXT_NAME}_build-${PHP_VERSION}"}
CPU_ONLY=0
if [ "${1:-}" = "--cpu-only" ]; then
   CPU_ONLY=1
elif [ "${1:-}" != "" ] && [ "$1" != "--require-gpu" ]; then
   echo "Usage: $0 [--require-gpu|--cpu-only]" >&2
   exit 2
fi

if [ ! -d "$BUILD_DIR" ]; then
   echo "Compile the extension before running tests"
   exit 1
fi

cd "$BUILD_DIR"
"$PHP_BIN" -n tests/check_api_surface.php
if [ "$CPU_ONLY" -eq 1 ]; then
   make test TESTS="tests/backend_contract.phpt tests/cpu_backend.phpt tests/namespaces.phpt" TEST_PHP_ARGS="-q"
   exit 0
fi
if [ "${1:-}" = "--require-gpu" ]; then
   if ! "$PHP_BIN" -n -d extension="$PWD/modules/phptensors.so" -r 'exit(extension_loaded("phptensors") && cuda_get_device_count() > 0 ? 0 : 1);'; then
      echo "GPU validation needs a working NVIDIA driver (libcuda.so.1) and a visible device" >&2
      exit 1
   fi
fi
POOL_TEST_BINARY=$(mktemp)
trap 'rm -f "$POOL_TEST_BINARY"' EXIT
${CC:-cc} -D_GNU_SOURCE -pthread $($PHP_CONFIG --includes) \
   -I"${CUDA_HOME:-/usr/local/cuda}/include" -Isrc/backends/cuda \
   tests/memory_pool_test.c -o "$POOL_TEST_BINARY"
"$POOL_TEST_BINARY"
NPY_TEST_BINARY=$(mktemp)
trap 'rm -f "$POOL_TEST_BINARY" "$NPY_TEST_BINARY"' EXIT
${CC:-cc} -D_GNU_SOURCE -ffunction-sections -fdata-sections \
   -Wl,--gc-sections $($PHP_CONFIG --includes) \
   -I"${CUDA_HOME:-/usr/local/cuda}/include" -Isrc -Isrc/backends/cuda -Isrc/cuda_array \
   tests/npy_header_test.c -o "$NPY_TEST_BINARY"
"$NPY_TEST_BINARY"
WHERE_TEST_BINARY=$(mktemp)
trap 'rm -f "$POOL_TEST_BINARY" "$NPY_TEST_BINARY" "$WHERE_TEST_BINARY"' EXIT
${CC:-cc} -D_GNU_SOURCE -ffunction-sections -fdata-sections \
   -Wl,--gc-sections $($PHP_CONFIG --includes) \
   -I"${CUDA_HOME:-/usr/local/cuda}/include" -Isrc -Isrc/backends/cuda -Isrc/cuda_array \
   tests/where_shape_test.c -o "$WHERE_TEST_BINARY"
"$WHERE_TEST_BINARY"
make test TEST_PHP_ARGS="-q"