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
elif [ "${1:-}" != "" ] && [ "$1" != "--require-gpu" ] && [ "$1" != "--cuda-api-only" ]; then
   echo "Usage: $0 [--require-gpu|--cpu-only|--cuda-api-only]" >&2
   exit 2
fi

if [ ! -d "$BUILD_DIR" ]; then
   echo "Compile the extension before running tests"
   exit 1
fi

cd "$BUILD_DIR"
"$PHP_BIN" -n tests/check_api_surface.php
"$PHP_BIN" -n tests/generate_arginfo.php --check
if [ "$CPU_ONLY" -eq 1 ]; then
   "$PHP_BIN" -n -d extension="$PWD/modules/phptensors.so" tests/check_native_api.php --cpu-only
   make test TESTS="tests/native_api_contract.phpt tests/backend_contract.phpt tests/cpu_backend.phpt tests/host_array_contract.phpt tests/host_array_storage.phpt tests/host_array_copy_cast.phpt tests/host_array_broadcast.phpt tests/host_array_views.phpt tests/host_array_math.phpt tests/host_array_reductions.phpt tests/host_array_lifecycle.phpt tests/host_array_operator_aliasing.phpt tests/host_array_serialization.phpt tests/namespaces.phpt tests/operator_overload_aliasing.phpt tests/operator_overload_protocol.phpt tests/tensor_shape_boundaries.phpt" TEST_PHP_ARGS="-q"
   exit 0
fi
"$PHP_BIN" -n -d extension="$PWD/modules/phptensors.so" tests/check_native_api.php --cuda
if [ "${1:-}" = "--cuda-api-only" ]; then
   make test TESTS="tests/native_api_contract.phpt tests/native_api_cuda_parsers.phpt tests/operator_overload_protocol.phpt tests/operator_overload_aliasing.phpt tests/host_array_operator_aliasing.phpt" TEST_PHP_ARGS="-q"
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
SHAPE_TEST_BINARY=$(mktemp)
LAUNCH_TEST_BINARY=$(mktemp)
trap 'rm -f "$POOL_TEST_BINARY" "$NPY_TEST_BINARY" "$WHERE_TEST_BINARY" "$SHAPE_TEST_BINARY" "$LAUNCH_TEST_BINARY"' EXIT
${CC:-cc} -D_GNU_SOURCE -ffunction-sections -fdata-sections \
   -Wl,--gc-sections $($PHP_CONFIG --includes) \
   -I"${CUDA_HOME:-/usr/local/cuda}/include" -Isrc -Isrc/backends/cuda -Isrc/cuda_array \
   tests/tensor_shape_test.c -o "$SHAPE_TEST_BINARY"
"$SHAPE_TEST_BINARY"
${CXX:-c++} -I"${CUDA_HOME:-/usr/local/cuda}/include" \
   tests/launch_config_test.cpp -o "$LAUNCH_TEST_BINARY"
"$LAUNCH_TEST_BINARY"
SCALAR_TEST_BINARY=$(mktemp)
trap 'rm -f "$POOL_TEST_BINARY" "$NPY_TEST_BINARY" "$WHERE_TEST_BINARY" "$SHAPE_TEST_BINARY" "$LAUNCH_TEST_BINARY" "$SCALAR_TEST_BINARY"' EXIT
${CC:-cc} -D_GNU_SOURCE -ffunction-sections -fdata-sections \
   -Wl,--gc-sections $($PHP_CONFIG --includes) \
   -I"${CUDA_HOME:-/usr/local/cuda}/include" -Isrc -Isrc/backends/cuda -Isrc/cuda_array \
   tests/scalar_cast_test.c -lm -o "$SCALAR_TEST_BINARY"
"$SCALAR_TEST_BINARY"
if [ "${1:-}" = "--require-gpu" ]; then
   KERNEL_INDEX_TEST_BINARY=$(mktemp)
   SCALAR_TYPES_OBJECT=$(mktemp --suffix=.o)
   trap 'rm -f "$POOL_TEST_BINARY" "$NPY_TEST_BINARY" "$WHERE_TEST_BINARY" "$SHAPE_TEST_BINARY" "$LAUNCH_TEST_BINARY" "$SCALAR_TEST_BINARY" "$KERNEL_INDEX_TEST_BINARY" "$SCALAR_TYPES_OBJECT"' EXIT
   ${CC:-cc} -D_GNU_SOURCE -ffunction-sections -fdata-sections \
      $($PHP_CONFIG --includes) -I"${CUDA_HOME:-/usr/local/cuda}/include" \
      -Isrc -Isrc/backends/cuda -Isrc/cuda_array \
      -c src/data_types.c -o "$SCALAR_TYPES_OBJECT"
   "${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}" -std=c++17 \
      tests/kernel_index_test.cu src/backends/cuda/reduction_ops.o src/backends/cuda/factory_kernels.o \
      "$SCALAR_TYPES_OBJECT" -Xlinker --gc-sections \
      -o "$KERNEL_INDEX_TEST_BINARY"
   "$KERNEL_INDEX_TEST_BINARY"
fi
make test TEST_PHP_ARGS="-q"