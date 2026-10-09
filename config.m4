PHP_ARG_WITH(cuda, for CUDA support,
[  --with-cuda             Include CUDA support])

if test "$PHP_CUDA" != "no"; then
    PHP_REQUIRE_CXX()
    if test "$PHP_CUDA" = "yes"; then
        if test -n "$CUDA_HOME"; then
            PHP_CUDA="$CUDA_HOME"
        else
            PHP_CUDA=/usr/local/cuda
        fi
    fi

    if test ! -f "$PHP_CUDA/include/cuda_runtime.h"; then
        AC_MSG_ERROR([CUDA headers not found in $PHP_CUDA/include; use --with-cuda=/path/to/toolkit])
    fi

    PATH="$PHP_CUDA/bin:$PATH"
    AC_PATH_PROG(NVCC, nvcc, no)
    if test "$NVCC" = "no"; then
        AC_MSG_ERROR([nvcc not found - please install CUDA toolkit])
    fi

    PHP_ADD_INCLUDE($PHP_CUDA/include)
    PHP_ADD_INCLUDE([src])
    PHP_ADD_INCLUDE([src/cuda])
    PHP_ADD_INCLUDE([src/cuda_array])
    PHP_ADD_LIBRARY(stdc++, 1, CUDA_SHARED_LIBADD)

    AC_DEFINE_UNQUOTED(CUDA_INCLUDE_PATH_STR, "-I$PHP_CUDA/include", [inclusion path for NVRTC JIT])
    AC_DEFINE_UNQUOTED(CUDA_CRT_INCLUDE_STR, "-I$PHP_CUDA/include/crt", [ C++ inclusion path for NVRTC JIT])
    AC_DEFINE(HAVE_CUDA, 1, [Define if CUDA support is enabled])

    CUDA_LIB_DIR="$PHP_CUDA/lib64"
    if test ! -f "$CUDA_LIB_DIR/libcudart.so"; then
        CUDA_LIB_DIR="$PHP_CUDA/targets/x86_64-linux/lib"
    fi
    if test ! -f "$CUDA_LIB_DIR/libcudart.so"; then
        AC_MSG_ERROR([CUDA runtime libraries not found under $PHP_CUDA])
    fi

    PHP_CHECK_LIBRARY(nvrtc, nvrtcCreateProgram, [
        PHP_ADD_LIBRARY_WITH_PATH(nvrtc, $CUDA_LIB_DIR, CUDA_SHARED_LIBADD)
    ], [
        AC_MSG_ERROR([libnvrtc not found under $CUDA_LIB_DIR])
    ], [-L$CUDA_LIB_DIR])

    CUDA_STUB_DIR="$CUDA_LIB_DIR/stubs"
    if test -f "$CUDA_STUB_DIR/libcuda.so"; then
        CUDA_DRIVER_LINK_DIR="$CUDA_STUB_DIR"
    else
        CUDA_DRIVER_LINK_DIR="$CUDA_LIB_DIR"
    fi
    PHP_CHECK_LIBRARY(cuda, cuInit, [], [
        AC_MSG_ERROR([libcuda not found for linking; install the CUDA toolkit stubs or NVIDIA driver development files])
    ], [-L$CUDA_DRIVER_LINK_DIR])
    CUDA_SHARED_LIBADD="$CUDA_SHARED_LIBADD -L$CUDA_DRIVER_LINK_DIR -lcuda"

    PHP_ADD_LIBRARY_WITH_PATH(cudart, $CUDA_LIB_DIR, CUDA_SHARED_LIBADD)
    PHP_ADD_LIBRARY_WITH_PATH(curand, $CUDA_LIB_DIR, CUDA_SHARED_LIBADD)

    CUDA_CUBLAS_FLAG=""
    AC_ARG_VAR([CUDA_USE_CUBLAS], [Enable cuBLAS (yes, no; default yes)])
    AC_ARG_VAR([CUDA_USE_CUBLASLT], [Enable cuBLASLt (yes, no; default yes)])
    AC_ARG_VAR([CUDA_CUDNN_ROOT], [Optional cuDNN prefix containing include and lib or lib64])
    AC_ARG_VAR([CUDA_USE_CUDNN], [Enable cuDNN (auto, yes, no; default auto)])
    case "${CUDA_USE_CUBLAS:-yes}" in
      yes|no) ;;
      *) AC_MSG_ERROR([CUDA_USE_CUBLAS must be yes or no]);;
    esac
    case "${CUDA_USE_CUBLASLT:-yes}" in
      yes|no) ;;
      *) AC_MSG_ERROR([CUDA_USE_CUBLASLT must be yes or no]);;
    esac
    case "${CUDA_USE_CUDNN:-auto}" in
      auto|yes|no) ;;
      *) AC_MSG_ERROR([CUDA_USE_CUDNN must be auto, yes or no]);;
    esac
    CUDA_BACKEND_FLAGS=""
    if test "${CUDA_USE_CUBLAS:-yes}" != "no" && test -f "$PHP_CUDA/include/cublas_v2.h"; then
        PHP_CHECK_LIBRARY(cublas, cublasCreate_v2, [
            CUDA_CUBLAS_FLAG="-DHAVE_CUBLAS"
            AC_DEFINE(HAVE_CUBLAS, 1, [cuBLAS available])
            PHP_ADD_LIBRARY_WITH_PATH(cublas, $CUDA_LIB_DIR, CUDA_SHARED_LIBADD)
        ], [
            AC_MSG_WARN([cuBLAS not found; using built-in matmul kernels])
        ], [-L$CUDA_LIB_DIR])
    fi
    PHP_SUBST(CUDA_CUBLAS_FLAG)
    if test -n "$CUDA_CUBLAS_FLAG" && test "${CUDA_USE_CUBLASLT:-yes}" != "no" && test -f "$PHP_CUDA/include/cublasLt.h"; then
        PHP_CHECK_LIBRARY(cublasLt, cublasLtMatmul, [
            CUDA_BACKEND_FLAGS="$CUDA_BACKEND_FLAGS -DHAVE_CUBLASLT"
            AC_DEFINE(HAVE_CUBLASLT, 1, [cuBLASLt available])
            PHP_ADD_LIBRARY_WITH_PATH(cublasLt, $CUDA_LIB_DIR, CUDA_SHARED_LIBADD)
        ], [AC_MSG_WARN([cuBLASLt unavailable; using cuBLAS])], [-L$CUDA_LIB_DIR])
    fi
    if test ! -f "$PHP_CUDA/include/cub/cub.cuh"; then
        AC_MSG_ERROR([CUB headers required; install CUDA toolkit development headers])
    fi
    CUDNN_ROOT="${CUDA_CUDNN_ROOT:-$PHP_CUDA}"
    CUDNN_LIB_DIR="$CUDNN_ROOT/lib64"
    if test ! -f "$CUDNN_LIB_DIR/libcudnn.so"; then
        CUDNN_LIB_DIR="$CUDNN_ROOT/lib"
    fi
    if test "${CUDA_USE_CUDNN:-auto}" != "no" && test -f "$CUDNN_ROOT/include/cudnn.h"; then
        PHP_CHECK_LIBRARY(cudnn, cudnnCreate, [
            CUDA_BACKEND_FLAGS="$CUDA_BACKEND_FLAGS -DHAVE_CUDNN -I$CUDNN_ROOT/include"
            AC_DEFINE(HAVE_CUDNN, 1, [cuDNN available])
            PHP_ADD_INCLUDE($CUDNN_ROOT/include)
            PHP_ADD_LIBRARY_WITH_PATH(cudnn, $CUDNN_LIB_DIR, CUDA_SHARED_LIBADD)
        ], [
            if test "$CUDA_USE_CUDNN" = "yes"; then
                AC_MSG_ERROR([Requested cuDNN library not found])
            fi
            AC_MSG_WARN([cuDNN unavailable; CNN methods will report unavailable backend])
        ], [-L$CUDNN_LIB_DIR])
    elif test "$CUDA_USE_CUDNN" = "yes"; then
        AC_MSG_ERROR([Requested cuDNN headers not found; set CUDA_CUDNN_ROOT])
    fi
    PHP_SUBST(CUDA_BACKEND_FLAGS)

    CXXFLAGS="$CXXFLAGS -O2"
    CFLAGS="$CFLAGS -O2"
    
    AC_MSG_CHECKING([for CUDA GPU architecture])
    if test -n "$CUDA_ARCH"; then
        CUDA_ARCH_FLAG="$CUDA_ARCH"
        AC_MSG_RESULT([using $CUDA_ARCH_FLAG (CUDA_ARCH)])
    else
        DETECTED_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '.')
        if test -z "$DETECTED_ARCH"; then
            CUDA_ARCH_FLAG="sm_70"
            AC_MSG_RESULT([none detected, using $CUDA_ARCH_FLAG; override with CUDA_ARCH=sm_XX])
        else
            CUDA_ARCH_FLAG="sm_$DETECTED_ARCH"
            AC_MSG_RESULT([detected $CUDA_ARCH_FLAG])
        fi
    fi

    case "$CUDA_ARCH_FLAG" in
      sm_@<:@0-9@:>@*) ;;
      *) AC_MSG_ERROR([Invalid CUDA_ARCH: $CUDA_ARCH_FLAG; expected sm_XX]);;
    esac

    PHP_SUBST(NVCC)
    PHP_SUBST(CUDA_ARCH_FLAG)

    dnl CUDA kernels are a static archive: its backend dependencies must follow it.
    CUDA_SHARED_LIBADD="-L. -lcudakernels $CUDA_SHARED_LIBADD"
    
    PHP_SUBST(CUDA_SHARED_LIBADD)
    SRC_FILES="\
    src/cuda.c \
    src/cuda_exceptions.c \
    src/kernel_ce.c \
    src/number_ce.c \
    src/nvidia_types.c \
    src/cuda_wrapper.cpp \
    src/cuda_array/cuda_array_ce.c \ 
    src/contiguous_array_ce.c \
    src/nn.c \
    src/optimizer.c \
    src/cuda_array/ca_private.c \
    src/cuda_array/tensor_transfer.c \
    src/data_types.c \
    src/autograd.c \
    src/tensor.c \
    src/cuda/memory_pool.c \
    src/cuda_array/tensor_factory.c \
    src/cuda_array/tensor_import.c \
    src/cuda_array/npy_import.c \
    src/cuda_array/tensor_where.c \
    src/operations.c \
    src/compiler_ce.c \
    src/fusion.c \
    src/fusion_capture.c \
    src/fusion_plan.c \
    src/fusion_codegen.c \
    src/fusion_execute.c \
    src/fusion_cache.c \
    src/module_ce.c"

    PHP_NEW_EXTENSION(cuda, $SRC_FILES, $ext_shared)
    PHP_ADD_MAKEFILE_FRAGMENT(Makefile.frag, $ext_srcdir)
fi