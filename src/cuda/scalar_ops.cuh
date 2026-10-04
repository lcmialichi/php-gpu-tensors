#ifndef SCALAR_OPS_CUH
#define SCALAR_OPS_CUH

#include <cuda_runtime.h>
#include "cuda_op_functors.cuh"
#include "cast.cuh"
#include "launch_config.cuh"

template <typename T, typename Op>
__global__ void scalar_kernel_strided(
    const void *base,
    dtype_t base_dtype,
    T scalar,
    T *result,
    size_t base_offset,
    const int *shape,
    const size_t *strides,
    int ndims,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    size_t offset = 0;
    size_t remaining = idx;

    for (int d = ndims - 1; d >= 0; d--)
    {
        size_t coord = remaining % shape[d];
        remaining /= shape[d];
        offset += coord * strides[d];
    }

    const T base_val = fetch_and_cast<T>(base, base_dtype, offset);
    result[idx] = Op::apply(base_val, scalar);
}

template <typename T, typename Op>
__global__ void inv_scalar_kernel_strided(
    const void *base,
    dtype_t base_dtype,
    T scalar,
    T *result,
    size_t base_offset,
    const int *shape,
    const size_t *strides,
    int ndims,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    size_t offset = 0;
    size_t remaining = idx;

    for (int d = ndims - 1; d >= 0; d--)
    {
        size_t coord = remaining % shape[d];
        remaining /= shape[d];
        offset += coord * strides[d];
    }

    const T base_val = fetch_and_cast<T>(base, base_dtype, offset);
    result[idx] = Op::apply(scalar, base_val);
}

template <typename T, typename Op>
__global__ void scalar_kernel_contiguous(
    const void *__restrict__ base,
    dtype_t base_dtype,
    T scalar,
    T *__restrict__ result,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        const T base_val = fetch_and_cast<T>(base, base_dtype, idx);
        result[idx] = Op::apply(base_val, scalar);
    }
}

template <typename T, typename Op>
__global__ void scalar_kernel_contiguous_inv(
    void *__restrict__ base,
    dtype_t base_dtype,
    T scalar,
    T *__restrict__ result,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        const T base_val = fetch_and_cast<T>(base, base_dtype, idx);
        result[idx] = Op::apply(scalar, base_val);
    }
}

template <typename T, typename Op>
void launch_inv_scalar_op(
    void *base,
    dtype_t base_dtype,
    T scalar,
    T *result,
    size_t base_offset,
    int *d_shape,
    size_t *d_strides,
    int ndims,
    size_t total_size,
    int is_contiguous)
{
    if (total_size == 0)
        return;

    if (is_contiguous == 1)
    {
        int minGridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scalar_kernel_contiguous_inv<T, Op>, 0, 0);

        scalar_kernel_contiguous_inv<T, Op><<<cuda_grid_1d(total_size, blockSize), blockSize>>>(
            base,
            base_dtype,
            scalar,
            result,
            total_size);

        return;
    }

    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, inv_scalar_kernel_strided<T, Op>, 0, 0);

    inv_scalar_kernel_strided<T, Op><<<cuda_grid_1d(total_size, blockSize), blockSize>>>(
        base,
        base_dtype,
        scalar,
        result,
        base_offset,
        d_shape,
        d_strides,
        ndims,
        total_size);
}

template <typename T, typename Op>
void launch_scalar_op(
    void *base,
    dtype_t base_dtype,
    T scalar,
    T *result,
    size_t base_offset,
    int *d_shape,
    size_t *d_strides,
    int ndims,
    size_t total_size,
    int is_contiguous)
{
    if (total_size == 0)
        return;

    if (is_contiguous == 1)
    {
        int minGridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scalar_kernel_contiguous<T, Op>, 0, 0);

        scalar_kernel_contiguous<T, Op><<<cuda_grid_1d(total_size, blockSize), blockSize>>>(
            base,
            base_dtype,
            scalar,
            result,
            total_size);

        return;
    }

    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scalar_kernel_strided<T, Op>, 0, 0);

    scalar_kernel_strided<T, Op><<<cuda_grid_1d(total_size, blockSize), blockSize>>>(
        base,
        base_dtype,
        scalar,
        result,
        base_offset,
        d_shape,
        d_strides,
        ndims,
        total_size);
}

#endif
