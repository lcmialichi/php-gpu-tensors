#ifndef BROADCAST_OPS_CUH
#define BROADCAST_OPS_CUH

#include <cuda_runtime.h>
#include "../data_types.h"
#include "cast.cuh"
#include "launch_config.cuh"

#define MAX_DIMS 10

struct BroadcastParams
{
    size_t a_strides[MAX_DIMS];
    size_t b_strides[MAX_DIMS];
    size_t shape[MAX_DIMS];
    int dims;
    bool is_contiguous;
};

template <typename T, typename Op>
__global__ void broadcast_kernel(
    const T *__restrict__ a,
    const T *__restrict__ b,
    T *__restrict__ result,
    const size_t total,
    const BroadcastParams params,
    const size_t a_base_offset,
    const size_t b_base_offset)
{
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x)
    {
        size_t a_idx = 0;
        size_t b_idx = 0;

        if (params.is_contiguous)
        {
            a_idx = idx;
            b_idx = idx;
        }
        else
        {
            size_t tmp = idx;
            #pragma unroll
            for (int i = MAX_DIMS - 1; i >= 0; --i)
            {
                if (i < params.dims)
                {
                    const size_t coord = tmp % params.shape[i];
                    tmp /= params.shape[i];
                    a_idx += coord * params.a_strides[i];
                    b_idx += coord * params.b_strides[i];
                }
            }
        }

        result[idx] = Op::apply(a[a_idx + a_base_offset], b[b_idx + b_base_offset]);
    }
}

template <typename T, typename Op>
__global__ void broadcast_kernel_with_cast(
    const void *__restrict__ a,
    const dtype_t dtype_a,
    const void *__restrict__ b,
    const dtype_t dtype_b,
    T *__restrict__ result,
    const size_t total,
    const BroadcastParams params,
    const size_t a_base_offset,
    const size_t b_base_offset)
{
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x)
    {
        size_t a_idx = 0;
        size_t b_idx = 0;

        if (params.is_contiguous)
        {
            a_idx = idx;
            b_idx = idx;
        }
        else
        {
            size_t tmp = idx;
            #pragma unroll
            for (int i = MAX_DIMS - 1; i >= 0; --i)
            {
                if (i < params.dims)
                {
                    const size_t coord = tmp % params.shape[i];
                    tmp /= params.shape[i];
                    a_idx += coord * params.a_strides[i];
                    b_idx += coord * params.b_strides[i];
                }
            }
        }

        const T val_a = fetch_and_cast<T>(a, dtype_a, a_idx + a_base_offset);
        const T val_b = fetch_and_cast<T>(b, dtype_b, b_idx + b_base_offset);

        result[idx] = Op::apply(val_a, val_b);
    }
}

static inline BroadcastParams setup_params(int *a_strides, int a_dims, int *b_strides, int b_dims, int *result_shape, int result_dims) {
    BroadcastParams h_params;
    h_params.dims = result_dims;
    h_params.is_contiguous = (a_dims == result_dims && b_dims == result_dims);
    size_t expected_stride = 1;
    for (int i = result_dims - 1; i >= 0 && h_params.is_contiguous; --i) {
        if ((size_t)a_strides[i] != expected_stride || (size_t)b_strides[i] != expected_stride)
            h_params.is_contiguous = false;
        expected_stride *= result_shape[i];
    }

    for (int i = 0; i < result_dims; i++) {
        int a_off = i - (result_dims - a_dims);
        int b_off = i - (result_dims - b_dims);

        h_params.a_strides[i] = (a_off >= 0) ? a_strides[a_off] : 0;
        h_params.b_strides[i] = (b_off >= 0) ? b_strides[b_off] : 0;
        h_params.shape[i] = result_shape[i];

    }
    return h_params;
}

template <typename T, typename Op>
void launch_broadcast_kernel_with_cast(void *a, dtype_t dtype_a, void *b, dtype_t dtype_b, T *result,
                                       int *a_strides, int a_dims,
                                       int *b_strides, int b_dims,
                                       int *result_shape, int result_dims,
                                       size_t total_elements,
                                       size_t a_base_offset,
                                       size_t b_base_offset)
{
    if (total_elements == 0)
        return;

    BroadcastParams h_params = setup_params(a_strides, a_dims, b_strides, b_dims, result_shape, result_dims);

    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, broadcast_kernel_with_cast<T, Op>, 0, 0);

    broadcast_kernel_with_cast<T, Op><<<cuda_grid_1d(total_elements, blockSize), blockSize>>>(
        a, dtype_a, b, dtype_b, result,
        total_elements, h_params, a_base_offset, b_base_offset);
}

template <typename T, typename Op>
void launch_broadcast_kernel(T *a, T *b, T *result,
                             int *a_strides, int a_dims,
                             int *b_strides, int b_dims,
                             int *result_shape, int result_dims,
                             size_t total_elements,
                             size_t a_base_offset,
                             size_t b_base_offset)
{
    if (total_elements == 0)
        return;

    BroadcastParams h_params = setup_params(a_strides, a_dims, b_strides, b_dims, result_shape, result_dims);

    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, broadcast_kernel<T, Op>, 0, 0);

    broadcast_kernel<T, Op><<<cuda_grid_1d(total_elements, blockSize), blockSize>>>(
        a, b, result,
        total_elements, h_params, a_base_offset, b_base_offset);
}

#endif