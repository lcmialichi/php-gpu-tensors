#ifndef UNARY_OPS_CUH
#define UNARY_OPS_CUH

#include <cuda_runtime.h>
#include "launch_config.cuh"

#define MAX_DIMS 10
struct UnaryParams
{
    int shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
    int ndims;
};

__constant__ UnaryParams d_unary_params;

template <typename T, typename Op>
__global__ void unary_kernel_strided(
    const T *base,
    T *result,
    size_t base_offset,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;

    size_t offset = 0;
    size_t remaining = idx;

    for (int d = d_unary_params.ndims - 1; d >= 0; d--)
    {
        size_t coord = remaining % d_unary_params.shape[d];
        remaining /= d_unary_params.shape[d];
        offset += coord * d_unary_params.strides[d];
    }

    result[idx] = Op::apply(base[base_offset + offset]);
}

template <typename T, typename Op>
void launch_unary_op_kernel(
    T *base,
    T *result,
    size_t base_offset,
    int *shape,
    size_t *strides,
    int ndims,
    size_t total_size)
{
    if (total_size == 0)
        return;

    UnaryParams h_params;

    cudaMemcpy(h_params.shape, shape, ndims * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_params.strides, strides, ndims * sizeof(size_t), cudaMemcpyDeviceToHost);

    h_params.ndims = ndims;

    cudaMemcpyToSymbol(d_unary_params, &h_params, sizeof(UnaryParams));
    unary_kernel_strided<T, Op><<<cuda_grid_1d(total_size), 256>>>(
        base,
        result,
        base_offset,
        total_size);
}

#endif
