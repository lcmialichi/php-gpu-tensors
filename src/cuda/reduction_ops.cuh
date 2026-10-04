#include "cuda_runtime.h"
#include <vector>
#include <algorithm>
#include "cuda_op_functors.cuh"
#include <string.h>
#include <float.h>
#include <cstdint>
#include <cmath>

#define MAX_DIMS 10
#define REDUCTION_BLOCK_SIZE 256
#define WARP_SIZE 32

struct ReductionParams
{
    int ndims;
    int reduce_axis;
    size_t total_elements_out;
    size_t reduce_dim_size;
    int d_shape[MAX_DIMS];
    size_t d_strides[MAX_DIMS];
    size_t output_offsets[MAX_DIMS];
    size_t output_strides[MAX_DIMS];
};

template <typename InputT, typename AccumT, typename Op>
__global__ void reduce_kernel(
    const InputT *__restrict__ input,
    AccumT *__restrict__ result,
    size_t input_base_offset,
    size_t divisor,
    ReductionParams d_reduce_params)
{
    extern __shared__ char sdata_raw[];
    AccumT *sdata = (AccumT *)sdata_raw;

    size_t idx_out = blockIdx.x;
    if (idx_out >= d_reduce_params.total_elements_out)
        return;

    int tid = threadIdx.x;
    int reduce_dim_size = d_reduce_params.reduce_dim_size;
    size_t axis_stride = d_reduce_params.d_strides[d_reduce_params.reduce_axis];

    size_t base_flat_index = input_base_offset;
    size_t temp_idx = idx_out;
    for (int i = d_reduce_params.ndims - 1; i >= 0; --i)
    {
        if (i != d_reduce_params.reduce_axis)
        {
            base_flat_index += (temp_idx % d_reduce_params.d_shape[i]) * d_reduce_params.d_strides[i];
            temp_idx /= d_reduce_params.d_shape[i];
        }
    }

    AccumT accumulator = ArgIdentity<AccumT, Op>::get_init_val();

    for (int current_idx = tid; current_idx < reduce_dim_size; current_idx += blockDim.x)
    {
        accumulator = Op::apply(accumulator, static_cast<AccumT>(input[base_flat_index + (size_t)current_idx * axis_stride]));
    }

    for (int s = WARP_SIZE / 2; s > 0; s >>= 1)
    {
        accumulator = Op::apply(accumulator, __shfl_xor_sync(0xffffffff, accumulator, s));
    }

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0)
        sdata[warp_id] = accumulator;

    __syncthreads();

    if (warp_id == 0)
    {
            accumulator = (tid < (blockDim.x / WARP_SIZE)) ? sdata[tid] : ArgIdentity<AccumT, Op>::get_init_val();

        for (int s = 16; s > 0; s >>= 1)
        {
            accumulator = Op::apply(accumulator, __shfl_xor_sync(0xffffffff, accumulator, s));
        }

        if (tid == 0)
            result[idx_out] = accumulator / static_cast<AccumT>(divisor);
    }
}

template <typename T, typename Op>
__global__ void arg_reduce_kernel(
    const T *__restrict__ input,
    int *__restrict__ result_idx,
    size_t input_base_offset,
    ReductionParams d_reduce_params)
{
    extern __shared__ char sdata_raw[];
    T *sdata_vals = (T *)sdata_raw;
    
    int num_warps = blockDim.x / 32;
    size_t vals_bytes = num_warps * sizeof(T);
    size_t indices_offset = (vals_bytes + 3) & ~3; 
    int *sdata_indices = (int *)&sdata_raw[indices_offset];

    size_t idx_out = blockIdx.x;
    if (idx_out >= d_reduce_params.total_elements_out) return;

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    size_t base_flat_index = input_base_offset;
    size_t temp_idx = idx_out;
    for (int i = d_reduce_params.ndims - 1; i >= 0; --i) {
        if (i != d_reduce_params.reduce_axis) {
            base_flat_index += (temp_idx % d_reduce_params.d_shape[i]) * d_reduce_params.d_strides[i];
            temp_idx /= d_reduce_params.d_shape[i];
        }
    }

    T best_val = ArgIdentity<T, Op>::get_init_val();
    int best_idx = 0;
    size_t axis_stride = d_reduce_params.d_strides[d_reduce_params.reduce_axis];
    int dim_size = d_reduce_params.reduce_dim_size;

    for (int i = tid; i < dim_size; i += blockDim.x) {
        T val = input[base_flat_index + (size_t)i * axis_stride];
        if (Op::apply(val, best_val)) {
            best_val = val;
            best_idx = i;
        } else if (val == best_val && i < best_idx) {
            best_idx = i;
        }
    }

    for (int s = 16; s > 0; s >>= 1) {
        T remote_val = __shfl_xor_sync(0xffffffff, best_val, s);
        int remote_idx = __shfl_xor_sync(0xffffffff, best_idx, s);

        if (Op::apply(remote_val, best_val)) {
            best_val = remote_val;
            best_idx = remote_idx;
        } else if (remote_val == best_val && remote_idx < best_idx) {
            best_idx = remote_idx;
        }
    }

    if (lane_id == 0) {
        sdata_vals[warp_id] = best_val;
        sdata_indices[warp_id] = best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        best_val = (tid < num_warps) ? sdata_vals[tid] : ArgIdentity<T, Op>::get_init_val();
        best_idx = (tid < num_warps) ? sdata_indices[tid] : 0;

        for (int s = 16; s > 0; s >>= 1) {
            T remote_val = __shfl_xor_sync(0xffffffff, best_val, s);
            int remote_idx = __shfl_xor_sync(0xffffffff, best_idx, s);

            if (Op::apply(remote_val, best_val)) {
                best_val = remote_val;
                best_idx = remote_idx;
            } else if (remote_val == best_val && remote_idx < best_idx) {
                best_idx = remote_idx;
            }
        }

        if (tid == 0) result_idx[idx_out] = best_idx;
    }
}

template <typename InputT, typename AccumT, typename Op>
void launch_reduce_op_kernel(InputT *input, AccumT *result,
                             int *input_shape, int input_ndims,
                             int *result_shape, size_t *input_strides, int result_ndims,
                             int axis,
                             size_t total_elements_out, size_t input_base_offset,
                             size_t divisor, cudaStream_t stream)
{
    if (total_elements_out == 0)
        return;

    ReductionParams h_params = {};
    h_params.ndims = input_ndims;
    h_params.reduce_axis = axis;
    h_params.total_elements_out = total_elements_out;
    h_params.reduce_dim_size = input_shape[axis];
    memcpy(h_params.d_shape, input_shape, input_ndims * sizeof(int));
    memcpy(h_params.d_strides, input_strides, input_ndims * sizeof(size_t));

    int threads = REDUCTION_BLOCK_SIZE;

    size_t shared_mem_size = (threads / 32) * sizeof(AccumT);

    reduce_kernel<InputT, AccumT, Op><<<total_elements_out, threads, shared_mem_size, stream>>>(
        input, result, input_base_offset, divisor, h_params);
}

template <typename T, typename Op>
void launch_arg_reduce_kernel(T *input, int *result_idx,
                              int *input_shape, int input_ndims,
                              size_t *input_strides,
                              int axis,
                              size_t total_elements_out, size_t input_base_offset, cudaStream_t stream)
{
    if (total_elements_out == 0)
        return;

    ReductionParams h_params = {};
    h_params.ndims = input_ndims;
    h_params.reduce_axis = axis;
    h_params.total_elements_out = total_elements_out;
    h_params.reduce_dim_size = input_shape[axis];
    memcpy(h_params.d_shape, input_shape, input_ndims * sizeof(int));
    memcpy(h_params.d_strides, input_strides, input_ndims * sizeof(size_t));

    int threads = REDUCTION_BLOCK_SIZE;

    int num_warps = threads / 32;

    size_t vals_size = num_warps * sizeof(T);
    size_t indices_offset = ((vals_size + 3) & ~3);
    size_t shared_mem_size = indices_offset + (num_warps * sizeof(int));

    int blocks = (int)total_elements_out;

    arg_reduce_kernel<T, Op><<<blocks, threads, shared_mem_size, stream>>>(
        input, result_idx, input_base_offset, h_params);
}