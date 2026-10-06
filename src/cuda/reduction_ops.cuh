#include "cuda_runtime.h"
#include <vector>
#include <algorithm>
#include "cuda_op_functors.cuh"
#include <string.h>
#include <float.h>
#include <cstdint>
#include <cmath>
#include <cub/cub.cuh>
#include "backend_info.h"

cudaError_t reduction_workspace(cudaStream_t stream, size_t bytes, void **storage);
extern thread_local cudaError_t reduction_error;
extern thread_local cuda_backend_info reduction_counters;

template <typename T, typename Op>
struct CubReductionOp {
    __device__ T operator()(T a, T b) const { return Op::apply(a, b); }
};

template <typename InputT, typename AccumT>
struct ReductionCast {
    __host__ __device__ AccumT operator()(InputT value) const { return static_cast<AccumT>(value); }
};

template <typename T>
__global__ void divide_reduction(T *value, size_t divisor)
{
    *value /= static_cast<T>(divisor);
}

template <typename T>
struct IndexedReductionValue {
    T value;
    int index;
};

template <typename T, typename Op>
struct IndexedReductionInput {
    const T *input;
    T identity;
    __host__ __device__ IndexedReductionValue<T> operator()(int index) const {
        T value = input[index];
        if constexpr (std::is_floating_point<T>::value)
            if (isnan(value)) return {identity, 0};
        return {value, index};
    }
};

template <typename T, typename Op>
struct IndexedReductionOp {
    __device__ IndexedReductionValue<T> operator()(IndexedReductionValue<T> a, IndexedReductionValue<T> b) const {
        if (Op::apply(b.value, a.value) || (a.value == b.value && b.index < a.index)) return b;
        return a;
    }
};

template <typename T>
__global__ void extract_reduction_index(const IndexedReductionValue<T> *value, int *result)
{
    *result = value->index;
}

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
__global__ void reduce_columns(const InputT *input, AccumT *output, ReductionParams params, size_t offset, size_t divisor)
{
    __shared__ AccumT partial[8][32];
    size_t index = (size_t)blockIdx.x * 32 + threadIdx.x;
    size_t base = offset, remaining = index;
    if (index < params.total_elements_out)
        for (int d = params.ndims - 1; d >= 0; d--)
            if (d != params.reduce_axis) {
                base += (remaining % params.d_shape[d]) * params.d_strides[d];
                remaining /= params.d_shape[d];
            }
    AccumT value = ArgIdentity<AccumT, Op>::get_init_val();
    if (index < params.total_elements_out)
        for (size_t row = threadIdx.y; row < params.reduce_dim_size; row += 8)
            value = Op::apply(value, static_cast<AccumT>(input[base + row * params.d_strides[params.reduce_axis]]));
    partial[threadIdx.y][threadIdx.x] = value;
    __syncthreads();
    if (!threadIdx.y && index < params.total_elements_out) {
        for (int i = 1; i < 8; i++) value = Op::apply(value, partial[i][threadIdx.x]);
        output[index] = value / static_cast<AccumT>(divisor);
    }
}

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
    constexpr bool ordered_float = std::is_floating_point<AccumT>::value &&
        (std::is_same<Op, MaxOpT<AccumT>>::value || std::is_same<Op, MinOpT<AccumT>>::value);
    if (!ordered_float && total_elements_out == 1 && input_strides[axis] == 1 && input_shape[axis] >= 4096)
    {
        using Iterator = cub::TransformInputIterator<AccumT, ReductionCast<InputT, AccumT>, const InputT *>;
        Iterator iterator(input + input_base_offset, ReductionCast<InputT, AccumT>());
        CubReductionOp<AccumT, Op> operation;
        AccumT identity;
        if constexpr (std::is_same<Op, MulOpT<AccumT>>::value) identity = static_cast<AccumT>(1);
        else if constexpr (std::is_same<Op, MaxOpT<AccumT>>::value) identity = std::numeric_limits<AccumT>::lowest();
        else if constexpr (std::is_same<Op, MinOpT<AccumT>>::value) identity = std::numeric_limits<AccumT>::max();
        else identity = static_cast<AccumT>(0);
        if constexpr (std::is_floating_point<AccumT>::value) {
            if constexpr (std::is_same<Op, MaxOpT<AccumT>>::value) identity = -std::numeric_limits<AccumT>::infinity();
            if constexpr (std::is_same<Op, MinOpT<AccumT>>::value) identity = std::numeric_limits<AccumT>::infinity();
        }
        size_t bytes = 0;
        reduction_error = cub::DeviceReduce::Reduce(nullptr, bytes, iterator, result, input_shape[axis], operation, identity, stream);
        void *workspace = nullptr;
        if (reduction_error == cudaSuccess) reduction_error = reduction_workspace(stream, bytes, &workspace);
        if (reduction_error == cudaSuccess)
            reduction_error = cub::DeviceReduce::Reduce(workspace, bytes, iterator, result, input_shape[axis], operation, identity, stream);
        if (reduction_error == cudaSuccess && divisor != 1) {
            divide_reduction<<<1, 1, 0, stream>>>(result, divisor);
            reduction_error = cudaGetLastError();
        }
        reduction_counters.cub_calls++;
        return;
    }
    if (!ordered_float && axis < input_ndims - 1 && input_strides[input_ndims - 1] == 1 &&
        total_elements_out >= 32 && input_shape[axis] >= 8)
    {
        reduce_columns<InputT, AccumT, Op><<< (total_elements_out + 31) / 32, dim3(32, 8), 0, stream >>>(
            input, result, h_params, input_base_offset, divisor);
        reduction_error = cudaGetLastError();
        reduction_counters.axis_calls++;
        return;
    }
    if (!ordered_float)
        while (threads > 32 && input_shape[axis] <= threads / 2) threads /= 2;

    size_t shared_mem_size = (threads / 32) * sizeof(AccumT);

    reduce_kernel<InputT, AccumT, Op><<<total_elements_out, threads, shared_mem_size, stream>>>(
        input, result, input_base_offset, divisor, h_params);
    reduction_error = cudaGetLastError();
    reduction_counters.generic_reduce_calls++;
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

    if (total_elements_out == 1 && input_strides[axis] == 1 && input_shape[axis] >= 4096) {
        T identity;
        if constexpr (std::is_same<Op, ArgMaxOpT<T>>::value)
            identity = std::is_floating_point<T>::value ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
        else
            identity = std::is_floating_point<T>::value ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
        using Value = IndexedReductionValue<T>;
        using Iterator = cub::TransformInputIterator<Value, IndexedReductionInput<T, Op>, cub::CountingInputIterator<int>>;
        Iterator iterator(cub::CountingInputIterator<int>(0), IndexedReductionInput<T, Op>{input + input_base_offset, identity});
        IndexedReductionOp<T, Op> operation;
        Value initial = {identity, 0};
        size_t bytes = 0;
        reduction_error = cub::DeviceReduce::Reduce(nullptr, bytes, iterator, (Value *)nullptr,
                                                    input_shape[axis], operation, initial, stream);
        void *workspace = nullptr;
        size_t offset = (bytes + 255) & ~size_t(255);
        if (reduction_error == cudaSuccess) reduction_error = reduction_workspace(stream, offset + sizeof(Value), &workspace);
        if (reduction_error == cudaSuccess) {
            Value *value = reinterpret_cast<Value *>(static_cast<char *>(workspace) + offset);
            reduction_error = cub::DeviceReduce::Reduce(workspace, bytes, iterator, value,
                                                       input_shape[axis], operation, initial, stream);
            if (reduction_error == cudaSuccess) {
                extract_reduction_index<<<1, 1, 0, stream>>>(value, result_idx);
                reduction_error = cudaGetLastError();
            }
        }
        reduction_counters.cub_calls++;
        return;
    }

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
    reduction_error = cudaGetLastError();
    reduction_counters.generic_reduce_calls++;
}