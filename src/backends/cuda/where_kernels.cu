#include "where_kernels.h"
#include "launch_config.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

struct WhereParams
{
    size_t condition_strides[MAX_DIMS];
    size_t true_strides[MAX_DIMS];
    size_t false_strides[MAX_DIMS];
    int shape[MAX_DIMS];
    int ndims;
    size_t total;
    int fast_path;
};

__device__ bool where_condition(const void *data, dtype_t dtype, size_t offset)
{
    switch (dtype)
    {
    case DTYPE_FLOAT32: return ((const float *)data)[offset] != 0;
    case DTYPE_FLOAT64: return ((const double *)data)[offset] != 0;
    case DTYPE_INT8: return ((const int8_t *)data)[offset] != 0;
    case DTYPE_INT16: return ((const int16_t *)data)[offset] != 0;
    case DTYPE_INT32: return ((const int32_t *)data)[offset] != 0;
    case DTYPE_INT64: return ((const int64_t *)data)[offset] != 0;
    case DTYPE_UINT8: return ((const uint8_t *)data)[offset] != 0;
    case DTYPE_UINT16: return ((const uint16_t *)data)[offset] != 0;
    case DTYPE_UINT32: return ((const uint32_t *)data)[offset] != 0;
    case DTYPE_UINT64: return ((const uint64_t *)data)[offset] != 0;
    case DTYPE_BOOL: return ((const bool *)data)[offset];
    case DTYPE_FLOAT16: return __half2float(__ushort_as_half(((const uint16_t *)data)[offset])) != 0;
    case DTYPE_BFLOAT16: return __bfloat162float(__ushort_as_bfloat16(((const uint16_t *)data)[offset])) != 0;
    default: return false;
    }
}

template <typename T>
__global__ void where_kernel(const void *condition, dtype_t condition_dtype,
                             const T *on_true, const T *on_false, T *output, WhereParams params)
{
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < params.total; index += (size_t)blockDim.x * gridDim.x)
    {
        size_t condition_offset = index;
        size_t true_offset = index;
        size_t false_offset = index;
        if (!params.fast_path)
        {
            condition_offset = true_offset = false_offset = 0;
            size_t remaining = index;
            for (int axis = params.ndims - 1; axis >= 0; axis--)
            {
                size_t coordinate = remaining % params.shape[axis];
                remaining /= params.shape[axis];
                condition_offset += coordinate * params.condition_strides[axis];
                true_offset += coordinate * params.true_strides[axis];
                false_offset += coordinate * params.false_strides[axis];
            }
        }

        output[index] = where_condition(condition, condition_dtype, condition_offset)
                            ? on_true[true_offset] : on_false[false_offset];
    }
}

extern "C" cudaError_t launch_where_kernel(const tensor_t *condition, const tensor_t *on_true,
                                             const tensor_t *on_false, tensor_t *output,
                                             const size_t *condition_strides, const size_t *true_strides,
                                             const size_t *false_strides, int fast_path)
{
    WhereParams params = {};
    params.ndims = output->ndims;
    params.total = output->total_size;
    params.fast_path = fast_path;
    for (int axis = 0; axis < params.ndims; axis++)
    {
        params.shape[axis] = output->shape[axis];
        params.condition_strides[axis] = condition_strides[axis];
        params.true_strides[axis] = true_strides[axis];
        params.false_strides[axis] = false_strides[axis];
    }

    dim3 grid = cuda_grid_1d(params.total);
    switch (output->dtype)
    {
#define CUDA_WHERE_CASE(dtype_enum, type) \
    case dtype_enum: where_kernel<type><<<grid, 256>>>(condition->data, condition->dtype, \
                     (const type *)on_true->data, (const type *)on_false->data, \
                     (type *)output->data, params); break
    CUDA_WHERE_CASE(DTYPE_FLOAT32, float);
    CUDA_WHERE_CASE(DTYPE_FLOAT64, double);
    CUDA_WHERE_CASE(DTYPE_FLOAT16, __half);
    CUDA_WHERE_CASE(DTYPE_BFLOAT16, __nv_bfloat16);
    CUDA_WHERE_CASE(DTYPE_INT8, int8_t);
    CUDA_WHERE_CASE(DTYPE_INT16, int16_t);
    CUDA_WHERE_CASE(DTYPE_INT32, int32_t);
    CUDA_WHERE_CASE(DTYPE_INT64, int64_t);
    CUDA_WHERE_CASE(DTYPE_UINT8, uint8_t);
    CUDA_WHERE_CASE(DTYPE_UINT16, uint16_t);
    CUDA_WHERE_CASE(DTYPE_UINT32, uint32_t);
    CUDA_WHERE_CASE(DTYPE_UINT64, uint64_t);
    CUDA_WHERE_CASE(DTYPE_BOOL, bool);
#undef CUDA_WHERE_CASE
    default: return cudaErrorInvalidValue;
    }
    return cuda_launch_status(true);
}