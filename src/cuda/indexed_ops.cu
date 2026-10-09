#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include "dispatcher.h"
#include "indexed_ops.h"
#include "launch_config.cuh"

struct IndexedParams
{
    int shape[MAX_DIMS];
    size_t input_strides[MAX_DIMS];
    size_t index_strides[MAX_DIMS];
    size_t update_strides[MAX_DIMS];
    int ndims;
    int axis;
    int axis_size;
};

__global__ static void validate_indices_kernel(const int *indices, IndexedParams params,
                                                size_t total, int *invalid)
{
    for (size_t flat = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         flat < total; flat += (size_t)blockDim.x * gridDim.x)
    {
        size_t remainder = flat;
        size_t offset = 0;
        for (int axis = params.ndims - 1; axis >= 0; axis--)
        {
            size_t coordinate = remainder % (size_t)params.shape[axis];
            remainder /= (size_t)params.shape[axis];
            offset += coordinate * params.index_strides[axis];
        }
        int value = indices[offset];
        if (value < 0 || value >= params.axis_size)
            atomicExch(invalid, 1);
    }
}

template <typename T>
__global__ static void gather_kernel(const T *input, const int *indices, T *output,
                                     IndexedParams params, size_t total)
{
    for (size_t flat = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         flat < total; flat += (size_t)blockDim.x * gridDim.x)
    {
        size_t remainder = flat;
        size_t input_offset = 0;
        size_t index_offset = 0;
        int selected = 0;
        for (int axis = params.ndims - 1; axis >= 0; axis--)
        {
            size_t coordinate = remainder % (size_t)params.shape[axis];
            remainder /= (size_t)params.shape[axis];
            index_offset += coordinate * params.index_strides[axis];
            if (axis != params.axis)
                input_offset += coordinate * params.input_strides[axis];
        }
        selected = indices[index_offset];
        if (selected < 0 || selected >= params.axis_size)
            output[flat] = T{};
        else
            output[flat] = input[input_offset + (size_t)selected * params.input_strides[params.axis]];
    }
}

template <typename T>
__global__ static void copy_logical_kernel(const T *input, T *output,
                                           IndexedParams params, size_t total)
{
    for (size_t flat = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         flat < total; flat += (size_t)blockDim.x * gridDim.x)
    {
        size_t remainder = flat;
        size_t offset = 0;
        for (int axis = params.ndims - 1; axis >= 0; axis--)
        {
            size_t coordinate = remainder % (size_t)params.shape[axis];
            remainder /= (size_t)params.shape[axis];
            offset += coordinate * params.input_strides[axis];
        }
        output[flat] = input[offset];
    }
}

template <typename T>
__device__ static void atomic_add_value(T *address, T value);

template <>
__device__ void atomic_add_value<float>(float *address, float value)
{
    atomicAdd(address, value);
}

template <>
__device__ void atomic_add_value<double>(double *address, double value)
{
    atomicAdd(address, value);
}

template <typename T>
__global__ static void scatter_add_kernel(const int *indices, const T *updates,
                                          T *output, IndexedParams params, size_t total)
{
    for (size_t flat = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         flat < total; flat += (size_t)blockDim.x * gridDim.x)
    {
        size_t remainder = flat;
        size_t index_offset = 0;
        size_t update_offset = 0;
        size_t output_offset = 0;
        int selected = 0;
        for (int axis = params.ndims - 1; axis >= 0; axis--)
        {
            size_t coordinate = remainder % (size_t)params.shape[axis];
            remainder /= (size_t)params.shape[axis];
            index_offset += coordinate * params.index_strides[axis];
            update_offset += coordinate * params.update_strides[axis];
            if (axis != params.axis)
                output_offset += coordinate * params.input_strides[axis];
        }
        selected = indices[index_offset];
        if (selected >= 0 && selected < params.axis_size)
            atomic_add_value(output + output_offset +
                             (size_t)selected * params.input_strides[params.axis], updates[update_offset]);
    }
}

static IndexedParams indexed_params(const int *shape, const size_t *input_strides,
                                    const size_t *index_strides, const size_t *update_strides, int ndims,
                                    int axis, int axis_size)
{
    IndexedParams params = {};
    params.ndims = ndims;
    params.axis = axis;
    params.axis_size = axis_size;
    for (int dimension = 0; dimension < ndims; dimension++)
    {
        params.shape[dimension] = shape[dimension];
        params.input_strides[dimension] = input_strides ? input_strides[dimension] : 0;
        params.index_strides[dimension] = index_strides ? index_strides[dimension] : 0;
        params.update_strides[dimension] = update_strides ? update_strides[dimension] : 0;
    }
    return params;
}

template <typename T>
static cudaError_t launch_gather_t(const void *input, const int *indices, void *output,
                                   IndexedParams params, size_t total)
{
    gather_kernel<T><<<cuda_grid_1d(total, 256), 256>>>(
        (const T *)input, indices, (T *)output, params, total);
    return cudaGetLastError();
}

template <typename T>
static cudaError_t launch_copy_t(const void *input, void *output,
                                 IndexedParams params, size_t total)
{
    copy_logical_kernel<T><<<cuda_grid_1d(total, 256), 256>>>(
        (const T *)input, (T *)output, params, total);
    return cudaGetLastError();
}

extern "C" cudaError_t cuda_validate_indices(const int *indices, const int *shape,
                                  const size_t *strides, int ndims, int axis,
                                  int axis_size, size_t total)
{
    if (!total) return cudaSuccess;
    int *device_invalid = NULL;
    int invalid = 0;
    cudaError_t status = cudaMalloc(&device_invalid, sizeof(int));
    if (status == cudaSuccess) status = cudaMemset(device_invalid, 0, sizeof(int));
    if (status == cudaSuccess)
    {
        IndexedParams params = indexed_params(shape, NULL, strides, NULL, ndims, axis, axis_size);
        validate_indices_kernel<<<cuda_grid_1d(total, 256), 256>>>(indices, params, total, device_invalid);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess)
        status = cudaMemcpy(&invalid, device_invalid, sizeof(int), cudaMemcpyDeviceToHost);
    if (device_invalid)
    {
        cudaError_t free_status = cudaFree(device_invalid);
        if (status == cudaSuccess) status = free_status;
    }
    if (status == cudaSuccess && invalid)
        return cudaErrorInvalidValue;
    return status;
}

extern "C" cudaError_t cuda_launch_gather(const void *input, const int *indices, void *output,
                               dtype_t dtype, const int *input_shape,
                               const size_t *input_strides, const int *index_shape,
                               const size_t *index_strides, int ndims, int axis,
                               size_t total)
{
    if (!total) return cudaSuccess;
    IndexedParams params = indexed_params(index_shape, input_strides, index_strides, NULL,
                                          ndims, axis, input_shape[axis]);
    switch (dtype)
    {
        case DTYPE_FLOAT32: return launch_gather_t<float>(input, indices, output, params, total);
        case DTYPE_FLOAT64: return launch_gather_t<double>(input, indices, output, params, total);
        case DTYPE_INT8: return launch_gather_t<int8_t>(input, indices, output, params, total);
        case DTYPE_INT16: return launch_gather_t<int16_t>(input, indices, output, params, total);
        case DTYPE_INT32: return launch_gather_t<int32_t>(input, indices, output, params, total);
        case DTYPE_INT64: return launch_gather_t<int64_t>(input, indices, output, params, total);
        case DTYPE_UINT8: return launch_gather_t<uint8_t>(input, indices, output, params, total);
        case DTYPE_UINT16: return launch_gather_t<uint16_t>(input, indices, output, params, total);
        case DTYPE_UINT32: return launch_gather_t<uint32_t>(input, indices, output, params, total);
        case DTYPE_UINT64: return launch_gather_t<uint64_t>(input, indices, output, params, total);
        case DTYPE_BOOL: return launch_gather_t<bool>(input, indices, output, params, total);
        default: return cudaErrorInvalidValue;
    }
}

extern "C" cudaError_t cuda_launch_scatter_copy(const void *input, void *output, dtype_t dtype,
                                     const int *shape, const size_t *strides,
                                     int ndims, size_t total)
{
    if (!total) return cudaSuccess;
    IndexedParams params = indexed_params(shape, strides, NULL, NULL, ndims, 0, 0);
    switch (dtype)
    {
        case DTYPE_FLOAT32: return launch_copy_t<float>(input, output, params, total);
        case DTYPE_FLOAT64: return launch_copy_t<double>(input, output, params, total);
        case DTYPE_INT8: return launch_copy_t<int8_t>(input, output, params, total);
        case DTYPE_INT16: return launch_copy_t<int16_t>(input, output, params, total);
        case DTYPE_INT32: return launch_copy_t<int32_t>(input, output, params, total);
        case DTYPE_INT64: return launch_copy_t<int64_t>(input, output, params, total);
        case DTYPE_UINT8: return launch_copy_t<uint8_t>(input, output, params, total);
        case DTYPE_UINT16: return launch_copy_t<uint16_t>(input, output, params, total);
        case DTYPE_UINT32: return launch_copy_t<uint32_t>(input, output, params, total);
        case DTYPE_UINT64: return launch_copy_t<uint64_t>(input, output, params, total);
        case DTYPE_BOOL: return launch_copy_t<bool>(input, output, params, total);
        default: return cudaErrorInvalidValue;
    }
}

extern "C" cudaError_t cuda_launch_scatter_add(const int *indices, const void *updates,
                                    void *output, dtype_t dtype,
                                    const int *index_shape, const size_t *output_strides,
                                    const size_t *index_strides, const size_t *update_strides,
                                    int ndims, int axis,
                                    int axis_size, size_t total)
{
    if (!total) return cudaSuccess;
    IndexedParams params = indexed_params(index_shape, output_strides, index_strides, update_strides,
                                          ndims, axis, axis_size);
    if (dtype == DTYPE_FLOAT32)
        scatter_add_kernel<float><<<cuda_grid_1d(total, 256), 256>>>(
            indices, (const float *)updates, (float *)output, params, total);
    else if (dtype == DTYPE_FLOAT64)
        scatter_add_kernel<double><<<cuda_grid_1d(total, 256), 256>>>(
            indices, (const double *)updates, (double *)output, params, total);
    else
        return cudaErrorInvalidValue;
    return cudaGetLastError();
}
