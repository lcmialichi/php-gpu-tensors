#include <cuda_runtime.h>
#include "factory_kernels.cuh"
#include "launch_config.cuh"
#include "dispatcher.h"
#include "../data_types.h"

struct transfer_layout
{
    int shape[10];
    size_t strides[10];
    int ndims;
};

template <typename T>
__global__ void pack_strided_kernel(const T *source, T *destination, size_t elements, transfer_layout layout)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < elements; i += (size_t)blockDim.x * gridDim.x)
    {
        size_t remaining = i, offset = 0;
        for (int d = layout.ndims - 1; d >= 0; d--)
        {
            offset += (remaining % layout.shape[d]) * layout.strides[d];
            remaining /= layout.shape[d];
        }
        destination[i] = source[offset];
    }
}

extern "C" cudaError_t launch_pack_strided(
    const void *source, void *destination, size_t elements, size_t element_size,
    const int *shape, const size_t *strides, int ndims)
{
    if (!elements) return cudaSuccess;
    transfer_layout layout;
    if (ndims < 1 || ndims > (int)(sizeof(layout.shape) / sizeof(layout.shape[0])))
        return cudaErrorInvalidValue;
    layout.ndims = ndims;
    for (int d = 0; d < ndims; d++)
    {
        layout.shape[d] = shape[d];
        layout.strides[d] = strides[d];
    }
    dim3 grid = cuda_grid_1d(elements);
    switch (element_size)
    {
        case 1: pack_strided_kernel<<<grid, 256>>>((const uint8_t *)source, (uint8_t *)destination, elements, layout); break;
        case 2: pack_strided_kernel<<<grid, 256>>>((const uint16_t *)source, (uint16_t *)destination, elements, layout); break;
        case 4: pack_strided_kernel<<<grid, 256>>>((const uint32_t *)source, (uint32_t *)destination, elements, layout); break;
        case 8: pack_strided_kernel<<<grid, 256>>>((const uint64_t *)source, (uint64_t *)destination, elements, layout); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

__global__ void bernoulli_kernel(
    const float *values,
    bool *output_data,
    size_t size,
    float p)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_data[idx] = values[idx] <= p;
    }
}

extern "C" void launch_bernoulli_kernel(
    float *values,
    bool *base,
    size_t total_elements,
    float p)
{
    if (total_elements == 0)
        return;

    bernoulli_kernel<<<cuda_grid_1d(total_elements), 256>>>(values, base, total_elements, p);
}

extern "C" void launch_assign_scalar_val_kernel(
    void *base,
    dtype_t dtype,
    scalar_value_t value,
    size_t total_elements)
{
    DISPATCH_DTYPE(dtype, {
        scalar_t scalar;

        switch (dtype)
        {
        case DTYPE_FLOAT32:
            scalar = (scalar_t)value.v.f32;
            break;
        case DTYPE_FLOAT64:
            scalar = (scalar_t)value.v.f64;
            break;
        case DTYPE_INT32:
            scalar = (scalar_t)value.v.i32;
            break;
        case DTYPE_INT64:
            scalar = (scalar_t)value.v.i64;
            break;
        case DTYPE_INT8:
            scalar = (scalar_t)value.v.i8;
            break;
        case DTYPE_BOOL:
            scalar = (scalar_t)value.v.b;
            break;
        default:
            scalar = (scalar_t)0;
            break;
        }

        launch_fill_kernel_with_scalar<scalar_t>(
            (scalar_t *)base,
            scalar,
            total_elements);
    });
}

extern "C" void launch_scale_range_kernel(
    float *values,
    void *base,
    dtype_t dtype,
    scalar_value_t min,
    scalar_value_t max,
    size_t total_elements)
{
    if (total_elements == 0)
        return;

    DISPATCH_DTYPE(dtype, {
        scalar_t s_min, s_max;

        switch (dtype)
        {
        case DTYPE_FLOAT32:
            s_min = (scalar_t)min.v.f32;
            s_max = (scalar_t)max.v.f32;
            break;
        case DTYPE_FLOAT64:
            s_min = (scalar_t)min.v.f64;
            s_max = (scalar_t)max.v.f64;
            break;
        case DTYPE_INT32:
            s_min = (scalar_t)min.v.i32;
            s_max = (scalar_t)max.v.i32;
            break;
        case DTYPE_INT64:
            s_min = (scalar_t)min.v.i64;
            s_max = (scalar_t)max.v.i64;
            break;
        case DTYPE_INT8:
            s_min = (scalar_t)min.v.i8;
            s_max = (scalar_t)max.v.i8;
            break;
        case DTYPE_BOOL:
            s_min = (scalar_t)min.v.b;
            s_max = (scalar_t)max.v.b;
            break;
        default:
            s_min = (scalar_t)0;
            s_max = (scalar_t)0;
            break;
        }

        scale_kernel<scalar_t><<<cuda_grid_1d(total_elements), 256>>>(
            values,
            (scalar_t *)base,
            total_elements,
            s_min,
            s_max);
    });
}
