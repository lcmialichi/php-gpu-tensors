#ifndef FACTORY_KERNELS_CUH
#define FACTORY_KERNELS_CUH

#include "../../data_types.h"
#include "launch_config.cuh"

template <typename T>
static __host__ T get_scalar_value(scalar_value_t s)
{
    if (std::is_same<T, float>::value)
        return (T)s.v.f32;
    if (std::is_same<T, double>::value)
        return (T)s.v.f64;
    if (std::is_same<T, int32_t>::value)
        return (T)s.v.i32;
    if (std::is_same<T, int64_t>::value)
        return (T)s.v.i64;
    if (std::is_same<T, int8_t>::value)
        return (T)s.v.i8;
    if (std::is_same<T, bool>::value)
        return (T)s.v.b;
    return (T)0;
}

template <typename T>
__global__ void assign_scalar_val_kernel(
    T *__restrict__ base,
    T scalar,
    size_t total_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        base[idx] = (T)scalar;
    }
}

template <typename T>
__global__ void scale_kernel(const float *values,
                             T *output_data,
                             size_t size,
                             T min_value,
                             T max_value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float raw_rand = values[idx];
        float f_min = (float)min_value;
        float f_max = (float)max_value;

        if (std::is_integral<T>::value)
        {
            output_data[idx] = (T)roundf(fmaf(raw_rand, (f_max - f_min), f_min));
        }
        else
        {
            output_data[idx] = (T)fmaf(raw_rand, (f_max - f_min), f_min);
        }
    }
}

__global__ void bernoulli_kernel(const float *values, bool *output_data, size_t size, float p);

template <typename T>
void launch_fill_kernel_with_scalar(T *base, T scalar, size_t total_elements)
{
    if (total_elements == 0)
        return;

    assign_scalar_val_kernel<T><<<cuda_grid_1d(total_elements), 256>>>(
        base, scalar, total_elements);
}

template <typename T>
void launch_scale_kernel(float *values, T *data, size_t size, T min_value, T max_value)
{
    if (size == 0)
        return;

    scale_kernel<T><<<cuda_grid_1d(size), 256>>>(values, data, size, min_value, max_value);
}

#endif