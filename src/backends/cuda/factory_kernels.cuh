#ifndef FACTORY_KERNELS_CUH
#define FACTORY_KERNELS_CUH

#include "../../data_types.h"
#include "launch_config.cuh"
#include <type_traits>

template <typename T>
static __host__ T get_scalar_value(scalar_value_t s)
{
    if constexpr (std::is_same<T, float>::value)
        return (T)s.v.f32;
    else if constexpr (std::is_same<T, double>::value)
        return (T)s.v.f64;
    else if constexpr (std::is_same<T, int32_t>::value)
        return (T)s.v.i32;
    else if constexpr (std::is_same<T, int64_t>::value)
        return (T)s.v.i64;
    else if constexpr (std::is_same<T, int8_t>::value)
        return (T)s.v.i8;
    else if constexpr (std::is_same<T, int16_t>::value)
        return s.v.i16;
    else if constexpr (std::is_same<T, uint8_t>::value)
        return s.v.u8;
    else if constexpr (std::is_same<T, uint16_t>::value)
        return s.v.u16;
    else if constexpr (std::is_same<T, uint32_t>::value)
        return s.v.u32;
    else if constexpr (std::is_same<T, uint64_t>::value)
        return s.v.u64;
    else if constexpr (std::is_same<T, bool>::value)
        return (T)s.v.b;
    else
        static_assert(sizeof(T) == 0, "Unsupported factory scalar type");
}

template <typename T>
__global__ void assign_scalar_val_kernel(
    T *__restrict__ base,
    T scalar,
    size_t total_size)
{
    size_t idx = cuda_linear_index(blockIdx.x, blockDim.x, threadIdx.x);
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
    size_t idx = cuda_linear_index(blockIdx.x, blockDim.x, threadIdx.x);
    if (idx < size)
    {
        if constexpr (std::is_integral<T>::value)
        {
            uint64_t numerator = (uint64_t)(values[idx] * 16777216.0f);
            uint64_t span = (uint64_t)max_value - (uint64_t)min_value;
            // Multiply by a 24-bit fraction without overflowing a 64-bit interval.
            uint64_t offset = (span >> 24) * numerator +
                (((span & 0xffffffULL) * numerator + 0x800000ULL) >> 24);
            if constexpr (std::is_signed<T>::value)
            {
                uint64_t magnitude = min_value < 0
                    ? (uint64_t)(-((int64_t)min_value + 1)) + 1 : 0;
                output_data[idx] = min_value < 0
                    ? (offset < magnitude ? (T)((int64_t)min_value + (int64_t)offset)
                                          : (T)(offset - magnitude))
                    : (T)((uint64_t)min_value + offset);
            }
            else
                output_data[idx] = (T)((uint64_t)min_value + offset);
        }
        else
        {
            T fraction = (T)values[idx];
            output_data[idx] = ((T)1 - fraction) * min_value + fraction * max_value;
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