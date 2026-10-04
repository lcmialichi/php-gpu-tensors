#ifndef CAST_CUH
#define CAST_CUH
#include <cuda_runtime.h>
#include "../data_types.h"

template <typename T>
__device__ __forceinline__ T fetch_and_cast(const void *data, const dtype_t type, const size_t idx)
{
    switch (type)
    {
    case DTYPE_FLOAT32:
        return static_cast<T>(static_cast<const float *>(data)[idx]);
    case DTYPE_FLOAT64:
        return static_cast<T>(static_cast<const double *>(data)[idx]);
    case DTYPE_INT32:
        return static_cast<T>(static_cast<const int32_t *>(data)[idx]);
    case DTYPE_INT64:
        return static_cast<T>(static_cast<const int64_t *>(data)[idx]);
    case DTYPE_INT8:
        return static_cast<T>(static_cast<const int8_t *>(data)[idx]);
    case DTYPE_INT16:
        return static_cast<T>(static_cast<const int16_t *>(data)[idx]);
    case DTYPE_UINT8:
        return static_cast<T>(static_cast<const uint8_t *>(data)[idx]);
    case DTYPE_UINT16:
        return static_cast<T>(static_cast<const uint16_t *>(data)[idx]);
    case DTYPE_UINT32:
        return static_cast<T>(static_cast<const uint32_t *>(data)[idx]);
    case DTYPE_UINT64:
        return static_cast<T>(static_cast<const uint64_t *>(data)[idx]);
    case DTYPE_BOOL:
        return static_cast<T>(static_cast<const bool *>(data)[idx]);
    default:
        return static_cast<T>(0);
    }
}

template <typename T>
__host__ T convert_union_to_type(scalar_value_t s)
{
    switch (s.dtype)
    {
    case DTYPE_FLOAT64:
        return (T)s.v.f64;
    case DTYPE_FLOAT32:
        return (T)s.v.f32;
    case DTYPE_INT64:
        return (T)s.v.i64;
    case DTYPE_INT32:
        return (T)s.v.i32;
    case DTYPE_INT8:
        return (T)s.v.i8;
    case DTYPE_BOOL:
        return (T)s.v.b;
    default:
        return (T)0;
    }
}

#endif