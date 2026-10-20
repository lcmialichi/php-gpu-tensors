#ifndef PHP_TENSORS_SCALAR_H
#define PHP_TENSORS_SCALAR_H

#include "data_types.h"
#include "cuda_exceptions.h"
#include <inttypes.h>

static inline int tensor_scalar_from_zval(zval *value, dtype_t dtype, scalar_value_t *result)
{
    ZVAL_DEREF(value);
    scalar_value_t input = {0};
    SCALAR_FROM_ZVAL(value, input);
    *result = cast_single_value(input, dtype);
    if (result->dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Numeric value is invalid or outside the destination dtype range");
        return FAILURE;
    }
    return SUCCESS;
}

static inline void tensor_int64_to_zval(zval *result, int64_t value)
{
    if (value < ZEND_LONG_MIN || value > ZEND_LONG_MAX)
        ZVAL_STR(result, zend_strpprintf(0, "%" PRId64, value));
    else
        ZVAL_LONG(result, (zend_long)value);
}

static inline void tensor_uint64_to_zval(zval *result, uint64_t value)
{
    if (value > (uint64_t)ZEND_LONG_MAX)
        ZVAL_STR(result, zend_strpprintf(0, "%" PRIu64, value));
    else
        ZVAL_LONG(result, (zend_long)value);
}

static inline int tensor_scalar_to_zval(const void *data, dtype_t dtype, zval *result)
{
    union {
        float f32; double f64;
        int8_t i8; int16_t i16; int32_t i32; int64_t i64;
        uint8_t u8; uint16_t u16; uint32_t u32; uint64_t u64;
    } value;
    size_t size = dtype_size(dtype);
    if (!data || !size || size > sizeof(value))
    {
        CUDA_THROW_INVALID("Unsupported scalar dtype: %d", (int)dtype);
        return FAILURE;
    }
    memcpy(&value, data, size);
    switch (dtype)
    {
        case DTYPE_FLOAT32: ZVAL_DOUBLE(result, value.f32); break;
        case DTYPE_FLOAT64: ZVAL_DOUBLE(result, value.f64); break;
        case DTYPE_FLOAT16: ZVAL_DOUBLE(result, float16_bits_to_float(value.u16)); break;
        case DTYPE_BFLOAT16: ZVAL_DOUBLE(result, bfloat16_bits_to_float(value.u16)); break;
        case DTYPE_INT8: ZVAL_LONG(result, value.i8); break;
        case DTYPE_INT16: ZVAL_LONG(result, value.i16); break;
        case DTYPE_INT32: ZVAL_LONG(result, value.i32); break;
        case DTYPE_INT64: tensor_int64_to_zval(result, value.i64); break;
        case DTYPE_UINT8: ZVAL_LONG(result, value.u8); break;
        case DTYPE_UINT16: ZVAL_LONG(result, value.u16); break;
        case DTYPE_UINT32: tensor_uint64_to_zval(result, value.u32); break;
        case DTYPE_UINT64: tensor_uint64_to_zval(result, value.u64); break;
        case DTYPE_BOOL: ZVAL_BOOL(result, value.u8 != 0); break;
        default:
            CUDA_THROW_INVALID("Unsupported scalar dtype: %d", (int)dtype);
            return FAILURE;
    }
    return SUCCESS;
}

#endif
