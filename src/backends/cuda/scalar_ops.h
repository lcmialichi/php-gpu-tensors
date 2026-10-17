#ifndef SCALAR_OPS_H
#define SCALAR_OPS_H

#include <stddef.h>
#include "../../operations.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void launch_scalar(
        void *base,
        dtype_t base_dtype,
        scalar_value_t scalar_val,
        void *result,
        dtype_t result_dtype,
        operation_type_t op_type,
        size_t base_offset,
        int *shape,
        size_t *strides,
        int ndims,
        size_t total_size,
        int is_contiguous);

    void launch_scalar_inv(
        void *base,
        dtype_t base_dtype,
        scalar_value_t scalar_val,
        void *result,
        dtype_t result_dtype,
        operation_type_t op_type,
        size_t base_offset,
        int *shape,
        size_t *strides,
        int ndims,
        size_t total_size,
        int is_contiguous);

#ifdef __cplusplus
}
#endif
#endif
