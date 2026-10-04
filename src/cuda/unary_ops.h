#ifndef UNARY_OPS_H
#define UNARY_OPS_H

#include <stddef.h>
#include "../operations.h"

#ifdef __cplusplus
extern "C"
{
#endif
    /* Shape and strides are host descriptors, copied into kernel parameters. */
    void launch_unary_op(
        void *base,
        void *result,
        size_t base_offset,
        dtype_t result_dtype,
        operation_type_t op_type,
        int *shape,
        size_t *strides,
        int ndims,
        size_t total_size);

#ifdef __cplusplus
}
#endif
#endif
