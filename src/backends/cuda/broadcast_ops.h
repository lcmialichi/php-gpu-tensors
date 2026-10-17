#ifndef BROADCAST_OPS_H
#define BROADCAST_OPS_H

#include "../../operations.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void launch_broadcast(
        void *a, dtype_t dtype_a, void *b, dtype_t dtype_b, void *result,
        dtype_t dtype, operation_type_t op_type,
        int *a_strides, int a_dims,
        int *b_strides, int b_dims,
        int *result_shape, int result_dims,
        size_t total_elements, size_t a_offset, size_t b_offset);

#ifdef __cplusplus
}
#endif

#endif