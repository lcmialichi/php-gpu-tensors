#include <cuda_runtime.h>
#include "cuda_op_functors.cuh"
#include "broadcast_ops.cuh"
#include "broadcast_ops.h"
#include "dispatcher.h"

template <typename OutputT>
static void dispatch_low_precision_broadcast(
    void *a, dtype_t dtype_a, void *b, dtype_t dtype_b, void *result,
    operation_type_t op_type, int *a_strides, int a_dims, int *b_strides,
    int b_dims, int *result_shape, int result_dims, size_t total_elements,
    size_t a_offset, size_t b_offset)
{
    typedef float scalar_t;
    DISPATCH_OP(op_type, {
        launch_broadcast_kernel_low_precision<OutputT, bin_op_t>(
            a, dtype_a, b, dtype_b, (OutputT *)result,
            a_strides, a_dims, b_strides, b_dims, result_shape, result_dims,
            total_elements, a_offset, b_offset);
    });
}

extern "C" void launch_broadcast(
    void *a, dtype_t dtype_a, void *b, dtype_t dtype_b, void *result,
    dtype_t dtype, operation_type_t op_type,
    int *a_strides, int a_dims,
    int *b_strides, int b_dims,
    int *result_shape, int result_dims,
    size_t total_elements, size_t a_offset, size_t b_offset)
{
    if (dtype == DTYPE_FLOAT16)
    {
        dispatch_low_precision_broadcast<__half>(
            a, dtype_a, b, dtype_b, result, op_type, a_strides, a_dims,
            b_strides, b_dims, result_shape, result_dims, total_elements,
            a_offset, b_offset);
        return;
    }
    if (dtype == DTYPE_BFLOAT16)
    {
        dispatch_low_precision_broadcast<__nv_bfloat16>(
            a, dtype_a, b, dtype_b, result, op_type, a_strides, a_dims,
            b_strides, b_dims, result_shape, result_dims, total_elements,
            a_offset, b_offset);
        return;
    }

    int is_same_type = (dtype_a == dtype) && (dtype_b == dtype) ? 1 : 0;
    DISPATCH_DTYPE(dtype, {
        DISPATCH_OP(op_type, {
            if (is_same_type == 1)
            {
                launch_broadcast_kernel<scalar_t, bin_op_t>(
                    (scalar_t *)a,
                    (scalar_t *)b,
                    (scalar_t *)result,
                    a_strides, a_dims,
                    b_strides, b_dims,
                    result_shape, result_dims,
                    total_elements,
                    a_offset,
                    b_offset);
            }
            else
            {
                launch_broadcast_kernel_with_cast<scalar_t, bin_op_t>(
                    a, dtype_a,
                    b, dtype_b,
                    (scalar_t *)result,
                    a_strides, a_dims,
                    b_strides, b_dims,
                    result_shape, result_dims,
                    total_elements,
                    a_offset,
                    b_offset);
            }
        });
    });
}