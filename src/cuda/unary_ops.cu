#include <cuda_runtime.h>
#include "unary_ops.h"
#include "unary_ops.cuh"
#include "cuda_op_functors.cuh"
#include "dispatcher.h"

extern "C" void launch_unary_op(
    void *base,
    void *result,
    size_t base_offset,
    dtype_t result_dtype,
    operation_type_t op_type,
    int *shape,
    size_t *strides,
    int ndims,
    size_t total_size)
{
    if (result_dtype == DTYPE_FLOAT16)
    {
        typedef float scalar_t;
        DISPATCH_UNARY_OP(op_type, {
            launch_unary_op_low_precision_kernel<__half, bin_op_t>(
                (__half *)base, (__half *)result, base_offset,
                shape, strides, ndims, total_size);
        });
        return;
    }
    if (result_dtype == DTYPE_BFLOAT16)
    {
        typedef float scalar_t;
        DISPATCH_UNARY_OP(op_type, {
            launch_unary_op_low_precision_kernel<__nv_bfloat16, bin_op_t>(
                (__nv_bfloat16 *)base, (__nv_bfloat16 *)result, base_offset,
                shape, strides, ndims, total_size);
        });
        return;
    }

    DISPATCH_DTYPE(result_dtype, {
        DISPATCH_UNARY_OP(op_type, {
            launch_unary_op_kernel<scalar_t, bin_op_t>(
                (scalar_t *)base,
                (scalar_t *)result,
                base_offset,
                shape,
                strides,
                ndims,
                total_size);
        });
    });
}
