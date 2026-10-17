#include <cuda_runtime.h>
#include "scalar_ops.h"
#include "scalar_ops.cuh"
#include "dispatcher.h"
#include "../../data_types.h"

template <typename OutputT>
static void launch_low_precision_scalar(
    void *base, dtype_t base_dtype, scalar_value_t scalar_val, void *result,
    operation_type_t op_type, size_t base_offset, int *shape, size_t *strides,
    int ndims, size_t total_size, int is_contiguous, int inverse)
{
    if (!total_size)
        return;

    typedef float scalar_t;
    DISPATCH_OP(op_type, {
        float scalar = convert_union_to_type<float>(scalar_val);
        if (inverse)
            scalar_kernel_low_precision<OutputT, bin_op_t, true><<<cuda_grid_1d(total_size), 256>>>(
                base, base_dtype, scalar, (OutputT *)result, base_offset,
                shape, strides, ndims, total_size, is_contiguous);
        else
            scalar_kernel_low_precision<OutputT, bin_op_t, false><<<cuda_grid_1d(total_size), 256>>>(
                base, base_dtype, scalar, (OutputT *)result, base_offset,
                shape, strides, ndims, total_size, is_contiguous);
    });
}

extern "C" void launch_scalar(
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
    int is_contiguous
)
{
    if (result_dtype == DTYPE_FLOAT16)
    {
        launch_low_precision_scalar<__half>(
            base, base_dtype, scalar_val, result, op_type, base_offset,
            shape, strides, ndims, total_size, is_contiguous, 0);
        return;
    }
    if (result_dtype == DTYPE_BFLOAT16)
    {
        launch_low_precision_scalar<__nv_bfloat16>(
            base, base_dtype, scalar_val, result, op_type, base_offset,
            shape, strides, ndims, total_size, is_contiguous, 0);
        return;
    }

    DISPATCH_DTYPE(result_dtype, {
        scalar_t val = convert_union_to_type<scalar_t>(scalar_val);
        DISPATCH_OP(op_type, {
            launch_scalar_op<scalar_t, bin_op_t>(
                base,
                base_dtype,
                val, 
                (scalar_t *)result, 
                base_offset, 
                shape, 
                strides, 
                ndims, 
                total_size,
                is_contiguous
            );
        });
    });
}

extern "C" void launch_scalar_inv(
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
    int is_contiguous
)
{
    if (result_dtype == DTYPE_FLOAT16)
    {
        launch_low_precision_scalar<__half>(
            base, base_dtype, scalar_val, result, op_type, base_offset,
            shape, strides, ndims, total_size, is_contiguous, 1);
        return;
    }
    if (result_dtype == DTYPE_BFLOAT16)
    {
        launch_low_precision_scalar<__nv_bfloat16>(
            base, base_dtype, scalar_val, result, op_type, base_offset,
            shape, strides, ndims, total_size, is_contiguous, 1);
        return;
    }

    DISPATCH_DTYPE(result_dtype, {
        scalar_t val = convert_union_to_type<scalar_t>(scalar_val);
        DISPATCH_OP(op_type, {
            launch_inv_scalar_op<scalar_t, bin_op_t>(
                base,
                base_dtype,
                val, 
                (scalar_t *)result, 
                base_offset, 
                shape, 
                strides, 
                ndims, 
                total_size,
                is_contiguous
            );
        });
    });
}
