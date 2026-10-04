#include <cuda_runtime.h>
#include "reduction_ops.cuh"
#include "reduction_ops.h"
#include "dispatcher.h"
#include <stdio.h>

extern "C" void launch_reduction(
    void *input, void *output,
    dtype_t dtype, operation_type_t op_type,
    int *input_shape, int input_ndims,
    int *result_shape,
    size_t *input_strides,
    int result_ndims, int axis,
    size_t total_elements_out, size_t input_base_offset, cudaStream_t stream)
{
    if (op_type == OP_REDUCE_MEAN)
    {
        if (dtype != DTYPE_FLOAT32)
        {
            DISPATCH_DTYPE(dtype, {
                launch_reduce_op_kernel<scalar_t, double, AddOpT<double>>(
                    (scalar_t *)input,
                    (double *)output,
                    input_shape,
                    input_ndims,
                    result_shape,
                    input_strides,
                    result_ndims,
                    axis,
                    total_elements_out,
                    input_base_offset,
                    input_shape[axis], stream);
            });
        }
        else
        {
            DISPATCH_DTYPE(dtype, {
                launch_reduce_op_kernel<scalar_t, float, AddOpT<float>>(
                    (scalar_t *)input,
                    (float *)output,
                    input_shape,
                    input_ndims,
                    result_shape,
                    input_strides,
                    result_ndims,
                    axis,
                    total_elements_out,
                    input_base_offset,
                    input_shape[axis], stream);
            });
        }
        return;
    }

    DISPATCH_DTYPE(dtype, {
        DISPATCH_OP_REDUCTION(op_type, {
            launch_reduce_op_kernel<scalar_t, scalar_t, bin_op_t>(
                (scalar_t *)input,
                (scalar_t *)output,
                input_shape,
                input_ndims,
                result_shape,
                input_strides,
                result_ndims,
                axis,
                total_elements_out,
                input_base_offset,
                1, stream);
        });
    });
}

extern "C" void launch_arg_reduction(
    void *input, int *output,
    dtype_t dtype, operation_type_t op_type,
    int *input_shape, int input_ndims,
    int *result_shape,
    size_t *input_strides,
    int result_ndims, int axis,
    size_t total_elements_out, size_t input_base_offset, cudaStream_t stream)
{
    DISPATCH_DTYPE(dtype, {
        DISPATCH_OP_ARG_REDUCTION(op_type, {
            launch_arg_reduce_kernel<scalar_t, bin_op_t>(
                (scalar_t *)input,
                output,
                input_shape,
                input_ndims,
                input_strides,
                axis,
                total_elements_out,
                input_base_offset, stream);
        });
    });
}
