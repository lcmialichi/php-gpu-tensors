#ifndef REDUCTION_OPS_H
#define REDUCTION_OPS_H

#include <stddef.h>
#include <cuda_runtime_api.h>
#include "../operations.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void launch_reduction(
        void *input, void *output,
        dtype_t dtype, operation_type_t op_type,
        int *input_shape, int input_ndims,
        int *result_shape,
        size_t *input_strides,
        int result_ndims, int axis,
        size_t total_elements_out, size_t input_base_offset, cudaStream_t stream);

    void launch_arg_reduction(
        void *input, int *output,
        dtype_t dtype, operation_type_t op_type,
        int *input_shape, int input_ndims,
        int *result_shape,
        size_t *input_strides,
        int result_ndims, int axis,
        size_t total_elements_out, size_t input_base_offset, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif