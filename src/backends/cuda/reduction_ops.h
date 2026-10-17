#ifndef REDUCTION_OPS_H
#define REDUCTION_OPS_H

#include <stddef.h>
#include <cuda_runtime_api.h>
#include "../../operations.h"

#ifdef __cplusplus
extern "C"
{
#endif
    cudaError_t cuda_reduction_status(void);
    void cuda_reduction_shutdown(void);
    cudaError_t cuda_reduction_release_stream(cudaStream_t stream);
    void launch_reduction(
        void *input, void *output,
        dtype_t dtype, operation_type_t op_type,
        int *input_shape, int input_ndims,
        int *result_shape,
        size_t *input_strides,
        int result_ndims, int axis,
        size_t total_elements_out, size_t input_base_offset, cudaStream_t stream);
    void launch_moments_reduction(
        const void *input, void *output,
        dtype_t input_dtype, dtype_t result_dtype,
        const int *input_shape, int input_ndims,
        const size_t *input_strides,
        int axis, size_t total_elements_out,
        int correction, int standard_deviation, cudaStream_t stream);

    void launch_arg_reduction(
        void *input, int *output,
        dtype_t dtype, operation_type_t op_type,
        int *input_shape, int input_ndims,
        int *result_shape,
        size_t *input_strides,
        int result_ndims, int axis,
        size_t total_elements_out, size_t input_base_offset, cudaStream_t stream);
    cudaError_t launch_cumulative_scan(
        const void *input, void *output, dtype_t input_dtype,
        dtype_t output_dtype, size_t total_items, size_t segment_length,
        size_t segment_count, operation_type_t operation, cudaStream_t stream);
    cudaError_t launch_cumulative_product_gradient(
        const void *input, dtype_t dtype, const int *input_shape,
        const size_t *input_strides, int input_ndims,
        const void *upstream, const size_t *upstream_strides,
        const void *prefix_product, void *input_gradient,
        size_t total_items, int axis, cudaStream_t stream);
    cudaError_t launch_cumulative_reorder(
        const void *input, void *output, dtype_t dtype,
        const int *shape, int ndims, int axis, size_t total_items,
        cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif