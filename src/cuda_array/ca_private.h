#ifndef CUDA_ARRAY_WRAPPER_H
#define CUDA_ARRAY_WRAPPER_H

#include <cuda_runtime.h>
#include "php.h"
#include "tensor.h"
#include "operations.h"
#include "../tensor_transform.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define REDUCE_GLOBAL_FLAG -999

    cudaError_t cuda_flatten_php_array_to_gpu(zval *data, float *gpu_data, int *index, size_t total_size);
    static void flatten_php_array_to_buffer(zval *data, float *buffer, int *index);
    tensor_t *cuda_tensor_reshape(tensor_t *original, int *new_shape, int new_ndims);
    tensor_t *cuda_tensor_broadcast_to(tensor_t *original, int *new_shape, int new_ndims);
    tensor_t *cuda_tensor_gather(tensor_t *input, tensor_t *indices, int axis);
    tensor_t *cuda_tensor_scatter_add(tensor_t *input, tensor_t *indices,
                                      tensor_t *updates, int axis);
    int cuda_tensor_validate_indexed_shapes(tensor_t *input, tensor_t *indices,
                                            tensor_t *updates, int axis,
                                            const char *operation);
    tensor_t *cuda_tensor_slice_scatter(tensor_t *updates, const int *input_shape,
                                        int input_ndims, const int *axes,
                                        const int *starts, const int *steps);

    tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, operation_type_t operation_type);
    tensor_t *cuda_scalar_op(tensor_t *a, scalar_value_t scalar, operation_type_t operation_type);
    tensor_t *cuda_inv_scalar_op(tensor_t *a, scalar_value_t scalar, operation_type_t operation_type);
    tensor_t *cuda_unary_op(tensor_t *a, operation_type_t operation_type);
    tensor_t *cuda_tensor_reduce(tensor_t *input, int axis, operation_type_t operation_type);
    tensor_t *cuda_tensor_reduce_arg(tensor_t *input, int axis, operation_type_t operation_type);
    tensor_t *cuda_tensor_cumulative(tensor_t *input, int axis,
                                     operation_type_t operation_type);
    dtype_t cuda_cumulative_result_dtype(dtype_t input_dtype);
    int cuda_tensor_cumulative_into(tensor_t *input, tensor_t *result,
                                    int axis, operation_type_t operation_type,
                                    cudaStream_t stream);
    tensor_t *cuda_tensor_cumulative_product_gradient(
        tensor_t *input, tensor_t *upstream, int axis);
    int cuda_tensor_cumulative_product_gradient_into(
        tensor_t *input, tensor_t *upstream, tensor_t *result, int axis);
    tensor_t *cuda_tensor_moments_reduce(tensor_t *input, int axis, int correction,
                                         int standard_deviation);

    cudaError_t cuda_tensor_scatter(
        tensor_t *dest_tensor,
        const int *indices,
        size_t num_indices,
        tensor_t *src_tensor);

    tensor_t *cuda_tensor_matmul(tensor_t *a, tensor_t *b);
    tensor_t *perform_broadcast_operation(tensor_t *a, tensor_t *b, int operation_type);

    tensor_t *cuda_tensor_transpose(tensor_t *tensor, int *axis, int axis_len);

    tensor_t *cuda_tensor_copy(tensor_t *tensor);
    tensor_t *cuda_tensor_index_transform(tensor_t *input,
                                          tensor_transform_spec_t *spec);
    tensor_t *cuda_tensor_index_transform_backward(
        tensor_t *upstream, const tensor_transform_spec_t *spec);
    tensor_t *cuda_tensor_repeat_counts(tensor_t *input,
                                        const int *prefix_counts,
                                        size_t count, int axis,
                                        const int *output_shape,
                                        int output_ndims);
    int cuda_tensor_repeat_counts_into(tensor_t *input, tensor_t *result,
                                       const int *prefix_counts,
                                       size_t count, int axis,
                                       cudaStream_t stream);
    tensor_t *cuda_tensor_repeat_counts_backward(
        tensor_t *upstream, tensor_t *input, const int *prefix_counts,
        size_t count, int axis);
    int cuda_tensor_repeat_counts_backward_into(
        tensor_t *upstream, tensor_t *result, const int *prefix_counts,
        size_t count, int axis);
    size_t cuda_tensor_size(tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif