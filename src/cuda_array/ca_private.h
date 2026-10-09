#ifndef CUDA_ARRAY_WRAPPER_H
#define CUDA_ARRAY_WRAPPER_H

#include <cuda_runtime.h>
#include "php.h"
#include "tensor.h"
#include "operations.h"

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

    tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, operation_type_t operation_type);
    tensor_t *cuda_scalar_op(tensor_t *a, scalar_value_t scalar, operation_type_t operation_type);
    tensor_t *cuda_inv_scalar_op(tensor_t *a, scalar_value_t scalar, operation_type_t operation_type);
    tensor_t *cuda_unary_op(tensor_t *a, operation_type_t operation_type);
    tensor_t *cuda_tensor_reduce(tensor_t *input, int axis, operation_type_t operation_type);
    tensor_t *cuda_tensor_reduce_arg(tensor_t *input, int axis, operation_type_t operation_type);

    cudaError_t cuda_tensor_scatter(
        tensor_t *dest_tensor,
        const int *indices,
        size_t num_indices,
        tensor_t *src_tensor);

    tensor_t *cuda_tensor_matmul(tensor_t *a, tensor_t *b);
    tensor_t *perform_broadcast_operation(tensor_t *a, tensor_t *b, int operation_type);

    tensor_t *cuda_tensor_transpose(tensor_t *tensor, int *axis, int axis_len);

    tensor_t *cuda_tensor_copy(tensor_t *tensor);
    size_t cuda_tensor_size(tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif