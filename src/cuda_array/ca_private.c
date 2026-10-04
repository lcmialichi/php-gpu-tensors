#include "ca_private.h"
#include "broadcast_ops.h"
#include "reduction_ops.h"
#include "unary_ops.h"
#include "tensor_factory.h"
#include "matmul_kernels.h"
#include "scalar_ops.h"
#include "operations.h"
#include <stdlib.h>
#include <string.h>
#include "php.h"
#include "tensor.h"
#include "cuda_exceptions.h"
#include "fusion.h"

tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    if (fusion_active()) return fusion_binary(a, b, operation_type);
    if (!fusion_materialize(a) || !fusion_materialize(b)) return NULL;

    int result_shape[MAX_DIMS];
    int result_dims;
    int a_strides[MAX_DIMS] = {0};
    int b_strides[MAX_DIMS] = {0};
    size_t total_elements;

    if (!prepare_broadcast_operation(a, b, result_shape, &result_dims,
                                     a_strides, b_strides, &total_elements))
    {
        CUDA_THROW_INVALID("Broadcast failed: operand shapes are incompatible");
        return NULL;
    }

    dtype_t promoted_type = promote_types_for_arithmetic(a->dtype, b->dtype, operation_type);
    if (!can_safely_cast_to(a->dtype, promoted_type))
    {
        CUDA_THROW_INVALID("Cannot safely promote operand A (%s) to %s",
                         dtype_to_string(a->dtype), dtype_to_string(promoted_type));
        return NULL;
    }

    if (!can_safely_cast_to(b->dtype, promoted_type))
    {
        CUDA_THROW_INVALID("Cannot safely promote operand B (%s) to %s",
                         dtype_to_string(b->dtype), dtype_to_string(promoted_type));
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty_dtype(result_shape, result_dims, promoted_type);
    if (!result)
    {
        CUDA_THROW_OOM("Failed to create result tensor for operation between %s and %s",
                   dtype_to_string(a->dtype), dtype_to_string(b->dtype));
        return NULL;
    }

    if (!total_elements) return result;
    if (a->data == NULL || b->data == NULL || result->data == NULL)
    {
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("Cannot operate on tensor without GPU data");
        return NULL;
    }

    launch_broadcast(a->data, a->dtype, b->data, b->dtype, result->data,
                     promoted_type, operation_type,
                     a_strides, a->ndims,
                     b_strides, b->ndims,
                     result_shape, result_dims,
                     total_elements, 0, 0);

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("Broadcast operation failed: %s", cudaGetErrorString(status));
        return NULL;
    }
    return result;
}

tensor_t *cuda_scalar_op(tensor_t *a, scalar_value_t scalar, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    if (fusion_active()) return fusion_scalar(a, scalar, operation_type, 0);
    if (!fusion_materialize(a)) return NULL;
    if (!is_contiguous(a) && !lazy_copy_metadata_to_gpu(a))
    {
        CUDA_THROW_RUNTIME("Failed to prepare scalar operation metadata");
        return NULL;
    }
    dtype_t promoted_type = promote_scalar_for_arithmetic(a->dtype, scalar.dtype, operation_type, scalar.is_neg);

    tensor_t *result = cuda_tensor_create_empty_dtype(a->shape, a->ndims, promoted_type);
    if (!result)
    {
        CUDA_THROW_OOM("Failed to create result tensor");
        return NULL;
    }

    launch_scalar(a->data,
                  a->dtype,
                  scalar,
                  result->data,
                  promoted_type,
                  operation_type,
                  a->offset,
                  a->d_shape,
                  a->d_strides,
                  a->ndims,
                  a->total_size,
                  is_contiguous(a));

    cudaError_t status = cudaDeviceSynchronize();

    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Scalar operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_inv_scalar_op(tensor_t *a, scalar_value_t scalar, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    if (fusion_active()) return fusion_scalar(a, scalar, operation_type, 1);
    if (!fusion_materialize(a)) return NULL;
    if (!is_contiguous(a) && !lazy_copy_metadata_to_gpu(a))
    {
        CUDA_THROW_RUNTIME("Failed to prepare inverse scalar operation metadata");
        return NULL;
    }

    dtype_t promoted_type = promote_scalar_for_arithmetic(a->dtype, scalar.dtype, operation_type, scalar.is_neg);

    tensor_t *result = cuda_tensor_create_empty_dtype(a->shape, a->ndims, promoted_type);
    if (!result)
    {
        CUDA_THROW_OOM("Failed to create result tensor");
        return NULL;
    }

    launch_scalar_inv(
        a->data,
        a->dtype,
        scalar,
        result->data,
        promoted_type,
        operation_type,
        a->offset,
        a->d_shape,
        a->d_strides,
        a->ndims,
        a->total_size,
        is_contiguous(a));

    cudaError_t status = cudaDeviceSynchronize();

    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Scalar operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_unary_op(tensor_t *a, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    if (fusion_active()) return fusion_unary(a, operation_type);
    if (!fusion_materialize(a)) return NULL;

    tensor_t *result = resolve_result_tensor(a);
    if (!result)
    {
        CUDA_THROW_OOM("Failed to create result tensor");
        return NULL;
    }

    launch_unary_op(a->data, result->data, 0, a->dtype, operation_type, a->shape, a->strides, a->ndims, a->total_size);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Unary operation failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_reduce_arg(tensor_t *input, int axis, operation_type_t operation_type)
{
    if (fusion_active()) return fusion_reduce(input, axis, operation_type, 1);
    if (!fusion_materialize(input)) return NULL;
    int result_shape_arr[MAX_DIMS];
    size_t total_elements_out;

    int result_ndims = calculate_reduction_shape(input, axis, result_shape_arr, &total_elements_out);
    if (result_ndims <= 0) return NULL;
    if (total_elements_out && input->shape[axis] == 0)
    {
        CUDA_THROW_INVALID("Arg reduction has no identity for an empty axis");
        return NULL;
    }

    tensor_t *result = NULL;
    cudaError_t err = cudaSuccess;
    result = cuda_tensor_create_int(result_shape_arr, result_ndims, NULL);
    if (!result)
        return NULL;

    launch_arg_reduction(
        input->data,
        result->data,
        input->dtype,
        operation_type,
        input->shape,
        input->ndims,
        result->shape,
        input->strides,
        result->ndims,
        axis,
        total_elements_out,
        0, NULL);

    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Failed to synchronize device after reduction: %s", cudaGetErrorString(err));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_reduce(tensor_t *input, int axis, operation_type_t operation_type)
{
    if (fusion_active()) return fusion_reduce(input, axis, operation_type, 0);
    if (!fusion_materialize(input)) return NULL;
    int result_shape_arr[MAX_DIMS];
    size_t total_elements_out;

    int result_ndims = calculate_reduction_shape(input, axis, result_shape_arr, &total_elements_out);
    if (result_ndims <= 0) return NULL;
    if (total_elements_out && input->shape[axis] == 0 &&
        (operation_type == OP_REDUCE_MAX || operation_type == OP_REDUCE_MIN))
    {
        CUDA_THROW_INVALID("Min/max reduction has no identity for an empty axis");
        return NULL;
    }

    dtype_t result_dtype = operation_type == OP_REDUCE_MEAN
                               ? (input->dtype == DTYPE_FLOAT64 || dtype_is_integer(input->dtype) || input->dtype == DTYPE_BOOL
                                      ? DTYPE_FLOAT64
                                      : DTYPE_FLOAT32)
                               : input->dtype;
    tensor_t *result = cuda_tensor_create_empty_dtype(result_shape_arr, result_ndims, result_dtype);
    if (!result)
        return NULL;

    launch_reduction(input->data, result->data, input->dtype, operation_type, input->shape, input->ndims,
                     result_shape_arr, input->strides, result_ndims, axis, total_elements_out, 0, NULL);

    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Failed to synchronize device after reduction: %s", cudaGetErrorString(err));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_reshape(tensor_t *original, int *new_shape, int new_ndims)
{
    if (original == NULL || new_shape == NULL || new_ndims <= 0)
    {
        return NULL;
    }
    if (!fusion_active() && !fusion_materialize(original)) return NULL;

    size_t original_size = 1;
    for (int i = 0; i < original->ndims; i++)
    {
        original_size *= original->shape[i];
    }

    size_t new_size_known = 1;
    int wildcard_index = -1;
    int final_shape[MAX_DIMS];

    for (int i = 0; i < new_ndims; i++)
    {
        final_shape[i] = new_shape[i];

        if (new_shape[i] < 0)
        {
            if (wildcard_index != -1)
            {
                CUDA_THROW_INVALID("Reshape allows only one wildcard dimension (-1) in the new shape.");
                return NULL;
            }
            wildcard_index = i;
        }
        else
        {
            new_size_known *= new_shape[i];
        }
    }

    if (wildcard_index != -1)
    {
        if (!new_size_known)
        {
            CUDA_THROW_INVALID("Cannot infer reshape dimension with zero known elements");
            return NULL;
        }
        if (original_size % new_size_known != 0)
        {
            CUDA_THROW_INVALID("Cannot reshape array of size %zu into shape with known elements %zu.", original_size, new_size_known);
            return NULL;
        }
        final_shape[wildcard_index] = original_size / new_size_known;
        new_size_known = original_size;
    }

    if (original_size != new_size_known)
    {
        CUDA_THROW_INVALID("Reshape requires that the number of elements remains the same. Original: %zu, New: %zu.", original_size, new_size_known);
        return NULL;
    }

    if (!is_contiguous(original))
    {
        CUDA_THROW_RUNTIME("Reshape of non-contiguous tensor requires a memory copy/reorder operation, which is not yet implemented.");
        return NULL;
    }

    size_t new_strides[MAX_DIMS];
    new_strides[new_ndims - 1] = 1;
    for (int i = new_ndims - 2; i >= 0; i--)
    {
        new_strides[i] = new_strides[i + 1] * final_shape[i + 1];
    }

    if (fusion_active())
        return fusion_view(original, OP_RESHAPE, final_shape, new_strides, new_ndims, NULL);

    tensor_t *reshaped = cuda_tensor_create_view(
        original,
        final_shape,
        new_strides,
        new_ndims,
        0,
        original->total_size);

    return reshaped;
}

tensor_t *cuda_tensor_transpose(tensor_t *tensor, int *axis, int axis_len)
{
    if (tensor == NULL || axis == NULL)
    {
        return NULL;
    }
    if (!fusion_active() && !fusion_materialize(tensor)) return NULL;

    if (axis_len != tensor->ndims)
    {
        return NULL;
    }

    for (int i = 0; i < axis_len; i++)
    {
        if (axis[i] < 0 || axis[i] >= tensor->ndims)
        {
            return NULL;
        }
    }

    int new_shape[MAX_DIMS];
    size_t new_strides[MAX_DIMS];
    int ndims = tensor->ndims;

    for (int i = 0; i < ndims; i++)
    {
        new_shape[i] = tensor->shape[axis[i]];
        new_strides[i] = tensor->strides[axis[i]];
    }

    if (fusion_active())
        return fusion_view(tensor, OP_TRANSPOSE, new_shape, new_strides, ndims, axis);

    tensor_t *transposed = cuda_tensor_create_view(
        tensor,
        new_shape,
        new_strides,
        tensor->ndims,
        0,
        tensor->total_size);

    return transposed ? transposed : NULL;
}

tensor_t *cuda_tensor_matmul_nd(tensor_t *a, tensor_t *b)
{
    CUDA_CHECK_AND_RETURN_NULL(a);

    int result_ndims;
    int result_shape[MAX_DIMS];
    if (prepare_matmul_result_shape(
            a->ndims,
            a->shape,
            b->ndims,
            b->shape,
            &result_ndims,
            result_shape) == 0)
    {
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty(result_shape, result_ndims);
    if (result == NULL)
    {
        return NULL;
    }

    int status = cuda_batched_matmul_nd_launcher(
        a->data, b->data, result->data,
        a->shape, a->strides, a->ndims,
        b->shape, b->strides, b->ndims,
        result->shape, result->strides, result->ndims, NULL);

    if (status == 0)
    {
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_matmul(tensor_t *a, tensor_t *b)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    if (fusion_active()) return fusion_matmul(a, b);
    if (!fusion_materialize(a) || !fusion_materialize(b)) return NULL;
    if (a->ndims < 2 || b->ndims < 2)
    {
        return NULL;
    }

    if (a->ndims != 2 || b->ndims != 2)
    {
        return cuda_tensor_matmul_nd(a, b);
    }

    if (a->shape[1] != b->shape[0])
    {
        return NULL;
    }

    int result_shape[2] = {a->shape[0], b->shape[1]};

    tensor_t *result = cuda_tensor_create_empty(result_shape, 2);
    if (result == NULL)
    {
        return NULL;
    }

    int status = cuda_matmul_launcher(
        a->data, b->data, result->data,
        a->shape[0], a->shape[1], b->shape[1],
        a->strides[0], a->strides[1],
        b->strides[0], b->strides[1],
        result->strides[0], result->strides[1], NULL);

    if (status == 0)
    {
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_copy(tensor_t *tensor)
{
    if (!tensor)
        return NULL;

    tensor_t *copy = cuda_tensor_create_empty(tensor->shape, tensor->ndims);
    if (!copy)
        return NULL;

    cudaError_t cuda_status = cudaMemcpy(
        copy->data, tensor->data,
        tensor->total_size * sizeof(float),
        cudaMemcpyDeviceToDevice);

    if (cuda_status != cudaSuccess)
    {
        cuda_tensor_destroy(copy);
        return NULL;
    }

    return copy;
}
