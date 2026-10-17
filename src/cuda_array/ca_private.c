#include "ca_private.h"
#include "broadcast_ops.h"
#include "reduction_ops.h"
#include "unary_ops.h"
#include "tensor_factory.h"
#include "matmul_kernels.h"
#include "backend_info.h"
#include "scalar_ops.h"
#include "operations.h"
#include <stdlib.h>
#include <string.h>
#include "php.h"
#include "tensor.h"
#include "cuda_exceptions.h"
#include "fusion.h"
#include "autograd.h"
#include "indexed_ops.h"
#include "factory_kernels.h"

static int is_low_precision_dtype(dtype_t dtype)
{
    return dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16;
}

tensor_t *cuda_tensor_op(tensor_t *a, tensor_t *b, operation_type_t operation_type)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    if (!b) return NULL;
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

    if (!total_elements)
    {
        autograd_record_binary(result, a, b, operation_type);
        return result;
    }
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

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("Broadcast kernel launch failed: %s", cudaGetErrorString(status));
        return NULL;
    }
    autograd_record_binary(result, a, b, operation_type);
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

    cudaError_t status = cudaGetLastError();

    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Scalar kernel launch failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    autograd_record_scalar(result, a, scalar, operation_type, 0);
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

    cudaError_t status = cudaGetLastError();

    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Scalar kernel launch failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    autograd_record_scalar(result, a, scalar, operation_type, 1);
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

    launch_unary_op(a->data, result->data, a->offset, a->dtype, operation_type,
                    a->shape, a->strides, a->ndims, a->total_size);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Unary kernel launch failed: %s", cudaGetErrorString(status));
        cuda_tensor_destroy(result);
        return NULL;
    }

    autograd_record_unary(result, a, operation_type);
    return result;
}

tensor_t *cuda_tensor_reduce_arg(tensor_t *input, int axis, operation_type_t operation_type)
{
    if (fusion_active())
        return fusion_reduce(input, axis, operation_type, 1);
    if (input && is_low_precision_dtype(input->dtype))
    {
        tensor_t *promoted = tensor_cast(input, DTYPE_FLOAT32);
        if (!promoted) return NULL;
        tensor_t *result = cuda_tensor_reduce_arg(promoted, axis, operation_type);
        cuda_tensor_destroy(promoted);
        return result;
    }
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

    err = cuda_reduction_status();
    if (err != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Arg reduction kernel launch failed: %s", cudaGetErrorString(err));
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_reduce(tensor_t *input, int axis, operation_type_t operation_type)
{
    if (fusion_active())
        return fusion_reduce(input, axis, operation_type, 0);
    if (input && is_low_precision_dtype(input->dtype))
    {
        tensor_t *promoted = tensor_cast(input, DTYPE_FLOAT32);
        if (!promoted) return NULL;
        tensor_t *result = cuda_tensor_reduce(promoted, axis, operation_type);
        cuda_tensor_destroy(promoted);
        return result;
    }
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

    dtype_t result_dtype = operation_type == OP_REDUCE_ALL || operation_type == OP_REDUCE_ANY
                               ? DTYPE_BOOL
                               : operation_type == OP_REDUCE_MEAN
                               ? (input->dtype == DTYPE_FLOAT64 || dtype_is_integer(input->dtype) || input->dtype == DTYPE_BOOL
                                      ? DTYPE_FLOAT64
                                      : DTYPE_FLOAT32)
                               : input->dtype;
    tensor_t *result = cuda_tensor_create_empty_dtype(result_shape_arr, result_ndims, result_dtype);
    if (!result)
        return NULL;

    launch_reduction(input->data, result->data, input->dtype, operation_type, input->shape, input->ndims,
                     result_shape_arr, input->strides, result_ndims, axis, total_elements_out, 0, NULL);

    cudaError_t err = cuda_reduction_status();
    if (err != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Reduction kernel launch failed: %s", cudaGetErrorString(err));
        cuda_tensor_destroy(result);
        return NULL;
    }

    autograd_record_reduce(result, input, operation_type, axis);
    return result;
}

dtype_t cuda_cumulative_result_dtype(dtype_t input_dtype)
{
    switch (input_dtype)
    {
        case DTYPE_FLOAT32:
        case DTYPE_FLOAT64:
            return input_dtype;
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
            return DTYPE_FLOAT32;
        case DTYPE_INT8:
        case DTYPE_INT16:
        case DTYPE_INT32:
        case DTYPE_UINT8:
        case DTYPE_UINT16:
        case DTYPE_UINT32:
        case DTYPE_BOOL:
            return DTYPE_INT64;
        case DTYPE_INT64:
            return DTYPE_INT64;
        case DTYPE_UINT64:
            return DTYPE_UINT64;
        default:
            return DTYPE_UNKNOWN;
    }
}

int cuda_tensor_cumulative_into(tensor_t *input, tensor_t *result, int axis,
                                operation_type_t operation_type,
                                cudaStream_t stream)
{
    if (!input || !result ||
        (operation_type != OP_CUMSUM && operation_type != OP_CUMPROD))
    {
        CUDA_THROW_INVALID("Cumulative operation requires valid tensors and operation");
        return 0;
    }
    dtype_t output_dtype = cuda_cumulative_result_dtype(input->dtype);
    if (output_dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Cumulative scans do not support dtype %s",
                           dtype_to_string(input->dtype));
        return 0;
    }
    if (result->dtype != output_dtype)
    {
        CUDA_THROW_INVALID("Cumulative output dtype does not match the input promotion rule");
        return 0;
    }
    if (axis < -1 || axis >= input->ndims)
    {
        CUDA_THROW_INVALID("Cumulative scan axis is out of range");
        return 0;
    }
    if (stream && (!is_contiguous(input) ||
        (axis != -1 && axis != input->ndims - 1)))
    {
        CUDA_THROW_INVALID("Asynchronous cumulative scans require contiguous input and a flattened or last-axis scan");
        return 0;
    }
    if (axis == -1)
    {
        if (result->ndims != 1 || result->shape[0] != (int)input->total_size)
        {
            CUDA_THROW_INVALID("Global cumulative scan output must be a flattened tensor");
            return 0;
        }
    }
    else
    {
        if (result->ndims != input->ndims)
        {
            CUDA_THROW_INVALID("Axis cumulative scan output rank mismatch");
            return 0;
        }
        for (int dim = 0; dim < input->ndims; dim++)
            if (result->shape[dim] != input->shape[dim])
            {
                CUDA_THROW_INVALID("Axis cumulative scan output shape mismatch");
                return 0;
            }
    }
    if (!input->total_size) return 1;

    if (axis == -1)
    {
        if (input->total_size > INT_MAX)
        {
            CUDA_THROW_INVALID("Flattened cumulative scan exceeds the supported dimension size");
            return 0;
        }
        tensor_t *packed = is_contiguous(input) ? NULL : cuda_tensor_clone(input);
        if (!is_contiguous(input) && !packed) return 0;
        cudaError_t status = launch_cumulative_scan(
            packed ? packed->data : input->data, result->data, input->dtype,
            output_dtype, input->total_size, input->total_size, 1,
            operation_type, stream);
        if (packed) cuda_tensor_destroy(packed);
        if (status != cudaSuccess)
        {
            CUDA_THROW_RUNTIME("Cumulative scan failed: %s", cudaGetErrorString(status));
            return 0;
        }
        return 1;
    }

    if (axis == input->ndims - 1 && is_contiguous(input))
    {
        size_t segment_length = (size_t)input->shape[axis];
        cudaError_t status = launch_cumulative_scan(
            input->data, result->data, input->dtype, output_dtype,
            input->total_size, segment_length,
            input->total_size / segment_length, operation_type, stream);
        if (status != cudaSuccess)
        {
            CUDA_THROW_RUNTIME("Cumulative scan failed: %s",
                               cudaGetErrorString(status));
            return 0;
        }
        return 1;
    }

    int permuted_shape[MAX_DIMS];
    size_t permuted_strides[MAX_DIMS];
    int permutation[MAX_DIMS];
    int destination = 0;
    for (int dim = 0; dim < input->ndims; dim++)
        if (dim != axis)
        {
            permutation[destination] = dim;
            permuted_shape[destination] = input->shape[dim];
            permuted_strides[destination] = input->strides[dim];
            destination++;
        }
    permutation[destination] = axis;
    permuted_shape[destination] = input->shape[axis];
    permuted_strides[destination] = input->strides[axis];

    tensor_t *view = cuda_tensor_create_view(
        input, permuted_shape, permuted_strides, input->ndims, 0,
        input->total_size);
    if (!view) return 0;
    tensor_t *packed = cuda_tensor_clone(view);
    cuda_tensor_destroy(view);
    if (!packed) return 0;
    tensor_t *scanned = cuda_tensor_create_empty_dtype(
        permuted_shape, input->ndims, output_dtype);
    if (!scanned)
    {
        cuda_tensor_destroy(packed);
        return 0;
    }

    size_t segment_length = (size_t)input->shape[axis];
    size_t segment_count = segment_length
        ? input->total_size / segment_length : 0;
    cudaError_t status = launch_cumulative_scan(
        packed->data, scanned->data, input->dtype, output_dtype,
        input->total_size, segment_length, segment_count,
        operation_type, stream);
    if (status == cudaSuccess)
        status = launch_cumulative_reorder(
            scanned->data, result->data, output_dtype, input->shape,
            input->ndims, axis, input->total_size, stream);
    cuda_tensor_destroy(scanned);
    cuda_tensor_destroy(packed);
    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Cumulative scan failed: %s", cudaGetErrorString(status));
        return 0;
    }
    return 1;
}

tensor_t *cuda_tensor_cumulative(tensor_t *input, int axis,
                                 operation_type_t operation_type)
{
    if (!input)
    {
        CUDA_THROW_INVALID("Cumulative operation requires an input tensor");
        return NULL;
    }
    if (operation_type != OP_CUMSUM && operation_type != OP_CUMPROD)
    {
        CUDA_THROW_INVALID("Unsupported cumulative operation");
        return NULL;
    }
    if (fusion_active())
        return fusion_scan(input, axis, operation_type);
    if (is_low_precision_dtype(input->dtype))
    {
        tensor_t *promoted = tensor_cast(input, DTYPE_FLOAT32);
        if (!promoted) return NULL;
        tensor_t *result = cuda_tensor_cumulative(
            promoted, axis, operation_type);
        cuda_tensor_destroy(promoted);
        return result;
    }
    if (!fusion_materialize(input)) return NULL;
    if (axis < -1 || axis >= input->ndims)
    {
        CUDA_THROW_INVALID("Cumulative scan axis is out of range");
        return NULL;
    }
    dtype_t output_dtype = cuda_cumulative_result_dtype(input->dtype);
    if (output_dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Cumulative scans do not support dtype %s",
                           dtype_to_string(input->dtype));
        return NULL;
    }
    int output_shape[MAX_DIMS];
    int output_ndims;
    if (axis == -1)
    {
        if (input->total_size > INT_MAX)
        {
            CUDA_THROW_INVALID("Flattened cumulative scan exceeds the supported dimension size");
            return NULL;
        }
        output_shape[0] = (int)input->total_size;
        output_ndims = 1;
    }
    else
    {
        memcpy(output_shape, input->shape, input->ndims * sizeof(int));
        output_ndims = input->ndims;
    }

    tensor_t *result = cuda_tensor_create_empty_dtype(
        output_shape, output_ndims, output_dtype);
    if (!result) return NULL;
    if (!cuda_tensor_cumulative_into(input, result, axis, operation_type, NULL))
    {
        cuda_tensor_destroy(result);
        return NULL;
    }
    autograd_record_scan(result, input, operation_type, axis);
    return result;
}

int cuda_tensor_cumulative_product_gradient_into(tensor_t *input,
                                                 tensor_t *upstream,
                                                 tensor_t *result, int axis)
{
    if (!input || !upstream || !result ||
        (input->dtype != DTYPE_FLOAT32 && input->dtype != DTYPE_FLOAT64) ||
        upstream->dtype != input->dtype || result->dtype != input->dtype ||
        axis < -1 || axis >= input->ndims || !is_contiguous(result))
    {
        CUDA_THROW_INVALID("Cumulative product gradients require matching float32/float64 tensors and a valid axis");
        return 0;
    }
    if (axis == -1)
    {
        if (input->total_size > INT_MAX ||
            upstream->ndims != 1 ||
            upstream->shape[0] != (int)input->total_size)
        {
            CUDA_THROW_INVALID("Flattened cumulative product gradient shape mismatch");
            return 0;
        }
    }
    else
    {
        if (upstream->ndims != input->ndims ||
            result->ndims != input->ndims)
        {
            CUDA_THROW_INVALID("Axis cumulative product gradient rank mismatch");
            return 0;
        }
        for (int dimension = 0; dimension < input->ndims; dimension++)
            if (upstream->shape[dimension] != input->shape[dimension] ||
                result->shape[dimension] != input->shape[dimension])
            {
                CUDA_THROW_INVALID("Axis cumulative product gradient shape mismatch");
                return 0;
            }
    }
    if (result->total_size != input->total_size)
    {
        CUDA_THROW_INVALID("Cumulative product gradient output size mismatch");
        return 0;
    }
    if (!input->total_size) return 1;

    int prefix_shape[MAX_DIMS];
    int prefix_ndims = axis == -1 ? 1 : input->ndims;
    if (axis == -1) prefix_shape[0] = (int)input->total_size;
    else memcpy(prefix_shape, input->shape, input->ndims * sizeof(int));
    tensor_t *prefix_product = cuda_tensor_create_empty_dtype(
        prefix_shape, prefix_ndims, input->dtype);
    if (!prefix_product) return 0;
    if (!cuda_tensor_cumulative_into(input, prefix_product, axis,
                                     OP_CUMPROD, NULL))
    {
        cuda_tensor_destroy(prefix_product);
        return 0;
    }

    cudaError_t status = launch_cumulative_product_gradient(
        input->data, input->dtype, input->shape, input->strides,
        input->ndims, upstream->data, upstream->strides,
        prefix_product->data, result->data, input->total_size, axis, NULL);
    cuda_tensor_destroy(prefix_product);
    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Cumulative product gradient failed: %s",
                           cudaGetErrorString(status));
        return 0;
    }
    return 1;
}

tensor_t *cuda_tensor_cumulative_product_gradient(tensor_t *input,
                                                  tensor_t *upstream,
                                                  int axis)
{
    if (!input || !upstream)
    {
        CUDA_THROW_INVALID("Cumulative product gradient requires input and upstream tensors");
        return NULL;
    }
    if (fusion_active())
        return fusion_scan_backward(input, upstream, axis);
    if (!fusion_materialize(input) || !fusion_materialize(upstream))
        return NULL;
    tensor_t *result = cuda_tensor_create_empty_dtype(
        input->shape, input->ndims, upstream->dtype);
    if (!result) return NULL;
    if (!cuda_tensor_cumulative_product_gradient_into(
            input, upstream, result, axis))
    {
        cuda_tensor_destroy(result);
        return NULL;
    }
    return result;
}

tensor_t *cuda_tensor_moments_reduce(tensor_t *input, int axis, int correction,
                                     int standard_deviation)
{
    if (!input || input->requires_grad || fusion_active())
    {
        CUDA_THROW_INVALID("Direct moments reduction requires an untracked tensor outside Fusion capture");
        return NULL;
    }
    if (is_low_precision_dtype(input->dtype))
    {
        tensor_t *promoted = tensor_cast(input, DTYPE_FLOAT32);
        if (!promoted) return NULL;
        tensor_t *result = cuda_tensor_moments_reduce(
            promoted, axis, correction, standard_deviation);
        cuda_tensor_destroy(promoted);
        return result;
    }
    if (!fusion_materialize(input)) return NULL;

    int result_shape[MAX_DIMS];
    int result_ndims;
    size_t total_elements_out;
    if (axis == REDUCE_GLOBAL_FLAG)
    {
        result_shape[0] = 1;
        result_ndims = 1;
        total_elements_out = 1;
    }
    else
    {
        result_ndims = calculate_reduction_shape(input, axis, result_shape, &total_elements_out);
        if (result_ndims <= 0) return NULL;
    }

    dtype_t result_dtype = input->dtype == DTYPE_FLOAT32 ? DTYPE_FLOAT32 : DTYPE_FLOAT64;
    tensor_t *result = cuda_tensor_create_empty_dtype(result_shape, result_ndims, result_dtype);
    if (!result) return NULL;

    launch_moments_reduction(
        input->data,
        result->data,
        input->dtype,
        result_dtype,
        input->shape,
        input->ndims,
        input->strides,
        axis == REDUCE_GLOBAL_FLAG ? -1 : axis,
        total_elements_out,
        correction,
        standard_deviation,
        NULL);

    cudaError_t error = cuda_reduction_status();
    if (error != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Variance reduction kernel failed: %s", cudaGetErrorString(error));
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

    if (reshaped) autograd_record_view(reshaped, original, OP_RESHAPE, NULL);
    return reshaped;
}

tensor_t *cuda_tensor_broadcast_to(tensor_t *original, int *new_shape, int new_ndims)
{
    if (!original || !new_shape || new_ndims < original->ndims || new_ndims > MAX_DIMS)
    {
        CUDA_THROW_INVALID("broadcastTo target rank must be at least the input rank and no greater than %d", MAX_DIMS);
        return NULL;
    }

    size_t strides[MAX_DIMS] = {0};
    size_t total_size = 1;
    int input_offset = new_ndims - original->ndims;
    for (int axis = 0; axis < new_ndims; axis++)
    {
        if (new_shape[axis] < 0)
        {
            CUDA_THROW_INVALID("broadcastTo dimensions must be non-negative");
            return NULL;
        }
        if (new_shape[axis] && total_size > SIZE_MAX / (size_t)new_shape[axis])
        {
            CUDA_THROW_INVALID("broadcastTo target shape size overflows");
            return NULL;
        }
        total_size *= (size_t)new_shape[axis];
        int input_axis = axis - input_offset;
        if (input_axis < 0)
            continue;
        int input_size = original->shape[input_axis];
        if (input_size != new_shape[axis] && input_size != 1)
        {
            CUDA_THROW_INVALID("Cannot broadcast dimension %d from %d to %d",
                               input_axis, input_size, new_shape[axis]);
            return NULL;
        }
        strides[axis] = input_size == 1 && new_shape[axis] != 1
            ? 0 : original->strides[input_axis];
    }

    if (fusion_active())
        return fusion_view(original, OP_BROADCAST, new_shape, strides, new_ndims, NULL);
    if (!fusion_materialize(original))
        return NULL;
    tensor_t *view = cuda_tensor_create_view(original, new_shape, strides, new_ndims, 0, total_size);
    if (view)
        autograd_record_view(view, original, OP_BROADCAST, NULL);
    return view;
}

int cuda_tensor_validate_indexed_shapes(tensor_t *input, tensor_t *indices,
                                       tensor_t *updates, int axis, const char *operation)
{
    if (!input || !indices || axis < 0 || axis >= input->ndims ||
        indices->ndims != input->ndims || indices->dtype != DTYPE_INT32)
    {
        CUDA_THROW_INVALID("%s requires int32 indices with the same rank and a valid axis", operation);
        return 0;
    }
    for (int dimension = 0; dimension < input->ndims; dimension++)
    {
        if (dimension != axis && input->shape[dimension] != indices->shape[dimension])
        {
            CUDA_THROW_INVALID("%s input and index shapes must match outside the indexed axis", operation);
            return 0;
        }
        if (updates && updates->shape[dimension] != indices->shape[dimension])
        {
            CUDA_THROW_INVALID("%s updates must have the same shape as indices", operation);
            return 0;
        }
    }
    if (updates && updates->dtype != input->dtype)
    {
        CUDA_THROW_INVALID("%s input and updates must have the same dtype", operation);
        return 0;
    }
    return 1;
}

tensor_t *cuda_tensor_gather(tensor_t *input, tensor_t *indices, int axis)
{
    if (fusion_active())
        return fusion_indexed_gather(input, indices, axis);
    if (!cuda_tensor_validate_indexed_shapes(input, indices, NULL, axis, "gather"))
        return NULL;
    if (!fusion_materialize(input) || !fusion_materialize(indices))
        return NULL;

    cudaError_t status = cuda_validate_indices(indices->data, indices->shape,
        indices->strides, indices->ndims, axis, input->shape[axis], indices->total_size);
    if (status != cudaSuccess)
    {
        if (status == cudaErrorInvalidValue)
            CUDA_THROW_INVALID("gather indices must be within [0, %d)", input->shape[axis]);
        else
            CUDA_THROW_RUNTIME("Failed to validate gather indices: %s", cudaGetErrorString(status));
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty_dtype(indices->shape, indices->ndims, input->dtype);
    if (!result) return NULL;
    status = cuda_launch_gather(input->data, indices->data, result->data, input->dtype,
        input->shape, input->strides, indices->shape, indices->strides,
        input->ndims, axis, indices->total_size, NULL);
    if (status != cudaSuccess)
    {
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("gather kernel failed: %s", cudaGetErrorString(status));
        return NULL;
    }
    autograd_record_gather(result, input, indices, axis);
    return result;
}

tensor_t *cuda_tensor_scatter_add(tensor_t *input, tensor_t *indices,
                                  tensor_t *updates, int axis)
{
    if (fusion_active())
        return fusion_indexed_scatter_add(input, indices, updates, axis);
    if (!cuda_tensor_validate_indexed_shapes(input, indices, updates, axis, "scatterAdd"))
        return NULL;
    if (is_low_precision_dtype(input->dtype))
    {
        tensor_t *promoted_input = tensor_cast(input, DTYPE_FLOAT32);
        tensor_t *promoted_updates = promoted_input
            ? tensor_cast(updates, DTYPE_FLOAT32) : NULL;
        tensor_t *result = promoted_updates
            ? cuda_tensor_scatter_add(promoted_input, indices, promoted_updates, axis)
            : NULL;
        if (promoted_input) cuda_tensor_destroy(promoted_input);
        if (promoted_updates) cuda_tensor_destroy(promoted_updates);
        return result;
    }
    if (input->dtype != DTYPE_FLOAT32 && input->dtype != DTYPE_FLOAT64)
    {
        CUDA_THROW_INVALID("scatterAdd currently supports float32 and float64 tensors");
        return NULL;
    }
    if (!fusion_materialize(input) || !fusion_materialize(indices) ||
        !fusion_materialize(updates))
        return NULL;

    cudaError_t status = cuda_validate_indices(indices->data, indices->shape,
        indices->strides, indices->ndims, axis, input->shape[axis], indices->total_size);
    if (status != cudaSuccess)
    {
        if (status == cudaErrorInvalidValue)
            CUDA_THROW_INVALID("scatterAdd indices must be within [0, %d)", input->shape[axis]);
        else
            CUDA_THROW_RUNTIME("Failed to validate scatterAdd indices: %s", cudaGetErrorString(status));
        return NULL;
    }

    tensor_t *result = cuda_tensor_create_empty_dtype(input->shape, input->ndims, input->dtype);
    if (!result) return NULL;
    status = cuda_launch_scatter_copy(input->data, result->data, input->dtype,
        input->shape, input->strides, input->ndims, input->total_size);
    if (status == cudaSuccess)
        status = cuda_launch_scatter_add(indices->data, updates->data, result->data,
            input->dtype, indices->shape, result->strides, indices->strides,
            updates->strides, input->ndims, axis, input->shape[axis],
            updates->total_size, NULL);
    if (status != cudaSuccess)
    {
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("scatterAdd kernel failed: %s", cudaGetErrorString(status));
        return NULL;
    }
    autograd_record_scatter_add(result, input, indices, updates, axis);
    return result;
}

tensor_t *cuda_tensor_slice_scatter(tensor_t *updates, const int *input_shape,
                                    int input_ndims, const int *axes,
                                    const int *starts, const int *steps)
{
    if (!updates || !input_shape || !starts || !steps ||
        input_ndims < 0 || input_ndims > MAX_DIMS ||
        updates->ndims < 0 || updates->ndims > MAX_DIMS ||
        (updates->ndims > 0 && (!axes || !updates->shape || !updates->strides)))
    {
        CUDA_THROW_INVALID("Invalid slice gradient metadata");
        return NULL;
    }
    if (updates->dtype != DTYPE_FLOAT32 && updates->dtype != DTYPE_FLOAT64)
    {
        CUDA_THROW_INVALID("Slice gradients currently support float32 and float64 tensors");
        return NULL;
    }
    if (fusion_active())
        return fusion_slice_scatter(updates, input_shape, input_ndims,
                                    axes, starts, steps);
    if (!fusion_materialize(updates))
        return NULL;

    scalar_value_t zero = {0};
    zero.dtype = updates->dtype;
    tensor_t *result = cuda_tensor_create_with_value((int *)input_shape, input_ndims,
                                                      zero, updates->dtype);
    if (!result)
        return NULL;

    cudaError_t status = cuda_launch_slice_scatter(
        updates->data, result->data, updates->dtype, input_shape, input_ndims,
        updates->strides, updates->shape, axes, starts, steps,
        updates->ndims, updates->total_size);
    if (status != cudaSuccess)
    {
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("Slice gradient kernel failed: %s", cudaGetErrorString(status));
        return NULL;
    }
    return result;
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

    if (transposed) autograd_record_view(transposed, tensor, OP_TRANSPOSE, axis);
    return transposed ? transposed : NULL;
}

tensor_t *cuda_tensor_matmul_nd(tensor_t *a, tensor_t *b)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    if (!b || a->dtype != b->dtype ||
        (a->dtype != DTYPE_FLOAT32 && a->dtype != DTYPE_FLOAT16 &&
         a->dtype != DTYPE_BFLOAT16))
    {
        CUDA_THROW_INVALID("matmul requires matching float32, float16 or bfloat16 operands");
        return NULL;
    }

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

    tensor_t *result = cuda_tensor_create_empty_dtype(result_shape, result_ndims, a->dtype);
    if (result == NULL)
    {
        return NULL;
    }

    int status = a->dtype == DTYPE_FLOAT32
        ? cuda_batched_matmul_nd_launcher(
            a->data, b->data, result->data,
            a->shape, a->strides, a->ndims,
            b->shape, b->strides, b->ndims,
            result->shape, result->strides, result->ndims, NULL)
        : cuda_low_precision_matmul_nd_launcher(
            a->data, b->data, result->data, a->dtype,
            a->shape, a->strides, a->ndims,
            b->shape, b->strides, b->ndims,
            result->shape, result->strides, result->ndims, NULL);

    if (status == 0)
    {
        cuda_backend_info info = {0};
        cuda_blas_info(&info);
        CUDA_THROW_RUNTIME("Matmul backend submission failed (cuBLAS status %d)", info.last_blas_status);
        cuda_tensor_destroy(result);
        return NULL;
    }

    return result;
}

tensor_t *cuda_tensor_matmul(tensor_t *a, tensor_t *b)
{
    CUDA_CHECK_AND_RETURN_NULL(a);
    if (fusion_active()) return fusion_matmul(a, b);
    if (!b || a->dtype != b->dtype ||
        (a->dtype != DTYPE_FLOAT32 && a->dtype != DTYPE_FLOAT16 &&
         a->dtype != DTYPE_BFLOAT16))
    {
        CUDA_THROW_INVALID("matmul requires matching float32, float16 or bfloat16 operands");
        return NULL;
    }
    if (!fusion_materialize(a) || !fusion_materialize(b)) return NULL;
    if (a->ndims < 2 || b->ndims < 2)
    {
        return NULL;
    }

    if (a->ndims != 2 || b->ndims != 2)
    {
        tensor_t *result = cuda_tensor_matmul_nd(a, b);
        if (result) autograd_record_matmul(result, a, b);
        return result;
    }

    if (a->dtype != DTYPE_FLOAT32)
    {
        tensor_t *result = cuda_tensor_matmul_nd(a, b);
        if (result) autograd_record_matmul(result, a, b);
        return result;
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
        cuda_backend_info info = {0};
        cuda_blas_info(&info);
        CUDA_THROW_RUNTIME("Matmul backend submission failed (cuBLAS status %d)", info.last_blas_status);
        cuda_tensor_destroy(result);
        return NULL;
    }

    autograd_record_matmul(result, a, b);
    return result;
}

tensor_t *cuda_tensor_copy(tensor_t *tensor)
{
    if (!tensor)
        return NULL;

    tensor_t *copy = cuda_tensor_create_empty_dtype(tensor->shape, tensor->ndims,
                                                     tensor->dtype);
    if (!copy)
        return NULL;

    cudaError_t cuda_status = is_contiguous(tensor)
        ? cudaMemcpy(copy->data, tensor->data,
                     tensor->total_size * tensor->element_size,
                     cudaMemcpyDeviceToDevice)
        : launch_pack_strided(tensor->data, copy->data,
                              tensor->total_size, tensor->element_size,
                              tensor->shape, tensor->strides, tensor->ndims);

    if (cuda_status != cudaSuccess)
    {
        cuda_tensor_destroy(copy);
        CUDA_THROW_RUNTIME("Tensor copy failed: %s", cudaGetErrorString(cuda_status));
        return NULL;
    }

    return copy;
}

static int tensor_transform_element_count(const int *shape, int ndims,
                                          size_t *elements)
{
    if (!shape || !elements || ndims < 1 || ndims > MAX_DIMS)
        return 0;
    size_t count = 1;
    for (int axis = 0; axis < ndims; axis++)
    {
        if (shape[axis] < 0 ||
            (shape[axis] != 0 && count > SIZE_MAX / (size_t)shape[axis]))
        {
            CUDA_THROW_INVALID("Tensor transform output shape is too large");
            return 0;
        }
        count *= (size_t)shape[axis];
    }
    *elements = count;
    return 1;
}

tensor_t *cuda_tensor_index_transform(tensor_t *input,
                                      tensor_transform_spec_t *spec)
{
    if (!input || !spec)
    {
        CUDA_THROW_INVALID("Tensor transform requires an input and operation metadata");
        return NULL;
    }
    if (spec->output_ndims < 1 || spec->output_ndims > MAX_DIMS)
    {
        CUDA_THROW_INVALID("Tensor transform output rank is unsupported");
        return NULL;
    }
    if (fusion_active())
    {
        spec->input_ndims = input->ndims;
        memcpy(spec->input_shape, input->shape, input->ndims * sizeof(int));
        memcpy(spec->input_strides, input->strides,
               input->ndims * sizeof(size_t));
        return fusion_transform(input, spec);
    }
    if (!fusion_materialize(input))
        return NULL;
    spec->input_ndims = input->ndims;
    memcpy(spec->input_shape, input->shape, input->ndims * sizeof(int));
    memcpy(spec->input_strides, input->strides,
           input->ndims * sizeof(size_t));

    size_t total_elements;
    if (!tensor_transform_element_count(spec->output_shape,
                                        spec->output_ndims, &total_elements))
        return NULL;

    tensor_t *result = cuda_tensor_create_empty_dtype(
        spec->output_shape, spec->output_ndims, input->dtype);
    if (!result) return NULL;
    cudaError_t status = launch_tensor_transform(
        input->data, result->data, input->dtype, spec, total_elements);
    if (status != cudaSuccess)
    {
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("Tensor transform kernel failed: %s",
                           cudaGetErrorString(status));
        return NULL;
    }
    autograd_record_index_transform(result, input, spec);
    return result;
}

tensor_t *cuda_tensor_index_transform_backward(
    tensor_t *upstream, const tensor_transform_spec_t *spec)
{
    if (!upstream || !spec ||
        (upstream->dtype != DTYPE_FLOAT32 && upstream->dtype != DTYPE_FLOAT64))
    {
        CUDA_THROW_INVALID("Tensor transform gradients require float32 or float64 data");
        return NULL;
    }
    if (fusion_active())
        return fusion_transform_backward(upstream, spec);
    if (!fusion_materialize(upstream))
        return NULL;
    if (upstream->ndims != spec->output_ndims)
    {
        CUDA_THROW_INVALID("Tensor transform gradient rank does not match its output");
        return NULL;
    }
    for (int axis = 0; axis < upstream->ndims; axis++)
    {
        if (upstream->shape[axis] != spec->output_shape[axis])
        {
            CUDA_THROW_INVALID("Tensor transform gradient shape does not match its output");
            return NULL;
        }
    }

    scalar_value_t zero = {0};
    zero.dtype = upstream->dtype;
    tensor_t *result = cuda_tensor_create_with_value(
        (int *)spec->input_shape, spec->input_ndims, zero, upstream->dtype);
    if (!result) return NULL;
    tensor_transform_spec_t gradient_spec = *spec;
    memcpy(gradient_spec.upstream_strides, upstream->strides,
           upstream->ndims * sizeof(size_t));
    cudaError_t status = launch_tensor_transform_gradient(
        upstream->data, result->data, upstream->dtype, &gradient_spec,
        upstream->total_size, NULL);
    if (status != cudaSuccess)
    {
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("Tensor transform gradient kernel failed: %s",
                           cudaGetErrorString(status));
        return NULL;
    }
    return result;
}

int cuda_tensor_repeat_counts_into(tensor_t *input, tensor_t *result,
                                   const int *prefix_counts, size_t count,
                                   int axis, cudaStream_t stream)
{
    if (!input || !result || axis < -1 || axis >= input->ndims)
    {
        CUDA_THROW_INVALID("Repeat requires valid input/output tensors and axis");
        return 0;
    }
    if (stream)
    {
        CUDA_THROW_INVALID("Per-element repeat currently requires synchronous execution");
        return 0;
    }
    size_t expected_count = axis < 0
        ? input->total_size : (size_t)input->shape[axis];
    if (count != expected_count || (count && !prefix_counts))
    {
        CUDA_THROW_INVALID("Repeat counts must match the selected input axis");
        return 0;
    }
    int output_extent = count ? prefix_counts[count - 1] : 0;
    if (output_extent < 0)
    {
        CUDA_THROW_INVALID("Repeat output extent cannot be negative");
        return 0;
    }
    if (result->dtype != input->dtype)
    {
        CUDA_THROW_INVALID("Per-element repeat cannot change the tensor dtype");
        return 0;
    }
    if (axis < 0)
    {
        if (result->ndims != 1 || result->shape[0] != output_extent)
        {
            CUDA_THROW_INVALID("Flattened repeat output shape mismatch");
            return 0;
        }
    }
    else
    {
        if (result->ndims != input->ndims)
        {
            CUDA_THROW_INVALID("Axis repeat output rank mismatch");
            return 0;
        }
        for (int dim = 0; dim < input->ndims; dim++)
        {
            int expected = dim == axis ? output_extent : input->shape[dim];
            if (result->shape[dim] != expected)
            {
                CUDA_THROW_INVALID("Axis repeat output shape mismatch");
                return 0;
            }
        }
    }
    if (!result->total_size) return 1;

    cudaError_t status = launch_tensor_repeat_counts(
        input->data, result->data, input->element_size, prefix_counts, count,
        axis, input->shape, input->strides, input->ndims, result->shape,
        result->ndims, result->total_size);
    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Per-element repeat failed: %s",
                           cudaGetErrorString(status));
        return 0;
    }
    return 1;
}

tensor_t *cuda_tensor_repeat_counts(tensor_t *input,
                                   const int *prefix_counts, size_t count,
                                   int axis, const int *output_shape,
                                   int output_ndims)
{
    if (!input || !output_shape || output_ndims < 1 ||
        output_ndims > MAX_DIMS || axis < -1 || axis >= input->ndims)
    {
        CUDA_THROW_INVALID("Per-element repeat requires valid input and shape metadata");
        return NULL;
    }
    if (fusion_active())
        return fusion_repeat_counts(input, prefix_counts, count, axis,
                                    output_shape, output_ndims);
    if (!fusion_materialize(input)) return NULL;

    tensor_t *result = cuda_tensor_create_empty_dtype(
        (int *)output_shape, output_ndims, input->dtype);
    if (!result) return NULL;
    if (!cuda_tensor_repeat_counts_into(input, result, prefix_counts, count,
                                        axis, NULL))
    {
        cuda_tensor_destroy(result);
        return NULL;
    }
    autograd_record_repeat_counts(result, input, prefix_counts, count, axis);
    return result;
}

int cuda_tensor_repeat_counts_backward_into(tensor_t *upstream,
                                             tensor_t *result,
                                             const int *prefix_counts,
                                             size_t count, int axis)
{
    if (!upstream || !result || axis < -1 || axis >= result->ndims)
    {
        CUDA_THROW_INVALID("Repeat-count gradient requires valid tensors and axis");
        return 0;
    }
    size_t expected_count = axis < 0
        ? result->total_size : (size_t)result->shape[axis];
    if (count != expected_count || (count && !prefix_counts))
    {
        CUDA_THROW_INVALID("Repeat-count gradient metadata does not match the input");
        return 0;
    }
    if (upstream->dtype != result->dtype ||
        (upstream->dtype != DTYPE_FLOAT32 && upstream->dtype != DTYPE_FLOAT64))
    {
        CUDA_THROW_INVALID("Per-element repeat gradients require float32 or float64 data");
        return 0;
    }
    int output_extent = count ? prefix_counts[count - 1] : 0;
    if (axis < 0)
    {
        if (upstream->ndims != 1 || upstream->shape[0] != output_extent)
        {
            CUDA_THROW_INVALID("Flattened repeat gradient shape mismatch");
            return 0;
        }
    }
    else
    {
        if (upstream->ndims != result->ndims)
        {
            CUDA_THROW_INVALID("Axis repeat gradient rank mismatch");
            return 0;
        }
        for (int dim = 0; dim < result->ndims; dim++)
        {
            int expected = dim == axis ? output_extent : result->shape[dim];
            if (upstream->shape[dim] != expected)
            {
                CUDA_THROW_INVALID("Axis repeat gradient shape mismatch");
                return 0;
            }
        }
    }
    if (!result->total_size) return 1;
    if (!upstream->total_size)
    {
        cudaError_t status = cudaMemset(result->data, 0,
                                        result->total_size * result->element_size);
        if (status != cudaSuccess)
        {
            CUDA_THROW_RUNTIME("Cannot zero per-element repeat gradient: %s",
                               cudaGetErrorString(status));
            return 0;
        }
        return 1;
    }
    cudaError_t status = launch_tensor_repeat_counts_gradient(
        upstream->data, result->data, result->dtype, result->shape,
        result->ndims, upstream->strides, prefix_counts, count, axis,
        result->total_size);
    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Per-element repeat gradient failed: %s",
                           cudaGetErrorString(status));
        return 0;
    }
    return 1;
}

tensor_t *cuda_tensor_repeat_counts_backward(tensor_t *upstream,
                                              tensor_t *input,
                                              const int *prefix_counts,
                                              size_t count, int axis)
{
    if (!upstream || !input)
    {
        CUDA_THROW_INVALID("Repeat-count gradient requires input and upstream tensors");
        return NULL;
    }
    if (fusion_active())
        return fusion_repeat_counts_backward(
            upstream, prefix_counts, count, axis, input->shape, input->ndims);
    if (!fusion_materialize(upstream)) return NULL;
    tensor_t *result = cuda_tensor_create_empty_dtype(
        input->shape, input->ndims, upstream->dtype);
    if (!result) return NULL;
    if (!cuda_tensor_repeat_counts_backward_into(
            upstream, result, prefix_counts, count, axis))
    {
        cuda_tensor_destroy(result);
        return NULL;
    }
    return result;
}
