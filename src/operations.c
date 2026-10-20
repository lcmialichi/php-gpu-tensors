#include "cuda_exceptions.h"
#include "operations.h"
#include "php.h"
#include "cuda.h"
#include "tensor_shape.h"

static int calculate_broadcast_shape(int *a_shape, int a_dims, int *b_shape, int b_dims, int *result_shape, int *result_dims);
static void calculate_tensor_strides(tensor_t *tensor, size_t *tensor_strides);

int prepare_broadcast_operation(tensor_t *a, tensor_t *b,
                                int *result_shape, int *result_dims,
                                size_t *a_strides, size_t *b_strides,
                                size_t *total_elements)
{
    if (!calculate_broadcast_shape(a->shape, a->ndims, b->shape, b->ndims,
                                   result_shape, result_dims))
    {
        return 0;
    }

    size_t bytes;
    if (!tensor_shape_metrics(result_shape, *result_dims, 1, NULL, total_elements, &bytes))
    {
        CUDA_THROW_INVALID("Broadcast shape or strides exceed supported limits");
        return 0;
    }

    calculate_tensor_strides(a, a_strides);
    calculate_tensor_strides(b, b_strides);
    return 1;
}

int prepare_matmul_result_shape(int a_ndims, int *a_shape, int b_ndims, int *b_shape, int *result_ndims, int *result_shape)
{
    if (a_ndims < 2 || a_ndims > MAX_DIMS || b_ndims < 2 || b_ndims > MAX_DIMS)
        return 0;
    int max_ndims = (a_ndims > b_ndims) ? a_ndims : b_ndims;
    int a_inner = a_shape[a_ndims - 1];
    int b_inner = b_shape[b_ndims - 2];

    if (a_inner != b_inner)
    {
        return 0;
    }

    for (int i = 0; i < max_ndims - 2; i++)
    {
        int a_idx = a_ndims - max_ndims + i;
        int b_idx = b_ndims - max_ndims + i;

        int a_dim = (a_idx < 0) ? 1 : a_shape[a_idx];
        int b_dim = (b_idx < 0) ? 1 : b_shape[b_idx];

        if (a_dim == b_dim)
        {
            result_shape[i] = a_dim;
        }
        else if (a_dim == 1)
        {
            result_shape[i] = b_dim;
        }
        else if (b_dim == 1)
        {
            result_shape[i] = a_dim;
        }
        else
        {
            return 0;
        }
    }

    result_shape[max_ndims - 2] = a_shape[a_ndims - 2];
    result_shape[max_ndims - 1] = b_shape[b_ndims - 1];
    *result_ndims = max_ndims;

    return 1;
}

int calculate_reduction_shape(tensor_t *input, int axis, int *result_shape, size_t *total_elements_out_ptr)
{
    if (axis < 0 || axis >= input->ndims)
    {
        CUDA_THROW_INVALID("Invalid axis %d for reduction operation. Must be between 0 and %d.", axis, input->ndims - 1);
        return 0;
    }

    int j = 0;
    for (int i = 0; i < input->ndims; i++)
    {
        if (i != axis)
        {
            result_shape[j++] = input->shape[i];
        }
    }

    if (j == 0)
    {
        j = 1;
        result_shape[0] = 1;
    }

    size_t bytes;
    if (!tensor_shape_metrics(result_shape, j, 1, NULL, total_elements_out_ptr, &bytes))
    {
        CUDA_THROW_INVALID("Reduction output shape or strides exceed supported limits");
        return 0;
    }
    return j;
}

static int calculate_broadcast_shape(int *a_shape, int a_dims, int *b_shape, int b_dims, int *result_shape, int *result_dims)
{
    *result_dims = (a_dims > b_dims) ? a_dims : b_dims;

    int offset_a = *result_dims - a_dims;
    int offset_b = *result_dims - b_dims;

    for (int i = 0; i < *result_dims; i++)
    {
        int a_dim = (i < offset_a) ? 1 : a_shape[i - offset_a];
        int b_dim = (i < offset_b) ? 1 : b_shape[i - offset_b];

        if (a_dim == b_dim)
        {
            result_shape[i] = a_dim;
        }
        else if (a_dim == 1)
        {
            result_shape[i] = b_dim;
        }
        else if (b_dim == 1)
        {
            result_shape[i] = a_dim;
        }
        else
        {
            return 0;
        }
    }

    return 1;
}

static void calculate_tensor_strides(tensor_t *tensor, size_t *tensor_strides)
{
    for (int i = tensor->ndims - 1; i >= 0; i--)
    {
        if (tensor->shape[i] == 1)
        {
            tensor_strides[i] = 0;
        }
        else
        {
            tensor_strides[i] = tensor->strides[i];
        }
    }
}
