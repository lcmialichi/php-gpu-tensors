#include "cuda_exceptions.h"
#include "operations.h"
#include "php.h"
#include "cuda.h"
#include "tensor_shape.h"

static void calculate_tensor_strides(tensor_t *tensor, size_t *tensor_strides);

int prepare_broadcast_operation(tensor_t *a, tensor_t *b,
                                int *result_shape, int *result_dims,
                                size_t *a_strides, size_t *b_strides,
                                size_t *total_elements)
{
    if (!tensor_broadcast_shape(a->shape, a->ndims, b->shape, b->ndims,
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
    return tensor_matmul_shape(a_shape, a_ndims, b_shape, b_ndims, result_shape, result_ndims);
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
