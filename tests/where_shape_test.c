#include "php.h"
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int errors;
static int fast_path_seen;
static int result_shape[2];
static size_t mask_strides[2], x_strides[2], y_strides[2];

#define CUDA_EXCEPTIONS_H
#define CUDA_THROW_INVALID(...) do { errors++; } while (0)
#define CUDA_THROW_RUNTIME(...) do { errors++; } while (0)
#include "../src/cuda_array/tensor_where.c"

int autograd_record_where(tensor_t *result, tensor_t *condition, tensor_t *x, tensor_t *y)
{
    (void)result; (void)condition; (void)x; (void)y;
    return 1;
}

tensor_t *cuda_tensor_create_empty_with_dtype(int *shape, int ndims, dtype_t dtype)
{
    tensor_t *tensor = calloc(1, sizeof(tensor_t));
    tensor->shape = malloc((size_t)ndims * sizeof(int));
    memcpy(tensor->shape, shape, (size_t)ndims * sizeof(int));
    tensor->ndims = ndims;
    tensor->dtype = dtype;
    tensor->total_size = 1;
    for (int axis = 0; axis < ndims; axis++) tensor->total_size *= shape[axis];
    tensor->data = calloc(tensor->total_size, sizeof(int));
    return tensor;
}

void cuda_tensor_destroy(tensor_t *tensor)
{
    free(tensor->data);
    free(tensor->shape);
    free(tensor);
}

int is_contiguous(tensor_t *tensor)
{
    size_t expected = 1;
    for (int axis = tensor->ndims - 1; axis >= 0; axis--)
    {
        if (tensor->strides[axis] != expected) return 0;
        expected *= tensor->shape[axis];
    }
    return 1;
}

cudaError_t launch_where_kernel(const tensor_t *condition, const tensor_t *on_true,
                                const tensor_t *on_false, tensor_t *output,
                                const size_t *condition_strides, const size_t *true_strides,
                                const size_t *false_strides, int fast_path)
{
    (void)condition; (void)on_true; (void)on_false; (void)output;
    fast_path_seen = fast_path;
    for (int axis = 0; axis < output->ndims; axis++)
    {
        result_shape[axis] = output->shape[axis];
        mask_strides[axis] = condition_strides[axis];
        x_strides[axis] = true_strides[axis];
        y_strides[axis] = false_strides[axis];
    }
    return cudaSuccess;
}

int main(void)
{
    int mask_shape[] = {2, 1}, x_shape[] = {1, 3}, y_shape[] = {2, 3};
    size_t mask_steps[] = {1, 1}, x_steps[] = {3, 1}, y_steps[] = {3, 1};
    bool mask_data[] = {true, false};
    int32_t x_data[] = {10, 20, 30}, y_data[] = {1, 2, 3, 4, 5, 6};
    tensor_t mask = {.data = mask_data, .shape = mask_shape, .strides = mask_steps, .ndims = 2, .dtype = DTYPE_BOOL};
    tensor_t x = {.data = x_data, .shape = x_shape, .strides = x_steps, .ndims = 2, .dtype = DTYPE_INT32};
    tensor_t y = {.data = y_data, .shape = y_shape, .strides = y_steps, .ndims = 2, .dtype = DTYPE_INT32};

    tensor_t *result = cuda_tensor_where(&mask, &x, &y);
    assert(result && result_shape[0] == 2 && result_shape[1] == 3);
    assert(!fast_path_seen && mask_strides[0] == 1 && mask_strides[1] == 0);
    assert(x_strides[0] == 0 && x_strides[1] == 1);
    assert(y_strides[0] == 3 && y_strides[1] == 1);
    cuda_tensor_destroy(result);

    int view_shape[] = {2, 2};
    size_t view_steps[] = {1, 2}, contiguous_steps[] = {2, 1};
    mask.shape = x.shape = y.shape = view_shape;
    mask.strides = view_steps;
    x.strides = y.strides = contiguous_steps;
    result = cuda_tensor_where(&mask, &x, &y);
    assert(result && !fast_path_seen && mask_strides[0] == 1 && mask_strides[1] == 2);
    cuda_tensor_destroy(result);

    mask.strides = contiguous_steps;
    result = cuda_tensor_where(&mask, &x, &y);
    assert(result && fast_path_seen);
    cuda_tensor_destroy(result);

    int invalid_shape[] = {3, 3};
    y.shape = invalid_shape;
    assert(cuda_tensor_where(&mask, &x, &y) == NULL && errors == 1);
    puts("where shape checks passed");
    return 0;
}