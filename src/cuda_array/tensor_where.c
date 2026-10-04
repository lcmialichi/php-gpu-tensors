#include "tensor_where.h"
#include "where_kernels.h"
#include "cuda_exceptions.h"
#include <stdint.h>

static int where_project_strides(const tensor_t *tensor, const int *shape, int ndims,
                                 size_t strides[MAX_DIMS])
{
    for (int axis = 0; axis < ndims; axis++)
    {
        int source_axis = axis - (ndims - tensor->ndims);
        int source_dim = source_axis < 0 ? 1 : tensor->shape[source_axis];
        if (source_dim != 1 && source_dim != shape[axis])
            return 0;
        strides[axis] = source_axis < 0 || source_dim == 1 ? 0 : tensor->strides[source_axis];
    }
    return 1;
}

tensor_t *cuda_tensor_where(tensor_t *condition, tensor_t *on_true, tensor_t *on_false)
{
    if (!condition || !on_true || !on_false ||
        (condition->total_size && !condition->data) ||
        (on_true->total_size && !on_true->data) || (on_false->total_size && !on_false->data))
    {
        CUDA_THROW_INVALID("where expects initialized CudaArray operands");
        return NULL;
    }
    if (on_true->dtype != on_false->dtype)
    {
        CUDA_THROW_INVALID("where requires x and y to have the same dtype");
        return NULL;
    }
    if (condition->dtype >= DTYPE_COUNT || on_true->dtype >= DTYPE_COUNT ||
        condition->ndims < 1 || on_true->ndims < 1 || on_false->ndims < 1)
    {
        CUDA_THROW_INVALID("where received unsupported tensor dtype or dimensions");
        return NULL;
    }

    int ndims = condition->ndims;
    if (on_true->ndims > ndims) ndims = on_true->ndims;
    if (on_false->ndims > ndims) ndims = on_false->ndims;
    if (ndims > MAX_DIMS)
    {
        CUDA_THROW_INVALID("where supports at most %d dimensions", MAX_DIMS);
        return NULL;
    }

    const tensor_t *inputs[] = {condition, on_true, on_false};
    int shape[MAX_DIMS];
    size_t total = 1;
    int fast_path = 1;
    for (int axis = 0; axis < ndims; axis++)
    {
        int dimension = 1;
        for (int input = 0; input < 3; input++)
        {
            int source_axis = axis - (ndims - inputs[input]->ndims);
            int size = source_axis < 0 ? 1 : inputs[input]->shape[source_axis];
            if (size < 0 || (size != 1 && dimension != 1 && size != dimension))
            {
                CUDA_THROW_INVALID("where operands have incompatible shapes");
                return NULL;
            }
            if (dimension == 1) dimension = size;
        }
        if (dimension && total > SIZE_MAX / (size_t)dimension)
        {
            CUDA_THROW_INVALID("where result size exceeds supported limits");
            return NULL;
        }
        shape[axis] = dimension;
        total *= (size_t)dimension;
        for (int input = 0; input < 3; input++)
        {
            int source_axis = axis - (ndims - inputs[input]->ndims);
            if (source_axis < 0 || inputs[input]->shape[source_axis] != dimension)
                fast_path = 0;
        }
    }

    size_t condition_strides[MAX_DIMS], true_strides[MAX_DIMS], false_strides[MAX_DIMS];
    if (!where_project_strides(condition, shape, ndims, condition_strides) ||
        !where_project_strides(on_true, shape, ndims, true_strides) ||
        !where_project_strides(on_false, shape, ndims, false_strides))
    {
        CUDA_THROW_INVALID("where operands have incompatible shapes");
        return NULL;
    }

    fast_path = fast_path && is_contiguous(condition) && is_contiguous(on_true) && is_contiguous(on_false);
    tensor_t *result = cuda_tensor_create_empty_with_dtype(shape, ndims, on_true->dtype);
    if (!result) return NULL;
    if (!total) return result;

    cudaError_t status = launch_where_kernel(condition, on_true, on_false, result,
                                              condition_strides, true_strides, false_strides, fast_path);
    if (status != cudaSuccess)
    {
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("where kernel failed: %s", cudaGetErrorString(status));
        return NULL;
    }
    return result;
}