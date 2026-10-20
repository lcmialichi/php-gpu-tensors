#include "concat_kernels.h"
#include "launch_config.cuh"
#include <string.h>

struct ConcatParams
{
    void *input_ptrs[MAX_CONCAT_TENSORS];
    int input_axis_sizes[MAX_CONCAT_TENSORS];
    size_t input_strides[MAX_CONCAT_TENSORS][MAX_DIMS];
    int output_shape[MAX_DIMS];
    int ndims;
    int axis;
    int num_tensors;
    size_t total_elements;
    size_t element_size;
};

static __global__ void concat_kernel(ConcatParams params, unsigned char *output)
{
    size_t idx = cuda_linear_index(blockIdx.x, blockDim.x, threadIdx.x);
    if (idx >= params.total_elements)
        return;

    int coordinates[MAX_DIMS];
    size_t remainder = idx;
    for (int dimension = params.ndims - 1; dimension >= 0; dimension--)
    {
        coordinates[dimension] = remainder % params.output_shape[dimension];
        remainder /= params.output_shape[dimension];
    }

    int tensor_idx = 0;
    size_t axis_offset = 0;
    for (; tensor_idx < params.num_tensors; tensor_idx++)
    {
        size_t axis_size = params.input_axis_sizes[tensor_idx];
        if ((size_t)coordinates[params.axis] < axis_offset + axis_size)
            break;
        axis_offset += axis_size;
    }

    if (tensor_idx >= params.num_tensors)
        return;

    size_t input_offset = 0;
    for (int dimension = 0; dimension < params.ndims; dimension++)
    {
        size_t coordinate = dimension == params.axis
            ? (size_t)coordinates[dimension] - axis_offset
            : (size_t)coordinates[dimension];
        input_offset += coordinate * params.input_strides[tensor_idx][dimension];
    }

    const unsigned char *input = (const unsigned char *)params.input_ptrs[tensor_idx];
    size_t output_byte_offset = idx * params.element_size;
    size_t input_byte_offset = input_offset * params.element_size;
    for (size_t byte = 0; byte < params.element_size; byte++)
        output[output_byte_offset + byte] = input[input_byte_offset + byte];
}

extern "C" int launch_concat_kernel_host(
    tensor_t **input_tensors, int num_tensors, tensor_t *output_tensor, int axis)
{
    if (!input_tensors || !output_tensor || num_tensors < 1 ||
        num_tensors > MAX_CONCAT_TENSORS || axis < 0 ||
        axis >= output_tensor->ndims || !output_tensor->element_size)
        return 1;

    if (output_tensor->total_size == 0)
        return 0;

    ConcatParams params = {};
    params.num_tensors = num_tensors;
    params.ndims = output_tensor->ndims;
    params.axis = axis;
    params.total_elements = output_tensor->total_size;
    params.element_size = output_tensor->element_size;
    for (int dimension = 0; dimension < params.ndims; dimension++)
        params.output_shape[dimension] = output_tensor->shape[dimension];

    for (int i = 0; i < num_tensors; i++)
    {
        if (!input_tensors[i] || (input_tensors[i]->total_size && !input_tensors[i]->data) ||
            input_tensors[i]->ndims != params.ndims ||
            input_tensors[i]->dtype != output_tensor->dtype)
            return 1;
        params.input_ptrs[i] = input_tensors[i]->data;
        params.input_axis_sizes[i] = input_tensors[i]->shape[axis];
        for (int dimension = 0; dimension < params.ndims; dimension++)
            if (dimension != axis &&
                input_tensors[i]->shape[dimension] != params.output_shape[dimension])
                return 1;
        memcpy(params.input_strides[i], input_tensors[i]->strides,
               params.ndims * sizeof(size_t));
    }

    concat_kernel<<<cuda_grid_1d(params.total_elements), 256>>>(
        params, (unsigned char *)output_tensor->data);

    if (cuda_launch_status(true) != cudaSuccess)
        return 1;
    return 0;
}