#ifndef CUDA_TENSOR_TRANSFORM_H
#define CUDA_TENSOR_TRANSFORM_H

#include "tensor.h"
#include "operations_strctures.h"

typedef struct
{
    operation_type_t op;
    int input_ndims;
    int output_ndims;
    int input_shape[MAX_DIMS];
    size_t input_strides[MAX_DIMS];
    size_t upstream_strides[MAX_DIMS];
    int output_shape[MAX_DIMS];
    int axis;
    int axis1;
    int axis2;
    int offset;
    int repeats;
    int axes[MAX_DIMS];
    int64_t shifts[MAX_DIMS];
    int factors[MAX_DIMS];
} tensor_transform_spec_t;

#endif
