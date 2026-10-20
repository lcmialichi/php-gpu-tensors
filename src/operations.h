#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "tensor.h"
#include <stdbool.h>
#include "operations_strctures.h"

int prepare_broadcast_operation(tensor_t *a, tensor_t *b,
                                int *result_shape, int *result_dims,
                                size_t *a_strides, size_t *b_strides,
                                size_t *total_elements);

int calculate_reduction_shape(tensor_t *input, int axis, int *result_shape, size_t *total_elements_out_ptr);
int prepare_matmul_result_shape(int a_ndims, int *a_shape, int b_ndims, int *b_shape, int *result_ndims, int *result_shape);
#endif