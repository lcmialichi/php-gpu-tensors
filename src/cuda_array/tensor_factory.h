#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H

#include "php.h"
#include "tensor.h"
#include "data_types.h"
#include <stdbool.h>

tensor_t *tensor_cast_string(tensor_t *tensor, const char *new_dtype_str);
tensor_t *create_tensor_from_php_array(zval *data, dtype_t dtype);
tensor_t *cuda_tensor_create_from_flat_array(zval *data, const int *shape, int ndims, dtype_t dtype);
tensor_t *cuda_tensor_create_from_host_buffer(const int *shape, int ndims, dtype_t dtype, const void *host_data, size_t byte_count);

tensor_t *cuda_tensor_create_with_value(int *shape, int ndims, scalar_value_t value, dtype_t dtype);
tensor_t *cuda_tensor_create(const int shape[], int ndims, const void *data, dtype_t dtype);
tensor_t *cuda_tensor_create_on_host(const int shape[], int ndims, void *data, dtype_t dtype);
tensor_t *cuda_tensor_create_on_host_pinned(const int shape[], int ndims, void *data, dtype_t dtype);
tensor_t *cuda_tensor_create_float(const int shape[], int ndims, const float data[]);
tensor_t *cuda_tensor_create_int(const int shape[], int ndims, const int data[]);
tensor_t *cuda_tensor_create_rand(
    int *shape,
    int ndims,
    scalar_value_t min_value,
    scalar_value_t max_value,
    dtype_t dtype,
    unsigned long long seed);

tensor_t *cuda_tensor_create_scalar(float value, int *shape, int ndims);
tensor_t *cuda_tensor_create_empty(const int shape[], int ndims);
tensor_t *cuda_tensor_create_empty_dtype(const int shape[], int ndims, dtype_t dtype);
tensor_t *resolve_result_tensor(tensor_t *t);
tensor_t *cuda_tensor_clone(tensor_t *base_tensor);

int cuda_tensor_get_scalar_value(tensor_t *t, float *result_val, int index);

#endif