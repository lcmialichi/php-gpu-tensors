#ifndef PHP_TENSORS_HOST_TENSOR_H
#define PHP_TENSORS_HOST_TENSOR_H

#include "tensor.h"

tensor_t *cuda_tensor_create_on_host(const int shape[], int ndims, void *data, dtype_t dtype);
tensor_t *cuda_tensor_create_on_host_pinned(const int shape[], int ndims, void *data, dtype_t dtype);
tensor_t *tensor_host_create_view(tensor_t *base, void *data, const int *shape,
                                  const size_t *strides, int ndims);
void tensor_host_destroy(tensor_t *tensor);

#endif
