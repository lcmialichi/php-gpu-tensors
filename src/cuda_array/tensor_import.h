#ifndef CUDA_TENSOR_IMPORT_H
#define CUDA_TENSOR_IMPORT_H

#include "php.h"
#include "tensor.h"
#include "main/php_streams.h"

int tensor_import_shape(zval *shape_array, int shape[MAX_DIMS], size_t *elements);
int tensor_import_serialized_shape(zval *shape_array, zend_long ndims,
                                   int shape[MAX_DIMS], size_t *elements);
tensor_t *tensor_import_file(zend_string *path, const int *shape, int ndims, dtype_t dtype, size_t bytes);
tensor_t *tensor_import_stream(php_stream *stream, const int *shape, int ndims, dtype_t dtype, size_t bytes);
tensor_t *tensor_import_npy(zend_string *path);

#endif