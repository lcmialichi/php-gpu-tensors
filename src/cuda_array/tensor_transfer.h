#ifndef CUDA_TENSOR_TRANSFER_H
#define CUDA_TENSOR_TRANSFER_H

#include "php.h"
#include "tensor.h"

void tensor_to_php_array(zval *result, const tensor_t *tensor);
void tensor_host_to_php_array(zval *result, const tensor_t *tensor, const void *data);
zend_string *tensor_to_buffer(const tensor_t *tensor);
tensor_t *tensor_copy_to_host(const tensor_t *tensor);

#endif