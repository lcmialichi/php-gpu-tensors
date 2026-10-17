#ifndef CONTIGUOUS_ARRAY_CE_H
#define CONTIGUOUS_ARRAY_CE_H

#include "php.h"
#include "zend_interfaces.h"
#include "tensor.h" 

extern zend_class_entry *contiguous_array_ce;

typedef struct _contiguous_array_object {
    tensor_t *tensor;
    char *cached_data_ptr;
    int *shape;
    size_t *strides;
    size_t total_elements;
    size_t element_size;
    int ndims;
    dtype_t dtype;
    size_t offset;
    uint8_t is_contiguous;
    zend_bool read_only;
    zend_object std;
} contiguous_array_object;

typedef struct {
    zend_object_iterator intern;
    zval host_array;
    zend_long current_idx;
    zend_long max_idx;
    zval current;
    void *current_data_ptr;
    void *extra_data;
    size_t stride_bytes;
    dtype_t dtype;
    uint8_t is_1d;
} contiguous_array_iterator;

int contiguous_array_init();
zend_object *contiguous_array_create_object(zend_class_entry *ce);
void contiguous_array_free_object(zend_object *object);
zend_object *contiguous_array_from_tensor(tensor_t *tensor);

size_t dtype_to_size(dtype_t dtype);
const char *dtype_to_string(dtype_t dtype);
void *allocate_for_dtype(dtype_t dtype, size_t count);

ZEND_METHOD(ContiguousArray, __construct);
ZEND_METHOD(ContiguousArray, fromBuffer);
ZEND_METHOD(ContiguousArray, fromFlatArray);
ZEND_METHOD(ContiguousArray, zeros);
ZEND_METHOD(ContiguousArray, ones);
ZEND_METHOD(ContiguousArray, full);
ZEND_METHOD(ContiguousArray, zerosLike);
ZEND_METHOD(ContiguousArray, onesLike);
ZEND_METHOD(ContiguousArray, get);
ZEND_METHOD(ContiguousArray, getShape);
ZEND_METHOD(ContiguousArray, getStrides);
ZEND_METHOD(ContiguousArray, getNdims);
ZEND_METHOD(ContiguousArray, getSize);
ZEND_METHOD(ContiguousArray, getDtype);
ZEND_METHOD(ContiguousArray, dtype);
ZEND_METHOD(ContiguousArray, getBackend);
ZEND_METHOD(ContiguousArray, toArray);
ZEND_METHOD(ContiguousArray, toGpu);
ZEND_METHOD(ContiguousArray, toHost);
ZEND_METHOD(ContiguousArray, toBuffer);
ZEND_METHOD(ContiguousArray, add);
ZEND_METHOD(ContiguousArray, subtract);
ZEND_METHOD(ContiguousArray, multiply);
ZEND_METHOD(ContiguousArray, divide);
ZEND_METHOD(ContiguousArray, __debugInfo);
ZEND_METHOD(ContiguousArray, isPinned);
ZEND_METHOD(ContiguousArray, getElementSize);
ZEND_METHOD(ContiguousArray, count);
ZEND_METHOD(ContiguousArray, at);
ZEND_METHOD(ContiguousArray, __unserialize);
ZEND_METHOD(ContiguousArray, __serialize);

#endif