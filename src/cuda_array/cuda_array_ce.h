#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H
#endif

#include "php.h"
#include "cuda_wrapper.h"
#include "tensor.h"
#include "cuda.h"
#include "ca_struct.h"

extern zend_class_entry *cuda_array_ce;

int cuda_array_init(size_t size);
void cuda_array_shutdown();

ZEND_METHOD(CudaArray, __construct);
ZEND_METHOD(CudaArray, fromBuffer);
ZEND_METHOD(CudaArray, fromFlatArray);
ZEND_METHOD(CudaArray, fromFile);
ZEND_METHOD(CudaArray, fromNpy);
ZEND_METHOD(CudaArray, where);
ZEND_METHOD(CudaArray, __serialize);
ZEND_METHOD(CudaArray, __unserialize);
ZEND_METHOD(CudaArray, __invoke);
ZEND_METHOD(CudaArray, slice);
ZEND_METHOD(CudaArray, __debugInfo);

ZEND_METHOD(CudaArray, zeros);
ZEND_METHOD(CudaArray, ones);
ZEND_METHOD(CudaArray, full);
ZEND_METHOD(CudaArray, rand);

ZEND_METHOD(CudaArray, matmul);
ZEND_METHOD(CudaArray, transpose);

ZEND_METHOD(CudaArray, multiply);
ZEND_METHOD(CudaArray, divide);
ZEND_METHOD(CudaArray, add);
ZEND_METHOD(CudaArray, subtract);
ZEND_METHOD(CudaArray, power);

ZEND_METHOD(CudaArray, sqrt);
ZEND_METHOD(CudaArray, exp);
ZEND_METHOD(CudaArray, log);
ZEND_METHOD(CudaArray, sin);
ZEND_METHOD(CudaArray, cos);
ZEND_METHOD(CudaArray, tan);
ZEND_METHOD(CudaArray, abs);
ZEND_METHOD(CudaArray, neg);
ZEND_METHOD(CudaArray, floor);
ZEND_METHOD(CudaArray, ceil);
ZEND_METHOD(CudaArray, round);

ZEND_METHOD(CudaArray, gt);
ZEND_METHOD(CudaArray, lt);
ZEND_METHOD(CudaArray, eq);
ZEND_METHOD(CudaArray, ne);
ZEND_METHOD(CudaArray, ge);
ZEND_METHOD(CudaArray, le);

ZEND_METHOD(CudaArray, sum);
ZEND_METHOD(CudaArray, mean);
ZEND_METHOD(CudaArray, max);
ZEND_METHOD(CudaArray, min);
ZEND_METHOD(CudaArray, prod);

ZEND_METHOD(CudaArray, argMax);
ZEND_METHOD(CudaArray, argMin);

ZEND_METHOD(CudaArray, reshape);
ZEND_METHOD(CudaArray, flatten);
ZEND_METHOD(CudaArray, astype);

ZEND_METHOD(CudaArray, concat);

ZEND_METHOD(CudaArray, getShape);
ZEND_METHOD(CudaArray, getSize);
ZEND_METHOD(CudaArray, getStrides);
ZEND_METHOD(CudaArray, getNdims);
ZEND_METHOD(CudaArray, toArray);
ZEND_METHOD(CudaArray, toBuffer);
ZEND_METHOD(CudaArray, toHost);
ZEND_METHOD(CudaArray, dtype);
