#ifndef CUDA_ARRAY_ARGINFO_H
#define CUDA_ARRAY_ARGINFO_H

#include "php.h"

#define CA_CLASS_NAME "Cuda\\CudaArray"

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_construct, 0, 0, 1)
ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_fromFlatArray, 0, 2, Cuda\\CudaArray, 0)
ZEND_ARG_TYPE_INFO(0, values, IS_ARRAY, 0)
ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_fromBuffer, 0, 2, Cuda\\CudaArray, 0)
ZEND_ARG_TYPE_INFO(0, bytes, IS_STRING, 0)
ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_fromFile, 0, 2, Cuda\\CudaArray, 0)
ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_fromNpy, 0, 1, Cuda\\CudaArray, 0)
ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_where, 0, 3, Cuda\\CudaArray, 0)
ZEND_ARG_INFO(0, condition)
ZEND_ARG_INFO(0, x)
ZEND_ARG_INFO(0, y)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_Tensor___debugInfo, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_serialize, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_unserialize, 0, 0, 1)
ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_multiply, 0, 1, Cuda\\CudaArray, 0)
ZEND_ARG_INFO(0, other)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_invoke, 0, 0, 0)
ZEND_ARG_VARIADIC_INFO(0, slices)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_slice, 0, 0, Cuda\\CudaArray, 0)
{ "selectors", ZEND_TYPE_INIT_MASK(MAY_BE_LONG | MAY_BE_STRING | MAY_BE_ARRAY | MAY_BE_NULL | _ZEND_ARG_INFO_FLAGS(0, 1, 0)), NULL },
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_subtract, 0, 1, Cuda\\CudaArray, 0)
ZEND_ARG_INFO(0, other)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_divide, 0, 1, Cuda\\CudaArray, 0)
ZEND_ARG_INFO(0, other)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_add, 0, 1, Cuda\\CudaArray, 0)
ZEND_ARG_INFO(0, other)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_matmul, 0, 0, 1)
ZEND_ARG_OBJ_INFO(0, other, Cuda\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_unary, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_binary, 0, 0, 1)
ZEND_ARG_INFO(0, other)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_getShape, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_getNdims, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_getStrides, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_getSize, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_transpose, 0, 0, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_toArray, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_toHost, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cuda_array_toBuffer, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_flatten, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_reduce, 0, 0, Cuda\\CudaArray, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_reshape, 0, 0, 1)
ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_zeros, 0, 1, Cuda\\CudaArray, 0)
ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_ones, 0, 1, Cuda\\CudaArray, 0)
ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_full, 0, 2, Cuda\\CudaArray, 0)
ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_ARG_TYPE_MASK(0, value, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_concat, 0, 0, 1)
ZEND_ARG_ARRAY_INFO(0, tensors, 0)
ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_rand, 0, 1, Cuda\\CudaArray, 0)
ZEND_ARG_ARRAY_INFO(0, shape, 0)
ZEND_ARG_TYPE_MASK(0, min, MAY_BE_LONG | MAY_BE_DOUBLE, "0")
ZEND_ARG_TYPE_MASK(0, max, MAY_BE_LONG | MAY_BE_DOUBLE, "1")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cuda_array_astype, 0, 1, Cuda\\CudaArray, 0)
ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_array_dtype, 0, 0, 0)
ZEND_END_ARG_INFO()

static zend_function_entry cuda_array_methods[] = {
    ZEND_ME(CudaArray, __construct, arginfo_cuda_array_construct, ZEND_ACC_PUBLIC | ZEND_ACC_CTOR)
    ZEND_ME(CudaArray, fromBuffer, arginfo_cuda_array_fromBuffer, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, fromFlatArray, arginfo_cuda_array_fromFlatArray, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, fromFile, arginfo_cuda_array_fromFile, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, fromNpy, arginfo_cuda_array_fromNpy, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, where, arginfo_cuda_array_where, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(CudaArray, __serialize, arginfo_cuda_array_serialize, ZEND_ACC_PUBLIC)
            ZEND_ME(CudaArray, __unserialize, arginfo_cuda_array_unserialize, ZEND_ACC_PUBLIC)
        ZEND_ME(CudaArray, __invoke, arginfo_cuda_array_invoke, ZEND_ACC_PUBLIC)
        ZEND_ME(CudaArray, slice, arginfo_cuda_array_slice, ZEND_ACC_PUBLIC)
            ZEND_ME(CudaArray, __debugInfo, arginfo_Tensor___debugInfo, ZEND_ACC_PUBLIC)

                ZEND_ME(CudaArray, concat, arginfo_cuda_array_concat, ZEND_ACC_PUBLIC)
                    ZEND_ME(CudaArray, multiply, arginfo_cuda_array_multiply, ZEND_ACC_PUBLIC)
                        ZEND_ME(CudaArray, divide, arginfo_cuda_array_divide, ZEND_ACC_PUBLIC)
                            ZEND_ME(CudaArray, add, arginfo_cuda_array_add, ZEND_ACC_PUBLIC)
                                ZEND_ME(CudaArray, subtract, arginfo_cuda_array_subtract, ZEND_ACC_PUBLIC)
                                    ZEND_ME(CudaArray, matmul, arginfo_cuda_array_matmul, ZEND_ACC_PUBLIC)
                                        ZEND_ME(CudaArray, transpose, arginfo_cuda_array_transpose, ZEND_ACC_PUBLIC)
                                            ZEND_ME(CudaArray, power, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
                                                ZEND_ME(CudaArray, sqrt, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                    ZEND_ME(CudaArray, exp, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                        ZEND_ME(CudaArray, log, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                            ZEND_ME(CudaArray, sin, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                                ZEND_ME(CudaArray, cos, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                                    ZEND_ME(CudaArray, tan, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                                        ZEND_ME(CudaArray, abs, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                                            ZEND_ME(CudaArray, neg, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                                                ZEND_ME(CudaArray, floor, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                                                    ZEND_ME(CudaArray, ceil, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                                                        ZEND_ME(CudaArray, round, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
                                                                                            ZEND_ME(CudaArray, gt, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
                                                                                                ZEND_ME(CudaArray, lt, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
                                                                                                    ZEND_ME(CudaArray, eq, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
                                                                                                        ZEND_ME(CudaArray, ne, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
                                                                                                            ZEND_ME(CudaArray, ge, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
                                                                                                                ZEND_ME(CudaArray, le, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)

                                                                                                                    ZEND_ME(CudaArray, sum, arginfo_cuda_array_reduce, ZEND_ACC_PUBLIC)
                                                                                                                        ZEND_ME(CudaArray, mean, arginfo_cuda_array_reduce, ZEND_ACC_PUBLIC)
                                                                                                                            ZEND_ME(CudaArray, max, arginfo_cuda_array_reduce, ZEND_ACC_PUBLIC)
                                                                                                                                ZEND_ME(CudaArray, min, arginfo_cuda_array_reduce, ZEND_ACC_PUBLIC)
                                                                                                                                    ZEND_ME(CudaArray, prod, arginfo_cuda_array_reduce, ZEND_ACC_PUBLIC)
                                                                                                                                        ZEND_ME(CudaArray, argMax, arginfo_cuda_array_reduce, ZEND_ACC_PUBLIC)
                                                                                                                                            ZEND_ME(CudaArray, argMin, arginfo_cuda_array_reduce, ZEND_ACC_PUBLIC)

                                                                                                                                                ZEND_ME(CudaArray, getShape, arginfo_cuda_array_getShape, ZEND_ACC_PUBLIC)
                                                                                                                                                    ZEND_ME(CudaArray, getStrides, arginfo_cuda_array_getStrides, ZEND_ACC_PUBLIC)
                                                                                                                                                        ZEND_ME(CudaArray, getNdims, arginfo_cuda_array_getNdims, ZEND_ACC_PUBLIC)
                                                                                                                                                            ZEND_ME(CudaArray, getSize, arginfo_cuda_array_getSize, ZEND_ACC_PUBLIC)

                                                                                                                                                                ZEND_ME(CudaArray, toArray, arginfo_cuda_array_toArray, ZEND_ACC_PUBLIC)
                                                                                                                                                                    ZEND_ME(CudaArray, toHost, arginfo_cuda_array_toHost, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, toBuffer, arginfo_cuda_array_toBuffer, ZEND_ACC_PUBLIC)
                                                                                                                                                                        ZEND_ME(CudaArray, reshape, arginfo_cuda_array_reshape, ZEND_ACC_PUBLIC)
                                                                                                                                                                            ZEND_ME(CudaArray, flatten, arginfo_cuda_array_flatten, ZEND_ACC_PUBLIC)

                                                                                                                                                                                ZEND_ME(CudaArray, zeros, arginfo_cuda_array_zeros, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
                                                                                                                                                                                    ZEND_ME(CudaArray, ones, arginfo_cuda_array_ones, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
                                                                                                                                                                                        ZEND_ME(CudaArray, full, arginfo_cuda_array_full, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
                                                                                                                                                                                            ZEND_ME(CudaArray, rand, arginfo_cuda_array_rand, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                                                                                                                                                                                                ZEND_ME(CudaArray, astype, arginfo_cuda_array_astype, ZEND_ACC_PUBLIC)
                                                                                                                                                                                                    ZEND_ME(CudaArray, dtype, arginfo_cuda_array_dtype, ZEND_ACC_PUBLIC)

                                                                                                                                                                                                        PHP_FE_END};

static zend_class_entry *register_cuda_array_class(void)
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, CA_CLASS_NAME, cuda_array_methods);
    cuda_array_ce = zend_register_internal_class(&ce);
    cuda_array_ce->ce_flags |= ZEND_ACC_FINAL;

    return cuda_array_ce;
}

#endif