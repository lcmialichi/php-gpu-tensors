#ifndef ARGINFO_contiguous_array_H
#define ARGINFO_contiguous_array_H

#include "php.h"
#include "host_array_unsupported.h"

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, values, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, pinned, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_contiguous_array_fromBuffer, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, bytes, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, pinned, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_isPinned, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_get, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, indices, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_at, 0, 0, 1)
    ZEND_ARG_VARIADIC_INFO(0, indices)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_shape, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_toArray, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_getNdims, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_getSize, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_getDtype, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_dtype, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_getBackend, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_getElementSize, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_count, 0, 0, 0)
ZEND_END_ARG_INFO();

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_toGpu, 0, 0, 0)
ZEND_END_ARG_INFO();

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_toBuffer, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_serialize, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_unserialize, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_contiguous_array_fromFlatArray, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, values, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_contiguous_array_fill, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, value, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_contiguous_array_zeros, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_contiguous_array_like, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_INFO(0, prototype, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_strides, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_contiguous_array_debug_info, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_contiguous_array_toHost, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_contiguous_array_binary, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, other, MAY_BE_OBJECT | MAY_BE_DOUBLE | MAY_BE_LONG | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

static const zend_function_entry contiguous_array_methods[] = {
    ZEND_ME(ContiguousArray, __construct, arginfo_contiguous_array_construct, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, fromBuffer, arginfo_contiguous_array_fromBuffer, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(ContiguousArray, fromFlatArray, arginfo_contiguous_array_fromFlatArray, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(ContiguousArray, zeros, arginfo_contiguous_array_zeros, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(ContiguousArray, ones, arginfo_contiguous_array_zeros, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(ContiguousArray, full, arginfo_contiguous_array_fill, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(ContiguousArray, zerosLike, arginfo_contiguous_array_like, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(ContiguousArray, onesLike, arginfo_contiguous_array_like, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(ContiguousArray, isPinned, arginfo_contiguous_array_isPinned, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, get, arginfo_contiguous_array_get, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, at, arginfo_contiguous_array_at, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getShape, arginfo_contiguous_array_shape, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getStrides, arginfo_contiguous_array_strides, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, toArray, arginfo_contiguous_array_toArray, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, __debugInfo, arginfo_contiguous_array_debug_info, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getNdims, arginfo_contiguous_array_getNdims, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getSize, arginfo_contiguous_array_getSize, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getDtype, arginfo_contiguous_array_getDtype, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, dtype, arginfo_contiguous_array_dtype, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getBackend, arginfo_contiguous_array_getBackend, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getElementSize, arginfo_contiguous_array_getElementSize, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, count, arginfo_contiguous_array_count, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, toGpu, arginfo_contiguous_array_toGpu, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, toHost, arginfo_contiguous_array_toHost, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, toBuffer, arginfo_contiguous_array_toBuffer, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, add, arginfo_contiguous_array_binary, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, subtract, arginfo_contiguous_array_binary, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, multiply, arginfo_contiguous_array_binary, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, divide, arginfo_contiguous_array_binary, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, __unserialize, arginfo_contiguous_array_unserialize, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, __serialize, arginfo_contiguous_array_serialize, ZEND_ACC_PUBLIC)
    HOST_ARRAY_UNSUPPORTED_METHODS
    ZEND_FE_END
};

#endif