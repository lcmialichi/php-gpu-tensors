#ifndef PHPTENSORS_HOST_ARRAY_UNSUPPORTED_H
#define PHPTENSORS_HOST_ARRAY_UNSUPPORTED_H

#include "php.h"

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_array_unsupported_tensor, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_VARIADIC_INFO(0, arguments)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_host_array_unsupported_nullable_tensor, 0, 0, PhpTensors\\HostArray, 1)
    ZEND_ARG_VARIADIC_INFO(0, arguments)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_host_array_unsupported_void, 0, 0, IS_VOID, 0)
    ZEND_ARG_VARIADIC_INFO(0, arguments)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_host_array_unsupported_scalar, 0, 0, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL)
    ZEND_ARG_VARIADIC_INFO(0, arguments)
ZEND_END_ARG_INFO()

ZEND_FUNCTION(php_tensors_host_array_unsupported);

#define HOST_ARRAY_UNSUPPORTED_METHOD(name, arginfo, flags) \
    ZEND_RAW_FENTRY(name, zif_php_tensors_host_array_unsupported, arginfo, flags)

#define HOST_ARRAY_UNSUPPORTED_METHODS \
    HOST_ARRAY_UNSUPPORTED_METHOD("fromFile", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("fromNpy", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("where", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("__invoke", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("slice", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("concat", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("maximum", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("minimum", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("clamp", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("matmul", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("transpose", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("copy", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("swapaxes", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("moveaxis", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("diagonal", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("repeat", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("tile", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("flip", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("roll", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("power", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("sqrt", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("exp", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("expm1", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("exp2", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("log", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("log1p", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("log2", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("log10", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("sin", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("cos", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("tan", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("asin", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("acos", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("atan", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("sinh", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("cosh", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("tanh", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("abs", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("neg", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("floor", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("ceil", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("round", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("gt", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("lt", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("eq", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("ne", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("ge", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("le", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("sum", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("mean", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("max", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("min", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("prod", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("cumsum", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("cumprod", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("argMax", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("argMin", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("all", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("any", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("var", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("std", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("softmax", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("layerNorm", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("gather", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("scatterAdd", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("reshape", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("squeeze", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("unsqueeze", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("broadcastTo", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("flatten", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("fullLike", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("eye", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("identity", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("arange", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("linspace", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("rand", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("randn", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("astype", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("requiresGrad", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("backward", arginfo_host_array_unsupported_void, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("grad", arginfo_host_array_unsupported_nullable_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("zeroGrad", arginfo_host_array_unsupported_void, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("detach", arginfo_host_array_unsupported_tensor, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD("item", arginfo_host_array_unsupported_scalar, ZEND_ACC_PUBLIC)

#endif
