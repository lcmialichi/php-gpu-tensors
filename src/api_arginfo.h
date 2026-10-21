/* Generated from PHP API stubs by tests/generate_arginfo.php. Do not edit. */
#ifndef PHPTENSORS_API_ARGINFO_H
#define PHPTENSORS_API_ARGINFO_H

#include "php.h"

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensorsTestFixtures_UnavailableCudaArray___call, 0, 2, IS_MIXED, 1)
    ZEND_ARG_TYPE_MASK(0, name, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, arguments, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensorsTestFixtures_UnavailableCudaArray___callStatic, 0, 2, IS_MIXED, 1)
    ZEND_ARG_TYPE_MASK(0, name, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, arguments, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_api_PhpTensorsTestFixtures_UnavailableCudaArray___construct, 0, 0, 0)
    { "arguments", ZEND_TYPE_INIT_MASK(MAY_BE_ANY | _ZEND_ARG_INFO_FLAGS(0, 1, 0)), NULL },
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_BackendInterface_createTensor, 0, 2, PhpTensors\\Tensor, 0)
    ZEND_ARG_TYPE_MASK(0, values, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_BackendInterface_execute, 0, 2, PhpTensors\\Tensor, 0)
    ZEND_ARG_TYPE_MASK(0, operation, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, inputs, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, arguments, MAY_BE_ARRAY, "array (\n)")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_BackendInterface_getName, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_BackendInterface_isAvailable, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_BackendInterface_supports, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_MASK(0, operation, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, context, MAY_BE_ARRAY, "array (\n)")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule___serialize, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule___unserialize, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_MASK(0, data, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_autoGrid, 0, 2, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, name, MAY_BE_STRING, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, elements, PhpTensors\\CudaArray, MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_cancelOperation, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_MASK(0, op_id, MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_cleanup, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_getAsyncStatus, 0, 0, IS_ARRAY, 1)
    ZEND_ARG_TYPE_MASK(0, op_id, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_getKernels, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_getPendingOperations, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_getPtx, 0, 0, IS_STRING, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_getStats, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_hasKernel, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_MASK(0, name, MAY_BE_STRING, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_initialize, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_isFinished, 0, 0, _IS_BOOL, 0)
    ZEND_ARG_TYPE_MASK(0, op_id, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_launch, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_MASK(0, name, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, config, MAY_BE_ARRAY | MAY_BE_NULL, "array (\n)")
    ZEND_ARG_TYPE_MASK(0, args, MAY_BE_ARRAY | MAY_BE_NULL, "array (\n)")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_launchAsync, 0, 1, IS_LONG, 0)
    ZEND_ARG_TYPE_MASK(0, name, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, config, MAY_BE_ARRAY | MAY_BE_NULL, "array (\n)")
    ZEND_ARG_TYPE_MASK(0, args, MAY_BE_ARRAY | MAY_BE_NULL, "array (\n)")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_api_PhpTensors_CompiledModule_launchAsyncBatch, 0, 1, MAY_BE_ARRAY | MAY_BE_BOOL)
    ZEND_ARG_TYPE_MASK(0, operations, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_save, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_MASK(0, filename, MAY_BE_STRING, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_sync, 0, 0, _IS_BOOL, 0)
    ZEND_ARG_TYPE_MASK(0, op_id, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CompiledModule_wait, 0, 0, _IS_BOOL, 0)
    ZEND_ARG_TYPE_MASK(0, op_id, MAY_BE_LONG | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, timeout_ms, MAY_BE_LONG, "-1")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_api_PhpTensors_Compiler___construct, 0, 0, 1)
    ZEND_ARG_TYPE_MASK(0, source, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, target, MAY_BE_STRING | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, optimization, MAY_BE_LONG, "2")
    ZEND_ARG_TYPE_MASK(0, debug, MAY_BE_BOOL, "false")
    ZEND_ARG_TYPE_MASK(0, fast_math, MAY_BE_BOOL, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Compiler_clearCache, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_Compiler_compile, 0, 0, PhpTensors\\CompiledModule, 0)
    ZEND_ARG_TYPE_MASK(0, optimize, MAY_BE_BOOL, "true")
    ZEND_ARG_TYPE_MASK(0, debug, MAY_BE_BOOL, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Compiler_getCacheStats, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Compiler_getKernels, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Compiler_header, 0, 1, IS_STATIC, 0)
    ZEND_ARG_TYPE_MASK(0, code, MAY_BE_STRING, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Compiler_kernel, 0, 1, IS_STATIC, 0)
    ZEND_ARG_TYPE_MASK(0, name, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, parameters, MAY_BE_ARRAY | MAY_BE_NULL, "array (\n)")
    ZEND_ARG_TYPE_MASK(0, headers, MAY_BE_ARRAY | MAY_BE_NULL, "array (\n)")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_api_PhpTensors_CudaArray___construct, 0, 0, 1)
    ZEND_ARG_TYPE_MASK(0, data, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray___debugInfo, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray___invoke, 0, 0, PhpTensors\\CudaArray, 0)
    { "slices", ZEND_TYPE_INIT_MASK(MAY_BE_ARRAY | MAY_BE_LONG | MAY_BE_NULL | _ZEND_ARG_INFO_FLAGS(0, 1, 0)), NULL },
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray___serialize, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray___unserialize, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_MASK(0, data, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_abs, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_acos, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_add, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_all, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_any, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_arange, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, start, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_MASK(0, stop, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, step, MAY_BE_LONG | MAY_BE_DOUBLE, "1")
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_argMax, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_argMin, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_asin, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_astype, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_atan, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray_backward, 0, 0, IS_VOID, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, gradient, PhpTensors\\CudaArray, MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, retainGraph, MAY_BE_BOOL, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_broadcastTo, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_ceil, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_clamp, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, min, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, max, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_concat, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, tensors, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_copy, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_cos, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_cosh, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_cumprod, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_cumsum, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_detach, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_diagonal, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, offset, MAY_BE_LONG, "0")
    ZEND_ARG_TYPE_MASK(0, axis1, MAY_BE_LONG, "0")
    ZEND_ARG_TYPE_MASK(0, axis2, MAY_BE_LONG, "1")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_divide, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray_dtype, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_eq, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_exp, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_exp2, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_expm1, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_eye, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, rows, MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, columns, MAY_BE_LONG | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, k, MAY_BE_LONG, "0")
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_flatten, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_flip, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_ARRAY | MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_floor, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_fromBuffer, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, bytes, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_fromFile, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, path, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_fromFlatArray, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, values, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_fromNpy, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, path, MAY_BE_STRING, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_full, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, value, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_fullLike, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, prototype, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, value, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_gather, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, indices, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_ge, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray_getBackend, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray_getNdims, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray_getShape, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray_getSize, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray_getStrides, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_grad, 0, 0, PhpTensors\\CudaArray, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_gt, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_identity, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, size, MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_api_PhpTensors_CudaArray_item, 0, 0, MAY_BE_STRING | MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_layerNorm, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axes, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, epsilon, MAY_BE_DOUBLE, "1.0E-5")
    ZEND_ARG_OBJ_TYPE_MASK(0, scale, PhpTensors\\CudaArray, MAY_BE_NULL, "NULL")
    ZEND_ARG_OBJ_TYPE_MASK(0, bias, PhpTensors\\CudaArray, MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_le, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_linspace, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, start, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_MASK(0, stop, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_MASK(0, num, MAY_BE_LONG, "50")
    ZEND_ARG_TYPE_MASK(0, endpoint, MAY_BE_BOOL, "true")
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float64'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_log, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_log10, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_log1p, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_log2, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_lt, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_matmul, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, 0, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_max, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_maximum, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_mean, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_min, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_minimum, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_moveaxis, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, source, MAY_BE_ARRAY | MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, destination, MAY_BE_ARRAY | MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_multiply, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_ne, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_neg, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_ones, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_onesLike, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, prototype, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_power, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_prod, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_rand, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, min, MAY_BE_LONG | MAY_BE_DOUBLE, "0")
    ZEND_ARG_TYPE_MASK(0, max, MAY_BE_LONG | MAY_BE_DOUBLE, "1")
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
    ZEND_ARG_TYPE_MASK(0, seed, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_randn, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
    ZEND_ARG_TYPE_MASK(0, seed, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_repeat, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, repeats, MAY_BE_ARRAY | MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_requiresGrad, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, requiresGrad, MAY_BE_BOOL, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_reshape, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_roll, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, shift, MAY_BE_ARRAY | MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_ARRAY | MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_round, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_scatterAdd, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, indices, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, updates, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_sin, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_sinh, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_slice, 0, 0, PhpTensors\\CudaArray, 0)
    { "selectors", ZEND_TYPE_INIT_MASK(MAY_BE_ARRAY | MAY_BE_STRING | MAY_BE_LONG | MAY_BE_NULL | _ZEND_ARG_INFO_FLAGS(0, 1, 0)), NULL },
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_softmax, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG, "-1")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_sqrt, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_squeeze, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_std, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, correction, MAY_BE_LONG, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_subtract, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\CudaArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_sum, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_swapaxes, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis1, MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, axis2, MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_tan, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_tanh, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_tile, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, reps, MAY_BE_ARRAY | MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray_toArray, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray_toBuffer, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_toHost, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_transpose, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_ARRAY | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_unsqueeze, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_var, 0, 0, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, correction, MAY_BE_LONG, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_where, 0, 3, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, condition, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, x, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, y, PhpTensors\\CudaArray, 0, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_CudaArray_zeroGrad, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_zeros, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_CudaArray_zerosLike, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, prototype, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_api_PhpTensors_Fusion___construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Fusion_clearCache, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_Fusion_compile, 0, 2, PhpTensors\\FusionGraph, 0)
    ZEND_ARG_TYPE_MASK(0, callback, MAY_BE_CALLABLE, NULL)
    ZEND_ARG_TYPE_MASK(0, inputs, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, cudaGraph, MAY_BE_BOOL, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Fusion_getCacheStats, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Fusion_run, 0, 1, IS_MIXED, 1)
    ZEND_ARG_TYPE_MASK(0, callback, MAY_BE_CALLABLE, NULL)
    ZEND_ARG_TYPE_MASK(0, enabled, MAY_BE_BOOL, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_api_PhpTensors_FusionExecution___construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_FusionExecution_isFinished, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_FusionExecution_wait, 0, 0, IS_MIXED, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_api_PhpTensors_FusionGraph___construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_FusionGraph_getPlan, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_FusionGraph_getSource, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_FusionGraph_getStats, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_FusionGraph_run, 0, 0, IS_MIXED, 1)
    ZEND_ARG_VARIADIC_OBJ_INFO(0, inputs, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_FusionGraph_runAsync, 0, 0, PhpTensors\\FusionExecution, 0)
    ZEND_ARG_VARIADIC_OBJ_INFO(0, inputs, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_FusionGraph_setProfiling, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_MASK(0, enabled, MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_api_PhpTensors_HostArray___construct, 0, 0, 1)
    ZEND_ARG_TYPE_MASK(0, values, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
    ZEND_ARG_TYPE_MASK(0, pinned, MAY_BE_BOOL, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray___debugInfo, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray___invoke, 0, 0, PhpTensors\\HostArray, 0)
    { "slices", ZEND_TYPE_INIT_MASK(MAY_BE_ARRAY | MAY_BE_LONG | MAY_BE_NULL | _ZEND_ARG_INFO_FLAGS(0, 1, 0)), NULL },
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray___serialize, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray___unserialize, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_MASK(0, data, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_abs, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_acos, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_add, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_all, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_any, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_arange, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, start, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_MASK(0, stop, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, step, MAY_BE_LONG | MAY_BE_DOUBLE, "1")
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_argMax, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_argMin, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_asin, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_astype, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_at, 0, 0, IS_MIXED, 1)
    { "indices", ZEND_TYPE_INIT_MASK(MAY_BE_LONG | _ZEND_ARG_INFO_FLAGS(0, 1, 0)), NULL },
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_atan, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_backward, 0, 0, IS_VOID, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, gradient, PhpTensors\\HostArray, MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, retainGraph, MAY_BE_BOOL, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_broadcastTo, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_ceil, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_clamp, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, min, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, max, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_concat, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, tensors, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_copy, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_cos, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_cosh, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_count, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_cumprod, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_cumsum, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_detach, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_diagonal, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, offset, MAY_BE_LONG, "0")
    ZEND_ARG_TYPE_MASK(0, axis1, MAY_BE_LONG, "0")
    ZEND_ARG_TYPE_MASK(0, axis2, MAY_BE_LONG, "1")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_divide, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_dtype, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_eq, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_exp, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_exp2, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_expm1, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_eye, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, rows, MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, columns, MAY_BE_LONG | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, k, MAY_BE_LONG, "0")
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_flatten, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_flip, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_ARRAY | MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_floor, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_fromBuffer, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, bytes, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
    ZEND_ARG_TYPE_MASK(0, pinned, MAY_BE_BOOL, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_fromFile, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, path, MAY_BE_STRING, NULL)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_fromFlatArray, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, values, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_fromNpy, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, path, MAY_BE_STRING, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_full, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, value, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_fullLike, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, prototype, PhpTensors\\HostArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, value, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_gather, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, indices, PhpTensors\\HostArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_ge, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_get, 0, 1, IS_MIXED, 1)
    ZEND_ARG_TYPE_MASK(0, indices, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_getBackend, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_getDtype, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_getElementSize, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_getIterator, 0, 0, Traversable, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_getNdims, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_getShape, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_getSize, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_getStrides, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_grad, 0, 0, PhpTensors\\HostArray, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_gt, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_identity, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, size, MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_isPinned, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_api_PhpTensors_HostArray_item, 0, 0, MAY_BE_STRING | MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_layerNorm, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axes, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, epsilon, MAY_BE_DOUBLE, "1.0E-5")
    ZEND_ARG_OBJ_TYPE_MASK(0, scale, PhpTensors\\HostArray, MAY_BE_NULL, "NULL")
    ZEND_ARG_OBJ_TYPE_MASK(0, bias, PhpTensors\\HostArray, MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_le, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_linspace, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, start, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_MASK(0, stop, MAY_BE_LONG | MAY_BE_DOUBLE, NULL)
    ZEND_ARG_TYPE_MASK(0, num, MAY_BE_LONG, "50")
    ZEND_ARG_TYPE_MASK(0, endpoint, MAY_BE_BOOL, "true")
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float64'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_log, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_log10, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_log1p, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_log2, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_lt, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_matmul, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, 0, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_max, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_maximum, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_mean, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_min, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_minimum, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_moveaxis, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, source, MAY_BE_ARRAY | MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, destination, MAY_BE_ARRAY | MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_multiply, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_ne, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_neg, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_ones, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_onesLike, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, prototype, PhpTensors\\HostArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_power, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_prod, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_rand, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, min, MAY_BE_LONG | MAY_BE_DOUBLE, "0")
    ZEND_ARG_TYPE_MASK(0, max, MAY_BE_LONG | MAY_BE_DOUBLE, "1")
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
    ZEND_ARG_TYPE_MASK(0, seed, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_randn, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
    ZEND_ARG_TYPE_MASK(0, seed, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_repeat, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, repeats, MAY_BE_ARRAY | MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_requiresGrad, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, requiresGrad, MAY_BE_BOOL, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_reshape, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_roll, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, shift, MAY_BE_ARRAY | MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_ARRAY | MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_round, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_scatterAdd, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, indices, PhpTensors\\HostArray, 0, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, updates, PhpTensors\\HostArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_sin, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_sinh, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_slice, 0, 0, PhpTensors\\HostArray, 0)
    { "selectors", ZEND_TYPE_INIT_MASK(MAY_BE_ARRAY | MAY_BE_STRING | MAY_BE_LONG | MAY_BE_NULL | _ZEND_ARG_INFO_FLAGS(0, 1, 0)), NULL },
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_softmax, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG, "-1")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_sqrt, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_squeeze, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_std, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, correction, MAY_BE_LONG, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_subtract, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, other, PhpTensors\\HostArray, MAY_BE_LONG | MAY_BE_DOUBLE | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_sum, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_swapaxes, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis1, MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, axis2, MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_tan, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_tanh, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_tile, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, reps, MAY_BE_ARRAY | MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_toArray, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_toBuffer, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_toGpu, 0, 0, PhpTensors\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_toHost, 0, 0, PhpTensors\\HostArray, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_transpose, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_ARRAY | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_unsqueeze, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_var, 0, 0, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, correction, MAY_BE_LONG, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_where, 0, 3, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, condition, PhpTensors\\HostArray, 0, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, x, PhpTensors\\HostArray, 0, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, y, PhpTensors\\HostArray, 0, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_HostArray_zeroGrad, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_zeros, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, shape, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "'float32'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_HostArray_zerosLike, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, prototype, PhpTensors\\HostArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, dtype, MAY_BE_STRING | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_api_PhpTensors_NN___construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_NN_conv2d, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, input, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, weights, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, bias, PhpTensors\\CudaArray, MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, stride, MAY_BE_ARRAY, "array (\n  0 => 1,\n  1 => 1,\n)")
    ZEND_ARG_TYPE_MASK(0, padding, MAY_BE_ARRAY, "array (\n  0 => 0,\n  1 => 0,\n)")
    ZEND_ARG_TYPE_MASK(0, dilation, MAY_BE_ARRAY, "array (\n  0 => 1,\n  1 => 1,\n)")
    ZEND_ARG_TYPE_MASK(0, groups, MAY_BE_LONG, "1")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_NN_pool2d, 0, 2, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, input, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_TYPE_MASK(0, window, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, stride, MAY_BE_ARRAY, "array (\n  0 => 2,\n  1 => 2,\n)")
    ZEND_ARG_TYPE_MASK(0, padding, MAY_BE_ARRAY, "array (\n  0 => 0,\n  1 => 0,\n)")
    ZEND_ARG_TYPE_MASK(0, mode, MAY_BE_STRING, "'max'")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_NN_softmax, 0, 1, PhpTensors\\CudaArray, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, input, PhpTensors\\CudaArray, 0, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_OperatorOverload___add, 0, 2, IS_MIXED, 1)
    ZEND_ARG_TYPE_MASK(0, left, MAY_BE_ANY, NULL)
    ZEND_ARG_TYPE_MASK(0, right, MAY_BE_ANY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_OperatorOverload___div, 0, 2, IS_MIXED, 1)
    ZEND_ARG_TYPE_MASK(0, left, MAY_BE_ANY, NULL)
    ZEND_ARG_TYPE_MASK(0, right, MAY_BE_ANY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_OperatorOverload___mod, 0, 2, IS_MIXED, 1)
    ZEND_ARG_TYPE_MASK(0, left, MAY_BE_ANY, NULL)
    ZEND_ARG_TYPE_MASK(0, right, MAY_BE_ANY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_OperatorOverload___mul, 0, 2, IS_MIXED, 1)
    ZEND_ARG_TYPE_MASK(0, left, MAY_BE_ANY, NULL)
    ZEND_ARG_TYPE_MASK(0, right, MAY_BE_ANY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_OperatorOverload___pow, 0, 2, IS_MIXED, 1)
    ZEND_ARG_TYPE_MASK(0, left, MAY_BE_ANY, NULL)
    ZEND_ARG_TYPE_MASK(0, right, MAY_BE_ANY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_OperatorOverload___sub, 0, 2, IS_MIXED, 1)
    ZEND_ARG_TYPE_MASK(0, left, MAY_BE_ANY, NULL)
    ZEND_ARG_TYPE_MASK(0, right, MAY_BE_ANY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_api_PhpTensors_Optimizer___construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_Optimizer_adamW, 0, 0, PhpTensors\\Optimizer, 0)
    ZEND_ARG_TYPE_MASK(0, learningRate, MAY_BE_DOUBLE, "0.001")
    ZEND_ARG_TYPE_MASK(0, beta1, MAY_BE_DOUBLE, "0.9")
    ZEND_ARG_TYPE_MASK(0, beta2, MAY_BE_DOUBLE, "0.999")
    ZEND_ARG_TYPE_MASK(0, epsilon, MAY_BE_DOUBLE, "1.0E-8")
    ZEND_ARG_TYPE_MASK(0, weightDecay, MAY_BE_DOUBLE, "0.01")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Optimizer_initState, 0, 1, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, parameters, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_api_PhpTensors_Optimizer_sgd, 0, 0, PhpTensors\\Optimizer, 0)
    ZEND_ARG_TYPE_MASK(0, learningRate, MAY_BE_DOUBLE, "0.01")
    ZEND_ARG_TYPE_MASK(0, momentum, MAY_BE_DOUBLE, "0.0")
    ZEND_ARG_TYPE_MASK(0, weightDecay, MAY_BE_DOUBLE, "0.0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Optimizer_step, 0, 3, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, parameters, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, gradients, MAY_BE_ARRAY, NULL)
    ZEND_ARG_TYPE_MASK(0, state, MAY_BE_ARRAY, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, learningRate, PhpTensors\\CudaArray, MAY_BE_NULL, "NULL")
    ZEND_ARG_TYPE_MASK(0, weightDecayMask, MAY_BE_ARRAY | MAY_BE_NULL, "NULL")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Optimizer_zeroGrad, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_MASK(0, parameters, MAY_BE_ARRAY, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Tensor_dtype, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Tensor_getBackend, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Tensor_getShape, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Tensor_getSize, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_PhpTensors_Tensor_toArray, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_clear_error, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_device_reset, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_get_backend_info, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_get_current_device, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_get_device_count, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_get_device_info, 0, 0, IS_ARRAY, 0)
    ZEND_ARG_TYPE_MASK(0, deviceId, MAY_BE_LONG, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_get_driver_version, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_get_last_error, 0, 0, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_get_memory_info, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_get_peer_access, 0, 2, _IS_BOOL, 0)
    ZEND_ARG_TYPE_MASK(0, device1, MAY_BE_LONG, NULL)
    ZEND_ARG_TYPE_MASK(0, device2, MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_get_runtime_version, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_set_device, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_MASK(0, deviceId, MAY_BE_LONG, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_set_matmul_precision, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_MASK(0, precision, MAY_BE_STRING, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_synchronize, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_api_cuda_tune_matmul, 0, 2, IS_ARRAY, 0)
    ZEND_ARG_OBJ_TYPE_MASK(0, left, PhpTensors\\CudaArray, 0, NULL)
    ZEND_ARG_OBJ_TYPE_MASK(0, right, PhpTensors\\CudaArray, 0, NULL)
ZEND_END_ARG_INFO()

#endif
