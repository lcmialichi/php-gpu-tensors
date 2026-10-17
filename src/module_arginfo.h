#ifndef MODULE_ARGINFO_H
#define MODULE_ARGINFO_H

#include "php.h"

#define MODULE_CLASS_NAME "PhpTensors\\CompiledModule"

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_initialize, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_launch, 0, 0, 1)
ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, config, IS_ARRAY, 1, "[]")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, args, IS_ARRAY, 1, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_autoGrid, 0, 0, 1)
ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_ARG_TYPE_MASK(0, elements, MAY_BE_LONG | MAY_BE_OBJECT, NULL)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_launchAsync, 0, 0, 1)
ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, config, IS_ARRAY, 1, "[]")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, args, IS_ARRAY, 1, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_launchAsyncBatch, 0, 0, 1)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, operations, IS_ARRAY, 1, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_sync, 0, 0, 0)
ZEND_ARG_TYPE_INFO(0, op_id, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_isFinished, 0, 0, 0)
ZEND_ARG_TYPE_INFO(0, op_id, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_wait, 0, 0, 0)
ZEND_ARG_TYPE_INFO(0, op_id, IS_LONG, 0)
ZEND_ARG_TYPE_INFO(0, timeout_ms, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_hasKernel, 0, 0, 1)
ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_getKernels, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_getPtx, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_save, 0, 0, 1)
ZEND_ARG_TYPE_INFO(0, filename, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_serialize, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_unserialize, 0, 0, 1)
ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_getAsyncStatus, 0, 0, 0)
ZEND_ARG_TYPE_INFO(0, op_id, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_getStats, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_getPendingOperations, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_cancelOperation, 0, 0, 1)
ZEND_ARG_TYPE_INFO(0, op_id, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_cleanup, 0, 0, 0)
ZEND_END_ARG_INFO()

static const zend_function_entry module_methods[] = {
    ZEND_ME(CompiledModule, initialize, arginfo_module_initialize, ZEND_ACC_PUBLIC)
    ZEND_ME(CompiledModule, autoGrid, arginfo_module_autoGrid, ZEND_ACC_PUBLIC)
        ZEND_ME(CompiledModule, launch, arginfo_module_launch, ZEND_ACC_PUBLIC)
            ZEND_ME(CompiledModule, launchAsync, arginfo_module_launchAsync, ZEND_ACC_PUBLIC)
            ZEND_ME(CompiledModule, launchAsyncBatch, arginfo_module_launchAsyncBatch, ZEND_ACC_PUBLIC)
                ZEND_ME(CompiledModule, sync, arginfo_module_sync, ZEND_ACC_PUBLIC)
                    ZEND_ME(CompiledModule, isFinished, arginfo_module_isFinished, ZEND_ACC_PUBLIC)
                        ZEND_ME(CompiledModule, wait, arginfo_module_wait, ZEND_ACC_PUBLIC)
                            ZEND_ME(CompiledModule, hasKernel, arginfo_module_hasKernel, ZEND_ACC_PUBLIC)
                                ZEND_ME(CompiledModule, getKernels, arginfo_module_getKernels, ZEND_ACC_PUBLIC)
                                    ZEND_ME(CompiledModule, getPtx, arginfo_module_getPtx, ZEND_ACC_PUBLIC)
                                        ZEND_ME(CompiledModule, save, arginfo_module_save, ZEND_ACC_PUBLIC)
                                            ZEND_ME(CompiledModule, __serialize, arginfo_module_serialize, ZEND_ACC_PUBLIC)
                                                ZEND_ME(CompiledModule, __unserialize, arginfo_module_unserialize, ZEND_ACC_PUBLIC)
                                                    ZEND_ME(CompiledModule, getAsyncStatus, arginfo_module_getAsyncStatus, ZEND_ACC_PUBLIC)
                                                        ZEND_ME(CompiledModule, getStats, arginfo_module_getStats, ZEND_ACC_PUBLIC)
                                                            ZEND_ME(CompiledModule, getPendingOperations, arginfo_module_getPendingOperations, ZEND_ACC_PUBLIC)
                                                                ZEND_ME(CompiledModule, cancelOperation, arginfo_module_cancelOperation, ZEND_ACC_PUBLIC)
                                                                    ZEND_ME(CompiledModule, cleanup, arginfo_module_cleanup, ZEND_ACC_PUBLIC)
                                                                        ZEND_FE_END};

#endif