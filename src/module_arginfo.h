#ifndef MODULE_ARGINFO_H
#define MODULE_ARGINFO_H

#include "php.h"
#include "api_arginfo.h"

#define MODULE_CLASS_NAME "PhpTensors\\CompiledModule"

static const zend_function_entry module_methods[] = {
    ZEND_ME(CompiledModule, initialize, arginfo_api_PhpTensors_CompiledModule_initialize, ZEND_ACC_PUBLIC)
    ZEND_ME(CompiledModule, autoGrid, arginfo_api_PhpTensors_CompiledModule_autoGrid, ZEND_ACC_PUBLIC)
        ZEND_ME(CompiledModule, launch, arginfo_api_PhpTensors_CompiledModule_launch, ZEND_ACC_PUBLIC)
            ZEND_ME(CompiledModule, launchAsync, arginfo_api_PhpTensors_CompiledModule_launchAsync, ZEND_ACC_PUBLIC)
            ZEND_ME(CompiledModule, launchAsyncBatch, arginfo_api_PhpTensors_CompiledModule_launchAsyncBatch, ZEND_ACC_PUBLIC)
                ZEND_ME(CompiledModule, sync, arginfo_api_PhpTensors_CompiledModule_sync, ZEND_ACC_PUBLIC)
                    ZEND_ME(CompiledModule, isFinished, arginfo_api_PhpTensors_CompiledModule_isFinished, ZEND_ACC_PUBLIC)
                        ZEND_ME(CompiledModule, wait, arginfo_api_PhpTensors_CompiledModule_wait, ZEND_ACC_PUBLIC)
                            ZEND_ME(CompiledModule, hasKernel, arginfo_api_PhpTensors_CompiledModule_hasKernel, ZEND_ACC_PUBLIC)
                                ZEND_ME(CompiledModule, getKernels, arginfo_api_PhpTensors_CompiledModule_getKernels, ZEND_ACC_PUBLIC)
                                    ZEND_ME(CompiledModule, getPtx, arginfo_api_PhpTensors_CompiledModule_getPtx, ZEND_ACC_PUBLIC)
                                        ZEND_ME(CompiledModule, save, arginfo_api_PhpTensors_CompiledModule_save, ZEND_ACC_PUBLIC)
                                            ZEND_ME(CompiledModule, __serialize, arginfo_api_PhpTensors_CompiledModule___serialize, ZEND_ACC_PUBLIC)
                                                ZEND_ME(CompiledModule, __unserialize, arginfo_api_PhpTensors_CompiledModule___unserialize, ZEND_ACC_PUBLIC)
                                                    ZEND_ME(CompiledModule, getAsyncStatus, arginfo_api_PhpTensors_CompiledModule_getAsyncStatus, ZEND_ACC_PUBLIC)
                                                        ZEND_ME(CompiledModule, getStats, arginfo_api_PhpTensors_CompiledModule_getStats, ZEND_ACC_PUBLIC)
                                                            ZEND_ME(CompiledModule, getPendingOperations, arginfo_api_PhpTensors_CompiledModule_getPendingOperations, ZEND_ACC_PUBLIC)
                                                                ZEND_ME(CompiledModule, cancelOperation, arginfo_api_PhpTensors_CompiledModule_cancelOperation, ZEND_ACC_PUBLIC)
                                                                    ZEND_ME(CompiledModule, cleanup, arginfo_api_PhpTensors_CompiledModule_cleanup, ZEND_ACC_PUBLIC)
                                                                        ZEND_FE_END};

#endif