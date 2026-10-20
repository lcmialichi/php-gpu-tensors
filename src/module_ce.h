#ifndef MODULE_H
#define MODULE_H

#include "php.h"
#include "kernel_types.h"

extern zend_class_entry *cuda_module_ce;

int module_init(void);
void module_shutdown(void);

ZEND_METHOD(CompiledModule, initialize);
ZEND_METHOD(CompiledModule, hasKernel);
ZEND_METHOD(CompiledModule, getKernels);
ZEND_METHOD(CompiledModule, getPtx);
ZEND_METHOD(CompiledModule, save);
ZEND_METHOD(CompiledModule, __serialize);
ZEND_METHOD(CompiledModule, __unserialize);
ZEND_METHOD(CompiledModule, autoGrid);
ZEND_METHOD(CompiledModule, launch);
ZEND_METHOD(CompiledModule, launchAsync);
ZEND_METHOD(CompiledModule, launchAsyncBatch);
ZEND_METHOD(CompiledModule, sync);
ZEND_METHOD(CompiledModule, isFinished);
ZEND_METHOD(CompiledModule, getStats);
ZEND_METHOD(CompiledModule, getAsyncStatus);
ZEND_METHOD(CompiledModule, getPendingOperations);
ZEND_METHOD(CompiledModule, cancelOperation);
ZEND_METHOD(CompiledModule, cleanup);
ZEND_METHOD(CompiledModule, wait);

#endif