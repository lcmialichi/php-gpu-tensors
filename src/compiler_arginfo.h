#ifndef COMPILER_ARGINFO_H
#define COMPILER_ARGINFO_H

#include "php.h"
#include "api_arginfo.h"

#define COMPILER_CLASS_NAME "PhpTensors\\Compiler"

static zend_function_entry compiler_methods[] = {
    ZEND_ME(Compiler, __construct, arginfo_api_PhpTensors_Compiler___construct, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, kernel, arginfo_api_PhpTensors_Compiler_kernel, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, header, arginfo_api_PhpTensors_Compiler_header, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, compile, arginfo_api_PhpTensors_Compiler_compile, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, getKernels, arginfo_api_PhpTensors_Compiler_getKernels, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, getCacheStats, arginfo_api_PhpTensors_Compiler_getCacheStats, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, clearCache, arginfo_api_PhpTensors_Compiler_clearCache, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static zend_class_entry *register_compiler_class(void)
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, COMPILER_CLASS_NAME, compiler_methods);
    zend_class_entry *compiler_ce = zend_register_internal_class(&ce);
    compiler_ce->ce_flags |= ZEND_ACC_FINAL;
    return compiler_ce;
}

#endif