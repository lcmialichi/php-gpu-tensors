#ifndef COMPILER_ARGINFO_H
#define COMPILER_ARGINFO_H

#include "php.h"

#define COMPILER_CLASS_NAME "PhpTensors\\Compiler"

ZEND_BEGIN_ARG_INFO_EX(arginfo_compiler_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, source, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, target, IS_STRING, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, optimization, IS_LONG, 0, "2")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, debug, _IS_BOOL, 0, "false")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, fast_math, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_compiler_kernel, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, parameters, IS_ARRAY, 1, "[]")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, headers, IS_ARRAY, 1, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_compiler_header, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, code, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_compiler_compile, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, optimize, _IS_BOOL, 0, "true")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, debug, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_compiler_get_kernels, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_compiler_getCacheStats, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_compiler_clearCache, 0, 0, 0)
ZEND_END_ARG_INFO()

static zend_function_entry compiler_methods[] = {
    ZEND_ME(Compiler, __construct, arginfo_compiler_construct, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, kernel, arginfo_compiler_kernel, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, header, arginfo_compiler_header, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, compile, arginfo_compiler_compile, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, getKernels, arginfo_compiler_get_kernels, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, getCacheStats, arginfo_compiler_getCacheStats, ZEND_ACC_PUBLIC)
    ZEND_ME(Compiler, clearCache, arginfo_compiler_clearCache, ZEND_ACC_PUBLIC)
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