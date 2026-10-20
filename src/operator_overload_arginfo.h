#ifndef PHPTENSORS_OPERATOR_OVERLOAD_ARGINFO_H
#define PHPTENSORS_OPERATOR_OVERLOAD_ARGINFO_H

#include "php.h"

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_operator_overload_binary, 0, 2, IS_MIXED, 0)
    ZEND_ARG_TYPE_INFO(0, left, IS_MIXED, 0)
    ZEND_ARG_TYPE_INFO(0, right, IS_MIXED, 0)
ZEND_END_ARG_INFO()

static const zend_function_entry operator_overload_methods[] = {
    ZEND_NAMED_ME(__add, zif_php_tensors_operator_unsupported, arginfo_operator_overload_binary, ZEND_ACC_PUBLIC)
    ZEND_NAMED_ME(__sub, zif_php_tensors_operator_unsupported, arginfo_operator_overload_binary, ZEND_ACC_PUBLIC)
    ZEND_NAMED_ME(__mul, zif_php_tensors_operator_unsupported, arginfo_operator_overload_binary, ZEND_ACC_PUBLIC)
    ZEND_NAMED_ME(__div, zif_php_tensors_operator_unsupported, arginfo_operator_overload_binary, ZEND_ACC_PUBLIC)
    ZEND_NAMED_ME(__pow, zif_php_tensors_operator_unsupported, arginfo_operator_overload_binary, ZEND_ACC_PUBLIC)
    ZEND_NAMED_ME(__mod, zif_php_tensors_operator_unsupported, arginfo_operator_overload_binary, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

#endif