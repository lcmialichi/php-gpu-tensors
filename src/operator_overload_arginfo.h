#ifndef PHPTENSORS_OPERATOR_OVERLOAD_ARGINFO_H
#define PHPTENSORS_OPERATOR_OVERLOAD_ARGINFO_H

#include "php.h"

ZEND_BEGIN_ARG_INFO_EX(arginfo_operator_overload_binary, 0, 0, 2)
    ZEND_ARG_INFO(0, left)
    ZEND_ARG_INFO(0, right)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_operator_overload_unary, 0, 0, 0)
ZEND_END_ARG_INFO()

static const zend_function_entry operator_overload_methods[] = {
    PHP_ABSTRACT_ME(OperatorOverload, __add, arginfo_operator_overload_binary)
    PHP_ABSTRACT_ME(OperatorOverload, __sub, arginfo_operator_overload_binary)
    PHP_ABSTRACT_ME(OperatorOverload, __mul, arginfo_operator_overload_binary)
    PHP_ABSTRACT_ME(OperatorOverload, __div, arginfo_operator_overload_binary)
    PHP_ABSTRACT_ME(OperatorOverload, __pow, arginfo_operator_overload_binary)
    PHP_ABSTRACT_ME(OperatorOverload, __mod, arginfo_operator_overload_binary)
    PHP_ABSTRACT_ME(OperatorOverload, __inc, arginfo_operator_overload_unary)
    PHP_ABSTRACT_ME(OperatorOverload, __dec, arginfo_operator_overload_unary)
    PHP_FE_END
};

#endif