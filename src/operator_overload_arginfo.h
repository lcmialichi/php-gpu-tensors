#ifndef PHPTENSORS_OPERATOR_OVERLOAD_ARGINFO_H
#define PHPTENSORS_OPERATOR_OVERLOAD_ARGINFO_H

#include "php.h"
#include "api_arginfo.h"


static const zend_function_entry operator_overload_methods[] = {
    ZEND_NAMED_ME(__add, zif_php_tensors_operator_unsupported, arginfo_api_PhpTensors_OperatorOverload___add, ZEND_ACC_PUBLIC)
    ZEND_NAMED_ME(__sub, zif_php_tensors_operator_unsupported, arginfo_api_PhpTensors_OperatorOverload___sub, ZEND_ACC_PUBLIC)
    ZEND_NAMED_ME(__mul, zif_php_tensors_operator_unsupported, arginfo_api_PhpTensors_OperatorOverload___mul, ZEND_ACC_PUBLIC)
    ZEND_NAMED_ME(__div, zif_php_tensors_operator_unsupported, arginfo_api_PhpTensors_OperatorOverload___div, ZEND_ACC_PUBLIC)
    ZEND_NAMED_ME(__pow, zif_php_tensors_operator_unsupported, arginfo_api_PhpTensors_OperatorOverload___pow, ZEND_ACC_PUBLIC)
    ZEND_NAMED_ME(__mod, zif_php_tensors_operator_unsupported, arginfo_api_PhpTensors_OperatorOverload___mod, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

#endif