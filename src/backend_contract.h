#ifndef PHPTENSORS_BACKEND_CONTRACT_H
#define PHPTENSORS_BACKEND_CONTRACT_H

#include "php.h"

extern zend_class_entry *php_tensors_tensor_ce;
extern zend_class_entry *php_tensors_backend_ce;
extern zend_class_entry *php_tensors_numbers_ce;
extern zend_class_entry *php_tensors_exception_ce;
extern zend_class_entry *php_tensors_invalid_argument_exception_ce;
extern zend_class_entry *php_tensors_runtime_exception_ce;
extern zend_class_entry *php_tensors_out_of_memory_exception_ce;
extern zend_class_entry *php_tensors_compilation_exception_ce;

int php_tensors_backend_contract_init(void);

#endif
