#include "cuda_exceptions.h"
#include "backend_contract.h"

zend_class_entry *cuda_exception_ce = NULL;
zend_class_entry *cuda_runtime_exception_ce = NULL;
zend_class_entry *cuda_invalid_argument_exception_ce = NULL;
zend_class_entry *cuda_out_of_memory_exception_ce = NULL;
zend_class_entry *cuda_compilation_exception_ce = NULL;

void cuda_register_exceptions(void)
{
    cuda_exception_ce = php_tensors_exception_ce;
    cuda_runtime_exception_ce = php_tensors_runtime_exception_ce;
    cuda_invalid_argument_exception_ce = php_tensors_invalid_argument_exception_ce;
    cuda_out_of_memory_exception_ce = php_tensors_out_of_memory_exception_ce;
    cuda_compilation_exception_ce = php_tensors_compilation_exception_ce;
}