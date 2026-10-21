#include "php.h"
#include "api_arginfo.h"
#include "backend_contract.h"
#include "zend_exceptions.h"

zend_class_entry *php_tensors_tensor_ce;
zend_class_entry *php_tensors_backend_ce;
zend_class_entry *php_tensors_operator_overloadable_ce;
zend_class_entry *php_tensors_exception_ce;
zend_class_entry *php_tensors_invalid_argument_exception_ce;
zend_class_entry *php_tensors_runtime_exception_ce;
zend_class_entry *php_tensors_unexpected_value_exception_ce;
zend_class_entry *php_tensors_out_of_memory_exception_ce;
zend_class_entry *php_tensors_compilation_exception_ce;

static const zend_function_entry tensor_interface_methods[] = {
    ZEND_ABSTRACT_ME(Tensor, getBackend, arginfo_api_PhpTensors_Tensor_getBackend)
    ZEND_ABSTRACT_ME(Tensor, getShape, arginfo_api_PhpTensors_Tensor_getShape)
    ZEND_ABSTRACT_ME(Tensor, getSize, arginfo_api_PhpTensors_Tensor_getSize)
    ZEND_ABSTRACT_ME(Tensor, dtype, arginfo_api_PhpTensors_Tensor_dtype)
    ZEND_ABSTRACT_ME(Tensor, toArray, arginfo_api_PhpTensors_Tensor_toArray)
    ZEND_FE_END
};

static const zend_function_entry backend_interface_methods[] = {
    ZEND_ABSTRACT_ME(Backend, getName, arginfo_api_PhpTensors_BackendInterface_getName)
    ZEND_ABSTRACT_ME(Backend, isAvailable, arginfo_api_PhpTensors_BackendInterface_isAvailable)
    ZEND_ABSTRACT_ME(Backend, supports, arginfo_api_PhpTensors_BackendInterface_supports)
    ZEND_ABSTRACT_ME(Backend, createTensor, arginfo_api_PhpTensors_BackendInterface_createTensor)
    ZEND_ABSTRACT_ME(Backend, execute, arginfo_api_PhpTensors_BackendInterface_execute)
    ZEND_FE_END
};

int php_tensors_backend_contract_init(void)
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\Exception", sizeof("PhpTensors\\Exception") - 1, NULL);
    php_tensors_exception_ce = zend_register_internal_class_ex(&ce, zend_ce_exception);
    if (!php_tensors_exception_ce) return FAILURE;

    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\InvalidArgumentException",
                        sizeof("PhpTensors\\InvalidArgumentException") - 1, NULL);
    php_tensors_invalid_argument_exception_ce =
        zend_register_internal_class_ex(&ce, php_tensors_exception_ce);
    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\RuntimeException",
                        sizeof("PhpTensors\\RuntimeException") - 1, NULL);
    php_tensors_runtime_exception_ce =
        zend_register_internal_class_ex(&ce, php_tensors_exception_ce);
    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\UnexpectedValueException",
                        sizeof("PhpTensors\\UnexpectedValueException") - 1, NULL);
    php_tensors_unexpected_value_exception_ce =
        zend_register_internal_class_ex(&ce, php_tensors_exception_ce);
    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\OutOfMemoryException",
                        sizeof("PhpTensors\\OutOfMemoryException") - 1, NULL);
    php_tensors_out_of_memory_exception_ce =
        zend_register_internal_class_ex(&ce, php_tensors_runtime_exception_ce);
    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\CompilationException",
                        sizeof("PhpTensors\\CompilationException") - 1, NULL);
    php_tensors_compilation_exception_ce =
        zend_register_internal_class_ex(&ce, php_tensors_runtime_exception_ce);
    if (!php_tensors_invalid_argument_exception_ce ||
        !php_tensors_runtime_exception_ce ||
        !php_tensors_unexpected_value_exception_ce ||
        !php_tensors_out_of_memory_exception_ce ||
        !php_tensors_compilation_exception_ce) {
        return FAILURE;
    }

    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\Tensor", sizeof("PhpTensors\\Tensor") - 1,
                        tensor_interface_methods);
    php_tensors_tensor_ce = zend_register_internal_interface(&ce);
    if (!php_tensors_tensor_ce) return FAILURE;

    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\BackendInterface",
                        sizeof("PhpTensors\\BackendInterface") - 1,
                        backend_interface_methods);
    php_tensors_backend_ce = zend_register_internal_interface(&ce);
    if (!php_tensors_backend_ce) return FAILURE;

    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\OperatorOverloadable",
                        sizeof("PhpTensors\\OperatorOverloadable") - 1, NULL);
    php_tensors_operator_overloadable_ce = zend_register_internal_interface(&ce);
    return php_tensors_operator_overloadable_ce ? SUCCESS : FAILURE;
}
