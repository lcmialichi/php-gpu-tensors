#include "php.h"
#include "backend_contract.h"
#include "zend_exceptions.h"

zend_class_entry *php_tensors_tensor_ce;
zend_class_entry *php_tensors_backend_ce;
zend_class_entry *php_tensors_numbers_ce;
zend_class_entry *php_tensors_exception_ce;
zend_class_entry *php_tensors_invalid_argument_exception_ce;
zend_class_entry *php_tensors_runtime_exception_ce;
zend_class_entry *php_tensors_unexpected_value_exception_ce;
zend_class_entry *php_tensors_out_of_memory_exception_ce;
zend_class_entry *php_tensors_compilation_exception_ce;

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_get_backend, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_get_shape, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_get_size, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_dtype, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_tensor_to_array, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_backend_get_name, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_backend_is_available, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_backend_supports, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_INFO(0, operation, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, context, IS_ARRAY, 0, "[]")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_backend_create_tensor, 0, 2, PhpTensors\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, values, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_backend_execute, 0, 2, PhpTensors\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, operation, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, inputs, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, arguments, IS_ARRAY, 0, "[]")
ZEND_END_ARG_INFO()

static const zend_function_entry tensor_interface_methods[] = {
    ZEND_ABSTRACT_ME(Tensor, getBackend, arginfo_tensor_get_backend)
    ZEND_ABSTRACT_ME(Tensor, getShape, arginfo_tensor_get_shape)
    ZEND_ABSTRACT_ME(Tensor, getSize, arginfo_tensor_get_size)
    ZEND_ABSTRACT_ME(Tensor, dtype, arginfo_tensor_dtype)
    ZEND_ABSTRACT_ME(Tensor, toArray, arginfo_tensor_to_array)
    ZEND_FE_END
};

static const zend_function_entry backend_interface_methods[] = {
    ZEND_ABSTRACT_ME(Backend, getName, arginfo_backend_get_name)
    ZEND_ABSTRACT_ME(Backend, isAvailable, arginfo_backend_is_available)
    ZEND_ABSTRACT_ME(Backend, supports, arginfo_backend_supports)
    ZEND_ABSTRACT_ME(Backend, createTensor, arginfo_backend_create_tensor)
    ZEND_ABSTRACT_ME(Backend, execute, arginfo_backend_execute)
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

    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\Numbers", sizeof("PhpTensors\\Numbers") - 1, NULL);
    php_tensors_numbers_ce = zend_register_internal_interface(&ce);
    return php_tensors_numbers_ce ? SUCCESS : FAILURE;
}
