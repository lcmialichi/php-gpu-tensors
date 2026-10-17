#include "php.h"
#include "zend_exceptions.h"
#include "zend_interfaces.h"
#include "backend_contract.h"
#include "cuda_globals.h"
#include <string.h>

#define PHP_TENSORS_BACKEND_HANDLE_KEY "\0PhpTensors\0backendHandle"

zend_class_entry *php_tensors_tensor_ce;
zend_class_entry *php_tensors_backend_ce;
zend_class_entry *php_tensors_numbers_ce;
static zend_class_entry *php_tensors_backend_manager_ce;
zend_class_entry *php_tensors_exception_ce;
zend_class_entry *php_tensors_invalid_argument_exception_ce;
zend_class_entry *php_tensors_runtime_exception_ce;
zend_class_entry *php_tensors_unexpected_value_exception_ce;
zend_class_entry *php_tensors_out_of_memory_exception_ce;
zend_class_entry *php_tensors_compilation_exception_ce;

static int backend_is_available(zval *backend);

static void backend_attach_handle(zval *tensor, zval *backend)
{
    zval backend_handle;
    ZVAL_COPY(&backend_handle, backend);
    zend_hash_str_update(Z_OBJPROP_P(tensor), PHP_TENSORS_BACKEND_HANDLE_KEY,
                         sizeof(PHP_TENSORS_BACKEND_HANDLE_KEY) - 1,
                         &backend_handle);
}

zend_result php_tensors_backend_attach_registered(zval *tensor, const char *name)
{
    if (!CUDA_G(custom_backends_initialized)) {
        zend_throw_exception(php_tensors_runtime_exception_ce,
                             "Backend registry is not initialized", 0);
        return FAILURE;
    }
    zval *backend = zend_hash_str_find(&CUDA_G(custom_backends), name, strlen(name));
    if (!backend) {
        zend_throw_exception_ex(php_tensors_invalid_argument_exception_ce, 0,
                                "Backend '%s' is not registered", name);
        return FAILURE;
    }
    backend_attach_handle(tensor, backend);
    return SUCCESS;
}

static int backend_invoke(zval *backend, const char *method, uint32_t param_count,
                          zval *params, zval *result)
{
    zval callable, object_copy;
    array_init_size(&callable, 2);
    ZVAL_COPY(&object_copy, backend);
    add_next_index_zval(&callable, &object_copy);
    add_next_index_string(&callable, method);

    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    if (zend_fcall_info_init(&callable, 0, &fci, &fcc, NULL, NULL) != SUCCESS)
    {
        zval_ptr_dtor(&callable);
        zend_throw_exception(php_tensors_runtime_exception_ce,
                             "Unable to invoke custom backend method", 0);
        return 0;
    }
    ZVAL_UNDEF(result);
    fci.retval = result;
    fci.params = params;
    fci.param_count = param_count;
    zend_result status = zend_call_function(&fci, &fcc);
    zval_ptr_dtor(&callable);
    if (status == FAILURE && !EG(exception))
    {
        zend_throw_exception(php_tensors_runtime_exception_ce,
                             "Custom backend method invocation failed", 0);
    }
    return status == SUCCESS && !EG(exception);
}

static zval *backend_lookup(zend_string *name)
{
    if (!CUDA_G(custom_backends_initialized))
    {
        zend_throw_exception(php_tensors_runtime_exception_ce, "Backend registry is not initialized", 0);
        return NULL;
    }
    zval *backend = zend_hash_find(&CUDA_G(custom_backends), name);
    if (!backend)
    {
        zend_throw_exception_ex(php_tensors_invalid_argument_exception_ce, 0,
                                "Backend '%s' is not registered", ZSTR_VAL(name));
        return NULL;
    }
    int available = backend_is_available(backend);
    if (available <= 0)
    {
        if (available == 0)
            zend_throw_exception_ex(php_tensors_runtime_exception_ce, 0,
                                    "Backend '%s' is no longer available", ZSTR_VAL(name));
        return NULL;
    }
    return backend;
}

static int backend_read_name(zval *backend, zend_string **name)
{
    zval result;
    ZVAL_UNDEF(&result);
    backend_invoke(backend, "getName", 0, NULL, &result);
    if (EG(exception))
    {
        if (!Z_ISUNDEF(result)) zval_ptr_dtor(&result);
        return 0;
    }
    if (Z_TYPE(result) != IS_STRING || Z_STRLEN(result) == 0)
    {
        zval_ptr_dtor(&result);
        zend_throw_exception(php_tensors_invalid_argument_exception_ce,
                             "Backend getName() must return a non-empty string", 0);
        return 0;
    }
    *name = zend_string_copy(Z_STR(result));
    zval_ptr_dtor(&result);
    return 1;
}

static int backend_is_available(zval *backend)
{
    zval result;
    ZVAL_UNDEF(&result);
    backend_invoke(backend, "isAvailable", 0, NULL, &result);
    if (EG(exception))
    {
        if (!Z_ISUNDEF(result)) zval_ptr_dtor(&result);
        return -1;
    }
    if (Z_TYPE(result) != IS_TRUE && Z_TYPE(result) != IS_FALSE)
    {
        zval_ptr_dtor(&result);
        zend_throw_exception(php_tensors_invalid_argument_exception_ce,
                             "Backend isAvailable() must return a boolean", 0);
        return -1;
    }
    int available = Z_TYPE(result) == IS_TRUE;
    zval_ptr_dtor(&result);
    return available;
}

static int backend_result_matches(zval *result, zend_string *backend_name)
{
    if (Z_TYPE_P(result) != IS_OBJECT ||
        !instanceof_function(Z_OBJCE_P(result), php_tensors_tensor_ce))
    {
        zend_throw_exception(php_tensors_unexpected_value_exception_ce,
                             "Backend operation must return a PhpTensors\\Tensor", 0);
        return 0;
    }

    zval actual_backend;
    ZVAL_UNDEF(&actual_backend);
    backend_invoke(result, "getBackend", 0, NULL, &actual_backend);
    if (EG(exception))
    {
        if (!Z_ISUNDEF(actual_backend)) zval_ptr_dtor(&actual_backend);
        return 0;
    }
    int matches = Z_TYPE(actual_backend) == IS_STRING &&
                  zend_string_equals(Z_STR(actual_backend), backend_name);
    zval_ptr_dtor(&actual_backend);
    if (!matches)
    {
        zend_throw_exception(php_tensors_unexpected_value_exception_ce,
                             "Backend operation returned a tensor owned by another backend", 0);
        return 0;
    }
    return 1;
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_backend_manager_register, 0, 0, 1)
    ZEND_ARG_INFO(0, backend)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_backend_manager_get, 0, 1, PhpTensors\\BackendInterface, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_backend_manager_available, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_backend_manager_unregister, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_backend_manager_create, 0, 3, PhpTensors\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, backend, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, values, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_backend_manager_for_tensor, 0, 1, PhpTensors\\BackendInterface, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, PhpTensors\\Tensor, 0)
ZEND_END_ARG_INFO()

ZEND_METHOD(PhpTensorsBackendManager, register)
{
    zval *backend;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(backend)
    ZEND_PARSE_PARAMETERS_END();

    if (Z_TYPE_P(backend) != IS_OBJECT ||
        !instanceof_function(Z_OBJCE_P(backend), php_tensors_backend_ce))
    {
        zend_throw_exception(php_tensors_invalid_argument_exception_ce,
                             "Backend must implement PhpTensors\\BackendInterface", 0);
        RETURN_THROWS();
    }

    zend_string *name = NULL;
    if (!backend_read_name(backend, &name)) RETURN_THROWS();
    if (zend_hash_exists(&CUDA_G(custom_backends), name))
    {
        zend_throw_exception_ex(php_tensors_invalid_argument_exception_ce, 0,
                                "Backend '%s' is already registered", ZSTR_VAL(name));
        zend_string_release(name);
        RETURN_THROWS();
    }
    int available = backend_is_available(backend);
    if (available < 0)
    {
        zend_string_release(name);
        RETURN_THROWS();
    }
    if (!available)
    {
        zend_throw_exception_ex(php_tensors_runtime_exception_ce, 0,
                                "Backend '%s' is not available", ZSTR_VAL(name));
        zend_string_release(name);
        RETURN_THROWS();
    }
    zval copy;
    ZVAL_COPY(&copy, backend);
    zend_hash_add(&CUDA_G(custom_backends), name, &copy);
    zend_string_release(name);
}

ZEND_METHOD(PhpTensorsBackendManager, get)
{
    zend_string *name;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();
    zval *backend = backend_lookup(name);
    if (!backend) RETURN_THROWS();
    RETURN_COPY(backend);
}

ZEND_METHOD(PhpTensorsBackendManager, available)
{
    ZEND_PARSE_PARAMETERS_NONE();
    array_init(return_value);
    zval *backend;
    zend_string *name;
    ZEND_HASH_FOREACH_STR_KEY_VAL(&CUDA_G(custom_backends), name, backend)
    {
        int available = backend_is_available(backend);
        if (available < 0)
        {
            zval_ptr_dtor(return_value);
            RETURN_THROWS();
        }
        if (available && name) add_next_index_str(return_value, zend_string_copy(name));
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(PhpTensorsBackendManager, unregister)
{
    zend_string *name;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();

    RETURN_BOOL(zend_hash_del(&CUDA_G(custom_backends), name) == SUCCESS);
}

static void backend_manager_create(INTERNAL_FUNCTION_PARAMETERS)
{
    zend_string *backend_name, *dtype = NULL;
    zval *values, *shape;
    ZEND_PARSE_PARAMETERS_START(3, 4)
        Z_PARAM_STR(backend_name)
        Z_PARAM_ARRAY(values)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();
    zval *backend = backend_lookup(backend_name);
    if (!backend) RETURN_THROWS();
    zval args[3], result;
    ZVAL_COPY(&args[0], values);
    ZVAL_COPY(&args[1], shape);
    if (dtype) ZVAL_STR_COPY(&args[2], dtype);
    else if (ZEND_NUM_ARGS() >= 4) ZVAL_NULL(&args[2]);
    else ZVAL_STRING(&args[2], "float32");
    ZVAL_UNDEF(&result);
    backend_invoke(backend, "createTensor", 3, args, &result);
    for (int i = 0; i < 3; i++) zval_ptr_dtor(&args[i]);
    if (EG(exception))
    {
        if (!Z_ISUNDEF(result)) zval_ptr_dtor(&result);
        RETURN_THROWS();
    }
    if (!backend_result_matches(&result, backend_name))
    {
        zval_ptr_dtor(&result);
        RETURN_THROWS();
    }
    backend_attach_handle(&result, backend);
    RETURN_ZVAL(&result, 0, 0);
}

ZEND_METHOD(PhpTensorsBackendManager, create)
{
    backend_manager_create(INTERNAL_FUNCTION_PARAM_PASSTHRU);
}

ZEND_METHOD(PhpTensorsBackendManager, createTensor)
{
    php_error_docref(NULL, E_DEPRECATED,
                     "BackendManager::createTensor() is deprecated; use create() instead.");
    backend_manager_create(INTERNAL_FUNCTION_PARAM_PASSTHRU);
}

ZEND_METHOD(PhpTensorsBackendManager, forTensor)
{
    zval *tensor;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor, php_tensors_tensor_ce)
    ZEND_PARSE_PARAMETERS_END();
    zval *backend = zend_hash_str_find(Z_OBJPROP_P(tensor),
                                      PHP_TENSORS_BACKEND_HANDLE_KEY,
                                      sizeof(PHP_TENSORS_BACKEND_HANDLE_KEY) - 1);
    if (!backend || Z_TYPE_P(backend) != IS_OBJECT ||
        !instanceof_function(Z_OBJCE_P(backend), php_tensors_backend_ce))
    {
        zend_throw_exception(php_tensors_invalid_argument_exception_ce,
                             "Tensor was not created by a registered backend factory", 0);
        RETURN_THROWS();
    }
    RETURN_COPY(backend);
}

static const zend_function_entry backend_manager_methods[] = {
    ZEND_ME(PhpTensorsBackendManager, register, arginfo_backend_manager_register, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(PhpTensorsBackendManager, unregister, arginfo_backend_manager_unregister, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(PhpTensorsBackendManager, get, arginfo_backend_manager_get, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(PhpTensorsBackendManager, available, arginfo_backend_manager_available, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(PhpTensorsBackendManager, create, arginfo_backend_manager_create, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(PhpTensorsBackendManager, createTensor, arginfo_backend_manager_create, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(PhpTensorsBackendManager, forTensor, arginfo_backend_manager_for_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_FE_END
};

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
    if (!php_tensors_exception_ce)
        return FAILURE;
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
        !php_tensors_compilation_exception_ce)
        return FAILURE;

    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\Tensor", sizeof("PhpTensors\\Tensor") - 1,
                        tensor_interface_methods);
    php_tensors_tensor_ce = zend_register_internal_interface(&ce);
    if (!php_tensors_tensor_ce)
        return FAILURE;

    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\BackendInterface",
                        sizeof("PhpTensors\\BackendInterface") - 1,
                        backend_interface_methods);
    php_tensors_backend_ce = zend_register_internal_interface(&ce);
    if (!php_tensors_backend_ce)
        return FAILURE;

    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\Numbers", sizeof("PhpTensors\\Numbers") - 1, NULL);
    php_tensors_numbers_ce = zend_register_internal_interface(&ce);
    if (!php_tensors_numbers_ce)
        return FAILURE;

    INIT_CLASS_ENTRY_EX(ce, "PhpTensors\\BackendManager",
                        sizeof("PhpTensors\\BackendManager") - 1,
                        backend_manager_methods);
    php_tensors_backend_manager_ce = zend_register_internal_class(&ce);
    if (php_tensors_backend_manager_ce)
        php_tensors_backend_manager_ce->ce_flags |= ZEND_ACC_FINAL;
    return php_tensors_backend_manager_ce ? SUCCESS : FAILURE;
}
