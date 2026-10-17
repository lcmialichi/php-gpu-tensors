#include "php.h"
#include "backend_contract.h"
#include "cpu_backend.h"
#include "cpu_host_array.h"
#include "cuda_globals.h"
#include "zend_exceptions.h"

static zend_class_entry *cpu_backend_ce;

static zend_result cpu_invoke(zval *callable, uint32_t param_count, zval *params, zval *result)
{
    ZVAL_UNDEF(result);
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    if (zend_fcall_info_init(callable, 0, &fci, &fcc, NULL, NULL) != SUCCESS) {
        zend_throw_exception(php_tensors_runtime_exception_ce,
                             "Unable to invoke CPU tensor operation", 0);
        return FAILURE;
    }
    fci.retval = result;
    fci.params = params;
    fci.param_count = param_count;
    return zend_call_function(&fci, &fcc);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_backend_name, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_backend_available, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_backend_supports, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_INFO(0, operation, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, context, IS_ARRAY, 0, "[]")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cpu_backend_create, 0, 2, PhpTensors\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, values, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cpu_backend_execute, 0, 2, PhpTensors\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, operation, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, inputs, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, arguments, IS_ARRAY, 0, "[]")
ZEND_END_ARG_INFO()

ZEND_METHOD(PhpTensorsCpuBackend, getName)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_STRING("cpu");
}

ZEND_METHOD(PhpTensorsCpuBackend, isAvailable)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_TRUE;
}

ZEND_METHOD(PhpTensorsCpuBackend, supports)
{
    zend_string *operation;
    HashTable *context = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_STR(operation)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT(context)
    ZEND_PARSE_PARAMETERS_END();
    (void)context;

    RETURN_BOOL(zend_string_equals_literal(operation, "add") ||
                zend_string_equals_literal(operation, "subtract") ||
                zend_string_equals_literal(operation, "multiply") ||
                zend_string_equals_literal(operation, "divide"));
}

ZEND_METHOD(PhpTensorsCpuBackend, createTensor)
{
    zval *values, *shape, args[3], callable, result;
    zend_string *dtype = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ARRAY(values)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();

    ZVAL_COPY(&args[0], values);
    ZVAL_COPY(&args[1], shape);
    if (dtype) ZVAL_STR_COPY(&args[2], dtype);
    else ZVAL_STRING(&args[2], "float32");
    array_init_size(&callable, 2);
    add_next_index_string(&callable, "PhpTensors\\HostArray");
    add_next_index_string(&callable, "fromFlatArray");
    zend_result status = cpu_invoke(&callable, 3, args, &result);
    zval_ptr_dtor(&callable);
    for (uint32_t i = 0; i < 3; i++) zval_ptr_dtor(&args[i]);
    if (status != SUCCESS || EG(exception)) {
        if (!Z_ISUNDEF(result)) zval_ptr_dtor(&result);
        RETURN_THROWS();
    }
    ZVAL_COPY_VALUE(return_value, &result);
    if (php_tensors_backend_attach_registered(return_value, "cpu") != SUCCESS) {
        zval_ptr_dtor(return_value);
        RETURN_THROWS();
    }
}

ZEND_METHOD(PhpTensorsCpuBackend, execute)
{
    zend_string *operation;
    zval *inputs, *arguments = NULL, *left, *right, callable, args[1], result;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_STR(operation)
        Z_PARAM_ARRAY(inputs)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(arguments)
    ZEND_PARSE_PARAMETERS_END();
    (void)arguments;

    if (!zend_array_is_list(Z_ARRVAL_P(inputs)) ||
        zend_hash_num_elements(Z_ARRVAL_P(inputs)) != 2) {
        zend_throw_exception(php_tensors_invalid_argument_exception_ce,
                             "CPU binary operations require exactly two inputs", 0);
        RETURN_THROWS();
    }
    if (arguments && zend_hash_num_elements(Z_ARRVAL_P(arguments)) != 0) {
        zend_throw_exception(php_tensors_invalid_argument_exception_ce,
                             "CPU binary operations do not accept additional arguments", 0);
        RETURN_THROWS();
    }

    if (!zend_string_equals_literal(operation, "add") &&
        !zend_string_equals_literal(operation, "subtract") &&
        !zend_string_equals_literal(operation, "multiply") &&
        !zend_string_equals_literal(operation, "divide")) {
        zend_throw_exception_ex(php_tensors_runtime_exception_ce, 0,
                                "CPU backend does not support operation '%s'",
                                ZSTR_VAL(operation));
        RETURN_THROWS();
    }
    left = zend_hash_index_find(Z_ARRVAL_P(inputs), 0);
    right = zend_hash_index_find(Z_ARRVAL_P(inputs), 1);
    if (!left || Z_TYPE_P(left) != IS_OBJECT ||
        !instanceof_function(Z_OBJCE_P(left), php_tensors_host_array_ce) || !right) {
        zend_throw_exception(php_tensors_invalid_argument_exception_ce,
                             "CPU binary operations require a HostArray and a right operand", 0);
        RETURN_THROWS();
    }
    array_init_size(&callable, 2);
    zval receiver;
    ZVAL_COPY(&receiver, left);
    add_next_index_zval(&callable, &receiver);
    add_next_index_str(&callable, zend_string_copy(operation));
    ZVAL_COPY(&args[0], right);
    zend_result status = cpu_invoke(&callable, 1, args, &result);
    zval_ptr_dtor(&callable);
    zval_ptr_dtor(&args[0]);
    if (status != SUCCESS || EG(exception)) {
        if (!Z_ISUNDEF(result)) zval_ptr_dtor(&result);
        RETURN_THROWS();
    }
    ZVAL_COPY_VALUE(return_value, &result);
    if (php_tensors_backend_attach_registered(return_value, "cpu") != SUCCESS) {
        zval_ptr_dtor(return_value);
        RETURN_THROWS();
    }
}

static const zend_function_entry cpu_backend_methods[] = {
    ZEND_ME(PhpTensorsCpuBackend, getName, arginfo_cpu_backend_name, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsCpuBackend, isAvailable, arginfo_cpu_backend_available, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsCpuBackend, supports, arginfo_cpu_backend_supports, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsCpuBackend, createTensor, arginfo_cpu_backend_create, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsCpuBackend, execute, arginfo_cpu_backend_execute, ZEND_ACC_PUBLIC)
    ZEND_FE_END
};

int cpu_backend_init(void)
{
    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "PhpTensors\\Internal", "CpuBackend", cpu_backend_methods);
    cpu_backend_ce = zend_register_internal_class(&ce);
    if (!cpu_backend_ce) {
        return FAILURE;
    }
    cpu_backend_ce->ce_flags |= ZEND_ACC_FINAL;
    zend_class_implements(cpu_backend_ce, 1, php_tensors_backend_ce);
    return SUCCESS;
}

int cpu_backend_register_request(void)
{
    zval backend;
    if (!CUDA_G(custom_backends_initialized)) {
        zend_throw_exception(php_tensors_runtime_exception_ce,
                             "Backend registry is not initialized", 0);
        return FAILURE;
    }
    object_init_ex(&backend, cpu_backend_ce);
    if (!zend_hash_str_add(&CUDA_G(custom_backends), "cpu", sizeof("cpu") - 1, &backend)) {
        zval_ptr_dtor(&backend);
        zend_throw_exception(php_tensors_runtime_exception_ce,
                             "Unable to register the built-in CPU backend", 0);
        return FAILURE;
    }
    return SUCCESS;
}
