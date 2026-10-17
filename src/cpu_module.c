#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "backend_contract.h"
#include "cpu_host_array.h"
#include "cuda.h"
#include "cuda_exceptions.h"
#include "ext/standard/info.h"
#include "number_ce.h"
#include "zend_exceptions.h"

ZEND_DECLARE_MODULE_GLOBALS(cuda);

static zend_class_entry *unavailable_cuda_array_ce;

ZEND_BEGIN_ARG_INFO_EX(arginfo_unavailable_cuda_array_construct, 0, 0, 0)
    ZEND_ARG_VARIADIC_INFO(0, arguments)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_unavailable_cuda_array_call, 0, 2, IS_MIXED, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, arguments, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

static void throw_cuda_unavailable(void)
{
    zend_throw_exception(cuda_runtime_exception_ce,
                         "CUDA backend is unavailable in this CPU-only build", 0);
}

ZEND_METHOD(PhpTensorsUnavailableCudaArray, __construct)
{
    zval *arguments;
    uint32_t argument_count;
    ZEND_PARSE_PARAMETERS_START(0, -1)
        Z_PARAM_VARIADIC('*', arguments, argument_count)
    ZEND_PARSE_PARAMETERS_END();
    throw_cuda_unavailable();
    RETURN_THROWS();
}

ZEND_METHOD(PhpTensorsUnavailableCudaArray, __call)
{
    zend_string *name;
    HashTable *arguments;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STR(name)
        Z_PARAM_ARRAY_HT(arguments)
    ZEND_PARSE_PARAMETERS_END();
    throw_cuda_unavailable();
    RETURN_THROWS();
}

ZEND_METHOD(PhpTensorsUnavailableCudaArray, __callStatic)
{
    zend_string *name;
    HashTable *arguments;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STR(name)
        Z_PARAM_ARRAY_HT(arguments)
    ZEND_PARSE_PARAMETERS_END();
    throw_cuda_unavailable();
    RETURN_THROWS();
}

static const zend_function_entry unavailable_cuda_array_methods[] = {
    ZEND_ME(PhpTensorsUnavailableCudaArray, __construct, arginfo_unavailable_cuda_array_construct, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsUnavailableCudaArray, __call, arginfo_unavailable_cuda_array_call, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsUnavailableCudaArray, __callStatic, arginfo_unavailable_cuda_array_call, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_FE_END
};

static int register_unavailable_cuda_array(void)
{
    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "PhpTensors", "CudaArray", unavailable_cuda_array_methods);
    unavailable_cuda_array_ce = zend_register_internal_class(&ce);
    if (!unavailable_cuda_array_ce) {
        return FAILURE;
    }
    unavailable_cuda_array_ce->ce_flags |= ZEND_ACC_FINAL;
    zend_class_implements(unavailable_cuda_array_ce, 1, php_tensors_numbers_ce);
    return SUCCESS;
}

static PHP_GINIT_FUNCTION(cuda)
{
    cuda_globals->memory_size = "0";
    cuda_globals->legacy_memory_size = "";
    cuda_globals->fusion_scope = NULL;
    cuda_globals->fusion_cache = NULL;
    cuda_globals->fusion_pending = 0;
}

PHP_MINIT_FUNCTION(cuda)
{
    if (php_tensors_backend_contract_init() != SUCCESS) {
        return FAILURE;
    }
    cuda_register_exceptions();
    register_number_class();
    if (cpu_host_array_init() != SUCCESS ||
        register_unavailable_cuda_array() != SUCCESS) {
        return FAILURE;
    }
    return SUCCESS;
}

PHP_MINFO_FUNCTION(phptensors)
{
    php_info_print_table_start();
    php_info_print_table_header(2, "PHP Tensors support", "enabled");
    php_info_print_table_row(2, "Default backend", "CPU");
    php_info_print_table_row(2, "CUDA backend", "not built");
    php_info_print_table_end();
}

zend_module_entry phptensors_module_entry = {
    STANDARD_MODULE_HEADER,
    PHP_PHPTENSORS_EXTNAME,
    NULL,
    PHP_MINIT(cuda),
    NULL,
    NULL,
    NULL,
    PHP_MINFO(phptensors),
    PHP_PHPTENSORS_VERSION,
    PHP_MODULE_GLOBALS(cuda),
    PHP_GINIT(cuda),
    NULL,
    NULL,
    STANDARD_MODULE_PROPERTIES_EX};

#ifdef COMPILE_DL_PHPTENSORS
ZEND_GET_MODULE(phptensors)
#endif
