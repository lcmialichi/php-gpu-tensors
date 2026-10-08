#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "number_ce.h"
#include "php_ini.h"
#include "zend_interfaces.h"
#include "zend_attributes.h"
#include "cuda_wrapper.h"
#include "cuda_arginfo.h"
#include "cuda_array_ce.h"
#include "cuda.h"
#include "compiler_ce.h"
#include "module_ce.h"
#include "contiguous_array_ce.h"
#include "matmul_kernels.h"
#include "cuda_exceptions.h"
#include "fusion.h"
#include "backend_info.h"
#include "reduction_ops.h"
#include "nn.h"

ZEND_DECLARE_MODULE_GLOBALS(cuda);

static PHP_GINIT_FUNCTION(cuda);

static size_t parse_size_string(const char *str);

static PHP_INI_MH(OnUpdateMemSize)
{
    if (stage == PHP_INI_STAGE_RUNTIME)
    {
        php_error_docref(NULL, E_WARNING,
                         "Changing cuda.memory_size at runtime (via ini_set) has no effect.");
    }

    return SUCCESS;
}

PHP_INI_BEGIN()
STD_PHP_INI_ENTRY(
    "cuda.memory_size",
    "3G",
    PHP_INI_ALL,
    OnUpdateMemSize,
    memory_size,
    zend_cuda_globals,
    cuda_globals)
PHP_INI_END()

static PHP_GINIT_FUNCTION(cuda)
{
    cuda_globals->memory_size = "3G";
    cuda_globals->fusion_scope = NULL;
    cuda_globals->fusion_cache = NULL;
    cuda_globals->fusion_pending = 0;
}

PHP_MINIT_FUNCTION(cuda)
{
    REGISTER_INI_ENTRIES();
    cuda_register_exceptions();

    const char *ini_value = INI_STR("cuda.memory_size");
    size_t pool_size = parse_size_string(ini_value);

    int count = cuda_wrapper_get_device_count();
    if (count < 0)
    {
        php_error_docref(NULL, E_WARNING, "CUDA initialization failed");
    }

    register_number_class();
    compiler_init();
    module_init();
    contiguous_array_init();
    fusion_init();
    cuda_nn_init();

    if (!cuda_array_init(pool_size))
    {
        return FAILURE;
    }

    return SUCCESS;
}

ZEND_FUNCTION(cuda_get_device_count)
{
    int count = cuda_wrapper_get_device_count();

    if (count < 0)
    {
        zend_throw_exception(cuda_runtime_exception_ce, "Failed to get device count", 0);
        RETURN_THROWS();
    }

    RETURN_LONG(count);
}

ZEND_FUNCTION(cuda_get_backend_info)
{
    ZEND_PARSE_PARAMETERS_NONE();
    cuda_backend_info info = {0};
    cuda_blas_info(&info);
    cuda_reduction_info(&info);
    array_init(return_value);
    add_assoc_bool(return_value, "cublas", info.cublas);
    add_assoc_bool(return_value, "cublasLt", info.cublas_lt);
    add_assoc_bool(return_value, "cub", 1);
    add_assoc_bool(return_value, "cudnn", cuda_nn_available());
    add_assoc_string(return_value, "precision", (char *)info.precision);
    add_assoc_string(return_value, "lastMatmul", info.last_matmul);
    add_assoc_long(return_value, "lastBlasStatus", info.last_blas_status);
    add_assoc_long(return_value, "cublasCalls", info.blas_calls);
    add_assoc_long(return_value, "cublasLtCalls", info.lt_calls);
    add_assoc_long(return_value, "builtinMatmulCalls", info.builtin_matmul_calls);
    add_assoc_long(return_value, "cubReductionCalls", info.cub_calls);
    add_assoc_long(return_value, "coalescedReductionCalls", info.axis_calls);
    add_assoc_long(return_value, "genericReductionCalls", info.generic_reduce_calls);
}

ZEND_FUNCTION(cuda_set_matmul_precision)
{
    zend_string *precision;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_STR(precision)
    ZEND_PARSE_PARAMETERS_END();

    if (!fusion_check_mutation()) RETURN_THROWS();

    int mode;
    if (zend_string_equals_literal(precision, "fp32-strict"))
    {
        mode = CUDA_MATMUL_PRECISION_FP32_STRICT;
    }
    else if (zend_string_equals_literal(precision, "tf32"))
    {
        cuda_backend_info info = {0};
        cuda_blas_info(&info);
        if (!info.cublas)
        {
            CUDA_THROW_RUNTIME("TF32 matmul precision requires a cuBLAS-enabled build");
            RETURN_THROWS();
        }

        int device = cuda_wrapper_get_current_device();
        char name[256];
        int major, minor;
        size_t total_mem;
        if (device < 0 || cuda_wrapper_get_device_properties(
                device, name, sizeof(name), &major, &minor, &total_mem) != 1)
        {
            CUDA_THROW_RUNTIME("Failed to query the current CUDA device for TF32 support");
            RETURN_THROWS();
        }
        if (major < 8)
        {
            CUDA_THROW_RUNTIME("TF32 matmul precision requires compute capability 8.0 or newer");
            RETURN_THROWS();
        }
        mode = CUDA_MATMUL_PRECISION_TF32;
    }
    else
    {
        CUDA_THROW_INVALID("Matmul precision must be 'fp32-strict' or 'tf32'");
        RETURN_THROWS();
    }

    if (!cuda_blas_set_precision(mode))
    {
        CUDA_THROW_RUNTIME("Failed to set matmul precision");
        RETURN_THROWS();
    }
    RETURN_NULL();
}

ZEND_FUNCTION(cuda_get_device_info)
{
    zend_long device_id = 0;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(device_id)
    ZEND_PARSE_PARAMETERS_END();

    char name[256];
    int major, minor;
    size_t total_mem;

    int ok = cuda_wrapper_get_device_properties(
        device_id, name, sizeof(name),
        &major, &minor, &total_mem);

    if (ok != 1)
    {
        zend_throw_exception(cuda_runtime_exception_ce, "Failed to get device properties", 0);
        RETURN_THROWS();
    }

    array_init(return_value);
    add_assoc_string(return_value, "name", name);
    add_assoc_long(return_value, "compute_capability_major", major);
    add_assoc_long(return_value, "compute_capability_minor", minor);
    add_assoc_long(return_value, "total_global_memory", total_mem);
}

ZEND_FUNCTION(cuda_set_device)
{
    if (!fusion_check_mutation()) RETURN_THROWS();
    zend_long device_id;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_LONG(device_id)
    ZEND_PARSE_PARAMETERS_END();

    if (!cuda_wrapper_set_device((int)device_id))
    {
        zend_throw_exception_ex(cuda_runtime_exception_ce, 0, "Failed to set device %d", (int)device_id);
        RETURN_THROWS();
    }

    RETURN_TRUE;
}

ZEND_FUNCTION(cuda_get_current_device)
{
    int device = cuda_wrapper_get_current_device();

    if (device == -1)
    {
        zend_throw_exception(cuda_runtime_exception_ce, "Failed to get current device", 0);
        RETURN_THROWS();
    }

    RETURN_LONG(device);
}

ZEND_FUNCTION(cuda_get_memory_info)
{
    size_t free_mem, total_mem;

    if (!cuda_wrapper_get_memory_info(&free_mem, &total_mem))
    {
        zend_throw_exception(cuda_runtime_exception_ce, "Failed to get memory info", 0);
        RETURN_THROWS();
    }

    array_init(return_value);
    add_assoc_long(return_value, "free_memory", free_mem);
    add_assoc_long(return_value, "total_memory", total_mem);
    add_assoc_long(return_value, "used_memory", total_mem - free_mem);
    add_assoc_double(
        return_value, "usage_percentage",
        ((double)(total_mem - free_mem) / total_mem) * 100.0);
}

ZEND_FUNCTION(cuda_device_reset)
{
    if (!fusion_check_mutation()) RETURN_THROWS();
    cuda_blas_shutdown();
    cuda_reduction_shutdown();
    cuda_nn_shutdown();
    if (!cuda_wrapper_device_reset())
    {
        zend_throw_exception(cuda_runtime_exception_ce, "Failed to reset device", 0);
        RETURN_THROWS();
    }

    RETURN_TRUE;
}

ZEND_FUNCTION(cuda_get_driver_version)
{
    int version = cuda_wrapper_get_driver_version();

    if (version == -1)
    {
        zend_throw_exception(cuda_runtime_exception_ce, "Failed to get driver version", 0);
        RETURN_THROWS();
    }

    array_init(return_value);
    add_assoc_long(return_value, "version", version);

    char str[16];
    snprintf(str, sizeof(str), "%d.%d", version / 1000, (version % 100) / 10);
    add_assoc_string(return_value, "version_string", str);
}

ZEND_FUNCTION(cuda_get_runtime_version)
{
    int version = cuda_wrapper_get_runtime_version();

    if (version == -1)
    {
        zend_throw_exception(cuda_runtime_exception_ce, "Failed to get runtime version", 0);
        RETURN_THROWS();
    }

    array_init(return_value);
    add_assoc_long(return_value, "version", version);

    char str[16];
    snprintf(str, sizeof(str), "%d.%d", version / 1000, (version % 100) / 10);
    add_assoc_string(return_value, "version_string", str);
}

ZEND_FUNCTION(cuda_synchronize)
{
    if (!cuda_wrapper_synchronize())
    {
        zend_throw_exception(cuda_runtime_exception_ce, "Failed to synchronize device", 0);
        RETURN_THROWS();
    }

    RETURN_TRUE;
}

ZEND_FUNCTION(cuda_get_last_error)
{
    int error = cuda_wrapper_error();

    if (error == 0)
    {
        RETURN_NULL();
    }

    array_init(return_value);
    add_assoc_long(return_value, "code", 1);
    add_assoc_string(return_value, "error_message", cuda_wrapper_get_error_string(error));
    add_assoc_string(return_value, "error_type", cuda_wrapper_get_error_type(error));
}

ZEND_FUNCTION(cuda_clear_error)
{
    cuda_wrapper_error();
    RETURN_TRUE;
}

ZEND_FUNCTION(cuda_get_peer_access)
{
    zend_long device1, device2;

    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_LONG(device1)
    Z_PARAM_LONG(device2)
    ZEND_PARSE_PARAMETERS_END();

    int result = cuda_wrapper_get_peer_access((int)device1, (int)device2);

    if (result == -1)
    {
        zend_throw_exception(cuda_runtime_exception_ce, "Failed to check peer access", 0);
        RETURN_THROWS();
    }

    RETURN_BOOL(result);
}

static zend_function_entry cuda_functions[] = {
    PHP_FE(cuda_get_backend_info, arginfo_cuda_get_backend_info)
        PHP_FE(cuda_set_matmul_precision, arginfo_cuda_set_matmul_precision)
    PHP_FE(cuda_get_device_count, arginfo_cuda_get_device_count)
        PHP_FE(cuda_get_device_info, arginfo_cuda_get_device_info)
            PHP_FE(cuda_set_device, arginfo_cuda_set_device)
                PHP_FE(cuda_get_current_device, arginfo_cuda_get_current_device)
                    PHP_FE(cuda_get_memory_info, arginfo_cuda_get_memory_info)
                        PHP_FE(cuda_device_reset, arginfo_cuda_device_reset)
                            PHP_FE(cuda_synchronize, arginfo_cuda_synchronize)
                                PHP_FE(cuda_get_driver_version, arginfo_cuda_get_driver_version)
                                    PHP_FE(cuda_get_runtime_version, arginfo_cuda_get_runtime_version)
                                        PHP_FE(cuda_get_last_error, arginfo_cuda_get_last_error)
                                            PHP_FE(cuda_clear_error, arginfo_cuda_clear_error)
                                                PHP_FE(cuda_get_peer_access, arginfo_cuda_get_peer_access)
                                                    PHP_FE_END};

PHP_MSHUTDOWN_FUNCTION(cuda)
{
    cuda_blas_shutdown();
    cuda_reduction_shutdown();
    cuda_nn_shutdown();
    cuda_array_shutdown();
    cuda_wrapper_device_reset();
    return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(cuda)
{
    fusion_request_shutdown();
    cuda_blas_shutdown();
    cuda_blas_set_precision(CUDA_MATMUL_PRECISION_FP32_STRICT);
    cuda_reduction_shutdown();
    cuda_nn_shutdown();
    return SUCCESS;
}

PHP_RINIT_FUNCTION(cuda)
{
    return SUCCESS;
}

zend_module_entry cuda_module_entry = {
    STANDARD_MODULE_HEADER,
    PHP_CUDA_EXTNAME,
    cuda_functions,
    PHP_MINIT(cuda),
    PHP_MSHUTDOWN(cuda),
    PHP_RINIT(cuda),
    PHP_RSHUTDOWN(cuda),
    NULL,
    PHP_CUDA_VERSION,
    PHP_MODULE_GLOBALS(cuda),
    PHP_GINIT(cuda),
    NULL,
    NULL,
    STANDARD_MODULE_PROPERTIES_EX};

static size_t parse_size_string(const char *str)
{
    char unit = str[strlen(str) - 1];
    size_t n = atol(str);

    switch (unit)
    {
    case 'G':
    case 'g':
        return n * 1024ULL * 1024ULL * 1024ULL;
    case 'M':
    case 'm':
        return n * 1024ULL * 1024ULL;
    case 'K':
    case 'k':
        return n * 1024ULL;
    default:
        return n;
    }
}

#ifdef COMPILE_DL_CUDA
ZEND_GET_MODULE(cuda)
#endif
