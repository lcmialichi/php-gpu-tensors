#include "cuda_exceptions.h"
#include "module_ce.h"
#include "php.h"
#include "ca_struct.h"
#include "zend_interfaces.h"
#include "zend_exceptions.h"
#include "module_arginfo.h"
#include "kernel_types.h"
#include "tensor.h"
#include "fusion.h"
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include "ext/standard/php_standard.h"
#include "ast_cuda_types.h"
#include "nvidia_types.h"
#include <stdbool.h>

#define MAX_KERNEL_ERROR_LEN 1024
#define ERROR_CHECK_INTERVAL 50
#define ASYNC_OP_TIMEOUT_MS 30000
#define INITIAL_STREAM_POOL_SIZE 1
#define MAX_STREAM_POOL_SIZE 100
#define MIN_KERNEL_SIZE_FOR_ASYNC 256
#define LAUNCH_CACHE_SIZE 64
#define BATCH_STREAM_TIMEOUT 1000
#define ASYNC_OP_POOL_SIZE 32
#define MODULE_CACHE_KEY "__cuda_module"

#define MAX_KERNEL_ARGS 128

extern zend_class_entry *cuda_array_ce;

zend_class_entry *cuda_module_ce;
static zend_object_handlers module_handlers;
static pthread_mutex_t g_cuda_global_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_shared_context_mutex = PTHREAD_MUTEX_INITIALIZER;
static zend_bool g_cuda_initialized = 0;
static CUdevice g_primary_device = 0;
static CUcontext g_shared_context = NULL;
static int g_shared_context_refcount = 0;

static launch_config_cache_t g_launch_cache[LAUNCH_CACHE_SIZE];
static int g_launch_cache_index = 0;
static pthread_mutex_t g_launch_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static zend_object *module_create_object(zend_class_entry *class_type);
static void module_free_object(zend_object *object);
static zend_bool module_initialize_cuda_context(cuda_module_object *module);
static void module_cleanup_cuda_resources(cuda_module_object *module);
static double module_get_current_time_ms(void);
static const char *module_dtype_to_string(dtype_t dtype);
static CUmodule module_get_or_load_module_cached(cuda_module_object *module, zend_string *kernel_name);
static zend_bool module_validate_launch_config(cuda_module_object *module, int grid[3], int block[3]);
static zend_bool module_validate_launch_config_cached(int grid[3], int block[3]);

// Nova assinatura: Buffer na stack
static zend_bool module_prepare_cuda_arguments(cuda_kernel_data *kernel, HashTable *args_ht,
                                               void **cuda_args, uint64_t *scalar_args);

static zend_bool module_execute_cuda_kernel(cuda_module_object *module,
                                            cuda_kernel_data *kernel,
                                            int grid[3], int block[3],
                                            void **cuda_args,
                                            CUstream stream);
static void module_check_cuda_error(cuda_module_object *module, CUresult result, const char *context);

// Nova assinatura: Sem alocação dinâmica de args
static int module_create_async_operation(cuda_module_object *module,
                                         zend_string *kernel_name,
                                         int grid[3],
                                         int block[3]);

static void module_cleanup_async_operation_by_id(cuda_module_object *module, int op_id);
static void module_cleanup_all_async_operations(cuda_module_object *module);
static void module_cleanup_timeout_operations(cuda_module_object *module);
static zend_bool module_validate_tensor_access(tensor_t *tensor, int total_threads);
static void module_log_error(const char *format, ...);
static CUstream module_get_stream_with_expansion(cuda_module_object *module);
static void module_return_stream_to_pool(cuda_module_object *module, CUstream stream);
static void module_initialize_stream_pool(cuda_module_object *module);
static void module_initialize_stream_pool_progressive(cuda_module_object *module);
static void module_expand_stream_pool(cuda_module_object *module);
static void module_destroy_stream_pool(cuda_module_object *module);
static zend_bool module_initialize_global_cuda(cuda_module_object *module);
static void module_prepare_launch_config(zval *config_zv, int grid[3], int block[3]);
static zend_bool module_validate_async_operation_count(cuda_module_object *module);
static zend_bool module_get_shared_context(cuda_module_object *module);
static void module_hash_launch_config(int grid[3], int block[3], size_t *hash);
static void free_parameter_list(func_parameter_list_t *params);
static void free_kernel_data(cuda_kernel_data *kernel);
static void module_ensure_context_set(cuda_module_object *module);

static func_parameter *create_parameter_from_array(HashTable *param_ht)
{
    zval *name_zv, *dtype_zv, *second_dtype_zv;

    if (!(name_zv = zend_hash_str_find(param_ht, "name", 4)) ||
        Z_TYPE_P(name_zv) != IS_STRING)
    {
        return NULL;
    }

    if (!(dtype_zv = zend_hash_str_find(param_ht, "dtype", 5)) ||
        Z_TYPE_P(dtype_zv) != IS_LONG)
    {
        return NULL;
    }

    func_parameter *param = (func_parameter *)emalloc(sizeof(func_parameter));
    if (!param)
    {
        return NULL;
    }
    memset(param, 0, sizeof(func_parameter));

    size_t name_len = Z_STRLEN_P(name_zv);
    if (name_len >= sizeof(param->name))
    {
        name_len = sizeof(param->name) - 1;
    }

    if (name_len > 0)
    {
        memcpy(param->name, Z_STRVAL_P(name_zv), name_len);
        param->name[name_len] = '\0';
    }

    param->dtype = (dtype_t)Z_LVAL_P(dtype_zv);

    if (param->dtype == DTYPE_LIST)
    {
        second_dtype_zv = zend_hash_str_find(param_ht, "second_dtype", 12);
        if (second_dtype_zv && Z_TYPE_P(second_dtype_zv) == IS_LONG)
        {
            param->second_dtype = (dtype_t)Z_LVAL_P(second_dtype_zv);
        }
    }

    return param;
}

static CUresult load_ptx(const char *ptx, CUmodule *module_out)
{
    CUresult result;

    result = cuModuleLoadData(module_out, ptx);
    if (result == CUDA_SUCCESS)
        return result;

    CUjit_option options[3];
    void *optionValues[3];

    options[0] = CU_JIT_FALLBACK_STRATEGY;
    optionValues[0] = (void *)CU_PREFER_PTX;

    options[1] = CU_JIT_OPTIMIZATION_LEVEL;
    optionValues[1] = (void *)4;

    options[2] = CU_JIT_TARGET_FROM_CUCONTEXT;
    optionValues[2] = (void *)1;

    result = cuModuleLoadDataEx(module_out, ptx,
                                3, options, optionValues);

    return result;
}

static void module_log_error(const char *format, ...)
{
    va_list args;
    char buffer[512];

    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    php_error_docref(NULL, E_WARNING, "[CUDA Module] %s", buffer);
}

static void module_check_cuda_error(cuda_module_object *module, CUresult result, const char *context)
{
    if (result != CUDA_SUCCESS)
    {
        const char *error_str = get_cuda_error_string(result);
        CUDA_THROW_RUNTIME(
                                "CUDA error in %s: %s (code: %d)",
                                context,
                                error_str ? error_str : "Unknown error",
                                result);
    }
}

static double module_get_current_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

static const char *module_dtype_to_string(dtype_t dtype)
{
    switch (dtype)
    {
    case DTYPE_FLOAT32: return "float32";
    case DTYPE_FLOAT64: return "float64";
    case DTYPE_INT8: return "int8";
    case DTYPE_INT16: return "int16";
    case DTYPE_INT32: return "int32";
    case DTYPE_INT64: return "int64";
    case DTYPE_UINT8: return "uint8";
    case DTYPE_UINT16: return "uint16";
    case DTYPE_UINT32: return "uint32";
    case DTYPE_UINT64: return "uint64";
    case DTYPE_BOOL: return "bool";
    case DTYPE_LIST: return "array";
    default: return "unknown";
    }
}

static void module_hash_launch_config(int grid[3], int block[3], size_t *hash)
{
    *hash = ((size_t)grid[0] << 32) | (grid[1] << 16) | grid[2];
    *hash ^= ((size_t)block[0] << 32) | (block[1] << 16) | block[2];
}

static zend_bool module_validate_launch_config_cached(int grid[3], int block[3])
{
    size_t hash;
    module_hash_launch_config(grid, block, &hash);

    pthread_mutex_lock(&g_launch_cache_mutex);
    for (int i = 0; i < LAUNCH_CACHE_SIZE; i++)
    {
        if (g_launch_cache[i].hash == hash)
        {
            if (g_launch_cache[i].grid[0] == grid[0] && g_launch_cache[i].block[0] == block[0])
            {
                zend_bool valid = g_launch_cache[i].valid;
                pthread_mutex_unlock(&g_launch_cache_mutex);
                return valid;
            }
        }
    }

    zend_bool valid = module_validate_launch_config(NULL, grid, block);

    int idx = g_launch_cache_index;
    g_launch_cache[idx].hash = hash;
    memcpy(g_launch_cache[idx].grid, grid, sizeof(int) * 3);
    memcpy(g_launch_cache[idx].block, block, sizeof(int) * 3);
    g_launch_cache[idx].valid = valid;

    g_launch_cache_index = (idx + 1) % LAUNCH_CACHE_SIZE;

    pthread_mutex_unlock(&g_launch_cache_mutex);
    return valid;
}

static zend_bool module_validate_tensor_access(tensor_t *tensor, int total_threads)
{
    if (!tensor)
        return 0;

    size_t tensor_elements = 1;
    for (int i = 0; i < tensor->ndims; i++)
    {
        tensor_elements *= tensor->shape[i];
    }

    if (total_threads > tensor_elements)
    {
        module_log_error("Potential out-of-bounds access: kernel configured for %d threads, "
                         "but tensor has only %zu elements",
                         total_threads, tensor_elements);
        return 0;
    }

    return 1;
}

static void module_prepare_launch_config(zval *config_zv, int grid[3], int block[3])
{
    grid[0] = grid[1] = grid[2] = 1;
    block[0] = block[1] = block[2] = 1;

    if (config_zv && Z_TYPE_P(config_zv) == IS_ARRAY)
    {
        zval *grid_zv = zend_hash_str_find(Z_ARR_P(config_zv), "grid", sizeof("grid") - 1);
        zval *block_zv = zend_hash_str_find(Z_ARR_P(config_zv), "block", sizeof("block") - 1);

        if (grid_zv && Z_TYPE_P(grid_zv) == IS_ARRAY)
        {
            zval *x = zend_hash_index_find(Z_ARR_P(grid_zv), 0);
            zval *y = zend_hash_index_find(Z_ARR_P(grid_zv), 1);
            zval *z = zend_hash_index_find(Z_ARR_P(grid_zv), 2);
            if (x) grid[0] = zval_get_long(x);
            if (y) grid[1] = zval_get_long(y);
            if (z) grid[2] = zval_get_long(z);
        }

        if (block_zv && Z_TYPE_P(block_zv) == IS_ARRAY)
        {
            zval *x = zend_hash_index_find(Z_ARR_P(block_zv), 0);
            zval *y = zend_hash_index_find(Z_ARR_P(block_zv), 1);
            zval *z = zend_hash_index_find(Z_ARR_P(block_zv), 2);
            if (x) block[0] = zval_get_long(x);
            if (y) block[1] = zval_get_long(y);
            if (z) block[2] = zval_get_long(z);
        }
    }
}

static zend_bool module_initialize_global_cuda(cuda_module_object *module)
{
    pthread_mutex_lock(&g_cuda_global_init_mutex);

    if (g_cuda_initialized)
    {
        pthread_mutex_unlock(&g_cuda_global_init_mutex);
        return 1;
    }

    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS)
    {
        pthread_mutex_unlock(&g_cuda_global_init_mutex);
        module_log_error("Failed to initialize CUDA driver: %s",
                         get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuDeviceGet(&g_primary_device, 0);
    if (cu_result != CUDA_SUCCESS)
    {
        pthread_mutex_unlock(&g_cuda_global_init_mutex);
        module_log_error("Failed to get CUDA device: %s",
                         get_cuda_error_string(cu_result));
        return 0;
    }

    g_cuda_initialized = 1;
    pthread_mutex_unlock(&g_cuda_global_init_mutex);

    return 1;
}

static zend_bool module_get_shared_context(cuda_module_object *module)
{
    pthread_mutex_lock(&g_shared_context_mutex);

    if (!g_shared_context)
    {
        if (!module_initialize_global_cuda(module))
        {
            pthread_mutex_unlock(&g_shared_context_mutex);
            return 0;
        }

        unsigned int ctx_flags = CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST | CU_CTX_LMEM_RESIZE_TO_MAX;
        CUresult result = cuCtxCreate(&g_shared_context, ctx_flags, g_primary_device);
        if (result != CUDA_SUCCESS)
        {
            pthread_mutex_unlock(&g_shared_context_mutex);
            module_log_error("Failed to create shared CUDA context: %s", get_cuda_error_string(result));
            return 0;
        }
    }

    CUresult context_result = cuCtxSetCurrent(g_shared_context);
    if (context_result != CUDA_SUCCESS)
    {
        if (g_shared_context_refcount == 0 && g_shared_context)
        {
            cuCtxDestroy(g_shared_context);
            g_shared_context = NULL;
        }
        pthread_mutex_unlock(&g_shared_context_mutex);
        module_log_error("Failed to set shared CUDA context on current thread: %s",
                         get_cuda_error_string(context_result));
        return 0;
    }

    g_shared_context_refcount++;
    module->cu_context = g_shared_context;
    module->uses_shared_context = 1;

    pthread_mutex_unlock(&g_shared_context_mutex);

    if (!module->cu_stream)
    {
        CUresult stream_result = cuStreamCreate(&module->cu_stream, CU_STREAM_NON_BLOCKING);
        if (stream_result != CUDA_SUCCESS)
        {
            pthread_mutex_lock(&g_shared_context_mutex);
            g_shared_context_refcount--;
            if (g_shared_context_refcount == 0 && g_shared_context)
            {
                cuCtxDestroy(g_shared_context);
                g_shared_context = NULL;
            }
            pthread_mutex_unlock(&g_shared_context_mutex);
            module->cu_context = NULL;
            module->uses_shared_context = 0;
            return 0;
        }
    }

    return 1;
}

static void module_initialize_stream_pool(cuda_module_object *module)
{
    if (!module->stream_pool)
    {
        module->stream_pool = (stream_pool_t *)emalloc(sizeof(stream_pool_t));
        memset(module->stream_pool, 0, sizeof(stream_pool_t));
        pthread_mutex_init(&module->stream_pool->mutex, NULL);
    }

    if (module->stream_pool->streams)
        return;

    module->stream_pool->actives = 0;
    module->stream_pool->capacity = MAX_STREAM_POOL_SIZE;
    module->stream_pool->size = INITIAL_STREAM_POOL_SIZE;
    module->stream_pool->streams = (pooled_stream_t *)ecalloc(module->stream_pool->capacity, sizeof(pooled_stream_t));

    for (int i = 0; i < INITIAL_STREAM_POOL_SIZE; i++)
    {
        CUresult cu_result = cuStreamCreate(&module->stream_pool->streams[i].stream, CU_STREAM_NON_BLOCKING);
        if (cu_result != CUDA_SUCCESS)
        {
            module_log_error("Failed to create stream %d for pool: %s",
                             i, get_cuda_error_string(cu_result));
            module->stream_pool->streams[i].stream = NULL;
        }
        else
        {
            module->stream_pool->streams[i].in_use = 0;
            module->stream_pool->streams[i].last_used = module_get_current_time_ms();
            module->stream_pool->streams[i].context = module->cu_context;
        }
    }
}

static void module_initialize_stream_pool_progressive(cuda_module_object *module)
{
    if (!module->stream_pool)
    {
        module->stream_pool = (stream_pool_t *)emalloc(sizeof(stream_pool_t));
        memset(module->stream_pool, 0, sizeof(stream_pool_t));
        pthread_mutex_init(&module->stream_pool->mutex, NULL);

        module->stream_pool->expand_threshold = 0.8;
        module->stream_pool->expand_lock = 0;
        module->stream_pool->capacity = MAX_STREAM_POOL_SIZE;

        module->stream_pool->streams = (pooled_stream_t *)ecalloc(MAX_STREAM_POOL_SIZE, sizeof(pooled_stream_t));
    }

    if (module->stream_pool->size > 0 && module->stream_pool->streams[0].stream != NULL)
        return;

    module->stream_pool->actives = 0;
    module->stream_pool->size = 1;

    CUresult cu_result = cuStreamCreate(&module->stream_pool->streams[0].stream, CU_STREAM_NON_BLOCKING);

    if (cu_result != CUDA_SUCCESS)
    {
        module_log_error("Failed to create initial stream for pool: %s",
                         get_cuda_error_string(cu_result));
        module->stream_pool->streams[0].stream = NULL;
    }
    else
    {
        module->stream_pool->streams[0].in_use = 0;
        module->stream_pool->streams[0].last_used = module_get_current_time_ms();
        module->stream_pool->streams[0].context = module->cu_context;
    }
}

static void module_expand_stream_pool(cuda_module_object *module)
{
    CUcontext old_context = NULL;
    CUresult cu_result;

    cu_result = cuCtxPushCurrent(module->cu_context);
    if (cu_result != CUDA_SUCCESS)
    {
        module_log_error("Failed to set context in expansion thread: %s",
                         get_cuda_error_string(cu_result));
        pthread_mutex_lock(&module->stream_pool->mutex);
        module->stream_pool->expand_lock = 0;
        pthread_mutex_unlock(&module->stream_pool->mutex);
        return;
    }

    pthread_mutex_lock(&module->stream_pool->mutex);

    int current_size = module->stream_pool->size;
    int target_size = current_size + 1;

    if (target_size > module->stream_pool->capacity)
    {
        module->stream_pool->expand_lock = 0;
        pthread_mutex_unlock(&module->stream_pool->mutex);
        cuCtxPopCurrent(&old_context);
        return;
    }

    cu_result = cuStreamCreate(
        &module->stream_pool->streams[current_size].stream,
        CU_STREAM_NON_BLOCKING);

    if (cu_result == CUDA_SUCCESS)
    {
        module->stream_pool->streams[current_size].in_use = 0;
        module->stream_pool->streams[current_size].context = module->cu_context;
        module->stream_pool->streams[current_size].last_used = module_get_current_time_ms();
        module->stream_pool->size = target_size;
    }

    module->stream_pool->expand_lock = 0;
    pthread_mutex_unlock(&module->stream_pool->mutex);

    cuCtxPopCurrent(&old_context);
}

static zend_bool module_ensure_ptx_loaded(cuda_module_object *module)
{
    if (!module->ptx_code || module->ptx_size == 0)
    {
        CUDA_THROW_RUNTIME("No PTX code available");
        return 0;
    }

    if (module->ptx_size < 10 || strstr(module->ptx_code, ".version") == NULL)
    {
        CUDA_THROW_INVALID("Invalid PTX code format");
        return 0;
    }

    return 1;
}

static void module_ensure_context_set(cuda_module_object *module)
{
    if (!module->cu_context)
        return;

    CUcontext current = NULL;
    CUresult result = cuCtxGetCurrent(&current);
    if (result != CUDA_SUCCESS || current != module->cu_context)
    {
        result = cuCtxSetCurrent(module->cu_context);
        if (result != CUDA_SUCCESS)
        {
            module_check_cuda_error(module, result, "setting context");
        }
    }
}

static zend_bool module_ensure_cuda_initialized(cuda_module_object *module)
{
    if (module->cu_context)
    {
        module_ensure_context_set(module);
        return EG(exception) == NULL;
    }

    if (module_get_shared_context(module))
    {
        module_initialize_stream_pool_progressive(module);
        return 1;
    }

    if (!module_initialize_global_cuda(module))
    {
        CUDA_THROW_RUNTIME("Failed to initialize CUDA driver");
        return 0;
    }

    CUresult cu_result = cuDevicePrimaryCtxRetain(&module->cu_context, g_primary_device);
    if (cu_result != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME("Failed to retain primary context: %s", get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuCtxSetCurrent(module->cu_context);
    if (cu_result != CUDA_SUCCESS)
    {
        cuDevicePrimaryCtxRelease(g_primary_device);
        module->cu_context = NULL;
        CUDA_THROW_RUNTIME("Failed to set primary CUDA context: %s", get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuStreamCreate(&module->cu_stream, CU_STREAM_NON_BLOCKING);
    if (cu_result != CUDA_SUCCESS)
    {
        cuDevicePrimaryCtxRelease(g_primary_device);
        module->cu_context = NULL;
        CUDA_THROW_RUNTIME("Failed to create stream: %s", get_cuda_error_string(cu_result));
        return 0;
    }

    module_initialize_stream_pool_progressive(module);
    return 1;
}

static void module_destroy_stream_pool(cuda_module_object *module)
{
    if (!module->stream_pool)
        return;

    if (module->stream_pool->streams)
    {
        pthread_mutex_lock(&module->stream_pool->mutex);

        for (int i = 0; i < module->stream_pool->size; i++)
        {
            if (module->stream_pool->streams[i].stream)
            {
                CUresult result = cuStreamDestroy(module->stream_pool->streams[i].stream);
                if (result != CUDA_SUCCESS && result != CUDA_ERROR_INVALID_HANDLE)
                {
                    module_log_error("Failed to destroy stream from pool: %s",
                                     get_cuda_error_string(result));
                }
            }
        }

        module->stream_pool->actives = 0;
        efree(module->stream_pool->streams);
        module->stream_pool->streams = NULL;
        module->stream_pool->size = 0;
        module->stream_pool->capacity = 0;

        pthread_mutex_unlock(&module->stream_pool->mutex);
    }

    pthread_mutex_destroy(&module->stream_pool->mutex);
    efree(module->stream_pool);
    module->stream_pool = NULL;
}

static zend_bool module_validate_async_operation_count(cuda_module_object *module)
{
    if (!module->stream_pool)
        return 0;

    if (module->stream_pool->actives >= MAX_STREAM_POOL_SIZE)
    {
        return 0;
    }

    return 1;
}

static CUstream module_get_stream_with_expansion(cuda_module_object *module)
{
    if (!module->stream_pool)
        return NULL;

    CUstream stream = NULL;
    double current_time = module_get_current_time_ms();
    CUcontext old_context = NULL;
    CUresult cu_result;

    cu_result = cuCtxGetCurrent(&old_context);
    if (cu_result != CUDA_SUCCESS)
    {
        module_log_error("Failed to get current CUDA context: %s",
                         get_cuda_error_string(cu_result));
        return NULL;
    }

    zend_bool context_changed = 0;
    if (old_context != module->cu_context)
    {
        cu_result = cuCtxPushCurrent(module->cu_context);
        if (cu_result != CUDA_SUCCESS)
        {
            module_log_error("Failed to set module CUDA context: %s",
                             get_cuda_error_string(cu_result));
            return NULL;
        }
        context_changed = 1;
    }

    pthread_mutex_lock(&module->stream_pool->mutex);

    if ((float)module->stream_pool->actives / module->stream_pool->size >
            module->stream_pool->expand_threshold &&
        module->stream_pool->size < module->stream_pool->capacity &&
        !module->stream_pool->expand_lock)
    {
        module->stream_pool->expand_lock = 1;
        pthread_mutex_unlock(&module->stream_pool->mutex);

        module_expand_stream_pool(module);

        pthread_mutex_lock(&module->stream_pool->mutex);
    }

    for (int i = 0; i < module->stream_pool->size; i++)
    {
        if (module->stream_pool->streams[i].stream != NULL)
        {
            if (!module->stream_pool->streams[i].in_use)
            {
                module->stream_pool->streams[i].in_use = 1;
                module->stream_pool->streams[i].last_used = current_time;
                module->stream_pool->streams[i].context = module->cu_context;
                stream = module->stream_pool->streams[i].stream;
                break;
            }

            CUresult query_result = cuStreamQuery(module->stream_pool->streams[i].stream);
            
            if (query_result == CUDA_SUCCESS)
            {
                module->stream_pool->streams[i].in_use = 1;
                module->stream_pool->streams[i].last_used = current_time;
                module->stream_pool->streams[i].context = module->cu_context;
                stream = module->stream_pool->streams[i].stream;
                module->stream_pool->actives--; 
                break;
            }
            else if (query_result != CUDA_ERROR_NOT_READY)
            {
                module_log_error("Stream query error: %s",
                                 get_cuda_error_string(query_result));
                module->stream_pool->streams[i].in_use = 0;
                module->stream_pool->streams[i].stream = NULL;
                module->stream_pool->actives--;
            }
        }
    }

    if (!stream && module->stream_pool->size < module->stream_pool->capacity)
    {
        int idx = module->stream_pool->size;
        cu_result = cuStreamCreate(&module->stream_pool->streams[idx].stream,
                                   CU_STREAM_NON_BLOCKING);

        if (cu_result == CUDA_SUCCESS)
        {
            module->stream_pool->streams[idx].in_use = 1;
            module->stream_pool->streams[idx].last_used = current_time;
            module->stream_pool->streams[idx].context = module->cu_context;
            module->stream_pool->size++;
            stream = module->stream_pool->streams[idx].stream;
        }
        else
        {
            module_log_error("Failed to create new stream: %s",
                             get_cuda_error_string(cu_result));
        }
    }

    if (stream)
    {
        module->stream_pool->actives++;
    }

    pthread_mutex_unlock(&module->stream_pool->mutex);

    if (!stream && module->stream_pool->size > 0)
    {
        pthread_mutex_lock(&module->stream_pool->mutex);

        if (!module->stream_pool->streams[0].in_use)
        {
            module->stream_pool->streams[0].in_use = 1;
            module->stream_pool->streams[0].last_used = current_time;
            module->stream_pool->streams[0].context = module->cu_context;
            stream = module->stream_pool->streams[0].stream;
            module->stream_pool->actives++;
        }

        pthread_mutex_unlock(&module->stream_pool->mutex);
    }

    if (context_changed)
    {
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
    }

    return stream;
}

static void free_parameter_list(func_parameter_list_t *params)
{
    if (!params)
        return;

    if (params->parameters)
    {
        for (int i = 0; i < params->total; i++)
        {
            func_parameter *param = params->parameters[i];

            if (param)
            {
                efree(param);
                params->parameters[i] = NULL;
            }
        }

        efree(params->parameters);
    }

    efree(params);
}

static void free_kernel_data(cuda_kernel_data *kernel)
{
    if (!kernel)
        return;

    if (kernel->name)
    {
        zend_string_release(kernel->name);
    }

    if (kernel->parameters)
    {
        free_parameter_list(kernel->parameters);
    }

    if (kernel->cuda_code)
    {
        efree(kernel->cuda_code);
    }

    efree(kernel);
}

static void module_return_stream_to_pool(cuda_module_object *module, CUstream stream)
{
    if (!stream || !module->stream_pool)
        return;

    pthread_mutex_lock(&module->stream_pool->mutex);

    for (int i = 0; i < module->stream_pool->size; i++)
    {
        if (module->stream_pool->streams[i].stream == stream)
        {
            module->stream_pool->actives--;
            module->stream_pool->streams[i].in_use = 0;
            module->stream_pool->streams[i].last_used = module_get_current_time_ms();

            if (module->stream_pool->streams[i].context != module->cu_context)
            {
                module_log_error("Stream context mismatch, destroying stream");
                cuStreamDestroy(stream);
                module->stream_pool->streams[i].stream = NULL;
            }

            break;
        }
    }

    pthread_mutex_unlock(&module->stream_pool->mutex);
}

static zend_bool module_initialize_cuda_context(cuda_module_object *module)
{
    CUresult cu_result;

    if (!module_initialize_global_cuda(module))
    {
        CUDA_THROW_RUNTIME(
                                "Failed to initialize CUDA driver");
        return 0;
    }

    if (module->cu_context)
    {
        module_ensure_context_set(module);
        return 1;
    }

    if (module_get_shared_context(module))
    {
        module_initialize_stream_pool_progressive(module);
        return 1;
    }

    unsigned int ctx_flags = CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST;

    cu_result = cuCtxCreate(&module->cu_context, ctx_flags, g_primary_device);
    if (cu_result != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME(
                                "Failed to create CUDA context: %s",
                                get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuStreamCreate(&module->cu_stream, CU_STREAM_DEFAULT);
    if (cu_result != CUDA_SUCCESS)
    {
        module_check_cuda_error(module, cu_result, "creating default stream");
        cuCtxDestroy(module->cu_context);
        module->cu_context = NULL;
        return 0;
    }

    module_initialize_stream_pool_progressive(module);

    return 1;
}

static void module_cleanup_cuda_resources(cuda_module_object *module)
{
    CUresult cu_result;

    if (module->loaded_modules)
    {
        zend_string *key;
        CUmodule *cu_module_ptr;

        ZEND_HASH_FOREACH_STR_KEY_PTR(module->loaded_modules, key, cu_module_ptr)
        {
            if (cu_module_ptr && *cu_module_ptr)
            {
                cu_result = cuModuleUnload(*cu_module_ptr);
                if (cu_result != CUDA_SUCCESS && cu_result != CUDA_ERROR_INVALID_HANDLE)
                {
                    module_log_error("Failed to unload CUDA module: %s",
                                     get_cuda_error_string(cu_result));
                }
                efree(cu_module_ptr);
            }
        }
        ZEND_HASH_FOREACH_END();

        zend_hash_destroy(module->loaded_modules);
        efree(module->loaded_modules);
        module->loaded_modules = NULL;
    }

    if (module->cu_stream)
    {
        cu_result = cuStreamDestroy(module->cu_stream);
        if (cu_result != CUDA_SUCCESS && cu_result != CUDA_ERROR_INVALID_HANDLE)
        {
            module_log_error("Failed to destroy CUDA stream: %s",
                             get_cuda_error_string(cu_result));
        }
        module->cu_stream = NULL;
    }

    if (module->cu_context)
    {
        if (module->uses_shared_context)
        {
            pthread_mutex_lock(&g_shared_context_mutex);
            g_shared_context_refcount--;
            if (g_shared_context_refcount == 0 && g_shared_context)
            {
                cu_result = cuCtxDestroy(g_shared_context);
                if (cu_result != CUDA_SUCCESS && cu_result != CUDA_ERROR_INVALID_HANDLE)
                {
                    module_log_error("Failed to destroy shared CUDA context: %s",
                                     get_cuda_error_string(cu_result));
                }
                g_shared_context = NULL;
            }
            pthread_mutex_unlock(&g_shared_context_mutex);
            module->cu_context = NULL;
            module->uses_shared_context = 0;
        }
        else
        {
            cu_result = cuCtxDestroy(module->cu_context);
            if (cu_result != CUDA_SUCCESS && cu_result != CUDA_ERROR_INVALID_HANDLE)
            {
                module_log_error("Failed to destroy CUDA context: %s",
                                 get_cuda_error_string(cu_result));
            }
            module->cu_context = NULL;
        }
    }
}

static CUmodule module_get_or_load_module_cached(cuda_module_object *module, zend_string *kernel_name)
{
    CUmodule *cached_module = (CUmodule *)zend_hash_str_find_ptr(module->loaded_modules, MODULE_CACHE_KEY, sizeof(MODULE_CACHE_KEY) - 1);
    if (cached_module)
    {
        return *cached_module;
    }

    if (!module->ptx_code || module->ptx_size == 0)
    {
        CUDA_THROW_RUNTIME(
                                "No PTX code available for kernel '%s'",
                                ZSTR_VAL(kernel_name));
        return NULL;
    }

    CUmodule cu_module = NULL;
    CUresult result = load_ptx(module->ptx_code, &cu_module);

    if (result != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME(
                                "Failed to load PTX module for kernel '%s': %s",
                                ZSTR_VAL(kernel_name),
                                get_cuda_error_string(result));
        return NULL;
    }

    CUmodule *module_ptr = (CUmodule *)emalloc(sizeof(CUmodule));
    *module_ptr = cu_module;

    if (zend_hash_str_add_ptr(module->loaded_modules, MODULE_CACHE_KEY, sizeof(MODULE_CACHE_KEY) - 1, module_ptr) == NULL)
    {
        efree(module_ptr);
        cuModuleUnload(cu_module);
        CUDA_THROW_RUNTIME(
                                "Failed to cache module for kernel '%s'",
                                ZSTR_VAL(kernel_name));
        return NULL;
    }

    return cu_module;
}

static zend_bool module_validate_launch_config(cuda_module_object *module, int grid[3], int block[3])
{
    static int max_threads = 0, max_block[3], max_grid[3];
    static zend_bool limits_loaded = 0;

    if (!limits_loaded)
    {
        pthread_mutex_lock(&g_cuda_global_init_mutex);
        zend_bool cuda_initialized = g_cuda_initialized;
        CUdevice primary_device = g_primary_device;
        pthread_mutex_unlock(&g_cuda_global_init_mutex);

        if (!cuda_initialized)
        {
            return 0;
        }

        cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, primary_device);
        cuDeviceGetAttribute(&max_block[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, primary_device);
        cuDeviceGetAttribute(&max_block[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, primary_device);
        cuDeviceGetAttribute(&max_block[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, primary_device);
        cuDeviceGetAttribute(&max_grid[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, primary_device);
        cuDeviceGetAttribute(&max_grid[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, primary_device);
        cuDeviceGetAttribute(&max_grid[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, primary_device);
        limits_loaded = 1;
    }

    if (block[0] <= 0 || block[1] <= 0 || block[2] <= 0)
        return 0;

    if (grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0)
        return 0;

    if (block[0] > max_block[0] || block[1] > max_block[1] || block[2] > max_block[2])
        return 0;

    if (grid[0] > max_grid[0] || grid[1] > max_grid[1] || grid[2] > max_grid[2])
        return 0;

    if ((block[0] * block[1] * block[2]) > max_threads)
        return 0;

    return 1;
}

static void module_cleanup_timeout_operations(cuda_module_object *module)
{
    double current_time = module_get_current_time_ms();

    if (current_time - module->last_timeout_check_ms < ERROR_CHECK_INTERVAL)
        return;

    module->last_timeout_check_ms = current_time;

    zend_ulong num_idx;
    cuda_async_operation *op;

    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
    {
        if (op && op->is_active)
        {
            CUcontext old_context = NULL;
            CUresult cu_result;

            cu_result = cuCtxPushCurrent(op->context);
            if (cu_result != CUDA_SUCCESS)
            {
                module_log_error("Failed to set context for timeout check: %s",
                                 get_cuda_error_string(cu_result));
                continue;
            }

            CUresult query_result = cuStreamQuery(op->stream);

            if (query_result == CUDA_SUCCESS)
            {
                cuCtxPopCurrent(&old_context);
                module_cleanup_async_operation_by_id(module, op->id);
                continue;
            }
            else if (query_result != CUDA_ERROR_NOT_READY)
            {
                module_log_error("Stream query error during timeout check: %s",
                                 get_cuda_error_string(query_result));
            }

            double elapsed = current_time - op->start_time;
            if (elapsed > ASYNC_OP_TIMEOUT_MS)
            {
                CUresult sync_result = cuStreamSynchronize(op->stream);

                if (sync_result != CUDA_SUCCESS)
                {
                    module_log_error("Failed to synchronize stream after timeout: %s",
                                     get_cuda_error_string(sync_result));
                }

                cuCtxPopCurrent(&old_context);

                module_cleanup_async_operation_by_id(module, op->id);
                continue;
            }

            cuCtxPopCurrent(&old_context);
        }
    }
    ZEND_HASH_FOREACH_END();
}

static int module_create_async_operation(cuda_module_object *module,
                                         zend_string *kernel_name,
                                         int grid[3],
                                         int block[3])
{
    CUcontext old_context = NULL;
    CUresult cu_result;

    cu_result = cuCtxPushCurrent(module->cu_context);
    if (cu_result != CUDA_SUCCESS)
    {
        module_log_error("Failed to set context for async operation: %s", get_cuda_error_string(cu_result));
        return 0;
    }

    cuda_async_operation *op = (cuda_async_operation *)emalloc(sizeof(cuda_async_operation));
    if (!op)
    {
        cuCtxPopCurrent(&old_context);
        return 0;
    }

    memset(op, 0, sizeof(cuda_async_operation));

    op->id = module->next_async_op_id++;
    op->kernel_name = kernel_name ? zend_string_copy(kernel_name) : NULL;
    
    op->cuda_args = NULL;
    op->temp_buffers = NULL;
    op->temp_buffers_count = 0;
    op->argc = 0;
    ZVAL_UNDEF(&op->retained_args);
    
    op->is_active = 1;
    op->start_time = module_get_current_time_ms();

    memcpy(op->grid, grid, sizeof(int) * 3);
    memcpy(op->block, block, sizeof(int) * 3);

    op->stream = module_get_stream_with_expansion(module);
    if (!op->stream)
    {
        CUDA_THROW_RUNTIME(
                                "Failed to get CUDA stream for async operation");
        cuCtxPopCurrent(&old_context);
        if (op->kernel_name)
            zend_string_release(op->kernel_name);
        efree(op);
        return 0;
    }

    op->last_error = CUDA_SUCCESS;
    op->context = module->cu_context;
    memset(op->error_message, 0, sizeof(op->error_message));

    zend_hash_index_update_ptr(module->async_operations, op->id, op);

    cuCtxPopCurrent(&old_context);

    return op->id;
}

static void module_cleanup_async_operation_by_id(cuda_module_object *module, int op_id)
{
    cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
    if (!op)
    {
        return;
    }

    if (op->stream)
    {
        module_return_stream_to_pool(module, op->stream);
    }

    if (op->kernel_name)
    {
        zend_string_release(op->kernel_name);
    }

    if (!Z_ISUNDEF(op->retained_args))
    {
        zval_ptr_dtor(&op->retained_args);
    }

    zend_hash_index_del(module->async_operations, op_id);
    efree(op);
}

static void module_cleanup_all_async_operations(cuda_module_object *module)
{
    if (!module->async_operations)
        return;

    zend_ulong num_idx;
    cuda_async_operation *op;

    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
    {
        if (op)
        {
            if (op->is_active && op->stream)
            {
                CUcontext old_context = NULL;
                if (cuCtxPushCurrent(op->context) == CUDA_SUCCESS)
                {
                    cuStreamSynchronize(op->stream);
                    cuCtxPopCurrent(&old_context);
                }
            }
            module_cleanup_async_operation_by_id(module, op->id);
        }
    }
    ZEND_HASH_FOREACH_END();
}

static zend_bool module_prepare_cuda_arguments(cuda_kernel_data *kernel, HashTable *args_ht,
                                               void **cuda_args, uint64_t *scalar_args)
{
    int argc = args_ht ? zend_hash_num_elements(args_ht) : 0;
    int expected_args = kernel->parameters ? kernel->parameters->total : 0;

    if (expected_args != argc)
    {
        CUDA_THROW_INVALID(
                                "Kernel '%s' expects %d arguments, %d given",
                                kernel->name ? ZSTR_VAL(kernel->name) : "unknown",
                                expected_args, argc);
        return 0;
    }

    if (argc == 0) return 1;

    if (argc > MAX_KERNEL_ARGS)
    {
        CUDA_THROW_INVALID("Too many arguments for kernel (maximum %d)", MAX_KERNEL_ARGS);
        return 0;
    }

    int i = 0;
    zval *arg;
    
    ZEND_HASH_FOREACH_VAL(args_ht, arg)
    {
        func_parameter *param = kernel->parameters->parameters[i];
        
        if (param->dtype == DTYPE_LIST)
        {
            if (Z_TYPE_P(arg) != IS_OBJECT ||
                !instanceof_function(Z_OBJCE_P(arg), cuda_array_ce))
            {
                CUDA_THROW_INVALID(
                                        "Argument %d '%s' must be a CudaArray",
                                        i + 1, param->name);
                return 0;
            }

            cuda_array_obj *array_obj = Z_CUDA_ARRAY_P(arg);
            if (!array_obj->tensor_handle)
            {
                CUDA_THROW_INVALID(
                                        "Argument %d '%s': CudaArray has no tensor data",
                                        i + 1, param->name);
                return 0;
            }

            tensor_t *tensor = array_obj->tensor_handle;
            if (!fusion_check_mutation() || !fusion_materialize(tensor)) return 0;
            if (tensor->dtype != param->second_dtype)
            {
                const char *expected = module_dtype_to_string(param->second_dtype);
                const char *actual = module_dtype_to_string(tensor->dtype);
                CUDA_THROW_INVALID(
                                        "Argument %d '%s': expected dtype %s, got %s",
                                        i + 1, param->name, expected, actual);
                return 0;
            }

            cuda_args[i] = &tensor->data;
        }
        else
        {
            switch (param->dtype)
            {
            case DTYPE_INT8:
            {
                int8_t val = (int8_t)zval_get_long(arg);
                memcpy(&scalar_args[i], &val, sizeof(int8_t));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            case DTYPE_INT16:
            {
                int16_t val = (int16_t)zval_get_long(arg);
                memcpy(&scalar_args[i], &val, sizeof(int16_t));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            case DTYPE_INT32:
            {
                int32_t val = (int32_t)zval_get_long(arg);
                memcpy(&scalar_args[i], &val, sizeof(int32_t));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            case DTYPE_UINT8:
            {
                uint8_t val = (uint8_t)zval_get_long(arg);
                memcpy(&scalar_args[i], &val, sizeof(uint8_t));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            case DTYPE_UINT16:
            {
                uint16_t val = (uint16_t)zval_get_long(arg);
                memcpy(&scalar_args[i], &val, sizeof(uint16_t));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            case DTYPE_UINT32:
            {
                uint32_t val = (uint32_t)zval_get_long(arg);
                memcpy(&scalar_args[i], &val, sizeof(uint32_t));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            case DTYPE_FLOAT32:
            {
                float val = (float)zval_get_double(arg);
                memcpy(&scalar_args[i], &val, sizeof(float));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            case DTYPE_FLOAT64:
            {
                double val = zval_get_double(arg);
                memcpy(&scalar_args[i], &val, sizeof(double));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            case DTYPE_INT64:
            {
                int64_t val = (int64_t)zval_get_long(arg);
                memcpy(&scalar_args[i], &val, sizeof(int64_t));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            case DTYPE_UINT64:
            {
                uint64_t val = (uint64_t)zval_get_long(arg);
                memcpy(&scalar_args[i], &val, sizeof(uint64_t));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            case DTYPE_BOOL:
            {
                bool val = zval_is_true(arg) ? true : false;
                memcpy(&scalar_args[i], &val, sizeof(bool));
                cuda_args[i] = &scalar_args[i];
                break;
            }
            default:
                CUDA_THROW_INVALID("Unsupported scalar dtype for argument %d", i + 1);
                return 0;
            }
        }
        i++;
    }
    ZEND_HASH_FOREACH_END();

    return 1;
}

static zend_bool module_execute_cuda_kernel(cuda_module_object *module,
                                            cuda_kernel_data *kernel,
                                            int grid[3], int block[3],
                                            void **cuda_args,
                                            CUstream stream)
{
    CUresult cu_result;
    CUfunction cu_function;
    CUcontext old_context = NULL;

    cu_result = cuCtxPushCurrent(module->cu_context);
    if (cu_result != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME(
                                "Failed to set CUDA context for kernel execution: %s",
                                get_cuda_error_string(cu_result));
        return 0;
    }

    if (!module_validate_launch_config_cached(grid, block))
    {
        cuCtxPopCurrent(&old_context);
        CUDA_THROW_INVALID("Invalid grid/block configuration");
        return 0;
    }

    CUmodule cu_module = module_get_or_load_module_cached(module, kernel->name);
    if (!cu_module)
    {
        cuCtxPopCurrent(&old_context);
        return 0;
    }

    cu_result = cuModuleGetFunction(&cu_function, cu_module, ZSTR_VAL(kernel->name));
    if (cu_result != CUDA_SUCCESS)
    {
        cuCtxPopCurrent(&old_context);
        CUDA_THROW_RUNTIME(
                                "Failed to get kernel function '%s': %s",
                                ZSTR_VAL(kernel->name),
                                get_cuda_error_string(cu_result));
        return 0;
    }

    cu_result = cuLaunchKernel(cu_function,
                               grid[0], grid[1], grid[2],
                               block[0], block[1], block[2],
                               0,
                               stream,
                               cuda_args,
                               NULL);

    cuCtxPopCurrent(&old_context);

    if (cu_result != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME(
                                "Failed to launch kernel '%s': %s",
                                ZSTR_VAL(kernel->name),
                                get_cuda_error_string(cu_result));
        return 0;
    }

    return 1;
}

ZEND_METHOD(CompiledModule, initialize)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    if (!module_ensure_cuda_initialized(module))
    {
        RETURN_FALSE;
    }

    if (!module_ensure_ptx_loaded(module))
    {
        RETURN_FALSE;
    }

    RETURN_TRUE;
}

ZEND_METHOD(CompiledModule, autoGrid)
{
    char *kernel_name_str;
    size_t kernel_name_len;
    zval *z_input;
    zend_long total_elements = 0;

    ZEND_PARSE_PARAMETERS_START(2, 2)
    Z_PARAM_STRING(kernel_name_str, kernel_name_len)
    Z_PARAM_ZVAL(z_input)
    ZEND_PARSE_PARAMETERS_END();

    if (Z_TYPE_P(z_input) == IS_LONG)
    {
        total_elements = Z_LVAL_P(z_input);
    }
    else if (Z_TYPE_P(z_input) == IS_OBJECT && instanceof_function(Z_OBJCE_P(z_input), cuda_array_ce))
    {
        cuda_array_obj *ca_obj = Z_CUDA_ARRAY_P(z_input);
        if (!ca_obj->tensor_handle)
        {
            CUDA_THROW_INVALID("CudaArray has no tensor data");
            return;
        }
        total_elements = ca_obj->tensor_handle->total_size;
    }
    else
    {
        CUDA_THROW_INVALID("invalid parameter type: 'elements'");
        return;
    }

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    cuda_kernel_data *kernel = zend_hash_str_find_ptr(module->kernel_functions, kernel_name_str, kernel_name_len);

    if (!kernel)
    {
        CUDA_THROW_INVALID("Kernel '%s' not found", kernel_name_str);
        return;
    }

    if (!module_ensure_cuda_initialized(module))
    {
        CUDA_THROW_RUNTIME("Failed to initialize CUDA context");
        RETURN_FALSE;
    }

    CUmodule cu_module = module_get_or_load_module_cached(module, kernel->name);
    if (!cu_module)
    {
        return;
    }

    CUfunction cu_func;
    CUresult res = cuModuleGetFunction(&cu_func, cu_module, kernel_name_str);
    if (res != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME("Failed to get function: %s", get_cuda_error_string(res));
        return;
    }

    int min_grid_size, block_size;
    res = cuOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, cu_func, NULL, 0, 0);

    if (res != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME("Occupancy failed: %s", get_cuda_error_string(res));
        return;
    }

    int grid_size = (total_elements + block_size - 1) / block_size;

    array_init(return_value);

    zval grid_arr, block_arr;

    array_init(&grid_arr);
    add_next_index_long(&grid_arr, grid_size);
    add_next_index_long(&grid_arr, 1);
    add_next_index_long(&grid_arr, 1);
    add_assoc_zval(return_value, "grid", &grid_arr);

    array_init(&block_arr);
    add_next_index_long(&block_arr, block_size);
    add_next_index_long(&block_arr, 1);
    add_next_index_long(&block_arr, 1);
    add_assoc_zval(return_value, "block", &block_arr);
}

ZEND_METHOD(CompiledModule, launch)
{
    if (!fusion_check_mutation()) RETURN_THROWS();
    zend_string *kernel_name;
    zval *config_zv = NULL;
    zval *args_zv = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 3)
    Z_PARAM_STR(kernel_name)
    Z_PARAM_OPTIONAL
    Z_PARAM_ARRAY_OR_NULL(config_zv)
    Z_PARAM_ARRAY_OR_NULL(args_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    if (!module_ensure_cuda_initialized(module))
    {
        CUDA_THROW_RUNTIME("Failed to initialize CUDA context");
        RETURN_FALSE;
    }

    if (!module->ptx_code || module->ptx_size == 0)
    {
        CUDA_THROW_RUNTIME("No PTX code available");
        RETURN_FALSE;
    }

    cuda_kernel_data *kernel = zend_hash_find_ptr(module->kernel_functions, kernel_name);
    if (!kernel)
    {
        CUDA_THROW_INVALID(
                                "Kernel '%s' not found in compiled module",
                                ZSTR_VAL(kernel_name));
        RETURN_FALSE;
    }

    HashTable *args_ht = (args_zv && Z_TYPE_P(args_zv) == IS_ARRAY) ? Z_ARRVAL_P(args_zv) : NULL;

    int grid[3], block[3];
    module_prepare_launch_config(config_zv, grid, block);

    // 🔥 PRE-ALLOCATION NA STACK (Super rápido, não vai para a HEAP do PHP)
    void *cuda_args[MAX_KERNEL_ARGS];
    uint64_t scalar_args[MAX_KERNEL_ARGS];

    if (!module_prepare_cuda_arguments(kernel, args_ht, cuda_args, scalar_args))
    {
        RETURN_FALSE;
    }

    double start_time = module_get_current_time_ms();
    
    zend_bool success = module_execute_cuda_kernel(module, kernel, grid, block,
                                                   cuda_args, module->cu_stream);

    if (success)
    {
        // Nota: Um launch síncrono requer a sincronização aqui no final.
        CUresult cu_result = cuStreamSynchronize(module->cu_stream);
        if (cu_result != CUDA_SUCCESS)
        {
            module_check_cuda_error(module, cu_result, "kernel execution");
            success = 0;
        }
        else
        {
            module->kernel_execution_count++;
            module->total_execution_time_ms += (module_get_current_time_ms() - start_time);
        }
    }

    RETURN_BOOL(success);
}

ZEND_METHOD(CompiledModule, launchAsync)
{
    if (!fusion_check_mutation()) RETURN_THROWS();
    zend_string *kernel_name;
    zval *config_zv = NULL;
    zval *args_zv = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 3)
    Z_PARAM_STR(kernel_name)
    Z_PARAM_OPTIONAL
    Z_PARAM_ARRAY_OR_NULL(config_zv)
    Z_PARAM_ARRAY_OR_NULL(args_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    if (!module_ensure_cuda_initialized(module))
    {
        CUDA_THROW_RUNTIME("Failed to initialize CUDA context");
        RETURN_FALSE;
    }

    if (!module->ptx_code || module->ptx_size == 0)
    {
        CUDA_THROW_RUNTIME("No PTX code available");
        RETURN_FALSE;
    }

    if (!module_validate_async_operation_count(module))
    {
        module_cleanup_timeout_operations(module);
        
        if (!module_validate_async_operation_count(module))
        {
            pthread_mutex_lock(&module->stream_pool->mutex);
            CUstream oldest_stream = NULL;
            double oldest_time = module_get_current_time_ms();
            int oldest_idx = -1;

            for (int i = 0; i < module->stream_pool->size; i++) {
                if (module->stream_pool->streams[i].in_use && 
                    module->stream_pool->streams[i].last_used < oldest_time) {
                    oldest_time = module->stream_pool->streams[i].last_used;
                    oldest_stream = module->stream_pool->streams[i].stream;
                    oldest_idx = i;
                }
            }
            pthread_mutex_unlock(&module->stream_pool->mutex);

            if (oldest_stream) {
                cuStreamSynchronize(oldest_stream);
                
                zend_ulong num_idx;
                cuda_async_operation *op;
                ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op) {
                    if (op && op->stream == oldest_stream) {
                        module_cleanup_async_operation_by_id(module, op->id);
                        break;
                    }
                } ZEND_HASH_FOREACH_END();
            }
        }
    }

    cuda_kernel_data *kernel = zend_hash_find_ptr(module->kernel_functions, kernel_name);
    if (!kernel)
    {
        CUDA_THROW_INVALID(
                                "Kernel '%s' not found in compiled module",
                                ZSTR_VAL(kernel_name));
        RETURN_FALSE;
    }

    HashTable *args_ht = (args_zv && Z_TYPE_P(args_zv) == IS_ARRAY) ? Z_ARRVAL_P(args_zv) : NULL;

    int grid[3], block[3];
    module_prepare_launch_config(config_zv, grid, block);

    // 🔥 PRE-ALLOCATION NA STACK
    void *cuda_args[MAX_KERNEL_ARGS];
    uint64_t scalar_args[MAX_KERNEL_ARGS];

    if (!module_prepare_cuda_arguments(kernel, args_ht, cuda_args, scalar_args))
    {
        RETURN_FALSE;
    }

    int op_id = module_create_async_operation(module, kernel_name, grid, block);
    
    if (!op_id)
    {
        RETURN_FALSE;
    }

    cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
    if (!op)
    {
        RETURN_FALSE;
    }

    if (args_zv)
    {
        ZVAL_COPY(&op->retained_args, args_zv);
    }

    double start_time = module_get_current_time_ms();
    
    // O cuLaunchKernel faz a cópia do `cuda_args` para a VRAM/Command Buffer *AGORA* antes de retornar
    zend_bool success = module_execute_cuda_kernel(module, kernel, grid, block,
                                                   cuda_args, op->stream);

    if (success)
    {
        module->kernel_execution_count++;
        module->total_execution_time_ms += (module_get_current_time_ms() - start_time);
        module->has_pending_operations = 1;
        RETURN_LONG(op_id);
    }
    else
    {
        module_cleanup_async_operation_by_id(module, op_id);
        RETURN_FALSE;
    }
}

ZEND_METHOD(CompiledModule, launchAsyncBatch)
{
    if (!fusion_check_mutation()) RETURN_THROWS();
    zval *operations;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY(operations)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    if (!module_ensure_cuda_initialized(module))
    {
        CUDA_THROW_RUNTIME("Failed to initialize CUDA context");
        RETURN_FALSE;
    }

    CUcontext old_context = NULL;
    CUresult cu_result = cuCtxPushCurrent(module->cu_context);
    if (cu_result != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME(
                                "Failed to set CUDA context for batch operations: %s",
                                get_cuda_error_string(cu_result));
        RETURN_FALSE;
    }

    CUstream batch_stream = module_get_stream_with_expansion(module);
    if (!batch_stream)
    {
        cuCtxPopCurrent(&old_context);
        CUDA_THROW_RUNTIME("Failed to get stream for batch operations");
        RETURN_FALSE;
    }

    array_init(return_value);
    zend_bool batch_success = 1;

    zval *op_item;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(operations), op_item)
    {
        if (Z_TYPE_P(op_item) != IS_ARRAY)
        {
            add_next_index_bool(return_value, 0);
            batch_success = 0;
            continue;
        }

        zval *kernel_name_zv = zend_hash_str_find(Z_ARR_P(op_item), "kernel", sizeof("kernel") - 1);
        zval *args_zv = zend_hash_str_find(Z_ARR_P(op_item), "args", sizeof("args") - 1);
        zval *config_zv = zend_hash_str_find(Z_ARR_P(op_item), "config", sizeof("config") - 1);

        if (!kernel_name_zv || Z_TYPE_P(kernel_name_zv) != IS_STRING)
        {
            add_next_index_bool(return_value, 0);
            batch_success = 0;
            continue;
        }

        zend_string *kernel_name = Z_STR_P(kernel_name_zv);
        cuda_kernel_data *kernel = zend_hash_find_ptr(module->kernel_functions, kernel_name);

        if (!kernel)
        {
            add_next_index_bool(return_value, 0);
            batch_success = 0;
            continue;
        }

        int grid[3], block[3];
        module_prepare_launch_config(config_zv, grid, block);

        HashTable *args_ht = (args_zv && Z_TYPE_P(args_zv) == IS_ARRAY) ? Z_ARRVAL_P(args_zv) : NULL;

        // 🔥 PRE-ALLOCATION NA STACK DENTRO DO LOOP 
        void *cuda_args[MAX_KERNEL_ARGS];
        uint64_t scalar_args[MAX_KERNEL_ARGS];

        if (!module_prepare_cuda_arguments(kernel, args_ht, cuda_args, scalar_args))
        {
            add_next_index_bool(return_value, 0);
            batch_success = 0;
            continue;
        }

        CUmodule cu_module = module_get_or_load_module_cached(module, kernel_name);
        if (!cu_module)
        {
            add_next_index_bool(return_value, 0);
            batch_success = 0;
            continue;
        }

        CUfunction cu_function;
        if (cuModuleGetFunction(&cu_function, cu_module, ZSTR_VAL(kernel_name)) != CUDA_SUCCESS)
        {
            add_next_index_bool(return_value, 0);
            batch_success = 0;
            continue;
        }

        cu_result = cuLaunchKernel(cu_function,
                                   grid[0], grid[1], grid[2],
                                   block[0], block[1], block[2],
                                   0,
                                   batch_stream,
                                   cuda_args,
                                   NULL);

        if (cu_result == CUDA_SUCCESS)
        {
            add_next_index_bool(return_value, 1);
            module->kernel_execution_count++;
        }
        else
        {
            add_next_index_bool(return_value, 0);
            batch_success = 0;
            module_log_error("Batch kernel launch failed: %s", get_cuda_error_string(cu_result));
        }

    }
    ZEND_HASH_FOREACH_END();

    cu_result = cuStreamSynchronize(batch_stream);

    cuCtxPopCurrent(&old_context);

    if (cu_result != CUDA_SUCCESS)
    {
        module_check_cuda_error(module, cu_result, "batch stream synchronization");
        batch_success = 0;
    }

    module_return_stream_to_pool(module, batch_stream);

    RETURN_BOOL(batch_success);
}

ZEND_METHOD(CompiledModule, sync)
{
    zend_long op_id = -1;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(op_id)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    CUresult cu_result;

    if (op_id == -1)
    {
        module_cleanup_timeout_operations(module);
        zend_ulong num_idx;
        cuda_async_operation *op;
        zend_bool all_success = 1;

        ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
        {
            if (op && op->is_active)
            {
                CUcontext old_context = NULL;
                cuCtxPushCurrent(op->context);
                cu_result = cuStreamSynchronize(op->stream);
                cuCtxPopCurrent(&old_context);

                if (cu_result != CUDA_SUCCESS)
                {
                    module_check_cuda_error(module, cu_result, "stream synchronization");
                    all_success = 0;
                }

                module_cleanup_async_operation_by_id(module, op->id);
            }
        }
        ZEND_HASH_FOREACH_END();

        module->has_pending_operations = 0;
        RETURN_BOOL(all_success);
    }
    else
    {
        cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
        if (!op)
        {
            CUDA_THROW_INVALID("Async operation %ld not found", op_id);
            RETURN_FALSE;
        }

        if (op->is_active)
        {
            CUcontext old_context = NULL;
            cuCtxPushCurrent(op->context);
            cu_result = cuStreamSynchronize(op->stream);
            cuCtxPopCurrent(&old_context);

            if (cu_result != CUDA_SUCCESS)
            {
                module_check_cuda_error(module, cu_result, "stream synchronization");
                RETURN_FALSE;
            }
            op->is_active = 0;
        }

        zend_ulong num_idx;
        cuda_async_operation *check_op;
        module->has_pending_operations = 0;
        ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, check_op)
        {
            if (check_op && check_op->is_active)
            {
                module->has_pending_operations = 1;
                break;
            }
        }
        ZEND_HASH_FOREACH_END();

        RETURN_TRUE;
    }
}

ZEND_METHOD(CompiledModule, isFinished)
{
    zend_long op_id = -1;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(op_id)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    module_cleanup_timeout_operations(module);

    if (op_id == -1)
    {
        zend_ulong num_idx;
        cuda_async_operation *op;
        zend_bool all_finished = 1;

        ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
        {
            if (op && op->is_active)
            {
                CUcontext old_context = NULL;
                cuCtxPushCurrent(op->context);
                CUresult cu_result = cuStreamQuery(op->stream);
                cuCtxPopCurrent(&old_context);

                if (cu_result == CUDA_SUCCESS)
                {
                    op->is_active = 0;
                }
                else if (cu_result == CUDA_ERROR_NOT_READY)
                {
                    all_finished = 0;
                }
                else
                {
                    module_check_cuda_error(module, cu_result, "stream query");
                    op->is_active = 0;
                }
            }
        }
        ZEND_HASH_FOREACH_END();

        module->has_pending_operations = !all_finished;
        RETURN_BOOL(all_finished);
    }
    else
    {
        cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
        if (!op)
        {
            RETURN_TRUE;
        }

        if (!op->is_active)
        {
            RETURN_TRUE;
        }

        CUresult cu_result = cuStreamQuery(op->stream);

        if (cu_result == CUDA_SUCCESS)
        {
            op->is_active = 0;
            RETURN_TRUE;
        }
        else if (cu_result == CUDA_ERROR_NOT_READY)
        {
            RETURN_FALSE;
        }
        else
        {
            module_check_cuda_error(module, cu_result, "stream query");
            RETURN_FALSE;
        }
    }
}

ZEND_METHOD(CompiledModule, wait)
{
    zend_long op_id = -1;
    zend_long timeout_ms = -1;

    ZEND_PARSE_PARAMETERS_START(0, 2)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(op_id)
    Z_PARAM_LONG(timeout_ms)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    module_cleanup_timeout_operations(module);

    double start_time = module_get_current_time_ms();

    if (op_id == -1)
    {
        zend_bool all_completed = 1;

        while (1)
        {
            all_completed = 1;
            zend_ulong num_idx;
            cuda_async_operation *op;

            ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
            {
                if (op && op->is_active)
                {
                    CUresult cu_result = cuStreamQuery(op->stream);

                    if (cu_result == CUDA_SUCCESS)
                    {
                        module_cleanup_async_operation_by_id(module, op->id);
                        op->is_active = 0;
                    }
                    else if (cu_result == CUDA_ERROR_NOT_READY)
                    {
                        all_completed = 0;
                    }
                    else
                    {
                        module_check_cuda_error(module, cu_result, "stream query");
                        op->is_active = 0;
                    }
                }
            }
            ZEND_HASH_FOREACH_END();

            if (all_completed)
            {
                module->has_pending_operations = 0;
                RETURN_TRUE;
            }

            if (timeout_ms >= 0)
            {
                double elapsed = module_get_current_time_ms() - start_time;
                if (elapsed > timeout_ms)
                {
                    CUDA_THROW_RUNTIME(
                                            "Timeout waiting for all async operations after %.2f ms",
                                            elapsed);
                    RETURN_FALSE;
                }
            }

            usleep(1000);
        }
    }
    else
    {
        cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
        if (!op)
        {
            RETURN_TRUE;
        }

        if (!op->is_active)
        {
            RETURN_TRUE;
        }

        while (op->is_active)
        {
            CUresult cu_result = cuStreamQuery(op->stream);

            if (cu_result == CUDA_SUCCESS)
            {
                op->is_active = 0;
                RETURN_TRUE;
            }
            else if (cu_result != CUDA_ERROR_NOT_READY)
            {
                module_check_cuda_error(module, cu_result, "stream query");
                RETURN_FALSE;
            }

            if (timeout_ms >= 0)
            {
                double elapsed = module_get_current_time_ms() - start_time;
                if (elapsed > timeout_ms)
                {
                    CUDA_THROW_RUNTIME(
                                            "Timeout waiting for async operation %ld after %.2f ms",
                                            op_id, elapsed);
                    RETURN_FALSE;
                }
            }

            usleep(1000);
        }

        RETURN_TRUE;
    }
}

ZEND_METHOD(CompiledModule, hasKernel)
{
    zend_string *name;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    RETURN_BOOL(zend_hash_exists(module->kernel_functions, name));
}

ZEND_METHOD(CompiledModule, getKernels)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    array_init(return_value);

    zend_string *key;
    ZEND_HASH_FOREACH_STR_KEY(module->kernel_functions, key)
    {
        add_next_index_string(return_value, ZSTR_VAL(key));
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(CompiledModule, getPtx)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    if (module->ptx_code)
    {
        RETURN_STRING(module->ptx_code);
    }
    else
    {
        RETURN_NULL();
    }
}

ZEND_METHOD(CompiledModule, save)
{
    zend_string *filename;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_STR(filename)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    if (!module->ptx_code)
    {
        CUDA_THROW_RUNTIME("No PTX code available to save");
        RETURN_THROWS();
    }

    FILE *file = fopen(ZSTR_VAL(filename), "w");
    if (!file)
    {
        CUDA_THROW_RUNTIME("Failed to open file for writing: %s", ZSTR_VAL(filename));
        RETURN_THROWS();
    }

    size_t written = fwrite(module->ptx_code, 1, module->ptx_size, file);
    fclose(file);

    if (written != module->ptx_size)
    {
        CUDA_THROW_RUNTIME("Failed to write complete PTX file: %s", ZSTR_VAL(filename));
        RETURN_THROWS();
    }
    RETURN_TRUE;
}

ZEND_METHOD(CompiledModule, __serialize)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    if (!module)
        RETURN_EMPTY_ARRAY();

    array_init(return_value);

    if (module->ptx_code)
    {
        add_assoc_stringl(return_value, "ptx", module->ptx_code, module->ptx_size);
    }

    zval kernels_zv;
    array_init(&kernels_zv);

    if (module->kernel_functions)
    {
        zend_string *key;
        cuda_kernel_data *kernel;

        ZEND_HASH_FOREACH_STR_KEY_PTR(module->kernel_functions, key, kernel)
        {
            zval kernel_data;
            array_init(&kernel_data);

            add_assoc_str(&kernel_data, "name", zend_string_copy(kernel->name));
            add_assoc_long(&kernel_data, "params_count", kernel->parameters ? kernel->parameters->total : 0);

            if (kernel->parameters && kernel->parameters->total > 0)
            {
                size_t p_size = sizeof(func_parameter);
                size_t total_bytes = kernel->parameters->total * p_size;

                zend_string *params_blob = zend_string_alloc(total_bytes, 0);
                char *dest = ZSTR_VAL(params_blob);

                for (int i = 0; i < kernel->parameters->total; i++)
                {
                    func_parameter *p = kernel->parameters->parameters[i];
                    if (p)
                    {
                        memcpy(dest + (i * p_size), p, p_size);
                    }
                }

                ZSTR_VAL(params_blob)[total_bytes] = '\0';
                add_assoc_str(&kernel_data, "params_blob", params_blob);
            }

            zend_hash_update(Z_ARRVAL(kernels_zv), key, &kernel_data);
        }
        ZEND_HASH_FOREACH_END();
    }

    add_assoc_zval(return_value, "kernels", &kernels_zv);
}

ZEND_METHOD(CompiledModule, __unserialize)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    HashTable *data;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();

    module->from_serialize = 1;
    zval *ptx_zv = zend_hash_str_find(data, "ptx", sizeof("ptx") - 1);
    if (ptx_zv && Z_TYPE_P(ptx_zv) == IS_STRING)
    {
        module->ptx_code = estrndup(Z_STRVAL_P(ptx_zv), Z_STRLEN_P(ptx_zv));
        module->ptx_size = Z_STRLEN_P(ptx_zv);
    }

    zval *kernels_zv = zend_hash_str_find(data, "kernels", sizeof("kernels") - 1);
    if (kernels_zv && Z_TYPE_P(kernels_zv) == IS_ARRAY)
    {

        ALLOC_HASHTABLE(module->kernel_functions);
        zend_hash_init(module->kernel_functions, zend_hash_num_elements(Z_ARRVAL_P(kernels_zv)), NULL, NULL, 0);

        zval *kernel_entry;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(kernels_zv), kernel_entry)
        {
            if (Z_TYPE_P(kernel_entry) != IS_ARRAY)
                continue;

            cuda_kernel_data *k = ecalloc(1, sizeof(cuda_kernel_data));

            zval *name_zv = zend_hash_str_find(Z_ARRVAL_P(kernel_entry), "name", sizeof("name") - 1);
            if (!name_zv || Z_TYPE_P(name_zv) != IS_STRING)
            {
                efree(k);
                continue;
            }

            k->name = zend_string_copy(Z_STR_P(name_zv));

            zval *blob_zv = zend_hash_str_find(Z_ARRVAL_P(kernel_entry), "params_blob", sizeof("params_blob") - 1);
            zval *count_zv = zend_hash_str_find(Z_ARRVAL_P(kernel_entry), "params_count", sizeof("params_count") - 1);

            uint32_t count = count_zv ? (uint32_t)zval_get_long(count_zv) : 0;
            k->parameters = ecalloc(1, sizeof(func_parameter_list_t));
            k->parameters->total = count;

            if (count > 0)
            {
                size_t p_size = sizeof(func_parameter);
                size_t expected_size = (size_t)count * p_size;

                if (!blob_zv || Z_TYPE_P(blob_zv) != IS_STRING || Z_STRLEN_P(blob_zv) != expected_size)
                {
                    free_kernel_data(k);
                    continue;
                }

                k->parameters->parameters = ecalloc(count, sizeof(func_parameter *));

                char *src_ptr = Z_STRVAL_P(blob_zv);

                for (uint32_t i = 0; i < count; i++)
                {
                    k->parameters->parameters[i] = ecalloc(1, p_size);
                    memcpy(k->parameters->parameters[i], src_ptr + (i * p_size), p_size);
                }
            }

            zend_hash_add_ptr(module->kernel_functions, k->name, k);
        }
        ZEND_HASH_FOREACH_END();
    }

    ALLOC_HASHTABLE(module->loaded_modules);
    zend_hash_init(module->loaded_modules, 4, NULL, NULL, 0);
    ALLOC_HASHTABLE(module->async_operations);
    zend_hash_init(module->async_operations, 4, NULL, NULL, 0);
}

ZEND_METHOD(CompiledModule, getAsyncStatus)
{
    zend_long op_id = -1;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(op_id)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    module_cleanup_timeout_operations(module);

    if (op_id == -1)
    {
        array_init(return_value);

        zend_ulong num_idx;
        cuda_async_operation *op;

        ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
        {
            if (op)
            {
                zval op_info;
                array_init(&op_info);

                add_assoc_long(&op_info, "id", op->id);
                add_assoc_bool(&op_info, "is_active", op->is_active);
                if (op->kernel_name)
                {
                    add_assoc_string(&op_info, "kernel", ZSTR_VAL(op->kernel_name));
                }
                add_assoc_double(&op_info, "elapsed_ms",
                                 module_get_current_time_ms() - op->start_time);
                
                add_assoc_long(&op_info, "temp_buffers_count", 0); 

                add_index_zval(return_value, num_idx, &op_info);
            }
        }
        ZEND_HASH_FOREACH_END();
    }
    else
    {
        cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);

        if (!op)
        {
            RETURN_NULL();
        }

        array_init(return_value);
        add_assoc_long(return_value, "id", op->id);
        add_assoc_bool(return_value, "is_active", op->is_active);
        if (op->kernel_name)
        {
            add_assoc_string(return_value, "kernel", ZSTR_VAL(op->kernel_name));
        }
        add_assoc_double(return_value, "elapsed_ms",
                         module_get_current_time_ms() - op->start_time);
        add_assoc_long(return_value, "temp_buffers_count", 0);
    }
}

ZEND_METHOD(CompiledModule, getPendingOperations)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    module_cleanup_timeout_operations(module);

    array_init(return_value);

    zend_ulong num_idx;
    cuda_async_operation *op;

    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
    {
        if (op && op->is_active)
        {
            zval op_info;
            array_init(&op_info);

            add_assoc_long(&op_info, "id", op->id);
            if (op->kernel_name)
            {
                add_assoc_string(&op_info, "kernel", ZSTR_VAL(op->kernel_name));
            }
            add_assoc_double(&op_info, "elapsed_ms",
                             module_get_current_time_ms() - op->start_time);

            add_index_zval(return_value, num_idx, &op_info);
        }
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(CompiledModule, getStats)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    array_init(return_value);

    add_assoc_long(return_value, "kernel_execution_count", module->kernel_execution_count);
    add_assoc_double(return_value, "total_execution_time_ms", module->total_execution_time_ms);

    if (module->kernel_execution_count > 0)
    {
        add_assoc_double(return_value, "avg_execution_time_ms",
                         module->total_execution_time_ms / module->kernel_execution_count);
    }

    int pending_count = 0;
    zend_ulong num_idx;
    cuda_async_operation *op;

    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
    {
        if (op && op->is_active)
        {
            pending_count++;
        }
    }
    ZEND_HASH_FOREACH_END();

    add_assoc_long(return_value, "pending_operations", pending_count);
    add_assoc_long(return_value, "total_operations", zend_hash_num_elements(module->async_operations));
    add_assoc_bool(return_value, "has_pending_operations", module->has_pending_operations);
    add_assoc_long(return_value, "stream_pool_size", module->stream_pool ? module->stream_pool->size : 0);
    add_assoc_long(return_value, "stream_pool_capacity", module->stream_pool ? module->stream_pool->capacity : 0);

    if (pending_count > 0)
    {
        zval active_ops;
        array_init(&active_ops);

        ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
        {
            if (op && op->is_active)
            {
                zval op_info;
                array_init(&op_info);

                add_assoc_long(&op_info, "id", op->id);
                if (op->kernel_name)
                {
                    add_assoc_string(&op_info, "kernel", ZSTR_VAL(op->kernel_name));
                }
                add_assoc_double(&op_info, "elapsed_ms",
                                 module_get_current_time_ms() - op->start_time);

                add_next_index_zval(&active_ops, &op_info);
            }
        }
        ZEND_HASH_FOREACH_END();

        add_assoc_zval(return_value, "active_operations", &active_ops);
    }
}

ZEND_METHOD(CompiledModule, cancelOperation)
{
    zend_long op_id;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_LONG(op_id)
    ZEND_PARSE_PARAMETERS_END();

    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    cuda_async_operation *op = zend_hash_index_find_ptr(module->async_operations, op_id);
    if (!op)
    {
        CUDA_THROW_INVALID("Async operation %ld not found", op_id);
        RETURN_FALSE;
    }

    if (!op->is_active)
    {
        RETURN_TRUE;
    }

    CUcontext old_context = NULL;
    CUresult cu_result = cuCtxPushCurrent(op->context);
    if (cu_result == CUDA_SUCCESS)
    {
        cu_result = cuStreamSynchronize(op->stream);
        cuCtxPopCurrent(&old_context);
    }
    if (cu_result != CUDA_SUCCESS)
    {
        module_log_error("Failed to synchronize stream before cancellation: %s",
                         get_cuda_error_string(cu_result));
        RETURN_FALSE;
    }

    op->is_active = 0;

    zend_ulong num_idx;
    cuda_async_operation *check_op;
    module->has_pending_operations = 0;
    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, check_op)
    {
        if (check_op && check_op->is_active)
        {
            module->has_pending_operations = 1;
            break;
        }
    }
    ZEND_HASH_FOREACH_END();

    RETURN_TRUE;
}

ZEND_METHOD(CompiledModule, cleanup)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);

    zend_ulong num_idx;
    cuda_async_operation *op;
    int cleaned_count = 0;

    ZEND_HASH_FOREACH_NUM_KEY_PTR(module->async_operations, num_idx, op)
    {
        if (op && !op->is_active)
        {
            module_cleanup_async_operation_by_id(module, op->id);
            cleaned_count++;
        }
    }
    ZEND_HASH_FOREACH_END();

    RETURN_LONG(cleaned_count);
}

static void module_free_object(zend_object *object)
{
    cuda_module_object *module = Z_CUDA_MODULE_FROM_OBJ(object);

    if (module->init_data.init_in_progress)
    {
        pthread_mutex_lock(&module->init_data.init_mutex);
        if (module->init_data.init_in_progress)
        {
            pthread_join(module->init_data.init_thread, NULL);
        }
        pthread_mutex_unlock(&module->init_data.init_mutex);
    }

    pthread_mutex_destroy(&module->init_data.init_mutex);
    pthread_cond_destroy(&module->init_data.init_cond);

    if (module->async_operations)
    {
        module_cleanup_all_async_operations(module);
        zend_hash_destroy(module->async_operations);
        efree(module->async_operations);
        module->async_operations = NULL;
    }

    module_destroy_stream_pool(module);
    module_cleanup_cuda_resources(module);

    if (module->ptx_code)
    {
        efree(module->ptx_code);
        module->ptx_code = NULL;
    }

    if (module->kernel_functions)
    {
        cuda_kernel_data *kernel;
        ZEND_HASH_FOREACH_PTR(module->kernel_functions, kernel)
        {
            free_kernel_data(kernel);
        }
        ZEND_HASH_FOREACH_END();

        zend_hash_destroy(module->kernel_functions);
        efree(module->kernel_functions);
        module->kernel_functions = NULL;
    }

    if (module->functions)
    {
        zend_hash_destroy(module->functions);
        efree(module->functions);
        module->functions = NULL;
    }

    if (module->loaded_modules)
    {
        zend_hash_destroy(module->loaded_modules);
        efree(module->loaded_modules);
        module->loaded_modules = NULL;
    }

    zend_object_std_dtor(&module->std);
}

static zend_object *module_create_object(zend_class_entry *class_type)
{
    cuda_module_object *module = (cuda_module_object *)ecalloc(1, sizeof(cuda_module_object));

    zend_object_std_init(&module->std, class_type);
    module->std.handlers = &module_handlers;

    module->ptx_code = NULL;
    module->ptx_size = 0;
    module->functions = NULL;
    module->kernel_functions = NULL;
    module->cu_device = 0;
    module->cu_context = NULL;
    module->cu_stream = NULL;
    module->stream_pool = NULL;

    ALLOC_HASHTABLE(module->loaded_modules);
    zend_hash_init(module->loaded_modules, 8, NULL, NULL, 0);

    module->from_serialize = 0;

    module->has_pending_operations = 0;
    ALLOC_HASHTABLE(module->async_operations);
    zend_hash_init(module->async_operations, 8, NULL, NULL, 0);
    module->next_async_op_id = 1;

    module->total_memory_allocated = 0;
    module->peak_memory_usage = 0;
    module->kernel_execution_count = 0;
    module->total_execution_time_ms = 0.0;

    module->uses_shared_context = 0;

    module->init_data.init_in_progress = 0;
    module->init_data.init_complete = 0;
    pthread_mutex_init(&module->init_data.init_mutex, NULL);
    pthread_cond_init(&module->init_data.init_cond, NULL);

    return &module->std;
}

int module_init(void)
{
    zend_class_entry ce;

    INIT_CLASS_ENTRY(ce, MODULE_CLASS_NAME, module_methods);
    cuda_module_ce = zend_register_internal_class(&ce);
    cuda_module_ce->create_object = module_create_object;
    cuda_module_ce->ce_flags |= ZEND_ACC_FINAL;

    memcpy(&module_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    module_handlers.offset = XtOffsetOf(cuda_module_object, std);
    module_handlers.free_obj = module_free_object;

    return 1;
}

void module_shutdown(void)
{
    pthread_mutex_lock(&g_cuda_global_init_mutex);
    g_cuda_initialized = 0;
    pthread_mutex_unlock(&g_cuda_global_init_mutex);

    pthread_mutex_lock(&g_shared_context_mutex);
    if (g_shared_context)
    {
        cuCtxDestroy(g_shared_context);
        g_shared_context = NULL;
    }
    pthread_mutex_unlock(&g_shared_context_mutex);

    pthread_mutex_destroy(&g_cuda_global_init_mutex);
    pthread_mutex_destroy(&g_shared_context_mutex);
    pthread_mutex_destroy(&g_launch_cache_mutex);
}