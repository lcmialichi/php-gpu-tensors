#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include "php.h"
#include "zend_compile.h"
#include "data_types.h"
#include <pthread.h>
#include <time.h>
#include <nvrtc.h>
#include <cuda.h>

typedef struct
{
    const char *cuda_name;
    dtype_t return_type;
} cuda_function_match_t;

typedef struct _kernel_obj
{
    zend_object obj;
} kernel_obj;

typedef struct _cached_ptx
{
    char *ptx;
    size_t ptx_size;
    time_t timestamp;
} cached_ptx_t;

typedef struct _cuda_compiler_object
{
    char *global_source;
    size_t global_source_len;
    zend_object std;
    char *target_device;
    int target_auto_detected;
    int optimization_level;
    zend_bool debug_mode;
    zend_bool fast_math;
    zend_bool strict_math;
    HashTable *headers;
    HashTable *kernels;
    HashTable *devices;
    HashTable *ptx_cache;
} cuda_compiler_object;

typedef struct _cuda_async_operation
{
    int id;
    zend_string *kernel_name;
    void **cuda_args;
    void **temp_buffers;
    int temp_buffers_count;
    zval retained_args;
    zend_bool is_active;
    double start_time;

    CUstream stream;
    CUcontext context;
    CUevent start_event;
    CUevent end_event;

    int grid[3];
    int block[3];
    int argc;

    CUmodule *cu_module_cache;

    zend_bool owns_module;

    CUresult last_error;
    char error_message[256];

    struct _cuda_async_operation *next;
    struct _cuda_async_operation *prev;
} cuda_async_operation;

typedef struct {
    CUstream stream;
    CUcontext context;
    zend_bool in_use;
    double last_used;
} pooled_stream_t;

typedef struct {
    int actives;
    pooled_stream_t *streams;
    int size;
    int capacity;
    pthread_mutex_t mutex;
    float expand_threshold;
    int expand_lock;
} stream_pool_t;

typedef struct {
    int grid[3];
    int block[3];
    size_t hash;
    zend_bool valid;
} launch_config_cache_t;

typedef struct {
    pthread_t init_thread;
    zend_bool init_in_progress;
    zend_bool init_complete;
    pthread_mutex_t init_mutex;
    pthread_cond_t init_cond;
} async_init_t;

typedef struct {
    zend_string *kernel_name;
    void **cuda_args;
    void **temp_buffers;
    int temp_buffers_count;
    int grid[3];
    int block[3];
    int argc;
    void *kernel_data;
} batch_operation_t;

typedef struct _cuda_module_object
{
    zend_object std;
    char *ptx_code;
    size_t ptx_size;
    HashTable *functions;
    HashTable *kernel_functions;

    CUdevice cu_device;
    CUcontext cu_context;
    CUstream cu_stream;

    HashTable *loaded_modules;
    int from_serialize;

    zend_bool has_pending_operations;
    int next_async_op_id;
    HashTable *async_operations;
    size_t total_memory_allocated;
    size_t peak_memory_usage;
    int kernel_execution_count;
    double total_execution_time_ms;
    double last_timeout_check_ms;

    stream_pool_t *stream_pool;
    zend_bool uses_shared_context;
    
    async_init_t init_data;
    
    HashTable *config_cache;
    
    int stream_expansions;
    double init_time_ms;
    zend_bool is_warmed_up;

} cuda_module_object;

#define Z_CUDA_MODULE_P(zv) ((cuda_module_object *)((char *)Z_OBJ_P(zv) - XtOffsetOf(cuda_module_object, std)))
#define Z_CUDA_MODULE_FROM_OBJ(obj) ((cuda_module_object *)((char *)(obj) - XtOffsetOf(cuda_module_object, std)))
#define Z_CUDA_COMPILER_P(zv) ((cuda_compiler_object *)((char *)Z_OBJ_P(zv) - XtOffsetOf(cuda_compiler_object, std)))
#define Z_CUDA_COMPILER_FROM_OBJ(obj) ((cuda_compiler_object *)((char *)(obj) - XtOffsetOf(cuda_compiler_object, std)))

#endif