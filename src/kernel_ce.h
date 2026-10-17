#ifndef KERNEL_CE_H
#define KERNEL_CE_H

#include "php.h"
#include "kernel_types.h"

#define KERNEL_CLASS_NAME "PhpTensors\\Kernel"

extern zend_class_entry *kernel_ce;

typedef struct _kernel_object
{
    zend_object std;
    char *ptx_code;
    size_t ptx_size;

    CUdevice cu_device;
    CUcontext cu_context;
    CUstream cu_stream;

    int from_serialize;

    zend_bool has_pending_operations;
    int next_async_op_id;
    HashTable *async_operations;
    size_t total_memory_allocated;
    size_t peak_memory_usage;
    int kernel_execution_count;
    double total_execution_time_ms;

    stream_pool_t *stream_pool;
    zend_bool uses_shared_context;
    
    async_init_t init_data;
    
    HashTable *config_cache;
    
    int stream_expansions;
    double init_time_ms;
    zend_bool is_warmed_up;

} kernel_object;


int kernel_init(void);

#endif