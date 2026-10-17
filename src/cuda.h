#ifndef CUDA_H
#define CUDA_H

#include "php.h"
#ifdef HAVE_CUDA
#include "operations.h"
#include <cuda_runtime.h>
#endif

#define PHP_PHPTENSORS_VERSION "0.1.0-beta.5"
#define PHP_PHPTENSORS_EXTNAME "phptensors"

extern zend_module_entry phptensors_module_entry;
#define phpext_phptensors_ptr &phptensors_module_entry

#ifdef ZTS
#include "TSRM.h"
#endif

ZEND_BEGIN_MODULE_GLOBALS(cuda)
char *memory_size;
char *legacy_memory_size;
struct fusion_scope *fusion_scope;
struct fusion_cache *fusion_cache;
size_t fusion_pending;
ZEND_END_MODULE_GLOBALS(cuda)

ZEND_EXTERN_MODULE_GLOBALS(cuda);

#ifdef ZTS
    #define CUDA_G(v) TSRMG(cuda_globals_id, zend_cuda_globals *, v)
#else
    extern zend_cuda_globals cuda_globals;
    #define CUDA_G(v) (cuda_globals.v)
#endif

ZEND_FUNCTION(cuda_get_device_count);
ZEND_FUNCTION(cuda_get_backend_info);
ZEND_FUNCTION(cuda_set_matmul_precision);
ZEND_FUNCTION(cuda_tune_matmul);
ZEND_FUNCTION(cuda_get_device_info);
ZEND_FUNCTION(cuda_set_device);
ZEND_FUNCTION(cuda_get_current_device);
ZEND_FUNCTION(cuda_get_memory_info);
ZEND_FUNCTION(cuda_device_reset);
ZEND_FUNCTION(cuda_get_driver_version);
ZEND_FUNCTION(cuda_get_runtime_version);
ZEND_FUNCTION(cuda_get_last_error);
ZEND_FUNCTION(cuda_clear_error);
ZEND_FUNCTION(cuda_get_peer_access);
ZEND_FUNCTION(cuda_synchronize);

PHP_MINIT_FUNCTION(cuda);
PHP_MSHUTDOWN_FUNCTION(cuda);

#endif
