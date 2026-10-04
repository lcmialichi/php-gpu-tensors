#include "fusion_internal.h"
#include "ext/standard/md5.h"

#define FUSION_CACHE_ENTRIES 16
#define FUSION_CACHE_BYTES (16 * 1024 * 1024)

typedef struct
{
    zend_string *ptx;
    size_t used;
} fusion_cache_entry;

typedef struct fusion_cache
{
    HashTable entries;
    size_t bytes;
    size_t clock;
    size_t hits;
    size_t misses;
    size_t compilations;
    size_t evictions;
} fusion_cache;

static void fusion_cache_entry_free(zval *value)
{
    fusion_cache_entry *entry = Z_PTR_P(value);
    zend_string_release(entry->ptx);
    efree(entry);
}

static fusion_cache *fusion_cache_get(void)
{
    if (!CUDA_G(fusion_cache))
    {
        CUDA_G(fusion_cache) = ecalloc(1, sizeof(fusion_cache));
        zend_hash_init(&CUDA_G(fusion_cache)->entries, 16, NULL, fusion_cache_entry_free, 0);
    }
    return CUDA_G(fusion_cache);
}

void fusion_cache_shutdown(void)
{
    fusion_cache *cache = CUDA_G(fusion_cache);
    if (!cache) return;
    zend_hash_destroy(&cache->entries);
    efree(cache);
    CUDA_G(fusion_cache) = NULL;
}

void fusion_cache_stats(zval *result)
{
    fusion_cache *cache = fusion_cache_get();
    array_init(result);
    add_assoc_long(result, "entries", zend_hash_num_elements(&cache->entries));
    add_assoc_long(result, "bytes", cache->bytes);
    add_assoc_long(result, "hits", cache->hits);
    add_assoc_long(result, "misses", cache->misses);
    add_assoc_long(result, "compilations", cache->compilations);
    add_assoc_long(result, "evictions", cache->evictions);
    add_assoc_long(result, "maxEntries", FUSION_CACHE_ENTRIES);
    add_assoc_long(result, "maxBytes", FUSION_CACHE_BYTES);
}

int fusion_load_module(fusion_plan *plan, const char **names)
{
    struct cudaDeviceProp properties;
    int driver, runtime;
    cudaError_t error = cudaGetDeviceProperties(&properties, plan->device);
    if (error == cudaSuccess) error = cudaDriverGetVersion(&driver);
    if (error == cudaSuccess) error = cudaRuntimeGetVersion(&runtime);
    if (error != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Cannot determine fusion cache target: %s", cudaGetErrorString(error));
        return 0;
    }
    char target[128];
    int length = snprintf(target, sizeof(target), "fusion-v2:sm_%d%d:driver%d:runtime%d:fast:fmad0",
                          properties.major, properties.minor, driver, runtime);
    PHP_MD5_CTX hash;
    unsigned char digest[16];
    PHP_MD5Init(&hash);
    PHP_MD5Update(&hash, (unsigned char *)target, length);
    PHP_MD5Update(&hash, (unsigned char *)ZSTR_VAL(plan->source), ZSTR_LEN(plan->source));
    PHP_MD5Final(digest, &hash);
    zend_string *key = zend_string_init((char *)digest, sizeof(digest), 0);
    fusion_cache *cache = fusion_cache_get();
    fusion_cache_entry *entry = zend_hash_find_ptr(&cache->entries, key);
    zend_string *ptx = NULL;
    if (entry)
    {
        cache->hits++;
        entry->used = ++cache->clock;
        plan->cache_hit = 1;
        ptx = zend_string_copy(entry->ptx);
    }
    else
    {
        cache->misses++;
        zval module;
        ZVAL_UNDEF(&module);
        int ok = cuda_compile_generated_source(plan->source, names, plan->kernel_count, &module);
        if (ok)
        {
            cuda_module_object *compiled = Z_CUDA_MODULE_P(&module);
            ptx = zend_string_init(compiled->ptx_code, compiled->ptx_size, 0);
            cache->compilations++;
        }
        if (!Z_ISUNDEF(module)) zval_ptr_dtor(&module);
        if (!ok)
        {
            zend_string_release(key);
            return 0;
        }
        if (ZSTR_LEN(ptx) <= FUSION_CACHE_BYTES)
        {
            while (zend_hash_num_elements(&cache->entries) >= FUSION_CACHE_ENTRIES ||
                   cache->bytes + ZSTR_LEN(ptx) > FUSION_CACHE_BYTES)
            {
                zend_string *oldest = NULL, *candidate;
                size_t age = SIZE_MAX;
                fusion_cache_entry *value;
                ZEND_HASH_FOREACH_STR_KEY_PTR(&cache->entries, candidate, value)
                {
                    if (value->used < age) { age = value->used; oldest = candidate; }
                }
                ZEND_HASH_FOREACH_END();
                ZEND_ASSERT(oldest);
                value = zend_hash_find_ptr(&cache->entries, oldest);
                cache->bytes -= ZSTR_LEN(value->ptx);
                zend_hash_del(&cache->entries, oldest);
                cache->evictions++;
            }
            entry = emalloc(sizeof(fusion_cache_entry));
            entry->ptx = zend_string_copy(ptx);
            entry->used = ++cache->clock;
            zend_hash_add_ptr(&cache->entries, key, entry);
            cache->bytes += ZSTR_LEN(ptx);
        }
    }
    zend_string_release(key);
    CUresult result = cuModuleLoadData(&plan->module, ZSTR_VAL(ptx));
    zend_string_release(ptx);
    if (result != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME("Failed to load cached fusion PTX (CUDA error %d)", result);
        return 0;
    }
    return 1;
}
