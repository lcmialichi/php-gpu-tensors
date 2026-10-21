#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include "php.h"
#include <cuda_runtime.h>
#include <cuda.h>
#undef NDEBUG
#include <assert.h>

#undef pemalloc
#undef pefree

static void *device_allocations[32];
static size_t device_allocation_sizes[32];
static size_t live_bytes;
static size_t device_allocation_calls;
static int invalid_frees;
static int fail_metadata;
static int owner_context_token;
static int foreign_context_token;
static int test_device;
static CUcontext test_context = (CUcontext)&owner_context_token;
static CUcontext saved_context;
static int saved_device;
static int context_pushes;
static int context_queries;

static cudaError_t test_get_device(int *device)
{
    context_queries++;
    *device = test_device;
    return cudaSuccess;
}

static CUresult test_get_context(CUcontext *context)
{
    context_queries++;
    *context = test_context;
    return CUDA_SUCCESS;
}

static CUresult test_push_context(CUcontext context)
{
    assert(context == (CUcontext)&owner_context_token);
    saved_context = test_context;
    saved_device = test_device;
    test_context = context;
    test_device = 0;
    context_pushes++;
    return CUDA_SUCCESS;
}

static CUresult test_pop_context(CUcontext *context)
{
    *context = test_context;
    test_context = saved_context;
    test_device = saved_device;
    return CUDA_SUCCESS;
}

static void test_cleanup_warning(const char *docref, int severity, const char *format, ...)
{
    (void)docref;
    (void)severity;
    (void)format;
    assert(!"Unexpected CUDA cleanup warning");
}

static void *test_malloc(size_t size, int persistent)
{
    (void)persistent;
    return fail_metadata ? NULL : malloc(size);
}

static void test_free(void *ptr, int persistent)
{
    (void)persistent;
    free(ptr);
}

static cudaError_t test_cuda_malloc(void **ptr, size_t size)
{
    for (int index = 0; index < 32; index++)
    {
        if (!device_allocations[index])
        {
            *ptr = malloc(size);
            if (!*ptr)
                return cudaErrorMemoryAllocation;
            device_allocations[index] = *ptr;
            device_allocation_sizes[index] = size;
            live_bytes += size;
            device_allocation_calls++;
            return cudaSuccess;
        }
    }
    return cudaErrorMemoryAllocation;
}

static cudaError_t test_cuda_free(void *ptr)
{
    assert(test_device == 0 && test_context == (CUcontext)&owner_context_token);
    for (int index = 0; index < 32; index++)
    {
        if (device_allocations[index] == ptr)
        {
            free(ptr);
            device_allocations[index] = NULL;
            live_bytes -= device_allocation_sizes[index];
            device_allocation_sizes[index] = 0;
            return cudaSuccess;
        }
    }
    invalid_frees++;
    return cudaErrorInvalidDevicePointer;
}

#define pemalloc test_malloc
#define pefree test_free
#define cudaMalloc test_cuda_malloc
#define cudaFree test_cuda_free
#define cudaGetDevice test_get_device
#define cuCtxGetCurrent test_get_context
#undef cuCtxPushCurrent
#undef cuCtxPopCurrent
#define cuCtxPushCurrent test_push_context
#define cuCtxPopCurrent test_pop_context
#define php_error_docref test_cleanup_warning
#include "../src/backends/cuda/memory_pool.c"

static void *reject_foreign_owner(void *allocation)
{
    int expected = getpid() == atomic_load(&owner_process)
        ? CUDA_OWNER_THREAD : CUDA_OWNER_FORKED;
    size_t bytes = live_bytes;
    size_t calls = device_allocation_calls;
    int queries = context_queries;
    assert(tensor_mem_owner_status() == expected);
    assert(tensor_mem_claim_owner() == expected);
    assert(!tensor_mem_init(1024 * 1024));
    assert(!tensor_mem_context_matches());
    assert(cuda_mem_alloc(128) == NULL);
    cuda_mem_free(allocation);
    tensor_mem_destroy();
    assert(live_bytes == bytes && device_allocation_calls == calls);
    assert(context_queries == queries && invalid_frees == 0);
    return NULL;
}

static void *claim_from_thread(void *result)
{
    *(int *)result = tensor_mem_claim_owner();
    return NULL;
}

int main(void)
{
    assert(tensor_mem_owner_status() == CUDA_OWNER_UNCLAIMED);
    pid_t worker = fork();
    assert(worker >= 0);
    if (worker == 0)
    {
        alarm(5);
        pthread_t thread;
        int result;
        assert(pthread_create(&thread, NULL, claim_from_thread, &result) == 0);
        assert(pthread_join(thread, NULL) == 0 && result == CUDA_OWNER_CURRENT);
        assert(tensor_mem_owner_status() == CUDA_OWNER_THREAD);
        assert(pthread_create(&thread, NULL, claim_from_thread, &result) == 0);
        assert(pthread_join(thread, NULL) == 0 && result == CUDA_OWNER_THREAD);
        assert(context_queries == 0 && device_allocation_calls == 0);
        _exit(0);
    }
    int status;
    assert(waitpid(worker, &status, 0) == worker);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    worker = fork();
    assert(worker >= 0);
    if (worker == 0)
    {
        alarm(5);
        assert(tensor_mem_init(1024 * 1024));
        assert(tensor_mem_owner_status() == CUDA_OWNER_CURRENT);
        tensor_mem_destroy();
        _exit(0);
    }
    assert(waitpid(worker, &status, 0) == worker);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(tensor_mem_owner_status() == CUDA_OWNER_UNCLAIMED);

    fail_metadata = 1;
    assert(!tensor_mem_init(64 * 1024 * 1024));
    assert(live_bytes == 0);
    fail_metadata = 0;

    assert(tensor_mem_init(64 * 1024 * 1024));
    assert(tensor_mem_device() == 0);
    assert(tensor_mem_context() == (CUcontext)&owner_context_token);
    assert(tensor_mem_context_matches());
    size_t owner_allocation_calls = device_allocation_calls;
    test_device = 1;
    assert(!tensor_mem_context_matches());
    assert(cuda_mem_alloc(128) == NULL);
    test_device = 0;
    test_context = (CUcontext)&foreign_context_token;
    assert(!tensor_mem_context_matches());
    assert(cuda_mem_alloc(128) == NULL);
    assert(device_allocation_calls == owner_allocation_calls);
    test_context = (CUcontext)&owner_context_token;
    size_t reserved_bytes = live_bytes;
    fail_metadata = 1;
    assert(cuda_mem_alloc(128) == NULL);
    assert(cuda_mem_alloc(2 * 1024 * 1024) == NULL);
    assert(live_bytes == reserved_bytes);
    fail_metadata = 0;

    void *small = cuda_mem_alloc(128);
    assert(live_bytes == reserved_bytes);
    void *pooled = cuda_mem_alloc(2 * 1024 * 1024);
    assert(live_bytes == reserved_bytes + 8 * 1024 * 1024);
    void *independent = cuda_mem_alloc(12 * 1024 * 1024);
    assert(small && pooled && independent);
    pthread_t thread;
    assert(pthread_create(&thread, NULL, reject_foreign_owner, small) == 0);
    assert(pthread_join(thread, NULL) == 0);
    assert(tensor_mem_owner_status() == CUDA_OWNER_CURRENT);
    pthread_mutex_lock(&central_mutex);
    worker = fork();
    assert(worker >= 0);
    if (worker == 0)
    {
        alarm(5);
        reject_foreign_owner(small);
        _exit(0);
    }
    pthread_mutex_unlock(&central_mutex);
    assert(waitpid(worker, &status, 0) == worker);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    fail_metadata = 1;
    cuda_mem_free(small);
    cuda_mem_free(independent);
    fail_metadata = 0;
    assert(cuda_mem_alloc(128) == small);
    cuda_mem_free(small);
    fail_metadata = 1;
    assert(cuda_mem_alloc(12 * 1024 * 1024) == NULL);
    fail_metadata = 0;
    void *reused = cuda_mem_alloc(12 * 1024 * 1024);
    assert(reused == independent);

    tensor_mem_destroy();
    assert(invalid_frees == 0);
    for (int index = 0; index < 32; index++)
        assert(device_allocations[index] == NULL);

    assert(tensor_mem_init(1024 * 1024));
    assert(live_bytes <= 1024 * 1024);
    tensor_mem_destroy();
    assert(live_bytes == 0);

    assert(tensor_mem_init(16 * 1024 * 1024));
    void *large_with_small_budget = cuda_mem_alloc(2 * 1024 * 1024);
    assert(large_with_small_budget);
    cuda_mem_free(large_with_small_budget);
    tensor_mem_destroy();
    assert(live_bytes == 0);

    assert(tensor_mem_init(64 * 1024 * 1024));
    void *main_pool_allocation = cuda_mem_alloc(2 * 1024 * 1024);
    void *cached_large = cuda_mem_alloc(32 * 1024 * 1024);
    assert(main_pool_allocation && cached_large);
    cuda_mem_free(cached_large);
    void *larger = cuda_mem_alloc(40 * 1024 * 1024);
    assert(larger);
    assert(live_bytes <= 64 * 1024 * 1024);
    cuda_mem_free(larger);
    cuda_mem_free(main_pool_allocation);
    tensor_mem_destroy();
    assert(live_bytes == 0);

    size_t allocation_calls_before_growth = device_allocation_calls;
    assert(tensor_mem_init(64 * 1024 * 1024));
    void *arena_blocks[8];
    for (int index = 0; index < 8; index++)
    {
        arena_blocks[index] = cuda_mem_alloc(2 * 1024 * 1024);
        assert(arena_blocks[index]);
    }
    assert(device_allocation_calls == allocation_calls_before_growth + 3);
    for (int index = 0; index < 8; index++)
        cuda_mem_free(arena_blocks[index]);
    size_t reserved_with_two_arenas = live_bytes;
    void *reused_arena_block = cuda_mem_alloc(2 * 1024 * 1024);
    assert(reused_arena_block);
    assert(live_bytes == reserved_with_two_arenas);
    cuda_mem_free(reused_arena_block);
    tensor_mem_destroy();
    assert(live_bytes == 0);
    assert(invalid_frees == 0);

    assert(tensor_mem_init(1024 * 1024));
    void *foreign_cleanup = cuda_mem_alloc(128);
    assert(foreign_cleanup);
    test_device = 1;
    test_context = (CUcontext)&foreign_context_token;
    cuda_mem_free(foreign_cleanup);
    assert(test_device == 1 && test_context == (CUcontext)&foreign_context_token);
    tensor_mem_destroy();
    assert(test_device == 1 && test_context == (CUcontext)&foreign_context_token);
    assert(context_pushes == 2);
    assert(tensor_mem_device() == -1 && tensor_mem_context() == NULL);
    assert(live_bytes == 0 && invalid_frees == 0);
    puts("memory pool checks passed");
    return 0;
}