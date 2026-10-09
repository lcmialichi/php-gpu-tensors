#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "php.h"
#include <cuda_runtime.h>
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
#include "../src/cuda/memory_pool.c"

int main(void)
{
    fail_metadata = 1;
    assert(!tensor_mem_init(64 * 1024 * 1024));
    assert(live_bytes == 0);
    fail_metadata = 0;

    assert(tensor_mem_init(64 * 1024 * 1024));
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
    puts("memory pool checks passed");
    return 0;
}