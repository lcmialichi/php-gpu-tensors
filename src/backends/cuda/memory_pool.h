#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cuda.h>

#define ALIGNMENT 256
#define MAX_CACHED_BLOCKS 64
#define SMALL_BLOCK_THRESHOLD (1024 * 1024)

typedef struct MemoryBlock {
    void* ptr;
    size_t size;
    void* pool_base;
    struct MemoryBlock* next;
} MemoryBlock;

typedef struct PoolSegment {
    void* ptr;
    size_t size;
    struct PoolSegment* next;
} PoolSegment;

typedef MemoryBlock AllocatedBlock;
typedef MemoryBlock FreeBlock;
typedef MemoryBlock CachedBlock;

enum {
    CUDA_OWNER_UNCLAIMED = 0,
    CUDA_OWNER_CURRENT = 1,
    CUDA_OWNER_FORKED = -1,
    CUDA_OWNER_THREAD = -2
};

int tensor_mem_claim_owner(void);
int tensor_mem_owner_status(void);
int tensor_mem_failed(void);
void tensor_mem_mark_failed(void);
int tensor_mem_init(size_t size);
int tensor_mem_device(void);
CUcontext tensor_mem_context(void);
int tensor_mem_context_matches(void);
void *cuda_mem_alloc(size_t size);
void cuda_mem_free(void *ptr);
void tensor_mem_destroy();

#endif