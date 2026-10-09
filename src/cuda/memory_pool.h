#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

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

int tensor_mem_init(size_t size);
void *cuda_mem_alloc(size_t size);
void cuda_mem_free(void *ptr);
void tensor_mem_destroy();

#endif