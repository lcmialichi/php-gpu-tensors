#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include "php.h"
#include "memory_pool.h"



static PoolSegment *main_pools = NULL;
static FreeBlock *free_list = NULL;
static AllocatedBlock *allocated_list = NULL;
static CachedBlock *cached_list = NULL;
static size_t cached_block_count = 0;
static pthread_mutex_t central_mutex;
static int initialized = 0;

static size_t max_pool_size = 0;
static size_t current_allocated = 0;

static void *small_pool_ptr = NULL;
static size_t small_pool_size = 0;
static FreeBlock *small_free_list = NULL;
static const size_t SMALL_POOL_CAPACITY = 16 * 1024 * 1024;

// --- Protótipos de Funções Internas ---
static size_t align_size(size_t s);
static int should_use_small_pool(size_t size);
static void *allocate_from_pool_logic(FreeBlock **pool_head, size_t aligned_size);
static void insert_into_pool_and_merge(FreeBlock **head, FreeBlock *new_block);
static void coalesce_free_list(FreeBlock **head);
static int __allocated_add_block(void *ptr, size_t size, void *pool_base);
static AllocatedBlock *__allocated_remove_block(void *ptr);
static CachedBlock *__cache_find_best_fit(size_t aligned_size);
static void __cache_add_block(CachedBlock *block);
static void __cache_evict_one(void);
static void __cache_release_block(void *ptr, size_t size);
static int expand_pool_if_needed(size_t aligned_size);
static int can_allocate_more(size_t requested_size);

static size_t align_size(size_t s) {
    return (s + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

static int should_use_small_pool(size_t size) {
    return (size <= SMALL_BLOCK_THRESHOLD) && (small_pool_ptr != NULL);
}

static int can_allocate_more(size_t requested_size) {
    return requested_size <= max_pool_size - current_allocated;
}

static void coalesce_free_list(FreeBlock **head) {
    if (!head || !*head) return;
    FreeBlock *curr = *head;
    while (curr && curr->next) {
        if (curr->pool_base == curr->next->pool_base &&
            (char *)curr->ptr + curr->size == (char *)curr->next->ptr) {
            FreeBlock *temp = curr->next;
            curr->size += temp->size;
            curr->next = temp->next;
            pefree(temp, 1);
        } else {
            curr = curr->next;
        }
    }
}

static void insert_into_pool_and_merge(FreeBlock **head, FreeBlock *new_block) {
    FreeBlock **curr_ptr = head;
    while (*curr_ptr && (uintptr_t)(*curr_ptr)->ptr < (uintptr_t)new_block->ptr) {
        curr_ptr = &(*curr_ptr)->next;
    }
    new_block->next = *curr_ptr;
    *curr_ptr = new_block;
    coalesce_free_list(head);
}

int tensor_mem_init(size_t size) {
    if (initialized) return 1;
    if (size == 0 || size > SIZE_MAX - (ALIGNMENT - 1)) return 0;

    if (pthread_mutex_init(&central_mutex, NULL) != 0) return 0;

    max_pool_size = align_size(size);
    current_allocated = 0;
    small_pool_size = (max_pool_size / 4 / ALIGNMENT) * ALIGNMENT;
    if (small_pool_size < ALIGNMENT) small_pool_size = max_pool_size;
    if (small_pool_size > SMALL_POOL_CAPACITY) small_pool_size = SMALL_POOL_CAPACITY;
    if (cudaMalloc(&small_pool_ptr, small_pool_size) != cudaSuccess) {
        small_pool_ptr = NULL;
        small_pool_size = 0;
        pthread_mutex_destroy(&central_mutex);
        return 0;
    }

    FreeBlock *small_initial = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
    if (!small_initial) {
        cudaFree(small_pool_ptr);
        small_pool_ptr = NULL;
        small_pool_size = 0;
        pthread_mutex_destroy(&central_mutex);
        return 0;
    }
    small_initial->ptr = small_pool_ptr;
    small_initial->size = small_pool_size;
    small_initial->pool_base = small_pool_ptr;
    small_initial->next = NULL;
    small_free_list = small_initial;
    current_allocated = small_pool_size;

    initialized = 1;
    return 1;
}

static int expand_pool_if_needed(size_t aligned_size) {
    if (!can_allocate_more(aligned_size)) return 0;

    size_t available = max_pool_size - current_allocated;
    size_t initial_pool_size = aligned_size > available / 4 ? available : aligned_size * 4;

    if (initial_pool_size < 1024 * 1024) return 0;

    PoolSegment *segment = (PoolSegment *)pemalloc(sizeof(PoolSegment), 1);
    if (!segment) return 0;

    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, initial_pool_size);
    if (err != cudaSuccess) {
        pefree(segment, 1);
        return 0;
    }

    FreeBlock *initial_block = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
    if (!initial_block) {
        cudaFree(ptr);
        pefree(segment, 1);
        return 0;
    }

    segment->ptr = ptr;
    segment->size = initial_pool_size;
    segment->next = main_pools;
    main_pools = segment;
    current_allocated += initial_pool_size;
    initial_block->ptr = ptr;
    initial_block->size = initial_pool_size;
    initial_block->pool_base = ptr;
    initial_block->next = NULL;
    insert_into_pool_and_merge(&free_list, initial_block);
    return 1;
}

void *cuda_mem_alloc(size_t size) {
    if (!initialized || size == 0 || size > SIZE_MAX - (ALIGNMENT - 1)) return NULL;

    pthread_mutex_lock(&central_mutex);
    void *ptr = NULL;
    size_t aligned_size = align_size(size);

    CachedBlock *cached_block = __cache_find_best_fit(aligned_size);
    if (cached_block) {
        ptr = cached_block->ptr;
        size_t actual_size = cached_block->size;
        if (!__allocated_add_block(ptr, actual_size, NULL)) {
            cached_block->next = cached_list;
            cached_list = cached_block;
            cached_block_count++;
            pthread_mutex_unlock(&central_mutex);
            return NULL;
        }
        pefree(cached_block, 1);
        pthread_mutex_unlock(&central_mutex);
        return ptr;
    }

    if (should_use_small_pool(aligned_size)) {
        ptr = allocate_from_pool_logic(&small_free_list, aligned_size);
    }

    if (!ptr && free_list)
        ptr = allocate_from_pool_logic(&free_list, aligned_size);

    while (!ptr && !can_allocate_more(aligned_size) && cached_list) {
        __cache_evict_one();
    }

    if (!ptr && can_allocate_more(aligned_size) &&
        expand_pool_if_needed(aligned_size))
        ptr = allocate_from_pool_logic(&free_list, aligned_size);

    if (!ptr && can_allocate_more(aligned_size)) {
        cudaError_t err = cudaMalloc(&ptr, aligned_size);
        while (err == cudaErrorMemoryAllocation && cached_list) {
            __cache_evict_one();
            err = cudaMalloc(&ptr, aligned_size);
        }
        if (err == cudaSuccess) {
            if (__allocated_add_block(ptr, aligned_size, NULL)) {
                current_allocated += aligned_size;
            } else {
                cudaFree(ptr);
                ptr = NULL;
            }
        }
    }

    pthread_mutex_unlock(&central_mutex);
    return ptr;
}

static void *allocate_from_pool_logic(FreeBlock **pool_head, size_t aligned_size) {
    FreeBlock *curr = *pool_head;
    FreeBlock *prev = NULL;
    FreeBlock *best_fit = NULL;
    FreeBlock *best_fit_prev = NULL;
    size_t min_diff = (size_t)-1;

    while (curr) {
        if (curr->size >= aligned_size) {
            size_t diff = curr->size - aligned_size;
            if (diff < min_diff) {
                min_diff = diff;
                best_fit = curr;
                best_fit_prev = prev;
                if (diff == 0) break;
            }
        }
        prev = curr;
        curr = curr->next;
    }

    if (!best_fit) return NULL;

    void *ptr = best_fit->ptr;
    size_t actual_size = best_fit->size;

    if (actual_size > aligned_size + ALIGNMENT) {
        actual_size = aligned_size;
    }

    if (!__allocated_add_block(ptr, actual_size, best_fit->pool_base)) return NULL;

    if (best_fit->size > aligned_size + ALIGNMENT) {
        best_fit->ptr = (char *)ptr + aligned_size;
        best_fit->size -= aligned_size;
    } else {
        if (best_fit_prev) best_fit_prev->next = best_fit->next;
        else *pool_head = best_fit->next;
        pefree(best_fit, 1);
    }

    return ptr;
}

void cuda_mem_free(void *ptr) {
    if (!initialized || !ptr) return;

    pthread_mutex_lock(&central_mutex);

    AllocatedBlock *ab = __allocated_remove_block(ptr);
    if (ab) {
        size_t size_to_free = ab->size;

        if (ab->pool_base == small_pool_ptr) {
            insert_into_pool_and_merge(&small_free_list, ab);
        }
        else if (ab->pool_base) {
            insert_into_pool_and_merge(&free_list, ab);
        }
        else {
            if (cached_block_count < MAX_CACHED_BLOCKS) {
                __cache_add_block(ab);
            } else {
                __cache_release_block(ptr, size_to_free);
                pefree(ab, 1);
            }
        }
    }

    pthread_mutex_unlock(&central_mutex);
}

void tensor_mem_destroy() {
    if (!initialized) return;

    pthread_mutex_lock(&central_mutex);

    FreeBlock *curr_small = small_free_list;
    while (curr_small) {
        FreeBlock *next = curr_small->next;
        pefree(curr_small, 1);
        curr_small = next;
    }
    CachedBlock *curr_cache = cached_list;
    while (curr_cache) {
        CachedBlock *next = curr_cache->next;
        cudaFree(curr_cache->ptr);
        pefree(curr_cache, 1);
        curr_cache = next;
    }

    AllocatedBlock *curr_alloc = allocated_list;
    while (curr_alloc) {
        AllocatedBlock *next = curr_alloc->next;
        if (!curr_alloc->pool_base) cudaFree(curr_alloc->ptr);
        pefree(curr_alloc, 1);
        curr_alloc = next;
    }

    FreeBlock *curr_free = free_list;
    while (curr_free) {
        FreeBlock *next = curr_free->next;
        pefree(curr_free, 1);
        curr_free = next;
    }

    if (small_pool_ptr) cudaFree(small_pool_ptr);
    PoolSegment *segment = main_pools;
    while (segment) {
        PoolSegment *next = segment->next;
        cudaFree(segment->ptr);
        pefree(segment, 1);
        segment = next;
    }

    small_free_list = NULL;
    cached_list = NULL;
    allocated_list = NULL;
    free_list = NULL;
    main_pools = NULL;
    small_pool_ptr = NULL;
    small_pool_size = 0;
    cached_block_count = 0;
    current_allocated = 0;
    max_pool_size = 0;

    initialized = 0;
    pthread_mutex_unlock(&central_mutex);
    pthread_mutex_destroy(&central_mutex);
}

static int __allocated_add_block(void *ptr, size_t size, void *pool_base) {
    AllocatedBlock *ab = (AllocatedBlock *)pemalloc(sizeof(AllocatedBlock), 1);
    if (!ab) return 0;
    ab->ptr = ptr;
    ab->size = size;
    ab->pool_base = pool_base;
    ab->next = allocated_list;
    allocated_list = ab;
    return 1;
}

static AllocatedBlock *__allocated_remove_block(void *ptr) {
    AllocatedBlock *curr = allocated_list, *prev = NULL;
    while (curr) {
        if (curr->ptr == ptr) {
            if (prev) prev->next = curr->next;
            else allocated_list = curr->next;
            return curr;
        }
        prev = curr; curr = curr->next;
    }
    return NULL;
}

static CachedBlock *__cache_find_best_fit(size_t aligned_size) {
    CachedBlock *curr = cached_list, *prev = NULL, *best = NULL, *best_prev = NULL;
    size_t min_diff = (size_t)-1;
    while (curr) {
        if (curr->size >= aligned_size && curr->size - aligned_size <= aligned_size) {
            size_t diff = curr->size - aligned_size;
            if (diff < min_diff) {
                min_diff = diff; best = curr; best_prev = prev;
            }
        }
        prev = curr; curr = curr->next;
    }
    if (best) {
        if (best_prev) best_prev->next = best->next;
        else cached_list = best->next;
        cached_block_count--;
        return best;
    }
    return NULL;
}

static void __cache_add_block(CachedBlock *block) {
    block->next = cached_list;
    cached_list = block;
    cached_block_count++;
}

static void __cache_evict_one(void) {
    CachedBlock *block = cached_list;
    cached_list = block->next;
    cached_block_count--;
    __cache_release_block(block->ptr, block->size);
    pefree(block, 1);
}

static void __cache_release_block(void *ptr, size_t size) {
    cudaFree(ptr);
    current_allocated -= size;
}