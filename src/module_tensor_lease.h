#ifndef MODULE_TENSOR_LEASE_H
#define MODULE_TENSOR_LEASE_H

#include "tensor.h"

#define MODULE_MAX_KERNEL_ARGS 128

typedef struct module_tensor_lease
{
    tensor_t *tensors[MODULE_MAX_KERNEL_ARGS];
    size_t count;
    struct module_tensor_lease *next;
} module_tensor_lease;

static inline int module_tensor_lease_acquire(module_tensor_lease *lease, tensor_t *tensor)
{
    if (!tensor || lease->count >= MODULE_MAX_KERNEL_ARGS) return 0;
    tensor->ref_count++;
    lease->tensors[lease->count++] = tensor;
    return 1;
}

static inline void module_tensor_lease_release(module_tensor_lease *lease)
{
    while (lease->count)
    {
        tensor_t *tensor = lease->tensors[--lease->count];
        lease->tensors[lease->count] = NULL;
        cuda_tensor_destroy(tensor);
    }
}

#endif
