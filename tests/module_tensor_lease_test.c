#include "module_tensor_lease.h"
#include <stdio.h>
#undef NDEBUG
#include <assert.h>

static int destroyed;

void cuda_tensor_destroy(tensor_t *tensor)
{
    assert(tensor && tensor->ref_count > 0);
    if (--tensor->ref_count) return;
    destroyed++;
    if (tensor->base_tensor) cuda_tensor_destroy(tensor->base_tensor);
}

int main(void)
{
    tensor_t original = {.ref_count = 1};
    module_tensor_lease lease = {0};
    assert(module_tensor_lease_acquire(&lease, &original));
    cuda_tensor_destroy(&original);
    assert(original.ref_count == 1 && destroyed == 0);
    module_tensor_lease_release(&lease);
    assert(original.ref_count == 0 && destroyed == 1);
    module_tensor_lease_release(&lease);
    assert(destroyed == 1);

    tensor_t base = {.ref_count = 3};
    tensor_t left = {.ref_count = 1, .base_tensor = &base};
    tensor_t right = {.ref_count = 1, .base_tensor = &base};
    module_tensor_lease first = {0}, second = {0};
    assert(module_tensor_lease_acquire(&first, &left));
    assert(module_tensor_lease_acquire(&first, &left));
    assert(module_tensor_lease_acquire(&second, &right));
    cuda_tensor_destroy(&left);
    cuda_tensor_destroy(&right);
    cuda_tensor_destroy(&base);
    assert(base.ref_count == 2 && left.ref_count == 2 && right.ref_count == 1);
    module_tensor_lease_release(&second);
    assert(base.ref_count == 1 && left.ref_count == 2);
    module_tensor_lease_release(&first);
    assert(base.ref_count == 0 && destroyed == 4);

    tensor_t shared = {.ref_count = 1};
    for (int i = 0; i < MODULE_MAX_KERNEL_ARGS; i++)
        assert(module_tensor_lease_acquire(&lease, &shared));
    assert(!module_tensor_lease_acquire(&lease, &shared));
    assert(!module_tensor_lease_acquire(&lease, NULL));
    assert(shared.ref_count == MODULE_MAX_KERNEL_ARGS + 1);
    module_tensor_lease_release(&lease);
    assert(shared.ref_count == 1 && lease.count == 0);
    cuda_tensor_destroy(&shared);
    assert(destroyed == 5);

    puts("Kernel tensor leases retain exact storage, shared views and duplicate arguments until release");
    return 0;
}
