#ifndef FACTORY_KERNELS_H
#define FACTORY_KERNELS_H
#include "../data_types.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

    cudaError_t launch_pack_strided(
        const void *source, void *destination, size_t elements, size_t element_size,
        const int *shape, const size_t *strides, int ndims);

    void launch_bernoulli_kernel(
        float *values,
        bool *base,
        size_t total_elements,
        float p);
        
    void launch_assign_scalar_val_kernel(
        void *base,
        dtype_t dtype,
        scalar_value_t value,
        size_t total_elements);

    void launch_scale_range_kernel(
        float *values,
        void *base,
        dtype_t dtype,
        scalar_value_t min,
        scalar_value_t max,
        size_t total_elements);

#ifdef __cplusplus
}
#endif

#endif