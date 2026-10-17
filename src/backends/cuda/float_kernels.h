#ifndef FLOAT_KERNELS_H
#define FLOAT_KERNELS_H
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void launch_fill_kernel(float *data, float value, size_t size);
    int launch_scale_kernel_host(float *data, size_t size, float min_value, float max_value);
#ifdef __cplusplus
}
#endif

#endif