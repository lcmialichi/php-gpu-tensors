#include <cuda_runtime.h>
#include "float_kernels.h"
#include "launch_config.cuh"

#define SUCCESS 0
#define FAILURE 1
extern "C"
{
    __global__ void fill_kernel(float *data, float value, size_t size)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            data[idx] = value;
    }

    __global__ void scale_kernel(float *data, size_t size, float min_value, float max_value)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < size)
        {
            float raw_rand = data[idx];
            float range = max_value - min_value;

            data[idx] = min_value + range * raw_rand;
        }
    }

    int launch_scale_kernel_host(float *data, size_t size, float min_value, float max_value)
    {
        if (size == 0)
            return SUCCESS;

        scale_kernel<<<cuda_grid_1d(size), 256>>>(data, size, min_value, max_value);

        if (cuda_launch_status(true) != cudaSuccess)
        {
            return FAILURE;
        }

        return SUCCESS;
    }

    void launch_fill_kernel(float *data, float value, size_t size)
    {
        if (size == 0)
            return;

        fill_kernel<<<cuda_grid_1d(size), 256>>>(data, value, size);
    }

}