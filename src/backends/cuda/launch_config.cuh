#ifndef CUDA_LAUNCH_CONFIG_CUH
#define CUDA_LAUNCH_CONFIG_CUH

#include <cuda_runtime.h>

static inline dim3 cuda_grid_1d(size_t elements, unsigned int threads = 256)
{
    return dim3(elements == 0 ? 0 : (unsigned int)(1 + (elements - 1) / threads));
}

static inline dim3 cuda_grid_2d(size_t columns, size_t rows,
                                unsigned int block_width, unsigned int block_height,
                                unsigned int batches = 1)
{
    return dim3((unsigned int)(1 + (columns - 1) / block_width),
                (unsigned int)(1 + (rows - 1) / block_height), batches);
}

static inline cudaError_t cuda_launch_status(bool synchronize = false)
{
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess || !synchronize)
        return status;
    return cudaDeviceSynchronize();
}

#endif