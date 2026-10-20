#ifndef CUDA_LAUNCH_CONFIG_CUH
#define CUDA_LAUNCH_CONFIG_CUH

#include <cuda_runtime.h>
#include <limits.h>

static __host__ __device__ inline size_t cuda_linear_index(
    unsigned int block, unsigned int threads, unsigned int lane)
{
    return static_cast<size_t>(block) * threads + lane;
}

static __host__ __device__ inline size_t cuda_grid_stride(
    unsigned int blocks, unsigned int threads)
{
    return static_cast<size_t>(blocks) * threads;
}

static __host__ __device__ inline size_t cuda_matrix_offset(
    size_t row, size_t row_stride, size_t column, size_t column_stride)
{
    return row * row_stride + column * column_stride;
}

static inline dim3 cuda_grid_blocks(size_t blocks)
{
    return dim3(blocks > INT_MAX ? 0 : (unsigned int)blocks);
}

static inline dim3 cuda_grid_1d(size_t elements, unsigned int threads = 256)
{
    if (!elements || !threads || threads > 1024) return dim3(0);
    size_t blocks = 1 + (elements - 1) / threads;
    return cuda_grid_blocks(blocks);
}

static inline dim3 cuda_grid_2d(size_t columns, size_t rows,
                                unsigned int block_width, unsigned int block_height,
                                unsigned int batches = 1)
{
    if (!columns || !rows || !block_width || !block_height ||
        block_width > 1024 || block_height > 1024 ||
        block_width > 1024 / block_height || !batches || batches > 65535)
        return dim3(0);
    size_t x = 1 + (columns - 1) / block_width;
    size_t y = 1 + (rows - 1) / block_height;
    if (x > INT_MAX || y > 65535) return dim3(0);
    return dim3((unsigned int)x, (unsigned int)y, batches);
}

static inline cudaError_t cuda_launch_status(bool synchronize = false)
{
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess || !synchronize)
        return status;
    return cudaDeviceSynchronize();
}

#endif