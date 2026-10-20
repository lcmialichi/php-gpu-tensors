#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../src/backends/cuda/launch_config.cuh"
#include "../src/backends/cuda/reduction_ops.h"
#include "../src/backends/cuda/factory_kernels.h"

static void require_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

static void require_invalid_grid(const char *operation)
{
    cudaError_t status = cuda_reduction_status();
    if (status != cudaErrorInvalidConfiguration)
    {
        fprintf(stderr, "%s must reject an oversized grid, got: %s\n",
                operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

static __global__ void wide_index_probe(unsigned long long *output)
{
    output[0] = cuda_linear_index(8388608, 256, 0);
    output[1] = cuda_linear_index(16777216, 256, 13);
    output[2] = cuda_linear_index(INT_MAX - 1, 1024, 1023);
    output[3] = cuda_grid_stride(INT_MAX, 1024);
    output[4] = cuda_matrix_offset(65536, 65536, 7, 1);
    output[5] = cuda_matrix_offset(1, 4294967297ULL, 3, 2147483648ULL);
}

static void check_integer_random_bounds(void)
{
    float fractions[] = {0.0f, 0.5f, 1.0f};
    float *input = NULL;
    uint64_t *output = NULL;
    uint64_t actual[3];
    require_cuda(cudaMalloc(&input, sizeof(fractions)), "Allocate random fractions");
    require_cuda(cudaMalloc(&output, sizeof(actual)), "Allocate integer random output");
    require_cuda(cudaMemcpy(input, fractions, sizeof(fractions), cudaMemcpyHostToDevice),
                 "Upload random fractions");
    scalar_value_t minimum = {}, maximum = {};
    minimum.dtype = maximum.dtype = DTYPE_INT64;
    minimum.v.i64 = INT64_MIN;
    maximum.v.i64 = INT64_MAX;
    launch_scale_range_kernel(input, output, DTYPE_INT64, minimum, maximum, 3);
    require_cuda(cudaGetLastError(), "Launch signed integer random bounds");
    require_cuda(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
                 "Read signed integer random bounds");
    const int64_t expected_signed[] = {INT64_MIN, 0, INT64_MAX};
    int64_t signed_actual[3];
    memcpy(signed_actual, actual, sizeof(actual));
    for (size_t i = 0; i < 3; i++)
        if (signed_actual[i] != expected_signed[i])
        {
            fprintf(stderr, "Signed integer random endpoint %zu was incorrect\n", i);
            exit(EXIT_FAILURE);
        }

    minimum.dtype = maximum.dtype = DTYPE_UINT64;
    minimum.v.u64 = 0;
    maximum.v.u64 = UINT64_MAX;
    launch_scale_range_kernel(input, output, DTYPE_UINT64, minimum, maximum, 3);
    require_cuda(cudaGetLastError(), "Launch unsigned integer random bounds");
    require_cuda(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
                 "Read unsigned integer random bounds");
    const uint64_t expected_unsigned[] = {0, UINT64_C(9223372036854775808), UINT64_MAX};
    for (size_t i = 0; i < 3; i++)
        if (actual[i] != expected_unsigned[i])
        {
            fprintf(stderr, "Unsigned integer random endpoint %zu was incorrect\n", i);
            exit(EXIT_FAILURE);
        }
    const dtype_t invalid_types[] = {DTYPE_UNKNOWN, (dtype_t)-1};
    const size_t counts[] = {0, 3};
    for (dtype_t dtype : invalid_types)
        for (size_t count : counts)
            if (launch_arange(output, dtype, count, 0, 1, 1, 0, 1) != cudaErrorInvalidValue ||
                launch_linspace(output, dtype, count, 0, 1, 1, 1, 0, 1) != cudaErrorInvalidValue)
            {
                fprintf(stderr, "Native ranges accepted an invalid dtype\n");
                exit(EXIT_FAILURE);
            }
    size_t oversized = (size_t)INT_MAX + 1;
    if (launch_arange(output, DTYPE_INT64, oversized, 0, 1, 1, 0, 1) != cudaErrorInvalidValue ||
        launch_linspace(output, DTYPE_INT64, oversized, 0, 1, 1, 1, 0, 1) != cudaErrorInvalidValue)
    {
        fprintf(stderr, "Native ranges accepted more than INT_MAX elements\n");
        exit(EXIT_FAILURE);
    }
    require_cuda(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost),
                 "Read rejected range sentinels");
    for (size_t i = 0; i < 3; i++)
        if (actual[i] != expected_unsigned[i])
        {
            fprintf(stderr, "Rejected range changed output %zu\n", i);
            exit(EXIT_FAILURE);
        }
    require_cuda(cudaFree(output), "Free integer random output");
    require_cuda(cudaFree(input), "Free random fractions");
}

int main(void)
{
    check_integer_random_bounds();
    unsigned long long *device_indices = NULL;
    require_cuda(cudaMalloc(&device_indices, 6 * sizeof(*device_indices)), "Allocate index probe");
    wide_index_probe<<<1, 1>>>(device_indices);
    require_cuda(cudaGetLastError(), "Launch index probe");
    unsigned long long indices[6];
    require_cuda(cudaMemcpy(indices, device_indices, sizeof(indices), cudaMemcpyDeviceToHost),
                 "Read index probe");
    const unsigned long long expected[] = {
        2147483648ULL, 4294967309ULL, 2199023254527ULL,
        2199023254528ULL, 4294967303ULL, 10737418241ULL
    };
    for (size_t i = 0; i < 6; i++)
    {
        if (indices[i] != expected[i])
        {
            fprintf(stderr, "GPU index %zu: expected %llu, got %llu\n", i, expected[i], indices[i]);
            return EXIT_FAILURE;
        }
    }
    require_cuda(cudaFree(device_indices), "Free index probe");

    float *input = NULL, *output = NULL;
    float host_input[8], host_output[64];
    for (size_t i = 0; i < 8; i++) host_input[i] = 3.0f;
    for (size_t i = 0; i < 64; i++) host_output[i] = -777.0f;
    require_cuda(cudaMalloc(&input, sizeof(host_input)), "Allocate reduction input");
    require_cuda(cudaMalloc(&output, sizeof(host_output)), "Allocate reduction output");
    require_cuda(cudaMemcpy(input, host_input, sizeof(host_input), cudaMemcpyHostToDevice),
                 "Upload reduction input");
    require_cuda(cudaMemcpy(output, host_output, sizeof(host_output), cudaMemcpyHostToDevice),
                 "Upload reduction sentinel");

    int shape[] = {1}, result_shape[] = {1};
    size_t strides[] = {1};
    // Narrowing this count produces one block, exposing false success with bounded buffers.
    const size_t overflowing_blocks = 4294967297ULL;
    launch_reduction(input, output, DTYPE_FLOAT32, OP_REDUCE_SUM,
                     shape, 1, result_shape, strides, 1, 0, overflowing_blocks, 0, NULL);
    require_invalid_grid("Generic reduction");
    launch_arg_reduction(input, reinterpret_cast<int *>(output), DTYPE_FLOAT32, OP_ARG_MAX,
                         shape, 1, result_shape, strides, 1, 0, overflowing_blocks, 0, NULL);
    require_invalid_grid("Arg reduction");
    launch_moments_reduction(input, output, DTYPE_FLOAT32, DTYPE_FLOAT32,
                            shape, 1, strides, 0, overflowing_blocks, 0, 0, NULL);
    require_invalid_grid("Moments reduction");

    int column_shape[] = {8, 1};
    size_t column_strides[] = {1, 1};
    launch_reduction(input, output, DTYPE_FLOAT32, OP_REDUCE_SUM,
                     column_shape, 2, result_shape, column_strides, 1, 0,
                     overflowing_blocks * 32, 0, NULL);
    require_invalid_grid("Column reduction");
    require_cuda(cudaMemcpy(host_output, output, sizeof(host_output), cudaMemcpyDeviceToHost),
                 "Read reduction sentinels");
    for (size_t i = 0; i < 64; i++)
    {
        if (host_output[i] != -777.0f)
        {
            fprintf(stderr, "Rejected reduction changed output %zu\n", i);
            return EXIT_FAILURE;
        }
    }

    launch_reduction(input, output, DTYPE_FLOAT32, OP_REDUCE_SUM,
                     shape, 1, result_shape, strides, 1, 0, 1, 0, NULL);
    require_cuda(cuda_reduction_status(), "Valid reduction after rejected launches");
    require_cuda(cudaMemcpy(host_output, output, sizeof(float), cudaMemcpyDeviceToHost),
                 "Read valid reduction");
    if (host_output[0] != 3.0f)
    {
        fprintf(stderr, "Valid reduction did not recover after rejected launches\n");
        return EXIT_FAILURE;
    }
    require_cuda(cudaFree(output), "Free reduction output");
    require_cuda(cudaFree(input), "Free reduction input");
    cuda_reduction_shutdown();
    puts("GPU integer random bounds, wide-index and reduction-grid checks passed");
    return EXIT_SUCCESS;
}
