#include "activation_kernels.h"
#include "launch_config.cuh"
#include <math.h>

static __global__ void clip_kernel(float *a, float min_val, float max_val, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        result[idx] = fminf(fmaxf(a[idx], min_val), max_val);
}

static __global__ void relu_kernel(float *a, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        result[idx] = fmaxf(a[idx], 0.0f);
}

static __global__ void sigmoid_kernel(float *a, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        result[idx] = 1.0f / (1.0f + expf(-a[idx]));
}

static __global__ void tanh_kernel(float *a, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        result[idx] = tanhf(a[idx]);
}

extern "C" void launch_clip_kernel(float *a, float min_val, float max_val, float *result, int n)
{
    if (n <= 0)
        return;
    clip_kernel<<<cuda_grid_1d(n), 256>>>(a, min_val, max_val, result, n);
}

extern "C" void launch_relu_kernel(float *a, float *result, int n)
{
    if (n <= 0)
        return;
    relu_kernel<<<cuda_grid_1d(n), 256>>>(a, result, n);
}

extern "C" void launch_sigmoid_kernel(float *a, float *result, int n)
{
    if (n <= 0)
        return;
    sigmoid_kernel<<<cuda_grid_1d(n), 256>>>(a, result, n);
}

extern "C" void launch_tanh_kernel(float *a, float *result, int n)
{
    if (n <= 0)
        return;
    tanh_kernel<<<cuda_grid_1d(n), 256>>>(a, result, n);
}