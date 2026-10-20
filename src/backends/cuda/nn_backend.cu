#include "nn_backend.h"
#include "launch_config.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <array>
#ifdef HAVE_CUDNN
#include <cudnn.h>
#endif

static thread_local char nn_error[256] = "cuDNN is not compiled into this extension";
extern "C" const char *cuda_nn_error(void) { return nn_error; }

static __global__ void nn_conv_fallback_kernel(
    const float *input, const float *weights, const float *bias, float *output,
    int in_channels, int in_height, int in_width,
    int out_channels, int kernel_height, int kernel_width,
    int out_height, int out_width, int stride_height, int stride_width,
    int pad_height, int pad_width, int dilation_height, int dilation_width,
    int groups, size_t total)
{
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int out_x = index % out_width;
    int out_y = (index / out_width) % out_height;
    int out_channel = (index / ((size_t)out_height * out_width)) % out_channels;
    int batch = index / ((size_t)out_channels * out_height * out_width);
    int channels_per_group = in_channels / groups;
    int outputs_per_group = out_channels / groups;
    int group = out_channel / outputs_per_group;
    int64_t base_y = (int64_t)out_y * stride_height - pad_height;
    int64_t base_x = (int64_t)out_x * stride_width - pad_width;
    int64_t kh_start = base_y < 0
        ? (-base_y + dilation_height - 1) / dilation_height : 0;
    int64_t kw_start = base_x < 0
        ? (-base_x + dilation_width - 1) / dilation_width : 0;
    int64_t kh_end = base_y >= in_height ? 0
        : (in_height - base_y + dilation_height - 1) / dilation_height;
    int64_t kw_end = base_x >= in_width ? 0
        : (in_width - base_x + dilation_width - 1) / dilation_width;
    if (kh_end > kernel_height) kh_end = kernel_height;
    if (kw_end > kernel_width) kw_end = kernel_width;
    float sum = bias ? bias[out_channel] : 0.0f;
    for (int local_channel = 0; local_channel < channels_per_group; local_channel++)
    {
        int in_channel = group * channels_per_group + local_channel;
        for (int64_t kh = kh_start; kh < kh_end; kh++)
        {
            int64_t in_y = base_y + kh * dilation_height;
            for (int64_t kw = kw_start; kw < kw_end; kw++)
            {
                int64_t in_x = base_x + kw * dilation_width;
                size_t input_index = ((size_t)batch * in_channels + in_channel) *
                    in_height * in_width + (size_t)in_y * in_width + in_x;
                size_t filter_index = ((size_t)out_channel * channels_per_group +
                    local_channel) * kernel_height * kernel_width +
                    (size_t)kh * kernel_width + kw;
                sum = fmaf(input[input_index], weights[filter_index], sum);
            }
        }
    }
    output[index] = sum;
}

static __global__ void nn_pool_fallback_kernel(
    const float *input, float *output,
    int channels, int in_height, int in_width, int out_height, int out_width,
    int window_height, int window_width, int stride_height, int stride_width,
    int pad_height, int pad_width, int average, size_t total)
{
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    int out_x = index % out_width;
    int out_y = (index / out_width) % out_height;
    size_t plane_size = (size_t)out_height * out_width;
    int channel = (index / plane_size) % channels;
    size_t batch = index / ((size_t)channels * plane_size);
    int64_t start_y = (int64_t)out_y * stride_height - pad_height;
    int64_t start_x = (int64_t)out_x * stride_width - pad_width;
    int64_t end_y = start_y + window_height;
    int64_t end_x = start_x + window_width;
    if (start_y < 0) start_y = 0;
    if (start_x < 0) start_x = 0;
    if (end_y > in_height) end_y = in_height;
    if (end_x > in_width) end_x = in_width;
    float value = average ? 0.0f : -INFINITY;
    size_t count = 0;
    int has_nan = 0;
    for (int64_t y = start_y; y < end_y; y++)
    {
        for (int64_t x = start_x; x < end_x; x++)
        {
            size_t input_index = ((batch * channels + channel) * in_height + y) *
                in_width + x;
            float item = input[input_index];
            if (average)
                value += item;
            else if (isnan(item))
            {
                value = item;
                has_nan = 1;
            }
            else if (!has_nan && item > value)
                value = item;
            count++;
        }
    }
    output[index] = average ? (count ? value / count : 0.0f) : value;
}

static __global__ void nn_softmax_fallback_kernel(
    const float *input, float *output, int channels, int height, int width,
    size_t positions)
{
    size_t position = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= positions) return;
    size_t plane_size = (size_t)height * width;
    size_t batch = position / plane_size;
    size_t spatial = position % plane_size;
    size_t base = batch * channels * plane_size + spatial;
    float maximum = -INFINITY;
    for (int channel = 0; channel < channels; channel++)
    {
        float value = input[base + (size_t)channel * plane_size];
        if (isnan(value))
        {
            maximum = value;
            break;
        }
        if (value > maximum) maximum = value;
    }
    float sum = 0.0f;
    for (int channel = 0; channel < channels; channel++)
        sum += expf(input[base + (size_t)channel * plane_size] - maximum);
    for (int channel = 0; channel < channels; channel++)
        output[base + (size_t)channel * plane_size] =
            expf(input[base + (size_t)channel * plane_size] - maximum) / sum;
}

static int nn_fallback_finish(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        snprintf(nn_error, sizeof(nn_error), "%s launch: %s", operation,
                 cudaGetErrorString(status));
        return 0;
    }
    status = cudaStreamSynchronize(nullptr);
    if (status != cudaSuccess)
    {
        snprintf(nn_error, sizeof(nn_error), "%s completion: %s", operation,
                 cudaGetErrorString(status));
        return 0;
    }
    return 1;
}

static int nn_fallback_conv(const float *input, const float *weights, const float *bias,
    float *output, const int *input_shape, const int *filter_shape,
    const int *output_shape, const int *stride, const int *padding,
    const int *dilation, int groups)
{
    size_t total = (size_t)output_shape[0] * output_shape[1] *
        output_shape[2] * output_shape[3];
    nn_conv_fallback_kernel<<<cuda_grid_1d(total), 256>>>(
        input, weights, bias, output, input_shape[1],
        input_shape[2], input_shape[3], filter_shape[0], filter_shape[2],
        filter_shape[3], output_shape[2], output_shape[3], stride[0],
        stride[1], padding[0], padding[1], dilation[0], dilation[1], groups,
        total);
    return nn_fallback_finish(cudaGetLastError(), "CUDA convolution fallback");
}

static int nn_fallback_pool(const float *input, float *output,
    const int *input_shape, const int *output_shape, const int *window,
    const int *stride, const int *padding, int average)
{
    size_t total = (size_t)output_shape[0] * output_shape[1] *
        output_shape[2] * output_shape[3];
    nn_pool_fallback_kernel<<<cuda_grid_1d(total), 256>>>(
        input, output, input_shape[1], input_shape[2], input_shape[3],
        output_shape[2], output_shape[3], window[0], window[1], stride[0],
        stride[1], padding[0], padding[1], average, total);
    return nn_fallback_finish(cudaGetLastError(), "CUDA pooling fallback");
}

static int nn_fallback_softmax(const float *input, float *output, const int *shape)
{
    size_t positions = (size_t)shape[0] * shape[2] * shape[3];
    nn_softmax_fallback_kernel<<<cuda_grid_1d(positions), 256>>>(
        input, output, shape[1], shape[2], shape[3], positions);
    return nn_fallback_finish(cudaGetLastError(), "CUDA softmax fallback");
}

#ifdef HAVE_CUDNN
static thread_local int nn_fallback_allowed = 0;
static thread_local cudnnHandle_t nn_handle = nullptr;
static thread_local int nn_device = -1;
static thread_local void *nn_workspace = nullptr;
static thread_local size_t nn_workspace_bytes = 0;
static constexpr size_t NN_WORKSPACE_LIMIT = 32 * 1024 * 1024;

static int nn_check(cudnnStatus_t status, const char *operation)
{
    if (status == CUDNN_STATUS_SUCCESS) return 1;
    nn_fallback_allowed = status == CUDNN_STATUS_NOT_SUPPORTED ||
        status == CUDNN_STATUS_ARCH_MISMATCH ||
        status == CUDNN_STATUS_ALLOC_FAILED;
    snprintf(nn_error, sizeof(nn_error), "%s: %s", operation, cudnnGetErrorString(status));
    return 0;
}
static int nn_cuda_check(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess) return 1;
    nn_fallback_allowed = status == cudaErrorMemoryAllocation;
    snprintf(nn_error, sizeof(nn_error), "%s: %s", operation, cudaGetErrorString(status));
    return 0;
}

struct NnDescriptors {
    cudnnTensorDescriptor_t input = nullptr, output = nullptr, bias = nullptr;
    cudnnFilterDescriptor_t filter = nullptr;
    cudnnConvolutionDescriptor_t conv = nullptr;
    cudnnPoolingDescriptor_t pool = nullptr;
    ~NnDescriptors() {
        if (input) cudnnDestroyTensorDescriptor(input);
        if (output) cudnnDestroyTensorDescriptor(output);
        if (bias) cudnnDestroyTensorDescriptor(bias);
        if (filter) cudnnDestroyFilterDescriptor(filter);
        if (conv) cudnnDestroyConvolutionDescriptor(conv);
        if (pool) cudnnDestroyPoolingDescriptor(pool);
    }
};
struct ConvPlan {
    std::array<int, 15> key;
    cudnnConvolutionFwdAlgo_t algorithm;
    size_t workspace;
};
static thread_local std::vector<ConvPlan> conv_plans;

extern "C" int cuda_nn_available(void) { return 1; }
extern "C" void cuda_nn_shutdown(void)
{
    int original;
    if (cudaGetDevice(&original) == cudaSuccess && nn_device >= 0) {
        cudaSetDevice(nn_device);
        if (nn_workspace) cudaFree(nn_workspace);
        if (nn_handle) cudnnDestroy(nn_handle);
        cudaSetDevice(original);
    }
    nn_workspace = nullptr;
    nn_workspace_bytes = 0;
    nn_handle = nullptr;
    nn_device = -1;
    conv_plans.clear();
}

static int nn_prepare()
{
    int device;
    if (!nn_cuda_check(cudaGetDevice(&device), "cudaGetDevice")) return 0;
    if (nn_device != device) cuda_nn_shutdown();
    if (!nn_handle) {
        nn_device = device;
        if (!nn_check(cudnnCreate(&nn_handle), "cudnnCreate")) return 0;
    }
    return nn_check(cudnnSetStream(nn_handle, nullptr), "cudnnSetStream");
}

static int nn_tensor(cudnnTensorDescriptor_t *descriptor, const int *shape)
{
    return nn_check(cudnnCreateTensorDescriptor(descriptor), "cudnnCreateTensorDescriptor") &&
        nn_check(cudnnSetTensor4dDescriptor(*descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           shape[0], shape[1], shape[2], shape[3]), "cudnnSetTensor4dDescriptor");
}

static int nn_cudnn_conv(const float *input, const float *weights, const float *bias, float *output,
                 const int *input_shape, const int *filter_shape, const int *output_shape,
                 const int *stride, const int *padding, const int *dilation, int groups)
{
    nn_fallback_allowed = 0;
    if (!nn_prepare()) return 0;
    NnDescriptors descriptors;
    int bias_shape[] = {1, output_shape[1], 1, 1};
    if (!nn_tensor(&descriptors.input, input_shape) || !nn_tensor(&descriptors.output, output_shape) ||
        !nn_check(cudnnCreateFilterDescriptor(&descriptors.filter), "cudnnCreateFilterDescriptor") ||
        !nn_check(cudnnSetFilter4dDescriptor(descriptors.filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                   filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]), "cudnnSetFilter4dDescriptor") ||
        !nn_check(cudnnCreateConvolutionDescriptor(&descriptors.conv), "cudnnCreateConvolutionDescriptor") ||
        !nn_check(cudnnSetConvolution2dDescriptor(descriptors.conv, padding[0], padding[1], stride[0], stride[1],
                   dilation[0], dilation[1], CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "cudnnSetConvolution2dDescriptor") ||
        !nn_check(cudnnSetConvolutionGroupCount(descriptors.conv, groups), "cudnnSetConvolutionGroupCount") ||
        !nn_check(cudnnSetConvolutionMathType(descriptors.conv, CUDNN_FMA_MATH), "cudnnSetConvolutionMathType"))
        return 0;
    std::array<int, 15> key;
    for (int i = 0; i < 4; i++) { key[i] = input_shape[i]; key[4+i] = filter_shape[i]; }
    for (int i = 0; i < 2; i++) { key[8+i] = stride[i]; key[10+i] = padding[i]; key[12+i] = dilation[i]; }
    key[14] = groups;
    ConvPlan plan = {};
    bool found = false;
    for (const auto &candidate : conv_plans) if (candidate.key == key) { plan = candidate; found = true; break; }
    if (!found) {
        cudnnConvolutionFwdAlgoPerf_t algorithms[8];
        int count;
        if (!nn_check(cudnnGetConvolutionForwardAlgorithm_v7(nn_handle, descriptors.input, descriptors.filter,
                        descriptors.conv, descriptors.output, 8, &count, algorithms), "cuDNN convolution heuristics")) return 0;
        for (int i = 0; i < count; i++)
            if (algorithms[i].status == CUDNN_STATUS_SUCCESS && algorithms[i].memory <= NN_WORKSPACE_LIMIT &&
                algorithms[i].mathType == CUDNN_FMA_MATH &&
                algorithms[i].determinism == CUDNN_DETERMINISTIC) {
                plan = {key, algorithms[i].algo, algorithms[i].memory};
                found = true;
                break;
            }
        if (!found) {
            nn_fallback_allowed = 1;
            snprintf(nn_error, sizeof(nn_error), "No deterministic FP32 cuDNN convolution plan within 32 MiB workspace");
            return 0;
        }
        if (conv_plans.size() >= 32) conv_plans.erase(conv_plans.begin());
        conv_plans.push_back(plan);
    }
    if (plan.workspace > nn_workspace_bytes) {
        if (nn_workspace && !nn_cuda_check(cudaFree(nn_workspace), "cuDNN workspace release")) return 0;
        nn_workspace = nullptr;
        nn_workspace_bytes = 0;
        if (!nn_cuda_check(cudaMalloc(&nn_workspace, plan.workspace), "cuDNN workspace allocation")) return 0;
        nn_workspace_bytes = plan.workspace;
    }
    const float alpha = 1, beta = 0;
    if (!nn_check(cudnnConvolutionForward(nn_handle, &alpha, descriptors.input, input,
        descriptors.filter, weights, descriptors.conv, plan.algorithm, nn_workspace, plan.workspace,
        &beta, descriptors.output, output), "cudnnConvolutionForward")) return 0;
    if (bias && (!nn_tensor(&descriptors.bias, bias_shape) ||
        !nn_check(cudnnAddTensor(nn_handle, &alpha, descriptors.bias, bias, &alpha, descriptors.output, output),
                   "cudnnAddTensor"))) return 0;
    return nn_cuda_check(cudaStreamSynchronize(nullptr), "cuDNN convolution completion");
}

static int nn_cudnn_pool(const float *input, float *output, const int *input_shape, const int *output_shape,
                 const int *window, const int *stride, const int *padding, int average)
{
    nn_fallback_allowed = 0;
    if (!nn_prepare()) return 0;
    NnDescriptors descriptors;
    if (!nn_tensor(&descriptors.input, input_shape) || !nn_tensor(&descriptors.output, output_shape) ||
        !nn_check(cudnnCreatePoolingDescriptor(&descriptors.pool), "cudnnCreatePoolingDescriptor") ||
        !nn_check(cudnnSetPooling2dDescriptor(descriptors.pool,
                   average ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_MAX_DETERMINISTIC,
                   CUDNN_PROPAGATE_NAN, window[0], window[1], padding[0], padding[1], stride[0], stride[1]),
                   "cudnnSetPooling2dDescriptor")) return 0;
    int shape[4];
    if (!nn_check(cudnnGetPooling2dForwardOutputDim(descriptors.pool, descriptors.input,
                   &shape[0], &shape[1], &shape[2], &shape[3]), "cuDNN pooling output shape")) return 0;
    for (int d = 0; d < 4; d++)
        if (shape[d] != output_shape[d]) {
            nn_fallback_allowed = 1;
            snprintf(nn_error, sizeof(nn_error), "cuDNN pooling output shape differs from requested shape");
            return 0;
        }
    const float alpha = 1, beta = 0;
    return nn_check(cudnnPoolingForward(nn_handle, descriptors.pool, &alpha, descriptors.input, input,
                   &beta, descriptors.output, output), "cudnnPoolingForward") &&
        nn_cuda_check(cudaStreamSynchronize(nullptr), "cuDNN pooling completion");
}

static int nn_cudnn_softmax(const float *input, float *output, const int *shape)
{
    nn_fallback_allowed = 0;
    if (!nn_prepare()) return 0;
    NnDescriptors descriptors;
    if (!nn_tensor(&descriptors.input, shape) || !nn_tensor(&descriptors.output, shape)) return 0;
    const float alpha = 1, beta = 0;
    return nn_check(cudnnSoftmaxForward(nn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                   &alpha, descriptors.input, input, &beta, descriptors.output, output), "cudnnSoftmaxForward") &&
        nn_cuda_check(cudaStreamSynchronize(nullptr), "cuDNN softmax completion");
}
#else
extern "C" int cuda_nn_available(void) { return 0; }
extern "C" void cuda_nn_shutdown(void) {}
#endif

#ifdef HAVE_CUDNN
static int nn_use_fallback(int cudnn_failed, const char *cudnn_error,
                           int fallback_allowed, int fallback_succeeded)
{
    if (fallback_succeeded) return 1;
    char fallback_error[sizeof(nn_error)];
    snprintf(fallback_error, sizeof(fallback_error), "%s", nn_error);
    if (cudnn_failed && !fallback_allowed)
    {
        snprintf(nn_error, sizeof(nn_error), "%s", cudnn_error);
        return 0;
    }
    if (cudnn_failed)
        snprintf(nn_error, sizeof(nn_error), "cuDNN failed (%.100s); CUDA fallback failed (%.100s)",
                 cudnn_error, fallback_error);
    return 0;
}
#endif

extern "C" int cuda_nn_conv(const float *input, const float *weights, const float *bias,
    float *output, const int *input_shape, const int *filter_shape,
    const int *output_shape, const int *stride, const int *padding,
    const int *dilation, int groups)
{
#ifdef HAVE_CUDNN
    if (cuda_nn_available())
    {
        if (nn_cudnn_conv(input, weights, bias, output, input_shape, filter_shape,
                          output_shape, stride, padding, dilation, groups))
            return 1;
        char cudnn_error[sizeof(nn_error)];
        snprintf(cudnn_error, sizeof(cudnn_error), "%s", nn_error);
        int fallback_allowed = nn_fallback_allowed;
        int fallback_succeeded = fallback_allowed && nn_fallback_conv(input, weights,
            bias, output, input_shape, filter_shape, output_shape, stride, padding,
            dilation, groups);
        return nn_use_fallback(1, cudnn_error, fallback_allowed, fallback_succeeded);
    }
#endif
    return nn_fallback_conv(input, weights, bias, output, input_shape, filter_shape,
                            output_shape, stride, padding, dilation, groups);
}

extern "C" int cuda_nn_pool(const float *input, float *output,
    const int *input_shape, const int *output_shape, const int *window,
    const int *stride, const int *padding, int average)
{
#ifdef HAVE_CUDNN
    if (cuda_nn_available())
    {
        if (nn_cudnn_pool(input, output, input_shape, output_shape, window, stride,
                          padding, average))
            return 1;
        char cudnn_error[sizeof(nn_error)];
        snprintf(cudnn_error, sizeof(cudnn_error), "%s", nn_error);
        int fallback_allowed = nn_fallback_allowed;
        int fallback_succeeded = fallback_allowed && nn_fallback_pool(input, output,
            input_shape, output_shape, window, stride, padding, average);
        return nn_use_fallback(1, cudnn_error, fallback_allowed, fallback_succeeded);
    }
#endif
    return nn_fallback_pool(input, output, input_shape, output_shape, window, stride,
                            padding, average);
}

extern "C" int cuda_nn_softmax(const float *input, float *output, const int *shape)
{
#ifdef HAVE_CUDNN
    if (cuda_nn_available())
    {
        if (nn_cudnn_softmax(input, output, shape)) return 1;
        char cudnn_error[sizeof(nn_error)];
        snprintf(cudnn_error, sizeof(cudnn_error), "%s", nn_error);
        int fallback_allowed = nn_fallback_allowed;
        int fallback_succeeded = fallback_allowed &&
            nn_fallback_softmax(input, output, shape);
        return nn_use_fallback(1, cudnn_error, fallback_allowed, fallback_succeeded);
    }
#endif
    return nn_fallback_softmax(input, output, shape);
}
