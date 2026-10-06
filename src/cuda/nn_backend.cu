#include "nn_backend.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <array>
#ifdef HAVE_CUDNN
#include <cudnn.h>
#endif

static thread_local char nn_error[256] = "cuDNN is not compiled into this extension";
extern "C" const char *cuda_nn_error(void) { return nn_error; }

#ifdef HAVE_CUDNN
static thread_local cudnnHandle_t nn_handle = nullptr;
static thread_local int nn_device = -1;
static thread_local void *nn_workspace = nullptr;
static thread_local size_t nn_workspace_bytes = 0;
static constexpr size_t NN_WORKSPACE_LIMIT = 32 * 1024 * 1024;

static int nn_check(cudnnStatus_t status, const char *operation)
{
    if (status == CUDNN_STATUS_SUCCESS) return 1;
    snprintf(nn_error, sizeof(nn_error), "%s: %s", operation, cudnnGetErrorString(status));
    return 0;
}
static int nn_cuda_check(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess) return 1;
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

extern "C" int cuda_nn_conv(const float *input, const float *weights, const float *bias, float *output,
                 const int *input_shape, const int *filter_shape, const int *output_shape,
                 const int *stride, const int *padding, const int *dilation, int groups)
{
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
        if (!found) { snprintf(nn_error, sizeof(nn_error), "No deterministic FP32 cuDNN convolution plan within 32 MiB workspace"); return 0; }
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

extern "C" int cuda_nn_pool(const float *input, float *output, const int *input_shape, const int *output_shape,
                 const int *window, const int *stride, const int *padding, int average)
{
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
            snprintf(nn_error, sizeof(nn_error), "cuDNN pooling output shape differs from requested shape");
            return 0;
        }
    const float alpha = 1, beta = 0;
    return nn_check(cudnnPoolingForward(nn_handle, descriptors.pool, &alpha, descriptors.input, input,
                   &beta, descriptors.output, output), "cudnnPoolingForward") &&
        nn_cuda_check(cudaStreamSynchronize(nullptr), "cuDNN pooling completion");
}

extern "C" int cuda_nn_softmax(const float *input, float *output, const int *shape)
{
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
extern "C" int cuda_nn_conv(const float *, const float *, const float *, float *, const int *, const int *, const int *,
                            const int *, const int *, const int *, int) { return 0; }
extern "C" int cuda_nn_pool(const float *, float *, const int *, const int *, const int *, const int *, const int *, int) { return 0; }
extern "C" int cuda_nn_softmax(const float *, float *, const int *) { return 0; }
#endif
