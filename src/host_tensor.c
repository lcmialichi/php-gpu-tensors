#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "host_tensor.h"
#include "tensor_shape.h"

static tensor_t *tensor_host_create(const int shape[], int ndims, void *data,
                                     dtype_t dtype, int pinned)
{
    size_t elements, bytes;
    size_t element_size = dtype_size(dtype);
    if (!tensor_shape_metrics(shape, ndims, element_size, NULL, &elements, &bytes))
    {
        CUDA_THROW_INVALID("Host tensor shape, strides or byte size exceed supported limits");
        return NULL;
    }
    if (pinned)
    {
#ifdef HAVE_CUDA
        if (!cuda_tensor_require_context()) return NULL;
#else
        CUDA_THROW_RUNTIME("Pinned host memory requires a CUDA-enabled build");
        return NULL;
#endif
    }
    tensor_t *tensor = ecalloc(1, sizeof(tensor_t));
    tensor->dtype = dtype;
    tensor->ndims = ndims;
    tensor->shape = emalloc(ndims * sizeof(int));
    memcpy(tensor->shape, shape, ndims * sizeof(int));
    tensor->strides = emalloc(ndims * sizeof(size_t));
    size_t stride = 1;
    for (int d = ndims - 1; d >= 0; d--)
    {
        tensor->strides[d] = stride;
        stride *= (size_t)shape[d];
    }
    tensor->total_size = elements;
    tensor->ref_count = 1;
    tensor->element_size = element_size;
    tensor->host_pinned = pinned;
    tensor->is_contiguous_cached = -1;
    tensor->allocated_size = bytes;
    if (!bytes) return tensor;

#ifdef HAVE_CUDA
    if (pinned)
    {
        if (cudaMallocHost(&tensor->data, bytes) != cudaSuccess)
            tensor->data = NULL;
    }
    else
#endif
        tensor->data = emalloc(bytes);
    if (!tensor->data)
    {
        efree(tensor->strides);
        efree(tensor->shape);
        efree(tensor);
        CUDA_THROW_OOM("Failed to allocate Host memory for tensor.");
        return NULL;
    }
    if (data) memcpy(tensor->data, data, bytes);
    else memset(tensor->data, 0, bytes);
    return tensor;
}

tensor_t *cuda_tensor_create_on_host(const int shape[], int ndims, void *data, dtype_t dtype)
{
    return tensor_host_create(shape, ndims, data, dtype, 0);
}

tensor_t *cuda_tensor_create_on_host_pinned(const int shape[], int ndims, void *data, dtype_t dtype)
{
    return tensor_host_create(shape, ndims, data, dtype, 1);
}

void tensor_host_destroy(tensor_t *tensor)
{
#ifdef HAVE_CUDA
    cuda_tensor_destroy(tensor);
#else
    if (!tensor || --tensor->ref_count > 0) return;
    if (tensor->base_tensor) tensor_host_destroy(tensor->base_tensor);
    else if (tensor->data) efree(tensor->data);
    if (tensor->shape) efree(tensor->shape);
    if (tensor->strides) efree(tensor->strides);
    if (tensor->slices) efree(tensor->slices);
    efree(tensor);
#endif
}
