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

tensor_t *tensor_host_create_view(tensor_t *base, void *data, const int *shape,
                                  const size_t *strides, int ndims)
{
    size_t elements, bytes;
    if (!tensor_shape_metrics(shape, ndims, base->element_size,
                              NULL, &elements, &bytes))
    {
        CUDA_THROW_INVALID("Host view shape or byte size exceed supported limits");
        return NULL;
    }
    tensor_t *view = ecalloc(1, sizeof(tensor_t));
    view->shape = emalloc(ndims * sizeof(int));
    view->strides = emalloc(ndims * sizeof(size_t));
    memcpy(view->shape, shape, ndims * sizeof(int));
    memcpy(view->strides, strides, ndims * sizeof(size_t));
    view->ndims = ndims;
    view->dtype = base->dtype;
    view->element_size = base->element_size;
    view->total_size = elements;
    view->data = data;
    view->host_pinned = base->host_pinned;
    view->is_view = 1;
    view->is_contiguous_cached = -1;
    view->ref_count = 1;
    view->base_tensor = base;
    base->ref_count++;
    return view;
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
