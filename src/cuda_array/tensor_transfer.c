#include "tensor_transfer.h"
#include "cuda_exceptions.h"
#include "contiguous_array_ce.h"

static void build_php_array(zval *result, const void *data, int dim, const tensor_t *tensor, size_t offset)
{
    array_init(result);
    int size = tensor->shape[dim];
    size_t stride = tensor->strides[dim];

    for (int i = 0; i < size; i++)
    {
        size_t child_offset = offset + i * stride;

        if (dim == tensor->ndims - 1)
        {
            zval val;
            switch (tensor->dtype)
            {
            case DTYPE_FLOAT32:
                ZVAL_DOUBLE(&val, (double)((const float *)data)[child_offset]);
                break;
            case DTYPE_FLOAT64:
                ZVAL_DOUBLE(&val, ((const double *)data)[child_offset]);
                break;
            case DTYPE_INT8:
                ZVAL_LONG(&val, (zend_long)((const int8_t *)data)[child_offset]);
                break;
            case DTYPE_INT16:
                ZVAL_LONG(&val, (zend_long)((const int16_t *)data)[child_offset]);
                break;
            case DTYPE_INT32:
                ZVAL_LONG(&val, (zend_long)((const int32_t *)data)[child_offset]);
                break;
            case DTYPE_INT64:
                ZVAL_LONG(&val, (zend_long)((const int64_t *)data)[child_offset]);
                break;
            case DTYPE_UINT8:
                ZVAL_LONG(&val, (zend_long)((const uint8_t *)data)[child_offset]);
                break;
            case DTYPE_UINT16:
                ZVAL_LONG(&val, (zend_long)((const uint16_t *)data)[child_offset]);
                break;
            case DTYPE_UINT32:
                ZVAL_LONG(&val, (zend_long)((const uint32_t *)data)[child_offset]);
                break;
            case DTYPE_UINT64:
                ZVAL_LONG(&val, (zend_long)((const uint64_t *)data)[child_offset]);
                break;
            case DTYPE_BOOL:
                ZVAL_BOOL(&val, ((const bool *)data)[child_offset]);
                break;
            default:
                ZVAL_NULL(&val);
                break;
            }
            zend_hash_index_update(Z_ARRVAL_P(result), i, &val);
        }
        else
        {
            zval sub;
            build_php_array(&sub, data, dim + 1, tensor, child_offset);
            zend_hash_index_update(Z_ARRVAL_P(result), i, &sub);
        }
    }
}

void tensor_to_php_array(zval *result, const tensor_t *tensor)
{
    size_t span = tensor->total_size ? 1 : 0;
    for (int d = 0; span && d < tensor->ndims; d++)
        span += (size_t)(tensor->shape[d] - 1) * tensor->strides[d];
    void *host_data = emalloc(span * tensor->element_size);
    cudaError_t status = span ? cudaMemcpy(host_data, tensor->data,
        span * tensor->element_size, cudaMemcpyDeviceToHost) : cudaSuccess;

    if (status != cudaSuccess)
    {
        efree(host_data);
        CUDA_THROW_RUNTIME("GPU Copy Failed: %s", cudaGetErrorString(status));
        return;
    }

    tensor_t descriptor = *tensor;
    int scalar_shape[] = {1};
    size_t scalar_stride[] = {1};
    if (!descriptor.ndims)
    {
        descriptor.ndims = 1;
        descriptor.shape = scalar_shape;
        descriptor.strides = scalar_stride;
    }
    build_php_array(result, host_data, 0, &descriptor, 0);
    efree(host_data);
}

tensor_t *tensor_copy_to_host(const tensor_t *tensor)
{
    tensor_t *host_tensor = ecalloc(1, sizeof(tensor_t));
    host_tensor->dtype = tensor->dtype;
    host_tensor->ndims = tensor->ndims;
    host_tensor->element_size = dtype_to_size(tensor->dtype);

    if (tensor->ndims > 0)
    {
        host_tensor->shape = emalloc(sizeof(int) * tensor->ndims);
        memcpy(host_tensor->shape, tensor->shape, sizeof(int) * tensor->ndims);

        host_tensor->strides = emalloc(sizeof(size_t) * tensor->ndims);
        size_t stride = 1;
        for (int d = tensor->ndims - 1; d >= 0; d--)
        {
            host_tensor->strides[d] = stride;
            stride *= tensor->shape[d];
        }
    }

    host_tensor->total_size = 1;
    for (int i = 0; i < tensor->ndims; i++)
    {
        host_tensor->total_size *= tensor->shape[i];
    }

    host_tensor->allocated_size = host_tensor->total_size * host_tensor->element_size;
    host_tensor->data = allocate_for_dtype(host_tensor->dtype, host_tensor->total_size);
    if (!host_tensor->data)
    {
        if (host_tensor->shape)
            efree(host_tensor->shape);
        if (host_tensor->strides)
            efree(host_tensor->strides);
        efree(host_tensor);
        CUDA_THROW_OOM("Failed to allocate host memory");
        return NULL;
    }

    size_t span = tensor->total_size ? 1 : 0;
    for (int d = 0; span && d < tensor->ndims; d++)
        span += (size_t)(tensor->shape[d] - 1) * tensor->strides[d];
    int contiguous = 1;
    for (int d = 0; tensor->total_size && d < tensor->ndims; d++)
        if (tensor->shape[d] > 1 && tensor->strides[d] != host_tensor->strides[d]) contiguous = 0;
    void *storage = contiguous ? host_tensor->data : emalloc(span * tensor->element_size);
    cudaError_t status = span ? cudaMemcpy(storage, tensor->data,
                                   span * tensor->element_size, cudaMemcpyDeviceToHost) : cudaSuccess;
    if (status == cudaSuccess && !contiguous)
        for (size_t i = 0; i < tensor->total_size; i++)
        {
            size_t remaining = i, offset = 0;
            for (int d = tensor->ndims - 1; d >= 0; d--)
            {
                offset += (remaining % tensor->shape[d]) * tensor->strides[d];
                remaining /= tensor->shape[d];
            }
            memcpy((char *)host_tensor->data + i * tensor->element_size,
                   (char *)storage + offset * tensor->element_size, tensor->element_size);
        }
    if (!contiguous) efree(storage);
    if (status != cudaSuccess)
    {
        efree(host_tensor->data);
        if (host_tensor->shape)
            efree(host_tensor->shape);
        if (host_tensor->strides)
            efree(host_tensor->strides);
        efree(host_tensor);
        CUDA_THROW_RUNTIME("CUDA error copying data to host: %s", cudaGetErrorString(status));
        return NULL;
    }

    host_tensor->is_on_gpu = 0;
    host_tensor->ref_count = 1;
    return host_tensor;
}