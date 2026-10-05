#include "tensor_transfer.h"
#include "cuda_exceptions.h"
#include "contiguous_array_ce.h"
#include "factory_kernels.h"
#include "memory_pool.h"

static void contiguous_descriptor(tensor_t *descriptor, const tensor_t *tensor, size_t strides[MAX_DIMS])
{
    *descriptor = *tensor;
    size_t stride = 1;
    for (int d = tensor->ndims - 1; d >= 0; d--)
    {
        strides[d] = stride;
        stride *= (size_t)tensor->shape[d];
    }
    descriptor->strides = strides;
}

static int tensor_transfer_is_contiguous(const tensor_t *tensor)
{
    size_t expected = 1;
    for (int d = tensor->ndims - 1; d >= 0; d--)
    {
        if (tensor->shape[d] > 1 && tensor->strides[d] != expected) return 0;
        expected *= (size_t)tensor->shape[d];
    }
    return 1;
}

static int tensor_download(const tensor_t *tensor, void *destination)
{
    size_t bytes = tensor->total_size * tensor->element_size;
    if (!bytes) return 1;
    size_t span = 1;
    for (int d = tensor->ndims - 1; d >= 0; d--)
    {
        span += (size_t)(tensor->shape[d] - 1) * tensor->strides[d];
    }
    size_t span_bytes = span * tensor->element_size;
    cudaError_t status;
    if (tensor_transfer_is_contiguous(tensor))
    {
        status = cudaMemcpy(destination, tensor->data, bytes, cudaMemcpyDeviceToHost);
    }
    else if (bytes >= 64 * 1024 || (span_bytes >= 64 * 1024 && span / tensor->total_size >= 4))
    {
        /* Pack on device when a host gather or copying gaps would dominate the transfer. */
        void *packed = cuda_mem_alloc(bytes);
        if (!packed)
        {
            CUDA_THROW_OOM("Failed to allocate strided transfer workspace");
            return 0;
        }
        status = launch_pack_strided(tensor->data, packed, tensor->total_size, tensor->element_size,
                                      tensor->shape, tensor->strides, tensor->ndims);
        if (status == cudaSuccess)
            status = cudaMemcpy(destination, packed, bytes, cudaMemcpyDeviceToHost);
        cuda_mem_free(packed);
    }
    else
    {
        void *storage = emalloc(span_bytes);
        status = cudaMemcpy(storage, tensor->data, span_bytes, cudaMemcpyDeviceToHost);
        if (status == cudaSuccess)
            for (size_t i = 0; i < tensor->total_size; i++)
            {
                size_t remaining = i, offset = 0;
                for (int d = tensor->ndims - 1; d >= 0; d--)
                {
                    offset += (remaining % tensor->shape[d]) * tensor->strides[d];
                    remaining /= tensor->shape[d];
                }
                memcpy((char *)destination + i * tensor->element_size,
                       (char *)storage + offset * tensor->element_size, tensor->element_size);
            }
        efree(storage);
    }
    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("GPU download failed: %s", cudaGetErrorString(status));
        return 0;
    }
    return 1;
}

typedef void (*php_leaf_writer)(HashTable *, const void *, size_t, size_t, int);

typedef struct {
    const tensor_t *tensor;
    void *data;
    size_t capacity, start, end;
} php_export_window;

#define DEFINE_PHP_LEAF(name, c_type, setter, cast_type) \
    static void php_leaf_##name(HashTable *array, const void *data, size_t offset, size_t stride, int size) \
    { \
        const c_type *source = (const c_type *)data + offset; \
        ZEND_HASH_FILL_PACKED(array); \
        for (int i = 0; i < size; i++) { \
            zval value; \
            setter(&value, (cast_type)source[(size_t)i * stride]); \
            ZEND_HASH_FILL_ADD(&value); \
        } \
        ZEND_HASH_FILL_END(); \
    }

DEFINE_PHP_LEAF(float32, float, ZVAL_DOUBLE, double)
DEFINE_PHP_LEAF(float64, double, ZVAL_DOUBLE, double)
DEFINE_PHP_LEAF(int8, int8_t, ZVAL_LONG, zend_long)
DEFINE_PHP_LEAF(int16, int16_t, ZVAL_LONG, zend_long)
DEFINE_PHP_LEAF(int32, int32_t, ZVAL_LONG, zend_long)
DEFINE_PHP_LEAF(int64, int64_t, ZVAL_LONG, zend_long)
DEFINE_PHP_LEAF(uint8, uint8_t, ZVAL_LONG, zend_long)
DEFINE_PHP_LEAF(uint16, uint16_t, ZVAL_LONG, zend_long)
DEFINE_PHP_LEAF(uint32, uint32_t, ZVAL_LONG, zend_long)
DEFINE_PHP_LEAF(uint64, uint64_t, ZVAL_LONG, zend_long)
DEFINE_PHP_LEAF(boolean, bool, ZVAL_BOOL, bool)

static const php_leaf_writer php_leaf_writers[] = {
    [DTYPE_FLOAT32] = php_leaf_float32, [DTYPE_FLOAT64] = php_leaf_float64,
    [DTYPE_INT8] = php_leaf_int8, [DTYPE_INT16] = php_leaf_int16,
    [DTYPE_INT32] = php_leaf_int32, [DTYPE_INT64] = php_leaf_int64,
    [DTYPE_UINT8] = php_leaf_uint8, [DTYPE_UINT16] = php_leaf_uint16,
    [DTYPE_UINT32] = php_leaf_uint32, [DTYPE_UINT64] = php_leaf_uint64,
    [DTYPE_BOOL] = php_leaf_boolean,
};

static int build_php_array(zval *result, const void *data, int dim, const tensor_t *tensor,
                           size_t offset, php_leaf_writer writer, php_export_window *window)
{
    int size = tensor->shape[dim];
    array_init_size(result, size);
    if (!size) return 1;
    HashTable *array = Z_ARRVAL_P(result);
    zend_hash_real_init_packed(array);
    if (dim == tensor->ndims - 1)
    {
        if (!window)
        {
            writer(array, data, offset, tensor->strides[dim], size);
            return 1;
        }
        size_t remaining = (size_t)size;
        while (remaining)
        {
            if (offset >= window->end)
            {
                size_t count = window->tensor->total_size - offset;
                if (count > window->capacity) count = window->capacity;
                cudaError_t status = cudaMemcpy(window->data,
                    (const char *)window->tensor->data + offset * tensor->element_size,
                    count * tensor->element_size, cudaMemcpyDeviceToHost);
                if (status != cudaSuccess)
                {
                    CUDA_THROW_RUNTIME("GPU array download failed: %s", cudaGetErrorString(status));
                    return 0;
                }
                window->start = offset;
                window->end = offset + count;
            }
            size_t count = window->end - offset;
            if (count > remaining) count = remaining;
            writer(array, window->data, offset - window->start, 1, (int)count);
            offset += count;
            remaining -= count;
        }
        return 1;
    }
    ZEND_HASH_FILL_PACKED(array);
    for (int i = 0; i < size; i++)
    {
        zval child;
        if (!build_php_array(&child, data, dim + 1, tensor,
                             offset + (size_t)i * tensor->strides[dim], writer, window))
        {
            zval_ptr_dtor(&child);
            ZEND_HASH_FILL_FINISH();
            return 0;
        }
        ZEND_HASH_FILL_ADD(&child);
    }
    ZEND_HASH_FILL_END();
    return 1;
}

void tensor_host_to_php_array(zval *result, const tensor_t *tensor, const void *data)
{
    if (tensor->dtype >= DTYPE_COUNT || !php_leaf_writers[tensor->dtype])
    {
        CUDA_THROW_INVALID("Unsupported dtype for PHP array export");
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
    build_php_array(result, data, 0, &descriptor, 0, php_leaf_writers[tensor->dtype], NULL);
}

void tensor_to_php_array(zval *result, const tensor_t *tensor)
{
    size_t bytes = tensor->total_size * tensor->element_size;
    if (bytes >= 16 * 1024 * 1024 && tensor_transfer_is_contiguous(tensor))
    {
        /* Bound staging RAM and keep conversion close to the CPU cache for large exports. */
        size_t staging_bytes = bytes < 32 * 1024 * 1024 ? bytes : 32 * 1024 * 1024;
        php_export_window window = {tensor, emalloc(staging_bytes),
                                    staging_bytes / tensor->element_size, 0, 0};
        int ok = build_php_array(result, NULL, 0, tensor, 0, php_leaf_writers[tensor->dtype], &window);
        efree(window.data);
        if (!ok)
        {
            zval_ptr_dtor(result);
            ZVAL_UNDEF(result);
        }
        return;
    }
    void *host_data = emalloc(bytes);
    if (!tensor_download(tensor, host_data))
    {
        efree(host_data);
        return;
    }
    tensor_t descriptor;
    size_t strides[MAX_DIMS];
    contiguous_descriptor(&descriptor, tensor, strides);
    tensor_host_to_php_array(result, &descriptor, host_data);
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
    host_tensor->data = emalloc(host_tensor->allocated_size);
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

    if (!tensor_download(tensor, host_tensor->data))
    {
        efree(host_tensor->data);
        if (host_tensor->shape)
            efree(host_tensor->shape);
        if (host_tensor->strides)
            efree(host_tensor->strides);
        efree(host_tensor);
        return NULL;
    }

    host_tensor->is_on_gpu = 0;
    host_tensor->ref_count = 1;
    return host_tensor;
}

zend_string *tensor_to_buffer(const tensor_t *tensor)
{
    size_t bytes = tensor->total_size * tensor->element_size;
    zend_string *result = zend_string_alloc(bytes, 0);
    ZSTR_VAL(result)[bytes] = '\0';
    if (!bytes) return result;

    if (!tensor_download(tensor, ZSTR_VAL(result)))
    {
        zend_string_release(result);
        return NULL;
    }
    return result;
}