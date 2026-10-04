#include "tensor_import.h"
#include "cuda_exceptions.h"
#include "tensor_factory.h"
#include "main/php_streams.h"
#include <limits.h>
#include <stdint.h>

#define PINNED_FILE_THRESHOLD (1024 * 1024)

tensor_t *tensor_import_stream(php_stream *stream, const int *shape, int ndims, dtype_t dtype, size_t bytes)
{
    void *buffer = NULL;
    int pinned = 0;
    if (bytes >= PINNED_FILE_THRESHOLD && cudaMallocHost(&buffer, bytes) == cudaSuccess)
    {
        pinned = 1;
    }
    else
    {
        cudaGetLastError();
        buffer = emalloc(bytes);
    }

    size_t read_bytes = 0;
    while (read_bytes < bytes)
    {
        size_t received = php_stream_read(stream, (char *)buffer + read_bytes, bytes - read_bytes);
        if (received == 0)
            break;
        read_bytes += received;
    }

    char trailing;
    size_t extra = read_bytes == bytes ? php_stream_read(stream, &trailing, 1) : 0;
    tensor_t *tensor = NULL;
    if (read_bytes == bytes && extra == 0)
        tensor = cuda_tensor_create_from_host_buffer((int *)shape, ndims, dtype, buffer, bytes);
    else
        CUDA_THROW_INVALID("Tensor file size does not match shape and dtype");

    if (pinned)
        cudaFreeHost(buffer);
    else
        efree(buffer);
    return tensor;
}

int tensor_import_shape(zval *shape_array, int shape[MAX_DIMS], size_t *elements)
{
    size_t count = zend_hash_num_elements(Z_ARRVAL_P(shape_array));
    if (count == 0 || count > MAX_DIMS)
    {
        CUDA_THROW_INVALID("Shape must contain between 1 and %d dimensions", MAX_DIMS);
        return 0;
    }

    *elements = 1;
    size_t index = 0;
    zval *dimension;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_array), dimension)
    {
        if (Z_TYPE_P(dimension) != IS_LONG || Z_LVAL_P(dimension) < 0 || Z_LVAL_P(dimension) > INT_MAX ||
            (Z_LVAL_P(dimension) && *elements > SIZE_MAX / (size_t)Z_LVAL_P(dimension)))
        {
            CUDA_THROW_INVALID("Shape dimensions must be nonnegative integers within supported limits");
            return 0;
        }
        shape[index++] = (int)Z_LVAL_P(dimension);
        *elements *= (size_t)Z_LVAL_P(dimension);
    }
    ZEND_HASH_FOREACH_END();

    return (int)count;
}

tensor_t *tensor_import_file(zend_string *path, const int *shape, int ndims, dtype_t dtype, size_t bytes)
{
    php_stream *stream = php_stream_open_wrapper(ZSTR_VAL(path), "rb", 0, NULL);
    if (!stream)
    {
        CUDA_THROW_RUNTIME("Cannot open tensor file: %s", ZSTR_VAL(path));
        return NULL;
    }

    tensor_t *tensor = tensor_import_stream(stream, shape, ndims, dtype, bytes);
    php_stream_close(stream);
    return tensor;
}