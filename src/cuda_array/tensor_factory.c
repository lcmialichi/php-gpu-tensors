#include "tensor_factory.h"
#include "cuda_exceptions.h"
#include "memory_pool.h"
#include <time.h>
#include <curand.h>
#include "data_types.h"
#include "factory_kernels.h"
#include <stdbool.h>

#define DEFINE_FLATTENER(type_name, c_type)                                              \
    static void flatten_php_array_to_##type_name(zval *data, c_type *buffer, int *index) \
    {                                                                                    \
        if (Z_TYPE_P(data) == IS_ARRAY)                                                  \
        {                                                                                \
            HashTable *ht = Z_ARRVAL_P(data);                                            \
            zval *current;                                                               \
            ZEND_HASH_FOREACH_VAL(ht, current)                                           \
            {                                                                            \
                flatten_php_array_to_##type_name(current, buffer, index);                \
            }                                                                            \
            ZEND_HASH_FOREACH_END();                                                     \
            return;                                                                      \
        }                                                                                \
        c_type value;                                                                    \
        if (Z_TYPE_P(data) == IS_DOUBLE)                                                 \
            value = (c_type)Z_DVAL_P(data);                                              \
        else if (Z_TYPE_P(data) == IS_LONG)                                              \
            value = (c_type)Z_LVAL_P(data);                                              \
        else if (Z_TYPE_P(data) == IS_TRUE)                                              \
            value = (c_type)1;                                                           \
        else if (Z_TYPE_P(data) == IS_FALSE)                                             \
            value = (c_type)0;                                                           \
        else                                                                             \
            value = (c_type)0;                                                           \
        buffer[(*index)++] = value;                                                      \
    }

DEFINE_FLATTENER(float32, float)
DEFINE_FLATTENER(float64, double)
DEFINE_FLATTENER(int8, int8_t)
DEFINE_FLATTENER(int16, int16_t)
DEFINE_FLATTENER(int32, int32_t)
DEFINE_FLATTENER(int64, int64_t)
DEFINE_FLATTENER(uint8, uint8_t)
DEFINE_FLATTENER(uint16, uint16_t)
DEFINE_FLATTENER(uint32, uint32_t)
DEFINE_FLATTENER(uint64, uint64_t)
DEFINE_FLATTENER(_bool, bool)

#define PINNED_TRANSFER_THRESHOLD_BYTES (1024 * 1024)

static void flatten_php_array(zval *data, float *flat_array, int *index);
static void extract_shape_from_array(zval *data, int *shape, int *ndims);
static size_t calculate_total_size(zval *data);
static cudaError_t cuda_flatten_php_array_to_gpu(zval *data, void *gpu_data, int *index, size_t total_size, dtype_t dtype);
static cudaError_t cuda_copy_host_buffer_to_gpu(void *gpu_data, const void *host_data, size_t byte_count);

tensor_t *tensor_cast_string(tensor_t *tensor, const char *new_dtype_str)
{
    if (!tensor || !new_dtype_str)
    {
        return NULL;
    }

    dtype_t new_dtype = dtype_from_string(new_dtype_str);
    if (new_dtype >= DTYPE_COUNT || new_dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Invalid dtype string: %s", new_dtype_str);
        return NULL;
    }

    return tensor_cast(tensor, new_dtype);
}

tensor_t *create_tensor_from_php_array(zval *data, dtype_t dtype)
{
    int shape[10] = {0};
    int ndims = 0;

    extract_shape_from_array(data, shape, &ndims);

    if (ndims == 0)
    {
        CUDA_THROW_INVALID("Invalid array: cannot determine dimensions");
        return NULL;
    }

    tensor_t *tensor = cuda_tensor_create_empty_with_dtype(shape, ndims, dtype);
    if (!tensor)
    {
        CUDA_THROW_OOM("Failed to create empty tensor");
        return NULL;
    }

    size_t total_size = calculate_total_size(data);
    int index = 0;

    cudaError_t cuda_status = cuda_flatten_php_array_to_gpu(
        data,
        tensor->data,
        &index,
        total_size,
        dtype);

    if (cuda_status != cudaSuccess)
    {
        cuda_tensor_destroy(tensor);
        CUDA_THROW_RUNTIME("Failed to copy data to GPU: %s", cudaGetErrorString(cuda_status));
        return NULL;
    }

    return tensor;
}

tensor_t *cuda_tensor_create_from_host_buffer(int *shape, int ndims, dtype_t dtype, const void *host_data, size_t byte_count)
{
    tensor_t *tensor = cuda_tensor_create_empty_with_dtype(shape, ndims, dtype);
    if (!tensor)
    {
        return NULL;
    }

    size_t expected_bytes = tensor->total_size * tensor->element_size;
    if (byte_count != expected_bytes)
    {
        cuda_tensor_destroy(tensor);
        CUDA_THROW_INVALID("Host buffer size mismatch: expected %zu bytes, got %zu", expected_bytes, byte_count);
        return NULL;
    }

    cudaError_t status = cuda_copy_host_buffer_to_gpu(tensor->data, host_data, byte_count);
    if (status != cudaSuccess)
    {
        cuda_tensor_destroy(tensor);
        CUDA_THROW_RUNTIME("Failed to copy host buffer to GPU: %s", cudaGetErrorString(status));
        return NULL;
    }

    return tensor;
}

int cuda_tensor_get_scalar_value(tensor_t *t, float *result_val, int index)
{
    size_t byte_offset = (size_t)index * t->element_size;
    void *gpu_source_ptr = (void *)((char *)t->data + byte_offset);
    if (byte_offset >= (size_t)t->total_size * t->element_size)
    {
        CUDA_THROW_INVALID("Index out of bounds.");
        return FAILURE;
    }

    cudaError_t status = cudaMemcpy(
        result_val,
        gpu_source_ptr,
        t->element_size,
        cudaMemcpyDeviceToHost);

    if (status != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Failed to copy scalar data from GPU: %s", cudaGetErrorString(status));
        return FAILURE;
    }

    return SUCCESS;
}

tensor_t *cuda_tensor_create_with_value(int *shape, int ndims, scalar_value_t value, dtype_t dtype)
{
    scalar_value_t scalar_value = cast_single_value(value, dtype);
    if (scalar_value.dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Failed create CudaArray object with dtype: %s, received %s as value.",
                         dtype_to_string(dtype),
                         dtype_to_string(value.dtype));

        return NULL;
    }

    tensor_t *tensor = cuda_tensor_create_empty_with_dtype(shape, ndims, dtype);
    if (!tensor)
    {
        return NULL;
    }

    launch_assign_scalar_val_kernel(tensor->data, tensor->dtype, scalar_value, tensor->total_size);
    return tensor;
}

void destroy_curand_generator(curandGenerator_t generator)
{
    if (generator)
    {
        curandDestroyGenerator(generator);
    }
}

int set_rand_tensor_data(void *data,
                         size_t size,
                         unsigned long long seed,
                         scalar_value_t min_value,
                         scalar_value_t max_value,
                         dtype_t dtype)
{
    curandGenerator_t generator = NULL;
    curandStatus_t status;

    float *temp = cuda_mem_alloc(sizeof(float) * size);
    if (temp == NULL)
    {
        CUDA_THROW_OOM("CUDA Out of Memory: Failed to allocate temporary buffer for Random.");
        return FAILURE;
    }

    status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS)
        return FAILURE;

    if (seed == 0)
    {
        seed = (unsigned long long)time(NULL);
    }

    status = curandSetPseudoRandomGeneratorSeed(generator, seed);
    if (status != CURAND_STATUS_SUCCESS)
    {
        destroy_curand_generator(generator);
        return FAILURE;
    }

    status = curandGenerateUniform(generator, temp, size);
    if (status != CURAND_STATUS_SUCCESS)
    {
        destroy_curand_generator(generator);
        return FAILURE;
    }

    cudaDeviceSynchronize();

    if (dtype == DTYPE_BOOL)
    {
        launch_bernoulli_kernel(temp, data, size, 0.5f);
    }
    else
    {
        launch_scale_range_kernel(temp, data, dtype, min_value, max_value, size);
    }

    cudaError_t err = cudaDeviceSynchronize();

    destroy_curand_generator(generator);
    cuda_mem_free(temp);

    if (err != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Kernel failed: %s", cudaGetErrorString(err));
        return FAILURE;
    }

    return SUCCESS;
}

tensor_t *cuda_tensor_create_rand(
    int *shape,
    int ndims,
    scalar_value_t min_value,
    scalar_value_t max_value,
    dtype_t dtype,
    unsigned long long seed)
{
    tensor_t *tensor = cuda_tensor_create_empty_with_dtype(shape, ndims, dtype);
    if (!tensor)
    {
        CUDA_THROW_OOM("Unable to create random tensor.");
        return NULL;
    }

    if (can_cast_unsafe(min_value.dtype, dtype) != 1 || can_cast_unsafe(max_value.dtype, dtype) != 1)
    {
        CUDA_THROW_INVALID("Failed to cast min and max value to dtype: %s.", dtype_to_string(dtype));
        return NULL;
    }

    scalar_value_t casted_min = cast_single_value(min_value, dtype);
    scalar_value_t casted_max = cast_single_value(max_value, dtype);

    if (set_rand_tensor_data(
            tensor->data,
            tensor->total_size,
            seed,
            casted_min,
            casted_max,
            dtype) != SUCCESS)
    {

        CUDA_THROW_RUNTIME("Kernel failed.");
        cuda_tensor_destroy(tensor);
        return NULL;
    }

    return tensor;
}

tensor_t *cuda_tensor_create_empty(const int shape[], int ndims)
{
    return cuda_tensor_create_float(shape, ndims, NULL);
}

tensor_t *cuda_tensor_create_empty_dtype(const int shape[], int ndims, dtype_t dtype)
{
    return cuda_tensor_create(shape, ndims, NULL, dtype);
}

tensor_t *cuda_tensor_create(const int shape[], int ndims, const void *data, dtype_t dtype)
{
    tensor_t *tensor = cuda_tensor_create_with_dtype((int *)shape, ndims, dtype);
    if (!tensor)
        return NULL;

    if (data)
    {
        cudaError_t err = cudaMemcpy(tensor->data, data,
                                     tensor->allocated_size,
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cuda_tensor_destroy(tensor);
            CUDA_THROW_RUNTIME("Failed to copy data to GPU: %s", cudaGetErrorString(err));
            return NULL;
        }
    }

    return tensor;
}

static tensor_t *cuda_tensor_create_on_host_impl(const int shape[], int ndims, void *data, dtype_t dtype, int pinned)
{
    tensor_t *tensor = (tensor_t *)ecalloc(1, sizeof(tensor_t));
    if (!tensor)
        return NULL;

    size_t element_size = dtype_size(dtype);

    tensor->dtype = dtype;
    tensor->ndims = ndims;
    tensor->shape = (int *)emalloc(ndims * sizeof(int));
    memcpy(tensor->shape, shape, ndims * sizeof(int));

    tensor->strides = (size_t *)emalloc(ndims * sizeof(size_t));

    size_t stride = 1;
    for (int i = ndims - 1; i >= 0; i--)
    {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    tensor->total_size = stride;
    tensor->is_view = 0;
    tensor->offset = 0;
    tensor->slices = NULL;
    tensor->num_slices = 0;
    tensor->ref_count = 1;
    tensor->d_shape = NULL;
    tensor->d_strides = NULL;
    tensor->element_size = element_size;
    tensor->is_on_gpu = 0;
    tensor->host_pinned = pinned;
    tensor->is_contiguous_cached = -1;

    size_t required_bytes = tensor->total_size * element_size;
    tensor->allocated_size = required_bytes;

    if (pinned)
    {
        if (cudaMallocHost(&tensor->data, required_bytes) != cudaSuccess)
            tensor->data = NULL;
    }
    else
    {
        tensor->data = emalloc(required_bytes);
    }
    if (!tensor->data)
    {
        efree(tensor->strides);
        efree(tensor->shape);
        efree(tensor);
        CUDA_THROW_OOM("Failed to allocate Host memory for tensor.");
        return NULL;
    }

    if (data)
    {
        memcpy(tensor->data, data, required_bytes);
    }
    else
    {
        memset(tensor->data, 0, required_bytes);
    }

    return tensor;
}

tensor_t *cuda_tensor_create_on_host(const int shape[], int ndims, void *data, dtype_t dtype)
{
    return cuda_tensor_create_on_host_impl(shape, ndims, data, dtype, 0);
}

tensor_t *cuda_tensor_create_on_host_pinned(const int shape[], int ndims, void *data, dtype_t dtype)
{
    return cuda_tensor_create_on_host_impl(shape, ndims, data, dtype, 1);
}

tensor_t *cuda_tensor_create_float(const int shape[], int ndims, const float data[])
{
    return cuda_tensor_create(shape, ndims, data, DTYPE_FLOAT32);
}

tensor_t *cuda_tensor_create_int(const int shape[], int ndims, const int data[])
{
    return cuda_tensor_create(shape, ndims, data, DTYPE_INT32);
}

tensor_t *cuda_tensor_create_scalar(float value, int *shape, int ndims)
{
    size_t total_size = 1;
    for (int i = 0; i < ndims; i++)
    {
        total_size *= shape[i];
    }

    float *host_data = (float *)emalloc(total_size * sizeof(float));
    for (size_t i = 0; i < total_size; i++)
    {
        host_data[i] = value;
    }

    tensor_t *tensor = cuda_tensor_create_float(shape, ndims, host_data);
    efree(host_data);

    return tensor;
}

tensor_t *resolve_result_tensor(tensor_t *t)
{
    return cuda_tensor_create_empty_dtype(t->shape, t->ndims, t->dtype);
}

static cudaError_t cuda_flatten_php_array_to_gpu(zval *data, void *gpu_data, int *index, size_t total_size, dtype_t dtype)
{
    void *host_data;
    size_t el_size = dtype_size(dtype);
    size_t total_bytes = total_size * el_size;
    bool use_pinned_host = total_bytes >= PINNED_TRANSFER_THRESHOLD_BYTES;

    if (el_size == 0)
    {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaSuccess;
    if (use_pinned_host)
    {
        status = cudaMallocHost(&host_data, total_bytes);
        if (status != cudaSuccess)
            return status;
    }
    else
    {
        host_data = emalloc(total_bytes);
    }

    int host_index = 0;

    switch (dtype)
    {
    case DTYPE_FLOAT32:
        flatten_php_array_to_float32(data, (float *)host_data, &host_index);
        break;
    case DTYPE_FLOAT64:
        flatten_php_array_to_float64(data, (double *)host_data, &host_index);
        break;
    case DTYPE_INT32:
        flatten_php_array_to_int32(data, (int32_t *)host_data, &host_index);
        break;
    case DTYPE_INT8:
        flatten_php_array_to_int8(data, (int8_t *)host_data, &host_index);
        break;
    case DTYPE_INT16:
        flatten_php_array_to_int16(data, (int16_t *)host_data, &host_index);
        break;
    case DTYPE_INT64:
        flatten_php_array_to_int64(data, (int64_t *)host_data, &host_index);
        break;
    case DTYPE_UINT8:
        flatten_php_array_to_uint8(data, (uint8_t *)host_data, &host_index);
        break;
    case DTYPE_UINT16:
        flatten_php_array_to_uint16(data, (uint16_t *)host_data, &host_index);
        break;
    case DTYPE_UINT32:
        flatten_php_array_to_uint32(data, (uint32_t *)host_data, &host_index);
        break;
    case DTYPE_UINT64:
        flatten_php_array_to_uint64(data, (uint64_t *)host_data, &host_index);
        break;
    case DTYPE_BOOL:
        flatten_php_array_to__bool(data, (bool *)host_data, &host_index);
        break;
    default:
        if (use_pinned_host)
            cudaFreeHost(host_data);
        else
            efree(host_data);
        return cudaErrorInvalidValue;
    }

    status = cudaMemcpy(gpu_data, host_data, total_bytes, cudaMemcpyHostToDevice);

    if (use_pinned_host)
        cudaFreeHost(host_data);
    else
        efree(host_data);
    *index = host_index;
    return status;
}

static cudaError_t cuda_copy_host_buffer_to_gpu(void *gpu_data, const void *host_data, size_t byte_count)
{
    if (byte_count == 0)
    {
        return cudaSuccess;
    }
    return cudaMemcpy(gpu_data, host_data, byte_count, cudaMemcpyHostToDevice);
}

static void flatten_php_array(zval *data, float *flat_array, int *index)
{
    if (Z_TYPE_P(data) != IS_ARRAY)
    {
        if (Z_TYPE_P(data) == IS_LONG)
        {
            flat_array[(*index)++] = (float)Z_LVAL_P(data);
        }
        else if (Z_TYPE_P(data) == IS_DOUBLE)
        {
            flat_array[(*index)++] = (float)Z_DVAL_P(data);
        }
        else if (Z_TYPE_P(data) == IS_TRUE)
        {
            flat_array[(*index)++] = 1.0f;
        }
        else if (Z_TYPE_P(data) == IS_FALSE)
        {
            flat_array[(*index)++] = 0.0f;
        }
        return;
    }

    HashTable *ht = Z_ARRVAL_P(data);
    zval *current;
    ZEND_HASH_FOREACH_VAL(ht, current)
    {
        flatten_php_array(current, flat_array, index);
    }
    ZEND_HASH_FOREACH_END();
}

static void extract_shape_from_array(zval *data, int *shape, int *ndims)
{
    *ndims = 0;

    void extract_shape_recursive(zval * arr, int current_depth)
    {
        if (Z_TYPE_P(arr) != IS_ARRAY)
            return;
        if (current_depth >= 10)
            return;

        HashTable *arr_ht = Z_ARRVAL_P(arr);
        int count = zend_array_count(arr_ht);

        if (count == 0)
            return;

        shape[current_depth] = count;
        if (current_depth >= *ndims)
        {
            *ndims = current_depth + 1;
        }

        if (count > 0)
        {
            zval *first = zend_hash_index_find(arr_ht, 0);
            if (first != NULL)
            {
                extract_shape_recursive(first, current_depth + 1);
            }
        }
    }

    extract_shape_recursive(data, 0);
}

static size_t calculate_total_size(zval *data)
{
    if (Z_TYPE_P(data) != IS_ARRAY)
    {
        return 1;
    }

    size_t total = 1;
    HashTable *ht = Z_ARRVAL_P(data);
    zval *first = zend_hash_index_find(ht, 0);

    if (first != NULL)
    {
        total = zend_array_count(ht) * calculate_total_size(first);
    }

    return total;
}

tensor_t *cuda_tensor_clone(tensor_t *base_tensor)
{
    if (base_tensor == NULL)
    {
        return NULL;
    }

    tensor_t *new_tensor = cuda_tensor_create(base_tensor->shape, base_tensor->ndims, base_tensor->data, base_tensor->dtype);

    if (new_tensor == NULL || new_tensor->data == NULL)
    {
        if (new_tensor)
            cuda_tensor_destroy(new_tensor);
        return NULL;
    }
    return new_tensor;
}