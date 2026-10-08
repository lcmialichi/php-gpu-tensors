#include "tensor.h"
#include "autograd.h"
#include "data_types.h"
#include "php.h"
#include "Zend/zend_API.h"
#include <string.h>
#include "memory_pool.h"
#include "operations.h"
#include "cuda_exceptions.h"
#include "fusion.h"
#include "tensor_transfer.h"
#include <math.h>

static int cuda_is_initialized = 0;
static tensor_t *handle_allocation_failure(tensor_t *tensor, const char *message, cudaError_t err_code);

static void compute_strides_from_shape(int *shape, size_t *strides, int ndims)
{
    if (ndims <= 0)
        return;

    size_t stride = 1;
    for (int i = ndims - 1; i >= 0; i--)
    {
        strides[i] = stride;
        stride *= (size_t)shape[i];
    }
}

static int compute_total_size(const int *shape, int ndims, size_t *total)
{
    size_t size = 1;
    for (int i = 0; i < ndims; i++)
    {
        if (shape[i] < 0 || (shape[i] != 0 && size > SIZE_MAX / (size_t)shape[i]))
            return 0;
        size *= (size_t)shape[i];
    }
    *total = size;
    return 1;
}

int is_contiguous(tensor_t *tensor)
{
    if (!tensor)
        return 0;
    if (!tensor->total_size) return 1;

    if (tensor->is_contiguous_cached != -1)
    {
        return tensor->is_contiguous_cached;
    }

    int result = 1;
    size_t expected_stride = 1;

    for (int i = tensor->ndims - 1; i >= 0; i--)
    {
        if (tensor->shape[i] == 0)
        {
            result = 1;
            break;
        }

        if (tensor->strides[i] != expected_stride)
        {
            result = 0;
            break;
        }
        expected_stride *= tensor->shape[i];
    }

    tensor->is_contiguous_cached = result;
    return result;
}

int tensor_can_cast_to(const tensor_t *tensor, dtype_t new_dtype)
{
    if (!tensor)
        return 0;

    if (tensor->dtype == new_dtype)
        return 1;

    return can_safely_cast_to(tensor->dtype, new_dtype);
}

tensor_t *tensor_cast(tensor_t *tensor, dtype_t new_dtype)
{
    if (!tensor)
        return NULL;

    if (tensor->dtype == new_dtype)
    {
        tensor->ref_count++;
        return tensor;
    }

    if (!tensor_can_cast_to(tensor, new_dtype))
    {
        CUDA_THROW_INVALID("Cannot safely cast from %s to %s",
                           dtype_to_string(tensor->dtype), dtype_to_string(new_dtype));
        return NULL;
    }

    if (fusion_active()) return fusion_cast(tensor, new_dtype);
    return fusion_cast_eager(tensor, new_dtype);
}

tensor_t *cuda_tensor_create_with_dtype(int *shape, int ndims, dtype_t dtype)
{
    if ((!shape && ndims > 0) || ndims < 0 || ndims > MAX_DIMS || dtype >= DTYPE_COUNT)
    {
        return NULL;
    }

    tensor_t *tensor = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!tensor)
    {
        return handle_allocation_failure(NULL, "Failed to allocate tensor_t structure", cudaSuccess);
    }

    memset(tensor, 0, sizeof(tensor_t));

    tensor->dtype = dtype;
    tensor->element_size = dtype_size(dtype);
    if (tensor->element_size == 0)
    {
        efree(tensor);
        CUDA_THROW_INVALID("Invalid dtype: %d", dtype);
        return NULL;
    }

    tensor->ndims = ndims;
    tensor->is_on_gpu = 1;
    tensor->is_contiguous_cached = -1;

    if (ndims > 0)
    {
        tensor->shape = (int *)emalloc(ndims * sizeof(int));
        if (!tensor->shape)
        {
            return handle_allocation_failure(tensor, "Failed to allocate shape array", cudaSuccess);
        }
        memcpy(tensor->shape, shape, ndims * sizeof(int));

        tensor->strides = (size_t *)emalloc(ndims * sizeof(size_t));
        if (!tensor->strides)
        {
            return handle_allocation_failure(tensor, "Failed to allocate strides array", cudaSuccess);
        }
        if (!compute_total_size(shape, ndims, &tensor->total_size))
        {
            CUDA_THROW_INVALID("Invalid tensor shape");
            return handle_allocation_failure(tensor, "Invalid tensor shape", cudaSuccess);
        }
        compute_strides_from_shape(shape, tensor->strides, ndims);
    }
    else
    {
        tensor->total_size = 1;
        tensor->shape = NULL;
        tensor->strides = NULL;
    }

    if (tensor->total_size > SIZE_MAX / tensor->element_size)
        return handle_allocation_failure(tensor, "Tensor size overflow", cudaSuccess);
    size_t total_bytes = tensor->total_size * tensor->element_size;
    if (total_bytes > 0)
    {
        tensor->data = cuda_mem_alloc(total_bytes);
        if (!tensor->data)
            return handle_allocation_failure(tensor, "Failed to allocate tensor data", cudaErrorMemoryAllocation);
        tensor->allocated_size = total_bytes;
    }

    tensor->ref_count = 1;

    return tensor;
}

tensor_t *cuda_tensor_create_empty_with_dtype(int *shape, int ndims, dtype_t dtype)
{
    tensor_t *tensor = cuda_tensor_create_with_dtype(shape, ndims, dtype);
    return tensor;
}

int tensors_have_same_dtype(const tensor_t *a, const tensor_t *b)
{
    if (!a || !b)
        return 0;
    return a->dtype == b->dtype;
}

int tensor_validate_dtype(tensor_t *tensor)
{
    if (!tensor)
        return 0;

    if (tensor->dtype >= DTYPE_COUNT)
    {
        CUDA_THROW_INVALID("Invalid dtype: %d", tensor->dtype);
        return 0;
    }

    size_t expected_size = dtype_size(tensor->dtype);
    if (tensor->element_size != expected_size)
    {
        tensor->element_size = expected_size;
    }

    return 1;
}

static inline size_t tensor_element_size(const tensor_t *tensor)
{
    if (!tensor)
        return 0;
    return dtype_size(tensor->dtype);
}

static inline size_t tensor_nbytes(const tensor_t *tensor)
{
    if (!tensor)
        return 0;
    return tensor->total_size * tensor_element_size(tensor);
}

void tensor_update_from_dtype(tensor_t *tensor)
{
    if (!tensor)
        return;

    tensor->element_size = dtype_size(tensor->dtype);
    tensor->is_contiguous_cached = -1;
}

tensor_t *cuda_tensor_create_view_with_dtype(tensor_t *base_tensor, dtype_t new_dtype,
                                             int *shape, size_t *strides,
                                             int dims, size_t offset, size_t total_size)
{
    if (!base_tensor)
        return NULL;
    tensor_t *view = cuda_tensor_create_view(base_tensor, shape, strides, dims, offset, total_size);
    if (!view)
        return NULL;
    view->dtype = new_dtype;
    view->element_size = dtype_size(new_dtype);

    return view;
}

tensor_t *cuda_tensor_allocate_base(const int shape[], int ndims)
{
    return cuda_tensor_create_with_dtype((int *)shape, ndims, DTYPE_FLOAT32);
}

tensor_t *cuda_tensor_create_view(tensor_t *base_tensor, int *shape, size_t *strides,
                                  int dims, size_t offset, size_t total_size)
{
    size_t byte_offset = offset * base_tensor->element_size;
    tensor_t *view = (tensor_t *)emalloc(sizeof(tensor_t));
    memset(view, 0, sizeof(tensor_t));

    view->is_view = 1;
    view->offset = offset;
    view->data = base_tensor->data ? (char *)base_tensor->data + byte_offset : NULL;
    view->total_size = total_size;
    view->ref_count = 1;
    view->ndims = dims;
    view->base_tensor = base_tensor;
    base_tensor->ref_count++;
    view->num_slices = 0;
    view->dtype = base_tensor->dtype;
    view->slices = NULL;
    view->element_size = base_tensor->element_size;
    view->allocated_size = base_tensor->allocated_size;
    view->is_on_gpu = 1;
    view->shape = NULL;
    view->strides = NULL;
    view->d_strides = NULL;
    view->d_shape = NULL;

    view->is_contiguous_cached = -1;

    if (dims > 0)
    {
        view->shape = (int *)emalloc(sizeof(int) * dims);
        memcpy(view->shape, shape, sizeof(int) * dims);

        view->strides = (size_t *)emalloc(sizeof(size_t) * dims);
        memcpy(view->strides, strides, sizeof(size_t) * dims);
    }

    return view;
}

static tensor_t *handle_allocation_failure(tensor_t *tensor, const char *message, cudaError_t err_code)
{
    if (err_code != cudaSuccess)
    {
        if (err_code == cudaErrorMemoryAllocation)
            CUDA_THROW_OOM("%s CUDA Error: %s", message, cudaGetErrorString(err_code));
        else
            CUDA_THROW_RUNTIME("%s CUDA Error: %s", message, cudaGetErrorString(err_code));
    }
    else
    {
        CUDA_THROW_OOM("%s Memory Error.", message);
    }

    if (tensor)
    {
        if (tensor->data && tensor->is_on_gpu)
            cuda_mem_free(tensor->data);
        if (tensor->d_strides)
            cuda_mem_free(tensor->d_strides);
        if (tensor->d_shape)
            cuda_mem_free(tensor->d_shape);
        if (tensor->strides)
            efree(tensor->strides);
        if (tensor->shape)
            efree(tensor->shape);
        efree(tensor);
    }
    return NULL;
}

int lazy_copy_metadata_to_gpu(tensor_t *t)
{
    if (t->ndims == 0) return 1;
    if (t->d_shape && t->d_strides)
        return 1;

    int *shape = t->d_shape ? t->d_shape : cuda_mem_alloc(t->ndims * sizeof(int));
    size_t *strides = t->d_strides ? t->d_strides : cuda_mem_alloc(t->ndims * sizeof(size_t));
    cudaError_t status = cudaSuccess;
    if (shape && strides)
    {
        status = cudaMemcpy(shape, t->shape, t->ndims * sizeof(int), cudaMemcpyHostToDevice);
        if (status == cudaSuccess)
            status = cudaMemcpy(strides, t->strides, t->ndims * sizeof(size_t), cudaMemcpyHostToDevice);
    }
    if (!shape || !strides || status != cudaSuccess)
    {
        if (!t->d_shape && shape)
            cuda_mem_free(shape);
        if (!t->d_strides && strides)
            cuda_mem_free(strides);
        if (!shape || !strides) CUDA_THROW_OOM("Failed to allocate tensor device metadata");
        else CUDA_THROW_RUNTIME("Failed to upload tensor metadata: %s", cudaGetErrorString(status));
        return 0;
    }

    t->d_shape = shape;
    t->d_strides = strides;
    return 1;
}

tensor_t *cuda_tensor_create_sliced_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices)
{
    if (!base_tensor || !slices)
    {
        return NULL;
    }

    size_t base_strides[MAX_DIMS];
    if (base_tensor->strides)
    {
        for (int i = 0; i < base_tensor->ndims; ++i)
        {
            base_strides[i] = base_tensor->strides[i];
        }
    }
    else
    {
        size_t stride = 1;
        for (int i = base_tensor->ndims - 1; i >= 0; --i)
        {
            base_strides[i] = stride;
            stride *= (size_t)base_tensor->shape[i];
        }
    }

    size_t element_offset = 0;

    int view_shape[MAX_DIMS];
    size_t view_strides[MAX_DIMS];
    int view_ndims = 0;

    for (int i = 0; i < base_tensor->ndims; ++i)
    {
        slice_info_t slice = (i < num_slices) ? slices[i] : (slice_info_t){.type = SLICE_ALL};

        switch (slice.type)
        {
        case SLICE_ALL:
            view_shape[view_ndims] = base_tensor->shape[i];
            view_strides[view_ndims] = base_strides[i];
            view_ndims++;
            break;

        case SLICE_INDEX:
        {
            int index = slice.data.index;
            if (index < 0 || index >= base_tensor->shape[i])
            {
                CUDA_THROW_INVALID("Index %d out of bounds for dimension %d (size %d)",
                                 index, i, base_tensor->shape[i]);
                return NULL;
            }

            view_strides[view_ndims] = 0;
            view_shape[view_ndims] = 1;
            element_offset += index * base_strides[i];
            view_ndims++;
            break;
        }

        case SLICE_RANGE:
        {
            int start = slice.data.range.start;
            int end = slice.data.range.end;
            if (start < 0 || end < start || end >= base_tensor->shape[i])
            {
                CUDA_THROW_INVALID("Range [%d:%d] out of bounds for dimension %d (size %d)",
                                 start, end, i, base_tensor->shape[i]);
                return NULL;
            }
            int len = (end - start + 1);
            element_offset += (size_t)start * base_strides[i];

            view_shape[view_ndims] = len;
            view_strides[view_ndims] = base_strides[i];
            view_ndims++;
            break;
        }

        default:
            CUDA_THROW_INVALID("Invalid slice type for dimension %d", i);
            return NULL;
        }
    }

    size_t view_total = 1;
    for (int i = 0; i < view_ndims; ++i)
        view_total *= (size_t)view_shape[i];

    tensor_t *view = cuda_tensor_create_view(base_tensor, view_shape, view_strides, view_ndims,
                                            view_total ? element_offset : 0, view_total);
    if (!view) return NULL;
    view->offset = 0;
    if (num_slices > 0)
    {
        view->num_slices = num_slices;
        view->slices = (slice_info_t *)emalloc(sizeof(slice_info_t) * num_slices);
        memcpy(view->slices, slices, sizeof(slice_info_t) * num_slices);
    }

    return view;
}

int cuda_tensor_set_scalar(tensor_t *tensor, size_t element_offset, scalar_value_t scalar)
{
    if (!fusion_check_tensor_mutation(tensor)) return FAILURE;
    if (!tensor->total_size)
    {
        CUDA_THROW_INVALID("Cannot write a scalar into empty tensor storage");
        return FAILURE;
    }
    size_t byte_offset = element_offset * tensor->element_size;

    void *gpu_destination = (char *)tensor->data + byte_offset;
    long double scalar_value;
    if (scalar.dtype == DTYPE_INT64) scalar_value = (long double)scalar.v.i64;
    else if (scalar.dtype == DTYPE_FLOAT64) scalar_value = (long double)scalar.v.f64;
    else
    {
        CUDA_THROW_INVALID("Scalar assignment expects an integer or float");
        return FAILURE;
    }
    union {
        float f32; double f64;
        int8_t i8; int16_t i16; int32_t i32; int64_t i64;
        uint8_t u8; uint16_t u16; uint32_t u32; uint64_t u64;
        bool boolean;
    } value;
    if (dtype_is_integer(tensor->dtype))
    {
        int bits = (int)(tensor->element_size * 8);
        int signed_type = dtype_is_signed(tensor->dtype);
        long double limit = ldexpl(1.0L, bits - signed_type);
        if (!isfinite(scalar_value) || scalar_value < (signed_type ? -limit : 0) || scalar_value >= limit)
        {
            CUDA_THROW_INVALID("Scalar value is outside the destination dtype range");
            return FAILURE;
        }
    }
    switch (tensor->dtype)
    {
        case DTYPE_FLOAT32: value.f32 = (float)scalar_value; break;
        case DTYPE_FLOAT64: value.f64 = (double)scalar_value; break;
        case DTYPE_INT8: value.i8 = (int8_t)scalar_value; break;
        case DTYPE_INT16: value.i16 = (int16_t)scalar_value; break;
        case DTYPE_INT32: value.i32 = (int32_t)scalar_value; break;
        case DTYPE_INT64: value.i64 = (int64_t)scalar_value; break;
        case DTYPE_UINT8: value.u8 = (uint8_t)scalar_value; break;
        case DTYPE_UINT16: value.u16 = (uint16_t)scalar_value; break;
        case DTYPE_UINT32: value.u32 = (uint32_t)scalar_value; break;
        case DTYPE_UINT64: value.u64 = (uint64_t)scalar_value; break;
        case DTYPE_BOOL: value.boolean = scalar_value != 0; break;
        default:
            CUDA_THROW_INVALID("Unsupported scalar dtype");
            return FAILURE;
    }
    cudaError_t err = cudaMemcpy(gpu_destination, &value, tensor->element_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        return FAILURE;
    }
    return SUCCESS;
}

int cuda_tensor_set_tensor(tensor_t *base_tensor, size_t element_offset, tensor_t *tensor)
{
    if (!fusion_check_tensor_mutation(base_tensor)) return FAILURE;
    if (base_tensor->element_size != tensor->element_size)
    {
        return FAILURE;
    }

    size_t total_bytes = tensor->total_size * tensor->element_size;
    if (!total_bytes) return SUCCESS;
    void *dest_ptr = (char *)base_tensor->data + element_offset * base_tensor->element_size;
    if (!is_contiguous(base_tensor) || !is_contiguous(tensor))
    {
        /* Stage the source so overlapping strided assignments cannot overwrite unread values. */
        tensor_t *host = tensor_copy_to_host(tensor);
        if (!host) return FAILURE;
        cudaError_t status = cudaSuccess;
        for (size_t i = 0; i < tensor->total_size && status == cudaSuccess; i++)
        {
            size_t remaining = i, offset = element_offset;
            for (int d = tensor->ndims - 1; d >= 0; d--)
            {
                offset += (remaining % tensor->shape[d]) * base_tensor->strides[d + 1];
                remaining /= tensor->shape[d];
            }
            status = cudaMemcpy((char *)base_tensor->data + offset * base_tensor->element_size,
                                (char *)host->data + i * tensor->element_size,
                                tensor->element_size, cudaMemcpyHostToDevice);
        }
        cuda_tensor_destroy(host);
        if (status != cudaSuccess)
        {
            CUDA_THROW_RUNTIME("Strided tensor assignment failed: %s", cudaGetErrorString(status));
            return FAILURE;
        }
        return SUCCESS;
    }

    cudaError_t err = cudaMemcpy(dest_ptr,
                                 tensor->data,
                                 total_bytes,
                                 cudaMemcpyDeviceToDevice);

    if (err != cudaSuccess)
    {
        return FAILURE;
    }

    return SUCCESS;
}

tensor_t *cuda_tensor_create_dim_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices)
{
    if (!base_tensor || !slices)
    {
        return NULL;
    }

    size_t base_strides[MAX_DIMS];
    if (base_tensor->strides)
    {
        for (int i = 0; i < base_tensor->ndims; ++i)
        {
            base_strides[i] = base_tensor->strides[i];
        }
    }
    else
    {
        size_t stride = 1;
        for (int i = base_tensor->ndims - 1; i >= 0; --i)
        {
            base_strides[i] = stride;
            stride *= (size_t)base_tensor->shape[i];
        }
    }

    size_t element_offset = 0;

    int view_shape[MAX_DIMS];
    size_t view_strides[MAX_DIMS];
    int view_ndims = 0;

    for (int i = 0; i < base_tensor->ndims; ++i)
    {
        slice_info_t slice = (i < num_slices) ? slices[i] : (slice_info_t){.type = SLICE_ALL};

        switch (slice.type)
        {
        case SLICE_ALL:
            view_shape[view_ndims] = base_tensor->shape[i];
            view_strides[view_ndims] = base_strides[i];
            view_ndims++;
            break;

        case SLICE_INDEX:
        {
            int index = slice.data.index;

            if (index < 0)
            {
                index = base_tensor->shape[i] + index;
            }

            if (index < 0 || index >= base_tensor->shape[i])
            {
                CUDA_THROW_INVALID("Index %d out of bounds for dimension %d (size %d)",
                                 index, i, base_tensor->shape[i]);
                return NULL;
            }

            size_t offset_increment = (size_t)index * base_strides[i];
            element_offset += offset_increment;
            break;
        }

        case SLICE_RANGE:
        {
            int start = slice.data.range.start;
            int end = slice.data.range.end;

            if (start < 0)
                start = base_tensor->shape[i] + start;
            if (end < 0)
                end = base_tensor->shape[i] + end;

            if (start < 0 || end < start || end >= base_tensor->shape[i])
            {
                CUDA_THROW_INVALID("Range [%d:%d] out of bounds for dimension %d (size %d)",
                                 start, end, i, base_tensor->shape[i]);
                return NULL;
            }
            int len = (end - start + 1);

            size_t offset_increment = (size_t)start * base_strides[i];

            element_offset += offset_increment;

            view_shape[view_ndims] = len;
            view_strides[view_ndims] = base_strides[i];
            view_ndims++;
            break;
        }

        default:
            CUDA_THROW_INVALID("Invalid slice type for dimension %d", i);
            return NULL;
        }
    }

    size_t view_total = 1;
    for (int i = 0; i < view_ndims; ++i)
        view_total *= (size_t)view_shape[i];

    tensor_t *view = cuda_tensor_create_view(base_tensor, view_shape, view_strides, view_ndims,
                                            view_total ? element_offset : 0, view_total);
    if (!view) return NULL;
    view->offset = 0;
    if (num_slices > 0)
    {
        view->num_slices = num_slices;
        view->slices = (slice_info_t *)emalloc(sizeof(slice_info_t) * num_slices);
        memcpy(view->slices, slices, sizeof(slice_info_t) * num_slices);
    }

    return view;
}

void cuda_tensor_destroy(tensor_t *tensor)
{
    if (!tensor)
    {
        return;
    }

    if (tensor->is_view && !tensor->base_tensor)
    {
        autograd_release_node(tensor);
        autograd_clear_gradient(tensor);
        if (tensor->shape)
        {
            efree(tensor->shape);
        }
        if (tensor->strides)
        {
            efree(tensor->strides);
        }
        if (tensor->slices)
        {
            efree(tensor->slices);
        }

        if (tensor->d_shape)
        {
            cuda_mem_free(tensor->d_shape);
        }

        if (tensor->d_strides)
        {
            cuda_mem_free(tensor->d_strides);
        }
        efree(tensor);
        return;
    }

    tensor->ref_count--;
    if (tensor->ref_count > 0)
    {
        return;
    }

    fusion_release_node(tensor);
    autograd_release_node(tensor);
    autograd_clear_gradient(tensor);

    if (tensor->is_view)
    {
        if (tensor->base_tensor)
        {
            cuda_tensor_destroy(tensor->base_tensor);
        }

        if (tensor->slices)
        {
            efree(tensor->slices);
        }
    }
    else
    {
        if (tensor->data && tensor->is_on_gpu)
        {
            cuda_mem_free(tensor->data);
            tensor->data = NULL;
        }
        else if (tensor->data && !tensor->is_on_gpu)
        {
            if (tensor->host_pinned)
                cudaFreeHost(tensor->data);
            else
                efree(tensor->data);
        }
    }

    if (tensor->d_shape)
    {
        cuda_mem_free(tensor->d_shape);
    }
    if (tensor->d_strides)
    {
        cuda_mem_free(tensor->d_strides);
    }

    if (tensor->shape)
    {
        efree(tensor->shape);
    }
    if (tensor->strides)
    {
        efree(tensor->strides);
    }

    efree(tensor);
}

char *tensor_shape_as_string(tensor_t *tensor)
{
    if (tensor->ndims == 0)
    {
        char *result = (char *)emalloc(8);
        strcpy(result, "scalar");
        return result;
    }

    int buffer_size = tensor->ndims * 12 + 2;
    char *result = (char *)emalloc(buffer_size);

    char *ptr = result;
    *ptr++ = '(';

    for (int i = 0; i < tensor->ndims; i++)
    {
        if (i > 0)
        {
            *ptr++ = ',';
            *ptr++ = ' ';
        }
        ptr += sprintf(ptr, "%d", tensor->shape[i]);
    }

    *ptr++ = ')';
    *ptr = '\0';

    return result;
}