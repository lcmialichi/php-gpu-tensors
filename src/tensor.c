#include "tensor.h"
#include <cuda_runtime.h>
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
#include "tensor_shape.h"
#include <math.h>

static int cuda_is_initialized = 0;
static tensor_t *handle_allocation_failure(tensor_t *tensor, const char *message, cudaError_t err_code);
static int tensor_storage_range_valid(tensor_t *tensor, size_t element_offset,
                                      size_t element_count);

int cuda_require_runtime_owner(void)
{
    if (tensor_mem_failed())
    {
        CUDA_THROW_RUNTIME("CUDA runtime is unusable after an unconfirmed execution failure; restart the PHP process");
        return 0;
    }
    int status = tensor_mem_claim_owner();
    if (status == CUDA_OWNER_CURRENT) return 1;
    if (status == CUDA_OWNER_FORKED)
        CUDA_THROW_RUNTIME("CUDA cannot be used after fork from a process that already accessed CUDA; fork before the first CUDA call or exec a fresh PHP worker");
    else
        CUDA_THROW_RUNTIME("CUDA is owned by another thread; only one CUDA owner thread per process is supported");
    return 0;
}

void cuda_fail_runtime(const char *operation, int error)
{
    tensor_mem_mark_failed();
    char message[384];
    snprintf(message, sizeof(message),
             "CUDA runtime disabled after %s (status %d); completion is unconfirmed, allocations remain reserved; restart the PHP process",
             operation, error);
    if (EG(exception))
        php_log_err(message);
    else
        CUDA_THROW_RUNTIME("%s", message);
}

int cuda_tensor_require_context(void)
{
    if (!cuda_require_runtime_owner()) return 0;
    if (tensor_mem_device() < 0 && !cuda_initialize_memory_pool()) return 0;
    if (tensor_mem_context_matches()) return 1;
    CUDA_THROW_RUNTIME("CUDA operations require the owning device %d and its original context; changing devices or contexts is not supported", tensor_mem_device());
    return 0;
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
    if (tensor->requires_grad && dtype_is_floating(tensor->dtype) &&
        !dtype_is_floating(new_dtype))
    {
        CUDA_THROW_INVALID("Cannot cast a gradient-tracked tensor to a non-floating dtype; detach it first");
        return NULL;
    }

    if (fusion_active()) return fusion_cast(tensor, new_dtype);
    return fusion_cast_eager(tensor, new_dtype);
}

tensor_t *cuda_tensor_create_with_dtype(int *shape, int ndims, dtype_t dtype)
{
    if (!cuda_tensor_require_context()) return NULL;
    size_t total_elements, total_bytes;
    size_t checked_strides[MAX_DIMS];
    size_t element_size = dtype_size(dtype);
    if (!tensor_shape_metrics(shape, ndims, element_size, checked_strides, &total_elements, &total_bytes))
    {
        CUDA_THROW_INVALID("Tensor shape, strides or byte size exceed supported limits");
        return NULL;
    }

    tensor_t *tensor = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!tensor)
    {
        return handle_allocation_failure(NULL, "Failed to allocate tensor_t structure", cudaSuccess);
    }

    memset(tensor, 0, sizeof(tensor_t));

    tensor->dtype = dtype;
    tensor->element_size = element_size;
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
        tensor->total_size = total_elements;
        memcpy(tensor->strides, checked_strides, ndims * sizeof(size_t));
    }
    else
    {
        tensor->total_size = 1;
        tensor->shape = NULL;
        tensor->strides = NULL;
    }

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

tensor_t *cuda_tensor_allocate_base(const int shape[], int ndims)
{
    return cuda_tensor_create_with_dtype((int *)shape, ndims, DTYPE_FLOAT32);
}

tensor_t *cuda_tensor_create_view(tensor_t *base_tensor, int *shape, size_t *strides,
                                  int dims, size_t offset, size_t total_size)
{
    size_t elements, bytes;
    if (!base_tensor || (dims && !strides) ||
        !tensor_shape_metrics(shape, dims, base_tensor->element_size, NULL, &elements, &bytes) ||
        total_size != elements || offset > (size_t)PTRDIFF_MAX / base_tensor->element_size)
    {
        CUDA_THROW_INVALID("Invalid tensor view shape or offset");
        return NULL;
    }
    size_t limit = (size_t)PTRDIFF_MAX / base_tensor->element_size;
    size_t span = elements ? 1 : 0;
    if (span > limit - offset)
    {
        CUDA_THROW_INVALID("Tensor view offset exceeds supported limits");
        return NULL;
    }
    for (int d = 0; d < dims; d++)
    {
        size_t extent = elements ? (size_t)(shape[d] - 1) : 0;
        if (strides[d] > (size_t)PTRDIFF_MAX ||
            (extent && strides[d] > (limit - offset - span) / extent))
        {
            CUDA_THROW_INVALID("Tensor view strides exceed supported limits");
            return NULL;
        }
        span += extent * strides[d];
    }
    if (!tensor_storage_range_valid(base_tensor, offset, span))
    {
        CUDA_THROW_INVALID("Tensor view footprint is outside its backing storage");
        return NULL;
    }
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
    int axes[MAX_DIMS], starts[MAX_DIMS], steps[MAX_DIMS];
    int view_ndims = 0;

    for (int i = 0; i < base_tensor->ndims; ++i)
    {
        slice_info_t slice = (i < num_slices) ? slices[i] : (slice_info_t){.type = SLICE_ALL};
        starts[i] = 0;
        steps[i] = 1;

        switch (slice.type)
        {
        case SLICE_ALL:
            view_shape[view_ndims] = base_tensor->shape[i];
            view_strides[view_ndims] = base_strides[i];
            axes[view_ndims] = i;
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
            starts[i] = index;
            axes[view_ndims] = i;
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
            starts[i] = start;
            element_offset += (size_t)start * base_strides[i];

            view_shape[view_ndims] = len;
            view_strides[view_ndims] = base_strides[i];
            axes[view_ndims] = i;
            view_ndims++;
            break;
        }

        default:
            CUDA_THROW_INVALID("Invalid slice type for dimension %d", i);
            return NULL;
        }
    }

    size_t view_total, view_bytes;
    if (!tensor_shape_metrics(view_shape, view_ndims, base_tensor->element_size,
                              NULL, &view_total, &view_bytes))
    {
        CUDA_THROW_INVALID("Slice shape or strides exceed supported limits");
        return NULL;
    }

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
    autograd_record_slice(view, base_tensor, axes, starts, steps);

    return view;
}

static int tensor_storage_range_valid(tensor_t *tensor, size_t element_offset,
                                      size_t element_count)
{
    tensor_t *root = tensor;
    while (root && root->base_tensor)
        root = root->base_tensor;
    if (!tensor || !root || !tensor->element_size)
        return 0;
    if (!tensor->data || !root->data)
        return !tensor->data && !root->data && !root->allocated_size &&
            !element_offset && !element_count;

    uintptr_t root_address = (uintptr_t)root->data;
    uintptr_t tensor_address = (uintptr_t)tensor->data;
    if (tensor_address < root_address)
        return 0;
    uintptr_t pointer_offset = tensor_address - root_address;
    if (pointer_offset > root->allocated_size)
        return 0;
    size_t available = root->allocated_size - (size_t)pointer_offset;
    if (element_offset > available / tensor->element_size)
        return 0;
    size_t byte_offset = element_offset * tensor->element_size;
    return element_count <= (available - byte_offset) / tensor->element_size;
}

int cuda_tensor_set_scalar(tensor_t *tensor, size_t element_offset, scalar_value_t scalar)
{
    if (!fusion_check_tensor_mutation(tensor)) return FAILURE;
    if (!tensor->total_size)
    {
        CUDA_THROW_INVALID("Cannot write a scalar into empty tensor storage");
        return FAILURE;
    }
    if (!tensor_storage_range_valid(tensor, element_offset, 1))
    {
        CUDA_THROW_INVALID("Scalar assignment offset is outside tensor storage");
        return FAILURE;
    }
    size_t byte_offset = element_offset * tensor->element_size;

    void *gpu_destination = (char *)tensor->data + byte_offset;
    scalar_value_t value = cast_single_value(scalar, tensor->dtype);
    if (value.dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Scalar value is invalid or outside the destination dtype range");
        return FAILURE;
    }
    autograd_mark_mutated(tensor);
    cudaError_t err = cudaMemcpy(gpu_destination, &value.v, tensor->element_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Scalar assignment failed: %s", cudaGetErrorString(err));
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
        for (size_t i = 0; i < tensor->total_size; i++)
        {
            size_t remaining = i, offset = element_offset;
            for (int d = tensor->ndims - 1; d >= 0; d--)
            {
                offset += (remaining % tensor->shape[d]) * base_tensor->strides[d + 1];
                remaining /= tensor->shape[d];
            }
            if (!tensor_storage_range_valid(base_tensor, offset, 1))
            {
                cuda_tensor_destroy(host);
                CUDA_THROW_INVALID("Tensor assignment exceeds destination storage");
                return FAILURE;
            }
        }
        autograd_mark_mutated(base_tensor);
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

    if (!tensor_storage_range_valid(base_tensor, element_offset, tensor->total_size))
    {
        CUDA_THROW_INVALID("Tensor assignment exceeds destination storage");
        return FAILURE;
    }
    autograd_mark_mutated(base_tensor);
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
    int axes[MAX_DIMS], starts[MAX_DIMS], steps[MAX_DIMS];
    int view_ndims = 0;

    for (int i = 0; i < base_tensor->ndims; ++i)
    {
        slice_info_t slice = (i < num_slices) ? slices[i] : (slice_info_t){.type = SLICE_ALL};
        starts[i] = 0;
        steps[i] = 1;

        switch (slice.type)
        {
        case SLICE_ALL:
            view_shape[view_ndims] = base_tensor->shape[i];
            view_strides[view_ndims] = base_strides[i];
            axes[view_ndims] = i;
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
            starts[i] = index;
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
            starts[i] = start;

            view_shape[view_ndims] = len;
            view_strides[view_ndims] = base_strides[i];
            axes[view_ndims] = i;
            view_ndims++;
            break;
        }

        default:
            CUDA_THROW_INVALID("Invalid slice type for dimension %d", i);
            return NULL;
        }
    }

    size_t view_total, view_bytes;
    if (!tensor_shape_metrics(view_shape, view_ndims, base_tensor->element_size,
                              NULL, &view_total, &view_bytes))
    {
        CUDA_THROW_INVALID("Slice shape or strides exceed supported limits");
        return NULL;
    }

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
    autograd_record_slice(view, base_tensor, axes, starts, steps);

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
            {
                if (tensor_mem_owner_status() == CUDA_OWNER_CURRENT && !tensor_mem_failed())
                    cudaFreeHost(tensor->data);
            }
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