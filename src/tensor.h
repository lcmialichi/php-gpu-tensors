#ifndef TENSOR_H
#define TENSOR_H

#include <cuda_runtime.h>
#include "data_types.h"
#ifndef __CUDACC__
#include "cuda_exceptions.h"
#endif

#define MAX_DIMS 10
#define MAX_CONCAT_TENSORS 10

struct _operation_t;
struct fusion_node;

typedef enum
{
    SLICE_ALL = 0,
    SLICE_RANGE = 1,
    SLICE_INDEX = 2
} slice_type_t;

typedef struct
{
    slice_type_t type;
    union
    {
        struct
        {
            int start;
            int end;
        } range;
        int index;
    } data;
} slice_info_t;

typedef struct tensor
{
    void *data;
    dtype_t dtype;
    int *shape;
    size_t element_size;
    size_t *strides;
    int ndims;
    size_t total_size;
    int ref_count;
    size_t allocated_size;
    int is_view;
    size_t offset;
    slice_info_t *slices;
    struct tensor *base_tensor;
    int num_slices;
    size_t *d_strides;
    int *d_shape;
    int is_on_gpu;
    int host_pinned;
    int is_dirty;
    int is_contiguous_cached;
    struct fusion_node *fusion;
    int fusion_failed;
} tensor_t;

#ifdef __cplusplus
extern "C"
{
#endif

#define CUDA_CHECK_AND_RETURN_NULL(__tensor__)                                           \
    do                                                                                   \
    {                                                                                    \
        if (UNEXPECTED((__tensor__) == NULL))                     \
        {                                                                                \
            CUDA_THROW_RUNTIME("CUDA not initialized or tensor is NULL"); \
            return NULL;                                                                 \
        }                                                                                \
    } while (0)

#define CUDA_CHECK_AND_RETURN_FAILURE(__tensor__)                                        \
    do                                                                                   \
    {                                                                                    \
        if (UNEXPECTED((__tensor__) == NULL))                     \
        {                                                                                \
            CUDA_THROW_RUNTIME("CUDA not initialized or tensor is NULL"); \
            return 0;                                                                    \
        }                                                                                \
    } while (0)

    int cuda_initialized();
    void cuda_set_initialized(int status);
    void cuda_tensor_destroy(tensor_t *tensor);
    int is_contiguous(tensor_t *tensor);
    int lazy_copy_metadata_to_gpu(tensor_t *t);
    char *tensor_shape_as_string(tensor_t *tensor);

    tensor_t *cuda_tensor_create_sliced_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices);
    tensor_t *cuda_tensor_create_view(tensor_t *base_tensor, int *shape, size_t *strides, int dims, size_t offset, size_t total_size);
    tensor_t *cuda_tensor_create_dim_view(tensor_t *base_tensor, slice_info_t *slices, int num_slices);

    int cuda_tensor_set_scalar(tensor_t *tensor, size_t element_offset, float scalar_value);
    int cuda_tensor_set_tensor(tensor_t *base_tensor, size_t element_offset, tensor_t *tensor);

    int tensor_can_cast_to(const tensor_t *tensor, dtype_t new_dtype);
    tensor_t *tensor_cast(tensor_t *tensor, dtype_t new_dtype);

    tensor_t *cuda_tensor_create_with_dtype(int *shape, int ndims, dtype_t dtype);
    tensor_t *cuda_tensor_create_empty_with_dtype(int *shape, int ndims, dtype_t dtype);

    int tensors_have_same_dtype(const tensor_t *a, const tensor_t *b);
    int tensor_validate_dtype(tensor_t *tensor);

    static inline size_t tensor_element_size(const tensor_t *tensor);
    static inline size_t tensor_nbytes(const tensor_t *tensor);
    static inline void tensor_invalidate_contiguity_cache(tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif