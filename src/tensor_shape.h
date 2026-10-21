#ifndef PHP_TENSORS_SHAPE_H
#define PHP_TENSORS_SHAPE_H

#include <stddef.h>
#include <stdint.h>

static inline int tensor_broadcast_shape(const int *a, int a_dims, const int *b,
                                        int b_dims, int *shape, int *ndims)
{
    *ndims = a_dims > b_dims ? a_dims : b_dims;
    int offset_a = *ndims - a_dims;
    int offset_b = *ndims - b_dims;
    for (int axis = 0; axis < *ndims; axis++)
    {
        int extent_a = axis < offset_a ? 1 : a[axis - offset_a];
        int extent_b = axis < offset_b ? 1 : b[axis - offset_b];
        if (extent_a != extent_b && extent_a != 1 && extent_b != 1)
            return 0;
        shape[axis] = extent_a == 1 ? extent_b : extent_a;
    }
    return 1;
}

static inline int tensor_shape_metrics(const int *shape, int ndims, size_t element_size,
                                      size_t *strides, size_t *elements, size_t *bytes)
{
    if (ndims < 0 || ndims > 10 || (ndims && !shape) || !element_size)
        return 0;

    size_t total = 1;
    for (int d = ndims - 1; d >= 0; d--)
    {
        if (shape[d] < 0 ||
            (shape[d] && total > (size_t)PTRDIFF_MAX / (size_t)shape[d]))
            return 0;
        if (strides) strides[d] = total;
        total *= (size_t)shape[d];
    }
    if (total > (size_t)PTRDIFF_MAX / element_size)
        return 0;
    *elements = total;
    *bytes = total * element_size;
    return 1;
}

static inline int tensor_matmul_shape(const int *a, int a_dims, const int *b, int b_dims,
                                     int *shape, int *ndims)
{
    if (a_dims < 2 || a_dims > 10 || b_dims < 2 || b_dims > 10 ||
        a[a_dims - 1] != b[b_dims - 2])
        return 0;
    int batch_dims;
    if (!tensor_broadcast_shape(a, a_dims - 2, b, b_dims - 2, shape, &batch_dims))
        return 0;
    shape[batch_dims] = a[a_dims - 2];
    shape[batch_dims + 1] = b[b_dims - 1];
    *ndims = batch_dims + 2;
    return 1;
}

#endif
