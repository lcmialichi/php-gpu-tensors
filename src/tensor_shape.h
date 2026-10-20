#ifndef PHP_TENSORS_SHAPE_H
#define PHP_TENSORS_SHAPE_H

#include <stddef.h>
#include <stdint.h>

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

#endif
