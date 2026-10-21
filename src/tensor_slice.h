#ifndef PHP_TENSORS_SLICE_H
#define PHP_TENSORS_SLICE_H

#include "php.h"
#include "cuda_exceptions.h"
#include "tensor_shape.h"
#include <limits.h>
#include <string.h>

typedef struct {
    int index, has_start, has_stop;
    zend_long start, stop, step;
} tensor_slice_selector;

typedef struct {
    int shape[10], axes[10], starts[10], steps[10], ndims;
    size_t strides[10], offset, total;
} tensor_slice_layout;

static inline void tensor_slice_trim(const char **text, size_t *length)
{
    while (*length && (**text == ' ' || **text == '\t' || **text == '\n' || **text == '\r'))
    {
        (*text)++;
        (*length)--;
    }
    while (*length && ((*text)[*length - 1] == ' ' || (*text)[*length - 1] == '\t' ||
                      (*text)[*length - 1] == '\n' || (*text)[*length - 1] == '\r'))
        (*length)--;
}

static inline int tensor_slice_integer(const char *text, size_t length, zend_long *value)
{
    tensor_slice_trim(&text, &length);
    if (!length) return 0;
    int negative = text[0] == '-';
    if (text[0] == '-' || text[0] == '+') { text++; length--; }
    if (!length) return 0;
    zend_ulong limit = (zend_ulong)ZEND_LONG_MAX + (negative ? 1 : 0), number = 0;
    for (size_t i = 0; i < length; i++)
    {
        if (text[i] < '0' || text[i] > '9') return 0;
        unsigned int digit = text[i] - '0';
        if (number > (limit - digit) / 10) return 0;
        number = number * 10 + digit;
    }
    *value = negative ? (number == (zend_ulong)ZEND_LONG_MAX + 1 ? ZEND_LONG_MIN : -(zend_long)number)
                      : (zend_long)number;
    return 1;
}

static inline int tensor_slice_text(const char *text, size_t length, tensor_slice_selector *selector)
{
    memset(selector, 0, sizeof(*selector));
    selector->step = 1;
    tensor_slice_trim(&text, &length);
    if (!memchr(text, ':', length))
    {
        selector->index = selector->has_start = 1;
        return tensor_slice_integer(text, length, &selector->start);
    }
    size_t start = 0;
    int field = 0;
    for (size_t i = 0; i <= length; i++)
    {
        if (i != length && text[i] != ':') continue;
        if (field >= 3) return 0;
        const char *part = text + start;
        size_t part_length = i - start;
        tensor_slice_trim(&part, &part_length);
        if (part_length)
        {
            zend_long number;
            if (!tensor_slice_integer(part, part_length, &number)) return 0;
            if (field == 0) { selector->start = number; selector->has_start = 1; }
            else if (field == 1) { selector->stop = number; selector->has_stop = 1; }
            else selector->step = number;
        }
        field++;
        start = i + 1;
    }
    return 1;
}

static inline int tensor_slice_value(zval *value, tensor_slice_selector *selector)
{
    ZVAL_DEREF(value);
    memset(selector, 0, sizeof(*selector));
    selector->step = 1;
    if (Z_TYPE_P(value) == IS_NULL) return 1;
    if (Z_TYPE_P(value) == IS_LONG)
    {
        selector->index = selector->has_start = 1;
        selector->start = Z_LVAL_P(value);
        return 1;
    }
    if (Z_TYPE_P(value) == IS_STRING)
        return tensor_slice_text(Z_STRVAL_P(value), Z_STRLEN_P(value), selector);
    if (Z_TYPE_P(value) != IS_ARRAY) return 0;
    size_t count = zend_hash_num_elements(Z_ARRVAL_P(value));
    if (count != 2 && count != 3) return 0;
    for (size_t i = 0; i < count; i++)
    {
        zval *part = zend_hash_index_find(Z_ARRVAL_P(value), i);
        if (!part) return 0;
        ZVAL_DEREF(part);
        if (Z_TYPE_P(part) == IS_NULL) continue;
        if (Z_TYPE_P(part) != IS_LONG) return 0;
        if (i == 0) { selector->start = Z_LVAL_P(part); selector->has_start = 1; }
        else if (i == 1) { selector->stop = Z_LVAL_P(part); selector->has_stop = 1; }
        else selector->step = Z_LVAL_P(part);
    }
    return 1;
}

static inline int tensor_slice_bound(zend_long value, int size)
{
    if (value < 0) value += size;
    if (value < 0) return 0;
    if (value > size) return size;
    return (int)value;
}

static inline int tensor_slice_prepare(const int *shape, const size_t *strides,
                                      int ndims, size_t element_size,
                                      zval *selectors, int count, tensor_slice_layout *result)
{
    tensor_slice_selector parsed[10];
    int parsed_count = 0;
    zval *first = count ? &selectors[0] : NULL;
    if (first) ZVAL_DEREF(first);
    if (count == 1 && Z_TYPE_P(first) == IS_STRING)
    {
        const char *text = Z_STRVAL_P(first);
        size_t length = Z_STRLEN_P(first), start = 0;
        for (size_t i = 0; i <= length; i++)
        {
            if (i != length && text[i] != ',') continue;
            if (parsed_count >= ndims ||
                !tensor_slice_text(text + start, i - start, &parsed[parsed_count]))
            {
                CUDA_THROW_INVALID("Invalid slice expression at axis %d", parsed_count);
                return FAILURE;
            }
            parsed_count++;
            start = i + 1;
        }
    }
    else
    {
        if (count > ndims)
        {
            CUDA_THROW_INVALID("Slice has %d selectors for %d axes", count, ndims);
            return FAILURE;
        }
        for (; parsed_count < count; parsed_count++)
            if (!tensor_slice_value(&selectors[parsed_count], &parsed[parsed_count]))
            {
                CUDA_THROW_INVALID("Invalid slice selector at axis %d", parsed_count);
                return FAILURE;
            }
    }
    memset(result, 0, sizeof(*result));
    for (int d = 0; d < ndims; d++)
    {
        tensor_slice_selector selector = {.step = 1};
        if (d < parsed_count) selector = parsed[d];
        int size = shape[d];
        if (selector.step <= 0 || selector.step > INT_MAX)
        {
            CUDA_THROW_INVALID("Slice step must be between 1 and INT_MAX; reverse slices are not supported");
            return FAILURE;
        }
        result->steps[d] = (int)selector.step;
        if (selector.index)
        {
            zend_long index = selector.start;
            if (index < 0) index += size;
            if (index < 0 || index >= size)
            {
                CUDA_THROW_INVALID("Slice index out of bounds at axis %d (size %d)", d, size);
                return FAILURE;
            }
            result->starts[d] = (int)index;
        }
        else
        {
            result->starts[d] = selector.has_start ? tensor_slice_bound(selector.start, size) : 0;
            int stop = selector.has_stop ? tensor_slice_bound(selector.stop, size) : size;
            int length = stop > result->starts[d]
                ? 1 + (stop - result->starts[d] - 1) / result->steps[d] : 0;
            if (strides[d] > SIZE_MAX / (size_t)result->steps[d])
            {
                CUDA_THROW_INVALID("Slice stride overflow at axis %d", d);
                return FAILURE;
            }
            int axis = result->ndims++;
            result->axes[axis] = d;
            result->shape[axis] = length;
            result->strides[axis] = strides[d] * (size_t)result->steps[d];
        }
        if (result->starts[d] && strides[d] > (SIZE_MAX - result->offset) / (size_t)result->starts[d])
        {
            CUDA_THROW_INVALID("Slice offset overflow");
            return FAILURE;
        }
        result->offset += (size_t)result->starts[d] * strides[d];
    }
    size_t bytes;
    if (!tensor_shape_metrics(result->shape, result->ndims, element_size,
                              NULL, &result->total, &bytes))
    {
        CUDA_THROW_INVALID("Slice shape or strides exceed supported limits");
        return FAILURE;
    }
    if (!result->total) result->offset = 0;
    return SUCCESS;
}

#endif
