#include "cuda_exceptions.h"
#include "php.h"
#include "contiguous_array_ce.h"
#include "contiguous_array_arginfo.h"
#include "backend_contract.h"
#include "zend_smart_str.h"
#include <stdint.h>
#include <string.h>
#include "tensor_factory.h"
#include "tensor_import.h"
#include "tensor_transfer.h"
#include "cuda_array_ce.h"
#include "ca_struct.h"
#include "zend_exceptions.h"
#include "operator_result.h"
#include "tensor_scalar.h"

static zend_object_handlers contiguous_array_handlers;
zend_class_entry *contiguous_array_ce;

typedef void (*dtype_getter_t)(void *ptr, zval *rv);

static zend_always_inline void get_f32(void *p, zval *rv) { ZVAL_DOUBLE(rv, (double)*(float *)p); }
static zend_always_inline void get_f64(void *p, zval *rv) { ZVAL_DOUBLE(rv, *(double *)p); }
static zend_always_inline void get_f16(void *p, zval *rv) { ZVAL_DOUBLE(rv, float16_bits_to_float(*(uint16_t *)p)); }
static zend_always_inline void get_bf16(void *p, zval *rv) { ZVAL_DOUBLE(rv, bfloat16_bits_to_float(*(uint16_t *)p)); }
static zend_always_inline void get_i8(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long)*(int8_t *)p); }
static zend_always_inline void get_i16(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long)*(int16_t *)p); }
static zend_always_inline void get_i32(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long) * (int32_t *)p); }
static zend_always_inline void get_i64(void *p, zval *rv) { tensor_int64_to_zval(rv, *(int64_t *)p); }
static zend_always_inline void get_u8(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long) * (uint8_t *)p); }
static zend_always_inline void get_u16(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long) * (uint16_t *)p); }
static zend_always_inline void get_u32(void *p, zval *rv) { tensor_uint64_to_zval(rv, *(uint32_t *)p); }
static zend_always_inline void get_u64(void *p, zval *rv) { tensor_uint64_to_zval(rv, *(uint64_t *)p); }
static zend_always_inline void get_bool(void *p, zval *rv) { ZVAL_BOOL(rv, *(uint8_t *)p != 0); }

static const dtype_getter_t dtype_getters[DTYPE_COUNT] = {
    [DTYPE_FLOAT32] = get_f32,
    [DTYPE_FLOAT64] = get_f64,
    [DTYPE_FLOAT16] = get_f16,
    [DTYPE_BFLOAT16] = get_bf16,
    [DTYPE_INT8] = get_i8,
    [DTYPE_INT16] = get_i16,
    [DTYPE_INT32] = get_i32,
    [DTYPE_INT64] = get_i64,
    [DTYPE_UINT8] = get_u8,
    [DTYPE_UINT16] = get_u16,
    [DTYPE_UINT32] = get_u32,
    [DTYPE_UINT64] = get_u64,
    [DTYPE_BOOL] = get_bool,
};

static zend_always_inline dtype_getter_t contiguous_array_dtype_getter(dtype_t dtype)
{
    if ((unsigned int)dtype >= DTYPE_COUNT || !dtype_getters[dtype])
    {
        CUDA_THROW_INVALID("Unsupported host array dtype: %d", (int)dtype);
        return NULL;
    }
    return dtype_getters[dtype];
}

static zend_always_inline contiguous_array_object *contiguous_array_from_obj(zend_object *zobj)
{
    return (contiguous_array_object *)((char *)zobj - contiguous_array_handlers.offset);
}

static contiguous_array_object *contiguous_array_fetch_valid_object(zend_object *object)
{
    contiguous_array_object *obj = contiguous_array_from_obj(object);
    if (!obj->tensor)
    {
        CUDA_THROW_RUNTIME("Attempting to access uninitialized HostArray");
        return NULL;
    }
    return obj;
}

static void contiguous_array_attach_tensor(contiguous_array_object *obj, tensor_t *tensor)
{
    obj->tensor = tensor;
    obj->ndims = tensor->ndims;
    obj->shape = tensor->shape;
    obj->strides = tensor->strides;
    obj->dtype = tensor->dtype;
    obj->element_size = tensor->element_size;
    obj->total_elements = tensor->total_size;
    obj->cached_data_ptr = (char *)tensor->data + tensor->offset * tensor->element_size;
    obj->offset = tensor->offset;
    obj->is_contiguous = 1;
    obj->read_only = 0;
}

static void contiguous_array_share_storage(contiguous_array_object *target,
                                           const contiguous_array_object *source)
{
    target->tensor = source->tensor;
    target->tensor->ref_count++;
    target->cached_data_ptr = source->cached_data_ptr;
    target->shape = source->shape;
    target->strides = source->strides;
    target->total_elements = source->total_elements;
    target->element_size = source->element_size;
    target->ndims = source->ndims;
    target->dtype = source->dtype;
    target->offset = source->offset;
    target->is_contiguous = source->is_contiguous;
    target->read_only = source->read_only;
}

static int contiguous_array_has_contiguous_layout(const contiguous_array_object *obj)
{
    size_t expected_stride = 1;
    for (int dimension = obj->ndims - 1; dimension >= 0; dimension--)
    {
        if (obj->shape[dimension] > 1 &&
            obj->strides[dimension] != expected_stride)
            return 0;
        expected_stride *= (size_t)obj->shape[dimension];
    }
    return 1;
}

static size_t contiguous_array_element_offset(const contiguous_array_object *obj,
                                              size_t linear_index)
{
    size_t offset = 0;
    for (int dimension = obj->ndims - 1; dimension >= 0; dimension--)
    {
        size_t extent = (size_t)obj->shape[dimension];
        offset += (linear_index % extent) * obj->strides[dimension];
        linear_index /= extent;
    }
    return offset;
}

static int contiguous_array_host_data(const contiguous_array_object *obj,
                                      const void **data, void **packed)
{
    *data = NULL;
    *packed = NULL;
    if (!obj->tensor)
    {
        CUDA_THROW_RUNTIME("Cannot transfer an uninitialized host array");
        return FAILURE;
    }
    if (obj->total_elements > SIZE_MAX / obj->element_size)
    {
        CUDA_THROW_INVALID("Host array transfer size exceeds the supported limit");
        return FAILURE;
    }
    size_t byte_count = obj->total_elements * obj->element_size;
    if (!byte_count)
        return SUCCESS;
    if (contiguous_array_has_contiguous_layout(obj))
    {
        *data = obj->cached_data_ptr;
        return SUCCESS;
    }

    char *buffer = emalloc(byte_count);
    for (size_t i = 0; i < obj->total_elements; i++)
    {
        size_t offset = contiguous_array_element_offset(obj, i);
        memcpy(buffer + i * obj->element_size,
               obj->cached_data_ptr + offset * obj->element_size,
               obj->element_size);
    }
    *data = buffer;
    *packed = buffer;
    return SUCCESS;
}

static zend_object *contiguous_array_clone_object(zend_object *object)
{
    /* The VM releases the returned object when cloning throws. */
    zend_object *clone = contiguous_array_create_object(object->ce);
    contiguous_array_object *source = contiguous_array_fetch_valid_object(object);
    if (!source) return clone;
    const void *data;
    void *packed;
    if (contiguous_array_host_data(source, &data, &packed) != SUCCESS)
        return clone;
    tensor_t *tensor = source->tensor->host_pinned
        ? cuda_tensor_create_on_host_pinned(source->shape, source->ndims, (void *)data, source->dtype)
        : cuda_tensor_create_on_host(source->shape, source->ndims, (void *)data, source->dtype);
    if (packed)
        efree(packed);
    if (!tensor) return clone;
    contiguous_array_attach_tensor(contiguous_array_from_obj(clone), tensor);
    zend_objects_clone_members(clone, object);
    return clone;
}

static int contiguous_array_validate_values(zval *value, int depth, int shape[MAX_DIMS], int *leaf_depth)
{
    if (Z_TYPE_P(value) != IS_ARRAY)
    {
        if (Z_TYPE_P(value) != IS_LONG && Z_TYPE_P(value) != IS_DOUBLE &&
            Z_TYPE_P(value) != IS_TRUE && Z_TYPE_P(value) != IS_FALSE)
            return 0;
        if (*leaf_depth == -1) *leaf_depth = depth;
        return depth == *leaf_depth;
    }

    if (depth >= MAX_DIMS || !zend_array_is_list(Z_ARRVAL_P(value)))
        return 0;
    size_t length = zend_hash_num_elements(Z_ARRVAL_P(value));
    if (length == 0 || length > INT_MAX || (shape[depth] && shape[depth] != (int)length))
        return 0;
    shape[depth] = (int)length;

    zval *child;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(value), child)
    {
        if (!contiguous_array_validate_values(child, depth + 1, shape, leaf_depth))
            return 0;
    }
    ZEND_HASH_FOREACH_END();
    return 1;
}

static int contiguous_array_copy_values(zval *value, void *buffer, dtype_t dtype, size_t *position)
{
    ZVAL_DEREF(value);
    if (Z_TYPE_P(value) == IS_ARRAY)
    {
        zval *child;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(value), child)
        {
            if (!contiguous_array_copy_values(child, buffer, dtype, position))
                return 0;
        }
        ZEND_HASH_FOREACH_END();
        return 1;
    }

    size_t index = (*position)++;
    switch (dtype)
    {
    case DTYPE_FLOAT32: ((float *)buffer)[index] = (float)zval_get_double(value); break;
    case DTYPE_FLOAT64: ((double *)buffer)[index] = zval_get_double(value); break;
    case DTYPE_FLOAT16: ((uint16_t *)buffer)[index] = float_to_float16_bits((float)zval_get_double(value)); break;
    case DTYPE_BFLOAT16: ((uint16_t *)buffer)[index] = float_to_bfloat16_bits((float)zval_get_double(value)); break;
    case DTYPE_BOOL: ((bool *)buffer)[index] = zend_is_true(value); break;
    default:
    {
        scalar_value_t converted;
        if (tensor_scalar_from_zval(value, dtype, &converted) != SUCCESS) return 0;
        memcpy((char *)buffer + index * dtype_size(dtype), &converted.v, dtype_size(dtype));
        break;
    }
    }
    return 1;
}

#define PHP_CUDA_LOOP_TO_ARRAY(type, zval_func)  \
    do                                           \
    {                                            \
        type *p = (type *)data_ptr;              \
        for (size_t i = 0; i < count; i++)       \
        {                                        \
            zval_func(return_value, (type) * p); \
            p = (type *)((char *)p + stride);    \
        }                                        \
    } while (0)

size_t dtype_to_size(dtype_t dtype)
{
    static const size_t sizes[] = {
        [DTYPE_FLOAT32] = 4, [DTYPE_FLOAT64] = 8, [DTYPE_INT8] = 1, [DTYPE_INT16] = 2,
        [DTYPE_INT32] = 4, [DTYPE_INT64] = 8, [DTYPE_UINT8] = 1, [DTYPE_UINT16] = 2,
        [DTYPE_UINT32] = 4, [DTYPE_UINT64] = 8, [DTYPE_BOOL] = 1,
        [DTYPE_FLOAT16] = 2, [DTYPE_BFLOAT16] = 2};
    return (dtype < sizeof(sizes) / sizeof(size_t)) ? sizes[dtype] : 8;
}

static zend_always_inline void get_value(void *ptr, zval *rv, dtype_t dtype)
{
    dtype_getter_t getter = contiguous_array_dtype_getter(dtype);
    if (getter)
        getter(ptr, rv);
}

int contiguous_array_count_elements(zend_object *object, zend_long *count)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(object);
    if (!obj) return FAILURE;
    if (obj->ndims <= 0)
    {
        *count = 0;
    }
    else
    {
        *count = (zend_long)obj->shape[0];
    }
    return SUCCESS;
}

static zval *contiguous_array_offset_get(zend_object *object, zval *offset, int type, zval *rv)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(object);
    if (!obj)
    {
        ZVAL_NULL(rv);
        return rv;
    }
    zend_long idx = zval_get_long(offset);

    if (UNEXPECTED(obj->ndims == 0 || idx < 0 || idx >= obj->shape[0]))
    {
        CUDA_THROW_INVALID("Index %ld out of bounds", idx);
        ZVAL_NULL(rv);
        return rv;
    }

    if (obj->ndims == 1)
    {
        get_value(obj->cached_data_ptr + (idx * obj->strides[0] * obj->element_size), rv, obj->dtype);
    }
    else
    {
        zend_object *zslice = contiguous_array_create_object(contiguous_array_ce);
        contiguous_array_object *slice = contiguous_array_from_obj(zslice);

        contiguous_array_share_storage(slice, obj);
        slice->ndims = obj->ndims - 1;
        slice->shape = obj->shape + 1;
        slice->strides = obj->strides + 1;
        slice->dtype = obj->dtype;
        slice->element_size = obj->element_size;
        slice->cached_data_ptr = obj->cached_data_ptr + (idx * obj->strides[0] * obj->element_size);
        slice->offset += idx * obj->strides[0];

        size_t total = slice->shape[0];
        for (int i = 1; i < slice->ndims; i++)
            total *= slice->shape[i];
        slice->total_elements = total;

        ZVAL_OBJ(rv, zslice);
    }
    return rv;
}

void *allocate_for_dtype(dtype_t dtype, size_t count)
{
    size_t size = dtype_to_size(dtype);
    return ecalloc(count, size);
}

static void contiguous_array_to_php_array(contiguous_array_object *obj, zval *return_value)
{
    if (!obj->tensor)
    {
        CUDA_THROW_RUNTIME("Cannot export an uninitialized host array");
        return;
    }
    tensor_t descriptor = *obj->tensor;
    descriptor.ndims = obj->ndims;
    descriptor.shape = obj->shape;
    descriptor.strides = obj->strides;
    descriptor.dtype = obj->dtype;
    tensor_host_to_php_array(return_value, &descriptor, obj->cached_data_ptr);
}

ZEND_METHOD(ContiguousArray, __serialize)
{
    ZEND_PARSE_PARAMETERS_NONE();
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    size_t strides[MAX_DIMS];
    size_t stride = 1;
    for (int d = obj->ndims - 1; d >= 0; d--)
    {
        if (stride > ZEND_LONG_MAX ||
            (obj->shape[d] && stride > (size_t)ZEND_LONG_MAX / (size_t)obj->shape[d]))
        {
            CUDA_THROW_INVALID("HostArray shape or strides exceed serialization limits");
            RETURN_THROWS();
        }
        strides[d] = stride;
        stride *= (size_t)obj->shape[d];
    }
    const void *host_data;
    void *packed_data;
    if (contiguous_array_host_data(obj, &host_data, &packed_data) != SUCCESS)
        RETURN_THROWS();
    array_init(return_value);
    add_assoc_stringl(return_value, "__contiguous_array_v1", "1", 1);

    add_assoc_long(return_value, "ndims", obj->ndims);
    add_assoc_long(return_value, "dtype", obj->dtype);
    add_assoc_long(return_value, "offset", 0);
    add_assoc_long(return_value, "total_elements", obj->total_elements);
    add_assoc_long(return_value, "element_size", obj->element_size);
    add_assoc_bool(return_value, "is_contiguous", 1);

    zval shape_array;
    array_init(&shape_array);
    for (int i = 0; i < obj->ndims; i++)
    {
        add_next_index_long(&shape_array, obj->shape[i]);
    }
    add_assoc_zval(return_value, "shape", &shape_array);

    zval strides_array;
    array_init(&strides_array);
    for (int i = 0; i < obj->ndims; i++)
    {
        add_next_index_long(&strides_array, strides[i]);
    }
    add_assoc_zval(return_value, "strides", &strides_array);

    size_t data_size = obj->total_elements * obj->element_size;
    add_assoc_stringl(return_value, "data", data_size ? host_data : "", data_size);
    if (packed_data)
        efree(packed_data);
}

ZEND_METHOD(ContiguousArray, __unserialize)
{
    HashTable *data;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();

    zval *version = zend_hash_str_find(data, "__contiguous_array_v1", sizeof("__contiguous_array_v1") - 1);
    if (!version || Z_TYPE_P(version) != IS_STRING ||
        !zend_string_equals_literal(Z_STR_P(version), "1"))
    {
        CUDA_THROW_INVALID("Invalid serialized data version");
        RETURN_THROWS();
    }

    zval *ndims_zv = zend_hash_str_find(data, "ndims", sizeof("ndims") - 1);
    zval *dtype_zv = zend_hash_str_find(data, "dtype", sizeof("dtype") - 1);
    zval *total_zv = zend_hash_str_find(data, "total_elements", sizeof("total_elements") - 1);
    zval *element_size_zv = zend_hash_str_find(data, "element_size", sizeof("element_size") - 1);
    zval *shape_zv = zend_hash_str_find(data, "shape", sizeof("shape") - 1);
    zval *strides_zv = zend_hash_str_find(data, "strides", sizeof("strides") - 1);
    zval *offset_zv = zend_hash_str_find(data, "offset", sizeof("offset") - 1);
    zval *contiguous_zv = zend_hash_str_find(data, "is_contiguous", sizeof("is_contiguous") - 1);
    zval *data_zv = zend_hash_str_find(data, "data", sizeof("data") - 1);
    if (!ndims_zv || Z_TYPE_P(ndims_zv) != IS_LONG ||
        !dtype_zv || Z_TYPE_P(dtype_zv) != IS_LONG ||
        !total_zv || Z_TYPE_P(total_zv) != IS_LONG ||
        !element_size_zv || Z_TYPE_P(element_size_zv) != IS_LONG ||
        !shape_zv || !strides_zv || Z_TYPE_P(strides_zv) != IS_ARRAY ||
        !offset_zv || Z_TYPE_P(offset_zv) != IS_LONG || Z_LVAL_P(offset_zv) != 0 ||
        !contiguous_zv || Z_TYPE_P(contiguous_zv) != IS_TRUE ||
        !data_zv || Z_TYPE_P(data_zv) != IS_STRING)
    {
        CUDA_THROW_INVALID("Malformed serialized HostArray payload");
        RETURN_THROWS();
    }

    if (Z_LVAL_P(dtype_zv) < 0 || Z_LVAL_P(dtype_zv) >= DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Invalid serialized HostArray dtype");
        RETURN_THROWS();
    }
    dtype_t dtype = (dtype_t)Z_LVAL_P(dtype_zv);
    size_t element_size = dtype_size(dtype);
    int shape[MAX_DIMS] = {0};
    size_t total_elements;
    if (!tensor_import_serialized_shape(shape_zv, Z_LVAL_P(ndims_zv), shape, &total_elements))
        RETURN_THROWS();
    int ndims = (int)Z_LVAL_P(ndims_zv);

    if (Z_LVAL_P(total_zv) < 0 || (size_t)Z_LVAL_P(total_zv) != total_elements ||
        Z_LVAL_P(element_size_zv) != (zend_long)element_size ||
        total_elements > SIZE_MAX / element_size ||
        Z_STRLEN_P(data_zv) != total_elements * element_size)
    {
        CUDA_THROW_INVALID("Invalid serialized HostArray size metadata");
        RETURN_THROWS();
    }

    if (!zend_array_is_list(Z_ARRVAL_P(strides_zv)) ||
        zend_hash_num_elements(Z_ARRVAL_P(strides_zv)) != (uint32_t)ndims)
    {
        CUDA_THROW_INVALID("Invalid serialized HostArray strides");
        RETURN_THROWS();
    }
    size_t expected_stride = 1;
    for (int d = ndims - 1; d >= 0; d--)
    {
        zval *value = zend_hash_index_find(Z_ARRVAL_P(strides_zv), d);
        if (Z_TYPE_P(value) != IS_LONG || Z_LVAL_P(value) < 0 ||
            (size_t)Z_LVAL_P(value) != expected_stride)
        {
            CUDA_THROW_INVALID("Serialized HostArray strides must describe contiguous data");
            RETURN_THROWS();
        }
        expected_stride *= (size_t)shape[d];
    }

    tensor_t *tensor = cuda_tensor_create_on_host(shape, ndims, Z_STRVAL_P(data_zv), dtype);
    if (!tensor) RETURN_THROWS();

    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(getThis()));
    if (obj->tensor)
        cuda_tensor_destroy(obj->tensor);
    contiguous_array_attach_tensor(obj, tensor);
}

ZEND_METHOD(ContiguousArray, toArray)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    contiguous_array_to_php_array(obj, return_value);
}

ZEND_METHOD(ContiguousArray, __debugInfo)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    array_init(return_value);
    zval shape;
    array_init_size(&shape, obj->ndims);
    for (int i = 0; i < obj->ndims; i++) {
        add_next_index_long(&shape, obj->shape[i]);
    }
    add_assoc_zval(return_value, "shape", &shape);
    add_assoc_string(return_value, "dtype", (char *)dtype_to_string(obj->dtype));
    add_assoc_long(return_value, "elements", (zend_long)obj->total_elements);
}

ZEND_METHOD(ContiguousArray, getStrides)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    array_init_size(return_value, obj->ndims);
    for (int i = 0; i < obj->ndims; i++) {
        add_next_index_long(return_value, (zend_long)obj->strides[i]);
    }
}

ZEND_METHOD(ContiguousArray, toHost)
{
    ZEND_PARSE_PARAMETERS_NONE();
    if (!contiguous_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS))) RETURN_THROWS();
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

ZEND_METHOD(ContiguousArray, toGpu)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    const void *host_data;
    void *packed_data;
    if (contiguous_array_host_data(obj, &host_data, &packed_data) != SUCCESS)
    {
        RETURN_THROWS();
    }
    size_t data_size = obj->total_elements * obj->element_size;
    tensor_t *gpu_tensor = cuda_tensor_create_from_host_buffer(
        obj->shape, obj->ndims, obj->dtype, host_data, data_size);
    if (packed_data)
        efree(packed_data);
    if (!gpu_tensor)
    {
        RETURN_THROWS();
    }

    object_init_ex(return_value, cuda_array_ce);
    cuda_array_obj *cuda_array = Z_CUDA_ARRAY_P(return_value);
    cuda_array->tensor_handle = gpu_tensor;

    cuda_array->shape = zend_new_array(obj->ndims);
    for (int i = 0; i < obj->ndims; i++)
    {
        zval dim;
        ZVAL_LONG(&dim, obj->shape[i]);
        zend_hash_index_update(cuda_array->shape, i, &dim);
    }

}

static zend_result contiguous_array_binary_apply(zval *return_value, zval *tensor_value,
                                                 zval *other, char operation,
                                                 zend_bool reverse)
{
    contiguous_array_object *left = contiguous_array_fetch_valid_object(Z_OBJ_P(tensor_value));
    if (!left) return FAILURE;
    contiguous_array_object *right = NULL;
    double scalar = 0.0;
    if (Z_TYPE_P(other) == IS_OBJECT &&
        instanceof_function(Z_OBJCE_P(other), contiguous_array_ce)) {
        right = contiguous_array_fetch_valid_object(Z_OBJ_P(other));
        if (!right) return FAILURE;
        if (left->ndims != right->ndims ||
            memcmp(left->shape, right->shape, left->ndims * sizeof(int)) != 0) {
            CUDA_THROW_INVALID("HostArray arithmetic requires identical shapes");
            return FAILURE;
        }
        if (left->dtype != right->dtype) {
            CUDA_THROW_INVALID("HostArray arithmetic requires identical dtypes");
            return FAILURE;
        }
    } else if (Z_TYPE_P(other) == IS_LONG || Z_TYPE_P(other) == IS_DOUBLE ||
               Z_TYPE_P(other) == IS_TRUE || Z_TYPE_P(other) == IS_FALSE) {
        scalar = zval_get_double(other);
    } else {
        CUDA_THROW_INVALID("HostArray arithmetic expects a scalar or HostArray");
        return FAILURE;
    }

    if (left->dtype != DTYPE_FLOAT32 && left->dtype != DTYPE_FLOAT64) {
        CUDA_THROW_INVALID("CPU HostArray arithmetic currently supports float32 and float64");
        return FAILURE;
    }
    if (operation == '/' && !right && scalar == 0.0) {
        CUDA_THROW_INVALID("Division by zero is not supported by HostArray");
        return FAILURE;
    }

    int shape[MAX_DIMS];
    memcpy(shape, left->shape, left->ndims * sizeof(int));
    tensor_t *result_tensor = cuda_tensor_create_on_host(
        shape, left->ndims, NULL, left->dtype);
    if (!result_tensor) return FAILURE;

    const void *left_data = left->cached_data_ptr;
    const void *right_data = right ? right->cached_data_ptr : NULL;
    int left_contiguous = contiguous_array_has_contiguous_layout(left);
    int right_contiguous = right && contiguous_array_has_contiguous_layout(right);
    for (size_t i = 0; i < left->total_elements; i++) {
        size_t left_offset = left_contiguous
            ? i
            : contiguous_array_element_offset(left, i);
        size_t right_offset = right
            ? (right_contiguous
                ? i
                : contiguous_array_element_offset(right, i))
            : 0;
        const char *left_element =
            (const char *)left_data + left_offset * left->element_size;
        const char *right_element = right
            ? (const char *)right_data + right_offset * right->element_size
            : NULL;
        double tensor_value = left->dtype == DTYPE_FLOAT32
            ? *(const float *)left_element
            : *(const double *)left_element;
        double other_tensor_value = right
            ? (left->dtype == DTYPE_FLOAT32
                ? *(const float *)right_element
                : *(const double *)right_element)
            : scalar;
        double lhs = reverse ? other_tensor_value : tensor_value;
        double rhs = reverse ? tensor_value : other_tensor_value;
        if (operation == '/' && rhs == 0.0) {
            cuda_tensor_destroy(result_tensor);
            CUDA_THROW_INVALID("Division by zero is not supported by HostArray");
            return FAILURE;
        }
        double value = operation == '+' ? lhs + rhs :
                       operation == '-' ? lhs - rhs :
                       operation == '*' ? lhs * rhs : lhs / rhs;
        if (left->dtype == DTYPE_FLOAT32) {
            ((float *)result_tensor->data)[i] = (float)value;
        } else {
            ((double *)result_tensor->data)[i] = value;
        }
    }

    zend_object *object = contiguous_array_from_tensor(result_tensor);
    if (!object) {
        cuda_tensor_destroy(result_tensor);
        CUDA_THROW_RUNTIME("Failed to create HostArray result");
        return FAILURE;
    }
    ZVAL_OBJ(return_value, object);
    return SUCCESS;
}

static void contiguous_array_binary(INTERNAL_FUNCTION_PARAMETERS, char operation)
{
    zval *other;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other)
    ZEND_PARSE_PARAMETERS_END();
    if (contiguous_array_binary_apply(return_value, ZEND_THIS, other,
                                      operation, 0) != SUCCESS) {
        RETURN_THROWS();
    }
}

ZEND_METHOD(ContiguousArray, add)
{
    contiguous_array_binary(INTERNAL_FUNCTION_PARAM_PASSTHRU, '+');
}

ZEND_METHOD(ContiguousArray, subtract)
{
    contiguous_array_binary(INTERNAL_FUNCTION_PARAM_PASSTHRU, '-');
}

ZEND_METHOD(ContiguousArray, multiply)
{
    contiguous_array_binary(INTERNAL_FUNCTION_PARAM_PASSTHRU, '*');
}

ZEND_METHOD(ContiguousArray, divide)
{
    contiguous_array_binary(INTERNAL_FUNCTION_PARAM_PASSTHRU, '/');
}

static zend_result contiguous_array_do_operation(zend_uchar opcode, zval *result,
                                                  zval *op1, zval *op2)
{
    char operation;
    switch (opcode) {
        case ZEND_ADD: operation = '+'; break;
        case ZEND_SUB: operation = '-'; break;
        case ZEND_MUL: operation = '*'; break;
        case ZEND_DIV: operation = '/'; break;
        default: return FAILURE;
    }
    zval value;
    ZVAL_UNDEF(&value);
    zend_result status;
    if (Z_TYPE_P(op1) == IS_OBJECT &&
        instanceof_function(Z_OBJCE_P(op1), contiguous_array_ce)) {
        status = contiguous_array_binary_apply(&value, op1, op2, operation, 0);
    } else if (Z_TYPE_P(op2) == IS_OBJECT &&
        instanceof_function(Z_OBJCE_P(op2), contiguous_array_ce)) {
        status = contiguous_array_binary_apply(&value, op2, op1, operation, 1);
    } else {
        return FAILURE;
    }
    return php_tensors_finish_operation(status, result, &value, op1, op2);
}

ZEND_METHOD(ContiguousArray, toBuffer)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    const void *host_data;
    void *packed_data;
    if (contiguous_array_host_data(obj, &host_data, &packed_data) != SUCCESS)
    {
        RETURN_THROWS();
    }
    size_t byte_count = obj->total_elements * obj->element_size;
    zend_string *buffer = zend_string_alloc(byte_count, 0);
    if (byte_count)
        memcpy(ZSTR_VAL(buffer), host_data, byte_count);
    ZSTR_VAL(buffer)[byte_count] = '\0';
    if (packed_data)
        efree(packed_data);
    RETURN_STR(buffer);
}

ZEND_METHOD(ContiguousArray, isPinned)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    RETURN_BOOL(obj->tensor->host_pinned);
}

ZEND_METHOD(ContiguousArray, fromBuffer)
{
    zend_string *bytes;
    zval *shape_array;
    zend_string *dtype_name = NULL;
    zend_bool pinned = 0;

    ZEND_PARSE_PARAMETERS_START(2, 4)
    Z_PARAM_STR(bytes)
    Z_PARAM_ARRAY(shape_array)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR(dtype_name)
    Z_PARAM_BOOL(pinned)
    ZEND_PARSE_PARAMETERS_END();

    dtype_t dtype = dtype_from_string(dtype_name ? ZSTR_VAL(dtype_name) : "float32");
    if (dtype >= DTYPE_COUNT || dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Invalid host array dtype");
        RETURN_THROWS();
    }

    int shape[MAX_DIMS];
    size_t elements;
    int ndims = tensor_import_shape(shape_array, shape, &elements);
    if (!ndims) RETURN_THROWS();
    size_t element_size = dtype_size(dtype);
    if (!element_size || elements > SIZE_MAX / element_size || ZSTR_LEN(bytes) != elements * element_size)
    {
        CUDA_THROW_INVALID("Host buffer size does not match shape and dtype");
        RETURN_THROWS();
    }

    tensor_t *tensor = pinned
        ? cuda_tensor_create_on_host_pinned(shape, ndims, ZSTR_VAL(bytes), dtype)
        : cuda_tensor_create_on_host(shape, ndims, ZSTR_VAL(bytes), dtype);
    if (!tensor) RETURN_THROWS();
    ZVAL_OBJ(return_value, contiguous_array_from_tensor(tensor));
}

ZEND_METHOD(ContiguousArray, at)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    uint32_t argc = ZEND_NUM_ARGS();

    if (UNEXPECTED(argc != (uint32_t)obj->ndims))
    {
        CUDA_THROW_INVALID("ContiguousArray: Expected %d indices, got %d", obj->ndims, argc);
        return;
    }

    zend_long *indices = (zend_long *)alloca(sizeof(zend_long) * argc);
    zval *args = (zval *)alloca(sizeof(zval) * argc);

    if (zend_get_parameters_array_ex(argc, args) == FAILURE)
    {
        return;
    }

    size_t final_offset = 0;
    for (uint32_t i = 0; i < argc; i++)
    {
        zend_long idx = zval_get_long(&args[i]);

        if (UNEXPECTED((zend_ulong)idx >= (zend_ulong)obj->shape[i]))
        {
            CUDA_THROW_INVALID("Index %ld out of bounds at dimension %d", idx, i);
            return;
        }
        final_offset += idx * obj->strides[i];
    }

    get_value(obj->cached_data_ptr + (final_offset * obj->element_size), return_value, obj->dtype);
}

ZEND_METHOD(ContiguousArray, get)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    zval *index_array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY(index_array)
    ZEND_PARSE_PARAMETERS_END();

    HashTable *ht = Z_ARRVAL_P(index_array);
    if (zend_hash_num_elements(ht) != obj->ndims)
    {
        CUDA_THROW_INVALID("Expected %d indices", obj->ndims);
        return;
    }

    size_t final_offset = 0;
    int dim = 0;
    zval *val;
    ZEND_HASH_FOREACH_VAL(ht, val)
    {
        zend_long idx = zval_get_long(val);
        if (idx < 0 || idx >= obj->shape[dim])
        {
            CUDA_THROW_INVALID("Index out of bounds");
            return;
        }
        final_offset += idx * obj->strides[dim];
        dim++;
    }
    ZEND_HASH_FOREACH_END();

    get_value(obj->cached_data_ptr + (final_offset * obj->element_size), return_value, obj->dtype);
}

ZEND_METHOD(ContiguousArray, getSize)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    RETURN_LONG(obj->total_elements);
}

ZEND_METHOD(ContiguousArray, getNdims)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    RETURN_LONG(obj->ndims);
}

ZEND_METHOD(ContiguousArray, getDtype)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    RETURN_STRING(dtype_to_string(obj->dtype));
}

ZEND_METHOD(ContiguousArray, dtype)
{
    ZEND_PARSE_PARAMETERS_NONE();
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    RETURN_STRING(dtype_to_string(obj->dtype));
}

ZEND_METHOD(ContiguousArray, getBackend)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_STRING("cpu");
}

ZEND_METHOD(ContiguousArray, getElementSize)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    RETURN_LONG(obj->element_size);
}

ZEND_METHOD(ContiguousArray, count)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    RETURN_LONG(obj->ndims > 0 ? obj->shape[0] : 0);
}

ZEND_METHOD(ContiguousArray, getShape)
{
    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(getThis()));
    if (!obj) RETURN_THROWS();
    array_init_size(return_value, obj->ndims);
    for (int i = 0; i < obj->ndims; i++)
        add_next_index_long(return_value, obj->shape[i]);
}

ZEND_METHOD(ContiguousArray, __construct)
{
    zval *values;
    zend_string *dtype_name = NULL;
    zend_bool pinned = 0;

    ZEND_PARSE_PARAMETERS_START(1, 3)
    Z_PARAM_ARRAY(values)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR(dtype_name)
    Z_PARAM_BOOL(pinned)
    ZEND_PARSE_PARAMETERS_END();

    dtype_t dtype = dtype_from_string(dtype_name ? ZSTR_VAL(dtype_name) : "float32");
    if (dtype >= DTYPE_COUNT || dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Invalid host array dtype");
        RETURN_THROWS();
    }

    int shape[MAX_DIMS] = {0};
    int ndims = -1;
    if (!contiguous_array_validate_values(values, 0, shape, &ndims) || ndims <= 0)
    {
        CUDA_THROW_INVALID("HostArray requires a nonempty, rectangular numeric array");
        RETURN_THROWS();
    }

    size_t elements = 1;
    for (int axis = 0; axis < ndims; axis++)
    {
        if (elements > SIZE_MAX / (size_t)shape[axis])
        {
            CUDA_THROW_INVALID("HostArray shape exceeds supported limits");
            RETURN_THROWS();
        }
        elements *= (size_t)shape[axis];
    }
    if (elements > SIZE_MAX / dtype_size(dtype))
    {
        CUDA_THROW_INVALID("HostArray byte size exceeds supported limits");
        RETURN_THROWS();
    }

    tensor_t *tensor = pinned
        ? cuda_tensor_create_on_host_pinned(shape, ndims, NULL, dtype)
        : cuda_tensor_create_on_host(shape, ndims, NULL, dtype);
    if (!tensor) RETURN_THROWS();
    size_t position = 0;
    if (!contiguous_array_copy_values(values, tensor->data, dtype, &position))
    {
        cuda_tensor_destroy(tensor);
        RETURN_THROWS();
    }

    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(ZEND_THIS));
    if (obj->tensor) cuda_tensor_destroy(obj->tensor);
    contiguous_array_attach_tensor(obj, tensor);
}

static int contiguous_array_fill_tensor(tensor_t *tensor, zval *value)
{
    scalar_value_t converted;
    if (tensor_scalar_from_zval(value, tensor->dtype, &converted) != SUCCESS)
        return FAILURE;
    for (size_t i = 0; i < tensor->total_size; i++) {
        memcpy((char *)tensor->data + i * tensor->element_size, &converted.v, tensor->element_size);
    }
    return SUCCESS;
}

static int contiguous_array_create_filled(zval *return_value, zval *shape_value,
                                          zval *fill_value, zend_string *dtype_name)
{
    if (Z_TYPE_P(fill_value) != IS_LONG && Z_TYPE_P(fill_value) != IS_DOUBLE &&
        Z_TYPE_P(fill_value) != IS_TRUE && Z_TYPE_P(fill_value) != IS_FALSE) {
        CUDA_THROW_INVALID("HostArray fill value must be numeric");
        return FAILURE;
    }
    dtype_t dtype = dtype_from_string(dtype_name ? ZSTR_VAL(dtype_name) : "float32");
    if (dtype >= DTYPE_COUNT || dtype == DTYPE_UNKNOWN) {
        CUDA_THROW_INVALID("Invalid host array dtype");
        return FAILURE;
    }

    int shape[MAX_DIMS];
    size_t elements;
    int ndims = tensor_import_shape(shape_value, shape, &elements);
    if (!ndims) return FAILURE;
    tensor_t *tensor = cuda_tensor_create_on_host(shape, ndims, NULL, dtype);
    if (!tensor) return FAILURE;
    if (contiguous_array_fill_tensor(tensor, fill_value) != SUCCESS)
    {
        cuda_tensor_destroy(tensor);
        return FAILURE;
    }
    zend_object *object = contiguous_array_from_tensor(tensor);
    if (!object) {
        cuda_tensor_destroy(tensor);
        CUDA_THROW_RUNTIME("Failed to create HostArray object");
        return FAILURE;
    }
    ZVAL_OBJ(return_value, object);
    return SUCCESS;
}

ZEND_METHOD(ContiguousArray, fromFlatArray)
{
    zval *values, *shape_array;
    zend_string *dtype_name = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ARRAY(values)
        Z_PARAM_ARRAY(shape_array)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR(dtype_name)
    ZEND_PARSE_PARAMETERS_END();

    int shape[MAX_DIMS];
    size_t elements;
    int ndims = tensor_import_shape(shape_array, shape, &elements);
    if (!ndims) RETURN_THROWS();
    if (!zend_array_is_list(Z_ARRVAL_P(values)) ||
        zend_hash_num_elements(Z_ARRVAL_P(values)) != elements) {
        CUDA_THROW_INVALID("Flat values count must match the HostArray shape");
        RETURN_THROWS();
    }
    zval *value;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(values), value) {
        if (Z_TYPE_P(value) != IS_LONG && Z_TYPE_P(value) != IS_DOUBLE &&
            Z_TYPE_P(value) != IS_TRUE && Z_TYPE_P(value) != IS_FALSE) {
            CUDA_THROW_INVALID("HostArray values must be numeric");
            RETURN_THROWS();
        }
    } ZEND_HASH_FOREACH_END();

    dtype_t dtype = dtype_from_string(dtype_name ? ZSTR_VAL(dtype_name) : "float32");
    if (dtype >= DTYPE_COUNT || dtype == DTYPE_UNKNOWN) {
        CUDA_THROW_INVALID("Invalid host array dtype");
        RETURN_THROWS();
    }
    tensor_t *tensor = cuda_tensor_create_on_host(shape, ndims, NULL, dtype);
    if (!tensor) RETURN_THROWS();
    size_t position = 0;
    if (!contiguous_array_copy_values(values, tensor->data, dtype, &position))
    {
        cuda_tensor_destroy(tensor);
        RETURN_THROWS();
    }
    zend_object *object = contiguous_array_from_tensor(tensor);
    if (!object) {
        cuda_tensor_destroy(tensor);
        CUDA_THROW_RUNTIME("Failed to create HostArray object");
        RETURN_THROWS();
    }
    ZVAL_OBJ(return_value, object);
}

ZEND_METHOD(ContiguousArray, zeros)
{
    zval *shape;
    zend_string *dtype = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();
    zval value;
    ZVAL_LONG(&value, 0);
    if (contiguous_array_create_filled(return_value, shape, &value, dtype) != SUCCESS) {
        RETURN_THROWS();
    }
}

ZEND_METHOD(ContiguousArray, ones)
{
    zval *shape;
    zend_string *dtype = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();
    zval value;
    ZVAL_LONG(&value, 1);
    if (contiguous_array_create_filled(return_value, shape, &value, dtype) != SUCCESS) {
        RETURN_THROWS();
    }
}

ZEND_METHOD(ContiguousArray, full)
{
    zval *shape, *value;
    zend_string *dtype = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_ZVAL(value)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();
    if (contiguous_array_create_filled(return_value, shape, value, dtype) != SUCCESS) {
        RETURN_THROWS();
    }
}

static int contiguous_array_create_like(zval *return_value, zval *prototype,
                                        zval *fill_value, zend_string *dtype_name)
{
    contiguous_array_object *source = contiguous_array_fetch_valid_object(Z_OBJ_P(prototype));
    if (!source) return FAILURE;
    zval shape;
    array_init_size(&shape, source->ndims);
    for (int i = 0; i < source->ndims; i++) {
        add_next_index_long(&shape, source->shape[i]);
    }
    zend_string *source_dtype = dtype_name
        ? zend_string_copy(dtype_name)
        : zend_string_init(dtype_to_string(source->dtype),
                           strlen(dtype_to_string(source->dtype)), 0);
    int result = contiguous_array_create_filled(return_value, &shape, fill_value, source_dtype);
    zend_string_release(source_dtype);
    zval_ptr_dtor(&shape);
    return result;
}

ZEND_METHOD(ContiguousArray, fullLike)
{
    zval *prototype, *value;
    zend_string *dtype = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_OBJECT_OF_CLASS(prototype, contiguous_array_ce)
        Z_PARAM_ZVAL(value)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();
    if (contiguous_array_create_like(return_value, prototype, value, dtype) != SUCCESS) {
        RETURN_THROWS();
    }
}

ZEND_METHOD(ContiguousArray, zerosLike)
{
    zval *prototype;
    zend_string *dtype = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_OBJECT_OF_CLASS(prototype, contiguous_array_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();
    zval value;
    ZVAL_LONG(&value, 0);
    if (contiguous_array_create_like(return_value, prototype, &value, dtype) != SUCCESS) {
        RETURN_THROWS();
    }
}

ZEND_METHOD(ContiguousArray, onesLike)
{
    zval *prototype;
    zend_string *dtype = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_OBJECT_OF_CLASS(prototype, contiguous_array_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();
    zval value;
    ZVAL_LONG(&value, 1);
    if (contiguous_array_create_like(return_value, prototype, &value, dtype) != SUCCESS) {
        RETURN_THROWS();
    }
}

static void contiguous_array_iterator_dtor(zend_object_iterator *iter)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;
    zval_ptr_dtor(&iterator->host_array);
    if (Z_TYPE(iterator->current) != IS_UNDEF)
        zval_ptr_dtor(&iterator->current);
}

static int contiguous_array_iterator_valid(zend_object_iterator *iter)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;
    return (iterator->current_idx < iterator->max_idx) ? SUCCESS : FAILURE;
}

static zval *contiguous_array_iterator_get_current_data(zend_object_iterator *iter)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;

    if (iterator->is_1d)
    {
        iterator->getter(iterator->current_data_ptr, &iterator->current);
    }
    else
    {
        zval offset;
        ZVAL_LONG(&offset, iterator->current_idx);
        contiguous_array_offset_get(Z_OBJ(iterator->host_array), &offset, BP_VAR_R, &iterator->current);
    }
    return &iterator->current;
}

static void contiguous_array_iterator_move_forward(zend_object_iterator *iter)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;
    if (iterator->is_1d)
        iterator->current_data_ptr = (char *)iterator->current_data_ptr + iterator->stride_bytes;
    iterator->current_idx++;
    if (Z_TYPE(iterator->current) != IS_UNDEF)
    {
        zval_ptr_dtor(&iterator->current);
        ZVAL_UNDEF(&iterator->current);
    }
}

static const zend_object_iterator_funcs contiguous_array_iterator_funcs = {
    contiguous_array_iterator_dtor,
    contiguous_array_iterator_valid,
    contiguous_array_iterator_get_current_data,
    NULL,
    contiguous_array_iterator_move_forward,
    NULL, NULL, NULL};

static zend_object_iterator *contiguous_array_get_iterator(zend_class_entry *ce, zval *object, int by_ref)
{
    if (by_ref)
    {
        CUDA_THROW_INVALID("An iterator cannot be used with foreach by reference");
        return NULL;
    }

    contiguous_array_object *obj = contiguous_array_fetch_valid_object(Z_OBJ_P(object));
    if (!obj) return NULL;
    dtype_getter_t getter = contiguous_array_dtype_getter(obj->dtype);
    if (!getter)
        return NULL;
    contiguous_array_iterator *iterator = emalloc(sizeof(contiguous_array_iterator));
    zend_iterator_init(&iterator->intern);

    zend_object *snapshot = contiguous_array_create_object(contiguous_array_ce);
    contiguous_array_share_storage(contiguous_array_from_obj(snapshot), obj);
    ZVAL_OBJ(&iterator->host_array, snapshot);
    iterator->current_idx = 0;
    iterator->max_idx = obj->ndims > 0 ? obj->shape[0] : 0;
    ZVAL_UNDEF(&iterator->current);

    iterator->is_1d = (obj->ndims == 1);
    if (iterator->is_1d)
    {
        iterator->current_data_ptr = obj->cached_data_ptr;
        iterator->stride_bytes = obj->strides[0] * obj->element_size;
        iterator->getter = getter;
    }

    iterator->intern.funcs = &contiguous_array_iterator_funcs;
    return &iterator->intern;
}

int contiguous_array_init()
{
    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "PhpTensors", "HostArray", contiguous_array_methods);
    contiguous_array_ce = zend_register_internal_class(&ce);
    if (!contiguous_array_ce) return FAILURE;
    contiguous_array_ce->create_object = contiguous_array_create_object;
    contiguous_array_ce->get_iterator = contiguous_array_get_iterator;
    contiguous_array_ce->ce_flags |= ZEND_ACC_FINAL;
    zend_class_implements(contiguous_array_ce, 2,
                          php_tensors_tensor_ce,
                          php_tensors_operator_overloadable_ce);

    memcpy(&contiguous_array_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    contiguous_array_handlers.offset = XtOffsetOf(contiguous_array_object, std);
    contiguous_array_handlers.free_obj = contiguous_array_free_object;
    contiguous_array_handlers.clone_obj = contiguous_array_clone_object;
    contiguous_array_handlers.read_dimension = contiguous_array_offset_get;
    contiguous_array_handlers.count_elements = contiguous_array_count_elements;
    contiguous_array_handlers.do_operation = contiguous_array_do_operation;

    return SUCCESS;
}

zend_object *contiguous_array_create_object(zend_class_entry *ce)
{
    contiguous_array_object *obj = zend_object_alloc(sizeof(contiguous_array_object), ce);
    zend_object_std_init(&obj->std, ce);
    object_properties_init(&obj->std, ce);
    obj->std.handlers = &contiguous_array_handlers;
    return &obj->std;
}

void contiguous_array_free_object(zend_object *object)
{
    contiguous_array_object *obj = contiguous_array_from_obj(object);
    if (obj->tensor)
        cuda_tensor_destroy(obj->tensor);
    zend_object_std_dtor(&obj->std);
}

zend_object *contiguous_array_from_tensor(tensor_t *tensor)
{
    zend_object *zobj = contiguous_array_create_object(contiguous_array_ce);
    contiguous_array_object *obj = contiguous_array_from_obj(zobj);

    contiguous_array_attach_tensor(obj, tensor);

    return zobj;
}