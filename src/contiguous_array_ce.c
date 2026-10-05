#include "cuda_exceptions.h"
#include "php.h"
#include "contiguous_array_ce.h"
#include "contiguous_array_arginfo.h"
#include "zend_smart_str.h"
#include <string.h>
#include "tensor_factory.h"
#include "tensor_import.h"
#include "tensor_transfer.h"
#include "cuda_array_ce.h"
#include "ca_struct.h"
#include "zend_exceptions.h"

static zend_object_handlers contiguous_array_handlers;
zend_class_entry *contiguous_array_ce;

typedef void (*dtype_getter_t)(void *ptr, zval *rv);

static zend_always_inline void get_f32(void *p, zval *rv) { ZVAL_DOUBLE(rv, (double)*(float *)p); }
static zend_always_inline void get_f64(void *p, zval *rv) { ZVAL_DOUBLE(rv, *(double *)p); }
static zend_always_inline void get_i8(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long)*(int8_t *)p); }
static zend_always_inline void get_i16(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long)*(int16_t *)p); }
static zend_always_inline void get_i32(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long) * (int32_t *)p); }
static zend_always_inline void get_i64(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long) * (int64_t *)p); }
static zend_always_inline void get_u8(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long) * (uint8_t *)p); }
static zend_always_inline void get_u16(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long) * (uint16_t *)p); }
static zend_always_inline void get_u32(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long)*(uint32_t *)p); }
static zend_always_inline void get_u64(void *p, zval *rv) { ZVAL_LONG(rv, (zend_long)*(uint64_t *)p); }
static zend_always_inline void get_bool(void *p, zval *rv) { ZVAL_BOOL(rv, *(uint8_t *)p != 0); }

static const dtype_getter_t dtype_getters[] = {
    [DTYPE_FLOAT32] = get_f32,
    [DTYPE_FLOAT64] = get_f64,
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

static zend_always_inline contiguous_array_object *contiguous_array_from_obj(zend_object *zobj)
{
    return (contiguous_array_object *)((char *)zobj - contiguous_array_handlers.offset);
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

static void contiguous_array_copy_values(zval *value, void *buffer, dtype_t dtype, size_t *position)
{
    if (Z_TYPE_P(value) == IS_ARRAY)
    {
        zval *child;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(value), child)
            contiguous_array_copy_values(child, buffer, dtype, position);
        ZEND_HASH_FOREACH_END();
        return;
    }

    size_t index = (*position)++;
    switch (dtype)
    {
    case DTYPE_FLOAT32: ((float *)buffer)[index] = (float)zval_get_double(value); break;
    case DTYPE_FLOAT64: ((double *)buffer)[index] = zval_get_double(value); break;
    case DTYPE_INT8: ((int8_t *)buffer)[index] = (int8_t)zval_get_long(value); break;
    case DTYPE_INT16: ((int16_t *)buffer)[index] = (int16_t)zval_get_long(value); break;
    case DTYPE_INT32: ((int32_t *)buffer)[index] = (int32_t)zval_get_long(value); break;
    case DTYPE_INT64: ((int64_t *)buffer)[index] = (int64_t)zval_get_long(value); break;
    case DTYPE_UINT8: ((uint8_t *)buffer)[index] = (uint8_t)zval_get_long(value); break;
    case DTYPE_UINT16: ((uint16_t *)buffer)[index] = (uint16_t)zval_get_long(value); break;
    case DTYPE_UINT32: ((uint32_t *)buffer)[index] = (uint32_t)zval_get_long(value); break;
    case DTYPE_UINT64: ((uint64_t *)buffer)[index] = (uint64_t)zval_get_long(value); break;
    case DTYPE_BOOL: ((bool *)buffer)[index] = zend_is_true(value); break;
    default: break;
    }
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
        [DTYPE_FLOAT32] = 4, [DTYPE_FLOAT64] = 8, [DTYPE_INT8] = 1, [DTYPE_INT16] = 2, [DTYPE_INT32] = 4, [DTYPE_INT64] = 8, [DTYPE_UINT8] = 1, [DTYPE_UINT16] = 2, [DTYPE_UINT32] = 4, [DTYPE_UINT64] = 8, [DTYPE_BOOL] = 1};
    return (dtype < sizeof(sizes) / sizeof(size_t)) ? sizes[dtype] : 8;
}

static zend_always_inline void get_value(void *ptr, zval *rv, dtype_t dtype)
{
    switch (dtype)
    {
    case DTYPE_FLOAT32:
        ZVAL_DOUBLE(rv, (double)*(float *)ptr);
        break;
    case DTYPE_FLOAT64:
        ZVAL_DOUBLE(rv, *(double *)ptr);
        break;
    case DTYPE_INT8:
        ZVAL_LONG(rv, (zend_long) * (int8_t *)ptr);
        break;
    case DTYPE_INT16:
        ZVAL_LONG(rv, (zend_long) * (int16_t *)ptr);
        break;
    case DTYPE_INT32:
        ZVAL_LONG(rv, (zend_long) * (int32_t *)ptr);
        break;
    case DTYPE_INT64:
        ZVAL_LONG(rv, (zend_long) * (int64_t *)ptr);
        break;
    case DTYPE_UINT8:
        ZVAL_LONG(rv, (zend_long) * (uint8_t *)ptr);
        break;
    case DTYPE_UINT16:
        ZVAL_LONG(rv, (zend_long) * (uint16_t *)ptr);
        break;
    case DTYPE_UINT32:
        ZVAL_LONG(rv, (zend_long) * (uint32_t *)ptr);
        break;
    case DTYPE_UINT64:
        ZVAL_LONG(rv, (zend_long) * (uint64_t *)ptr);
        break;
    case DTYPE_BOOL:
        ZVAL_BOOL(rv, *(uint8_t *)ptr != 0);
        break;
    default:
        ZVAL_NULL(rv);
    }
}

int contiguous_array_count_elements(zend_object *object, zend_long *count)
{
    contiguous_array_object *obj = contiguous_array_from_obj(object);
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
    contiguous_array_object *obj = contiguous_array_from_obj(object);
    zend_long idx = zval_get_long(offset);

    if (UNEXPECTED(idx < 0 || idx >= obj->shape[0]))
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

        obj->tensor->ref_count++;
        slice->tensor = obj->tensor;
        slice->ndims = obj->ndims - 1;
        slice->shape = obj->shape + 1;
        slice->strides = obj->strides + 1;
        slice->dtype = obj->dtype;
        slice->element_size = obj->element_size;
        slice->cached_data_ptr = obj->cached_data_ptr + (idx * obj->strides[0] * obj->element_size);
        slice->is_contiguous = obj->is_contiguous;

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
    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(getThis()));
    array_init(return_value);
    add_assoc_stringl(return_value, "__contiguous_array_v1", "1", 1);

    add_assoc_long(return_value, "ndims", obj->ndims);
    add_assoc_long(return_value, "dtype", obj->dtype);
    add_assoc_long(return_value, "offset", obj->offset);
    add_assoc_long(return_value, "total_elements", obj->total_elements);
    add_assoc_long(return_value, "element_size", obj->element_size);
    add_assoc_bool(return_value, "is_contiguous", obj->is_contiguous);

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
        add_next_index_long(&strides_array, obj->strides[i]);
    }
    add_assoc_zval(return_value, "strides", &strides_array);

    void *data_ptr = obj->tensor ? obj->tensor->data : obj->cached_data_ptr;
    size_t data_size = obj->total_elements * obj->element_size;

    if (data_ptr && data_size > 0)
    {
        add_assoc_stringl(return_value, "data", data_ptr, data_size);
    }
    else
    {
        add_assoc_stringl(return_value, "data", "", 0);
    }
}

ZEND_METHOD(ContiguousArray, __unserialize)
{
    HashTable *data;

    if (zend_parse_parameters(ZEND_NUM_ARGS(), "h", &data) == FAILURE)
    {
        RETURN_NULL();
    }

    zval *version = zend_hash_str_find(data, "__contiguous_array_v1", sizeof("__contiguous_array_v1") - 1);
    if (!version)
    {
        CUDA_THROW_INVALID("Invalid serialized data version");
        RETURN_NULL();
    }

    zval *tmp;
    int ndims = 0;
    dtype_t dtype = 0;
    size_t total_elements = 0, element_size = 0;

    if ((tmp = zend_hash_str_find(data, "ndims", sizeof("ndims") - 1)) == NULL || Z_TYPE_P(tmp) != IS_LONG)
    {
        CUDA_THROW_INVALID("Missing or invalid ndims");
        RETURN_NULL();
    }
    ndims = Z_LVAL_P(tmp);

    if ((tmp = zend_hash_str_find(data, "dtype", sizeof("dtype") - 1)) == NULL || Z_TYPE_P(tmp) != IS_LONG)
    {
        CUDA_THROW_INVALID("Missing or invalid dtype");
        RETURN_NULL();
    }
    dtype = (dtype_t)Z_LVAL_P(tmp);

    if ((tmp = zend_hash_str_find(data, "total_elements", sizeof("total_elements") - 1)) == NULL || Z_TYPE_P(tmp) != IS_LONG)
    {
        CUDA_THROW_INVALID("Missing or invalid total_elements");
        RETURN_NULL();
    }
    total_elements = (size_t)Z_LVAL_P(tmp);

    if ((tmp = zend_hash_str_find(data, "element_size", sizeof("element_size") - 1)) == NULL || Z_TYPE_P(tmp) != IS_LONG)
    {
        CUDA_THROW_INVALID("Missing or invalid element_size");
        RETURN_NULL();
    }
    element_size = (size_t)Z_LVAL_P(tmp);

    if (ndims <= 0 || total_elements == 0 || element_size == 0)
    {
        CUDA_THROW_INVALID("Invalid array dimensions or size");
        RETURN_NULL();
    }

    zval *shape_zv = zend_hash_str_find(data, "shape", sizeof("shape") - 1);
    if (!shape_zv || Z_TYPE_P(shape_zv) != IS_ARRAY || (int)zend_hash_num_elements(Z_ARRVAL_P(shape_zv)) != ndims)
    {
        CUDA_THROW_INVALID("Missing or invalid shape");
        RETURN_NULL();
    }

    int *shape = safe_emalloc(ndims, sizeof(int), 0);
    int i = 0;
    zval *item;
    size_t calculated_elements = 1;

    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_zv), item)
    {
        if (i >= ndims)
            break;
        shape[i] = Z_LVAL_P(item);
        if (shape[i] <= 0)
        {
            efree(shape);
            CUDA_THROW_INVALID("Invalid shape value");
            RETURN_NULL();
        }
        calculated_elements *= shape[i];
        i++;
    }
    ZEND_HASH_FOREACH_END();

    if (calculated_elements != total_elements)
    {
        efree(shape);
        CUDA_THROW_INVALID("Shape does not match total_elements");
        RETURN_NULL();
    }

    zval *strides_zv = zend_hash_str_find(data, "strides", sizeof("strides") - 1);
    if (!strides_zv || Z_TYPE_P(strides_zv) != IS_ARRAY || (int)zend_hash_num_elements(Z_ARRVAL_P(strides_zv)) != ndims)
    {
        efree(shape);
        CUDA_THROW_INVALID("Missing or invalid strides");
        RETURN_NULL();
    }

    size_t *strides = safe_emalloc(ndims, sizeof(size_t), 0);
    i = 0;

    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(strides_zv), item)
    {
        if (i >= ndims)
            break;
        strides[i] = (size_t)Z_LVAL_P(item);
        i++;
    }
    ZEND_HASH_FOREACH_END();

    zval *data_zv = zend_hash_str_find(data, "data", sizeof("data") - 1);
    if (!data_zv || Z_TYPE_P(data_zv) != IS_STRING)
    {
        efree(shape);
        efree(strides);
        CUDA_THROW_INVALID("Missing or invalid data");
        RETURN_NULL();
    }

    size_t expected_data_size = total_elements * element_size;
    if (Z_STRLEN_P(data_zv) != expected_data_size)
    {
        efree(shape);
        efree(strides);
        CUDA_THROW_INVALID("Data size mismatch");
        RETURN_NULL();
    }

    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(getThis()));
    if (obj->tensor)
    {
        cuda_tensor_destroy(obj->tensor);
        obj->tensor = NULL;
    }

    obj->shape = NULL;
    obj->strides = NULL;
    obj->cached_data_ptr = NULL;

    void *array_data = safe_emalloc(total_elements, element_size, 0);
    memcpy(array_data, Z_STRVAL_P(data_zv), total_elements * element_size);

    tensor_t *tensor = cuda_tensor_create_on_host(shape, ndims, array_data, dtype);
    efree(array_data);
    
    if (!tensor)
    {
        efree(shape);
        efree(strides);
        CUDA_THROW_RUNTIME("Failed to create tensor");
        RETURN_NULL();
    }

    obj->tensor = tensor;
    obj->ndims = ndims;
    obj->dtype = dtype;
    obj->total_elements = total_elements;
    obj->element_size = element_size;
    obj->shape = tensor->shape;
    obj->strides = tensor->strides;
    obj->cached_data_ptr = (char *)tensor->data + (tensor->offset * element_size);
    obj->offset = tensor->offset;
    obj->is_contiguous = 1;
    obj->read_only = 0;

    efree(shape);
    efree(strides);
}

ZEND_METHOD(ContiguousArray, toArray)
{
    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(getThis()));
    contiguous_array_to_php_array(obj, return_value);
}

ZEND_METHOD(ContiguousArray, toGpu)
{
    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(getThis()));
    if (!obj->tensor || !obj->is_contiguous)
    {
        CUDA_THROW_RUNTIME("Only initialized contiguous host arrays can be copied to GPU");
        RETURN_THROWS();
    }
    size_t data_size = obj->total_elements * obj->element_size;
    tensor_t *gpu_tensor = cuda_tensor_create_from_host_buffer(obj->shape, obj->ndims, obj->dtype, obj->cached_data_ptr, data_size);
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

ZEND_METHOD(ContiguousArray, toBuffer)
{
    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(getThis()));
    if (!obj->tensor || !obj->is_contiguous)
    {
        CUDA_THROW_RUNTIME("Only initialized contiguous host arrays can be exported as bytes");
        RETURN_THROWS();
    }
    RETURN_STRINGL(obj->cached_data_ptr, obj->total_elements * obj->element_size);
}

ZEND_METHOD(ContiguousArray, isPinned)
{
    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(getThis()));
    RETURN_BOOL(obj->tensor && obj->tensor->host_pinned);
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
    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(getThis()));
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

    dtype_getters[obj->dtype](obj->cached_data_ptr + (final_offset * obj->element_size), return_value);
}

ZEND_METHOD(ContiguousArray, get)
{
    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(getThis()));
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
    RETURN_LONG(contiguous_array_from_obj(Z_OBJ_P(getThis()))->total_elements);
}

ZEND_METHOD(ContiguousArray, getNdims)
{
    RETURN_LONG(contiguous_array_from_obj(Z_OBJ_P(getThis()))->ndims);
}

ZEND_METHOD(ContiguousArray, getDtype)
{
    RETURN_STRING(dtype_to_string(contiguous_array_from_obj(Z_OBJ_P(getThis()))->dtype));
}

ZEND_METHOD(ContiguousArray, getElementSize)
{
    RETURN_LONG(contiguous_array_from_obj(Z_OBJ_P(getThis()))->element_size);
}

ZEND_METHOD(ContiguousArray, count)
{
    RETURN_LONG(contiguous_array_from_obj(Z_OBJ_P(getThis()))->shape[0]);
}

ZEND_METHOD(ContiguousArray, getShape)
{
    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(getThis()));
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
    contiguous_array_copy_values(values, tensor->data, dtype, &position);

    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(ZEND_THIS));
    if (obj->tensor) cuda_tensor_destroy(obj->tensor);
    contiguous_array_attach_tensor(obj, tensor);
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
        dtype_getter_t getter = (dtype_getter_t)iterator->extra_data;
        if (getter)
        {
            getter(iterator->current_data_ptr, &iterator->current);
        }
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

    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(object));
    contiguous_array_iterator *iterator = emalloc(sizeof(contiguous_array_iterator));
    zend_iterator_init(&iterator->intern);

    ZVAL_COPY(&iterator->host_array, object);
    iterator->current_idx = 0;
    iterator->max_idx = obj->ndims > 0 ? obj->shape[0] : 0;
    ZVAL_UNDEF(&iterator->current);

    iterator->is_1d = (obj->ndims == 1);
    if (iterator->is_1d)
    {
        iterator->current_data_ptr = obj->cached_data_ptr;
        iterator->stride_bytes = obj->strides[0] * obj->element_size;
        iterator->extra_data = (void *)((obj->dtype < sizeof(dtype_getters) / sizeof(dtype_getter_t)) ? dtype_getters[obj->dtype] : NULL);
    }

    iterator->intern.funcs = &contiguous_array_iterator_funcs;
    return &iterator->intern;
}

int contiguous_array_init()
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, "Cuda\\ContiguousArray", contiguous_array_methods);
    contiguous_array_ce = zend_register_internal_class(&ce);
    zend_register_class_alias("Cuda\\HostArray", contiguous_array_ce);
    contiguous_array_ce->create_object = contiguous_array_create_object;
    contiguous_array_ce->get_iterator = contiguous_array_get_iterator;
    contiguous_array_ce->ce_flags |= ZEND_ACC_FINAL;

    memcpy(&contiguous_array_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    contiguous_array_handlers.offset = XtOffsetOf(contiguous_array_object, std);
    contiguous_array_handlers.free_obj = contiguous_array_free_object;
    contiguous_array_handlers.read_dimension = contiguous_array_offset_get;
    contiguous_array_handlers.count_elements = contiguous_array_count_elements;

    return SUCCESS;
}

zend_object *contiguous_array_create_object(zend_class_entry *ce)
{
    contiguous_array_object *obj = zend_object_alloc(sizeof(contiguous_array_object), ce);
    zend_object_std_init(&obj->std, ce);
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