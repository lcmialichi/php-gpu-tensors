#include "php.h"
#include "zend_exceptions.h"
#include "backend_contract.h"
#include "cuda_exceptions.h"
#include "cpu_host_array.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define CPU_ARRAY_MAX_DIMS 10

typedef struct {
    double *values;
    zend_long shape[CPU_ARRAY_MAX_DIMS];
    uint32_t ndims;
    size_t size;
    zend_bool float32;
    zend_object std;
} cpu_host_array_object;

zend_class_entry *php_tensors_host_array_ce;
static zend_object_handlers cpu_host_array_handlers;

static zend_always_inline cpu_host_array_object *cpu_host_array_from_obj(zend_object *object)
{
    return (cpu_host_array_object *)((char *)object - cpu_host_array_handlers.offset);
}

static zend_object *cpu_host_array_create_object(zend_class_entry *ce)
{
    cpu_host_array_object *object = zend_object_alloc(sizeof(*object), ce);
    zend_object_std_init(&object->std, ce);
    object->std.handlers = &cpu_host_array_handlers;
    return &object->std;
}

static void cpu_host_array_free_object(zend_object *object)
{
    cpu_host_array_object *array = cpu_host_array_from_obj(object);
    if (array->values) {
        efree(array->values);
    }
    zend_object_std_dtor(&array->std);
}

static int cpu_host_array_parse_dtype(zend_string *dtype, zend_bool *float32)
{
    if (!dtype || zend_string_equals_literal(dtype, "float32")) {
        *float32 = 1;
        return SUCCESS;
    }
    if (zend_string_equals_literal(dtype, "float64")) {
        *float32 = 0;
        return SUCCESS;
    }
    zend_throw_exception_ex(cuda_invalid_argument_exception_ce, 0,
                            "CPU HostArray currently supports only float32 and float64, got '%s'",
                            ZSTR_VAL(dtype));
    return FAILURE;
}

static int cpu_host_array_parse_shape(zval *shape_value, zend_long shape[CPU_ARRAY_MAX_DIMS],
                                     uint32_t *ndims, size_t *size)
{
    if (!zend_array_is_list(Z_ARRVAL_P(shape_value))) {
        zend_throw_exception(cuda_invalid_argument_exception_ce,
                             "Shape must be a list of non-negative integers", 0);
        return FAILURE;
    }

    *ndims = (uint32_t)zend_hash_num_elements(Z_ARRVAL_P(shape_value));
    if (*ndims == 0 || *ndims > CPU_ARRAY_MAX_DIMS) {
        zend_throw_exception_ex(cuda_invalid_argument_exception_ce, 0,
                                "CPU HostArray supports ranks from 1 to %d", CPU_ARRAY_MAX_DIMS);
        return FAILURE;
    }

    *size = 1;
    zval *dimension;
    uint32_t index = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_value), dimension) {
        if (Z_TYPE_P(dimension) != IS_LONG || Z_LVAL_P(dimension) < 0) {
            zend_throw_exception(cuda_invalid_argument_exception_ce,
                                 "Shape must be a list of non-negative integers", 0);
            return FAILURE;
        }
        shape[index] = Z_LVAL_P(dimension);
        if (shape[index] != 0 && *size > SIZE_MAX / (size_t)shape[index]) {
            zend_throw_exception(cuda_invalid_argument_exception_ce,
                                 "Shape exceeds the CPU HostArray size limit", 0);
            return FAILURE;
        }
        *size *= (size_t)shape[index];
        if (*size > ZEND_LONG_MAX) {
            zend_throw_exception(cuda_invalid_argument_exception_ce,
                                 "Shape exceeds the PHP integer size limit", 0);
            return FAILURE;
        }
        index++;
    } ZEND_HASH_FOREACH_END();
    return SUCCESS;
}

static int cpu_host_array_values_count(zval *values, size_t expected)
{
    if (!zend_array_is_list(Z_ARRVAL_P(values)) ||
        zend_hash_num_elements(Z_ARRVAL_P(values)) != expected) {
        zend_throw_exception(cuda_invalid_argument_exception_ce,
                             "Flat values count must match the shape", 0);
        return FAILURE;
    }
    zval *value;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(values), value) {
        if (Z_TYPE_P(value) != IS_LONG && Z_TYPE_P(value) != IS_DOUBLE &&
            Z_TYPE_P(value) != IS_TRUE && Z_TYPE_P(value) != IS_FALSE) {
            zend_throw_exception(cuda_invalid_argument_exception_ce,
                                 "Tensor values must be numeric", 0);
            return FAILURE;
        }
    } ZEND_HASH_FOREACH_END();
    return SUCCESS;
}

static int cpu_host_array_collect_nested(zval *value, uint32_t depth,
                                         zend_long shape[CPU_ARRAY_MAX_DIMS],
                                         zend_bool dimensions_set[CPU_ARRAY_MAX_DIMS],
                                         uint32_t *ndims, int *leaf_depth, zval *flat)
{
    if (depth > CPU_ARRAY_MAX_DIMS ||
        (depth == CPU_ARRAY_MAX_DIMS && Z_TYPE_P(value) == IS_ARRAY)) {
        zend_throw_exception_ex(cuda_invalid_argument_exception_ce, 0,
                                "CPU HostArray supports at most %d dimensions",
                                CPU_ARRAY_MAX_DIMS);
        return FAILURE;
    }
    if (Z_TYPE_P(value) == IS_ARRAY) {
        if (!zend_array_is_list(Z_ARRVAL_P(value))) {
            zend_throw_exception(cuda_invalid_argument_exception_ce,
                                 "HostArray values must be rectangular lists", 0);
            return FAILURE;
        }
        zend_long length = (zend_long)zend_hash_num_elements(Z_ARRVAL_P(value));
        if (dimensions_set[depth] && shape[depth] != length) {
            zend_throw_exception(cuda_invalid_argument_exception_ce,
                                 "HostArray values must be rectangular lists", 0);
            return FAILURE;
        }
        dimensions_set[depth] = 1;
        shape[depth] = length;
        if (*ndims <= depth) {
            *ndims = depth + 1;
        }
        zval *child;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(value), child) {
            if (cpu_host_array_collect_nested(child, depth + 1, shape,
                                              dimensions_set, ndims,
                                              leaf_depth, flat) != SUCCESS) {
                return FAILURE;
            }
        } ZEND_HASH_FOREACH_END();
        return SUCCESS;
    }
    if (Z_TYPE_P(value) != IS_LONG && Z_TYPE_P(value) != IS_DOUBLE &&
        Z_TYPE_P(value) != IS_TRUE && Z_TYPE_P(value) != IS_FALSE) {
        zend_throw_exception(cuda_invalid_argument_exception_ce,
                             "HostArray values must be numeric", 0);
        return FAILURE;
    }
    if (*leaf_depth >= 0 && *leaf_depth != (int)depth) {
        zend_throw_exception(cuda_invalid_argument_exception_ce,
                             "HostArray values must be rectangular lists", 0);
        return FAILURE;
    }
    *leaf_depth = (int)depth;
    add_next_index_double(flat, zval_get_double(value));
    return SUCCESS;
}

static int cpu_host_array_assign(cpu_host_array_object *array, zval *values,
                                 const zend_long *shape, uint32_t ndims, size_t size,
                                 zend_bool float32)
{
    array->values = size ? safe_emalloc(size, sizeof(double), 0) : NULL;
    array->size = size;
    array->ndims = ndims;
    array->float32 = float32;
    memcpy(array->shape, shape, ndims * sizeof(zend_long));

    zval *value;
    size_t index = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(values), value) {
        double number = zval_get_double(value);
        array->values[index++] = float32 ? (double)(float)number : number;
    } ZEND_HASH_FOREACH_END();
    return SUCCESS;
}

static int cpu_host_array_assign_buffer(cpu_host_array_object *array, zend_string *bytes,
                                        const zend_long *shape, uint32_t ndims, size_t size,
                                        zend_bool float32)
{
    size_t element_size = float32 ? sizeof(float) : sizeof(double);
    if (size > SIZE_MAX / element_size ||
        ZSTR_LEN(bytes) != size * element_size) {
        zend_throw_exception(cuda_invalid_argument_exception_ce,
                             "Buffer length must match the shape and dtype", 0);
        return FAILURE;
    }

    array->values = size ? safe_emalloc(size, sizeof(double), 0) : NULL;
    array->size = size;
    array->ndims = ndims;
    array->float32 = float32;
    memcpy(array->shape, shape, ndims * sizeof(zend_long));
    const char *buffer = ZSTR_VAL(bytes);
    for (size_t i = 0; i < size; i++) {
        if (float32) {
            float value;
            memcpy(&value, buffer + i * sizeof(float), sizeof(float));
            array->values[i] = value;
        } else {
            double value;
            memcpy(&value, buffer + i * sizeof(double), sizeof(double));
            array->values[i] = value;
        }
    }
    return SUCCESS;
}

static int cpu_host_array_fill(zval *return_value, zval *values, zval *shape_value,
                               zend_string *dtype)
{
    zend_long shape[CPU_ARRAY_MAX_DIMS] = {0};
    uint32_t ndims = 0;
    size_t size = 0;
    zend_bool float32 = 0;
    if (cpu_host_array_parse_shape(shape_value, shape, &ndims, &size) != SUCCESS ||
        cpu_host_array_values_count(values, size) != SUCCESS ||
        cpu_host_array_parse_dtype(dtype, &float32) != SUCCESS) {
        return FAILURE;
    }

    object_init_ex(return_value, php_tensors_host_array_ce);
    cpu_host_array_object *array = cpu_host_array_from_obj(Z_OBJ_P(return_value));
    cpu_host_array_assign(array, values, shape, ndims, size, float32);
    if (php_tensors_backend_attach_registered(return_value, "cpu") != SUCCESS) {
        zval_ptr_dtor(return_value);
        ZVAL_UNDEF(return_value);
        return FAILURE;
    }
    return SUCCESS;
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_cpu_host_array_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, values, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, pinned, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cpu_host_array_factory, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, values, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cpu_host_array_fill, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cpu_host_array_from_buffer, 0, 2, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_INFO(0, bytes, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "\"float32\"")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, pinned, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_is_pinned, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_get, 0, 1, IS_MIXED, 0)
    ZEND_ARG_TYPE_INFO(0, indices, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(arginfo_cpu_host_array_at, 0, 0, 1)
    ZEND_ARG_VARIADIC_TYPE_INFO(0, indices, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_ndims, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_element_size, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_to_buffer, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_serialize, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_unserialize, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_shape, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_size, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_dtype, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_backend, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_cpu_host_array_to_array, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cpu_host_array_binary, 0, 1, PhpTensors\\HostArray, 0)
    ZEND_ARG_TYPE_MASK(0, other, MAY_BE_OBJECT | MAY_BE_DOUBLE | MAY_BE_LONG | MAY_BE_BOOL, NULL)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_cpu_host_array_to_gpu, 0, 0, PhpTensors\\Cuda\\CudaArray, 0)
ZEND_END_ARG_INFO()

ZEND_METHOD(PhpTensorsHostArray, __construct)
{
    zval *values;
    zend_string *dtype = NULL;
    zend_bool pinned = 0;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ARRAY(values)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
        Z_PARAM_BOOL(pinned)
    ZEND_PARSE_PARAMETERS_END();
    if (pinned) {
        zend_throw_exception(cuda_runtime_exception_ce,
                             "Pinned host memory is unavailable in the CPU-only build", 0);
        RETURN_THROWS();
    }

    zend_long shape[CPU_ARRAY_MAX_DIMS] = {0};
    zend_bool dimensions_set[CPU_ARRAY_MAX_DIMS] = {0};
    uint32_t ndims = 0;
    size_t size = 0;
    zend_bool float32 = 0;
    int leaf_depth = -1;
    zval flat;
    array_init(&flat);
    if (cpu_host_array_parse_dtype(dtype, &float32) != SUCCESS ||
        cpu_host_array_collect_nested(values, 0, shape, dimensions_set,
                                      &ndims, &leaf_depth, &flat) != SUCCESS) {
        zval_ptr_dtor(&flat);
        if (!EG(exception)) {
            zend_throw_exception(cuda_invalid_argument_exception_ce,
                                 "HostArray values must be a rectangular list", 0);
        }
        RETURN_THROWS();
    }
    if (ndims == 0 || (leaf_depth >= 0 && leaf_depth != (int)ndims) ||
        cpu_host_array_values_count(&flat, zend_hash_num_elements(Z_ARRVAL(flat))) != SUCCESS) {
        zval_ptr_dtor(&flat);
        if (!EG(exception)) {
            zend_throw_exception(cuda_invalid_argument_exception_ce,
                                 "HostArray values must be non-empty rectangular lists", 0);
        }
        RETURN_THROWS();
    }
    size = zend_hash_num_elements(Z_ARRVAL(flat));
    cpu_host_array_object *array = cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS));
    cpu_host_array_assign(array, &flat, shape, ndims, size, float32);
    zval_ptr_dtor(&flat);
    if (php_tensors_backend_attach_registered(ZEND_THIS, "cpu") != SUCCESS) {
        RETURN_THROWS();
    }
}

ZEND_METHOD(PhpTensorsHostArray, fromFlatArray)
{
    zval *values, *shape;
    zend_string *dtype = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ARRAY(values)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();
    if (cpu_host_array_fill(return_value, values, shape, dtype) != SUCCESS) {
        RETURN_THROWS();
    }
}

ZEND_METHOD(PhpTensorsHostArray, fromBuffer)
{
    zend_string *bytes, *dtype = NULL;
    zval *shape_value;
    zend_bool pinned = 0;
    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_STR(bytes)
        Z_PARAM_ARRAY(shape_value)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
        Z_PARAM_BOOL(pinned)
    ZEND_PARSE_PARAMETERS_END();
    if (pinned) {
        zend_throw_exception(cuda_runtime_exception_ce,
                             "Pinned host memory is unavailable in the CPU-only build", 0);
        RETURN_THROWS();
    }

    zend_long shape[CPU_ARRAY_MAX_DIMS] = {0};
    uint32_t ndims = 0;
    size_t size = 0;
    zend_bool float32 = 0;
    if (cpu_host_array_parse_shape(shape_value, shape, &ndims, &size) != SUCCESS ||
        cpu_host_array_parse_dtype(dtype, &float32) != SUCCESS) {
        RETURN_THROWS();
    }
    object_init_ex(return_value, php_tensors_host_array_ce);
    cpu_host_array_object *array = cpu_host_array_from_obj(Z_OBJ_P(return_value));
    if (cpu_host_array_assign_buffer(array, bytes, shape, ndims, size, float32) != SUCCESS ||
        php_tensors_backend_attach_registered(return_value, "cpu") != SUCCESS) {
        zval_ptr_dtor(return_value);
        ZVAL_UNDEF(return_value);
        RETURN_THROWS();
    }
}

ZEND_METHOD(PhpTensorsHostArray, isPinned)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_FALSE;
}

static zend_result cpu_host_array_filled(zval *return_value, zval *shape_value, double fill,
                                         zend_string *dtype)
{
    zend_long shape[CPU_ARRAY_MAX_DIMS] = {0};
    uint32_t ndims = 0;
    size_t size = 0;
    zend_bool float32 = 0;
    if (cpu_host_array_parse_shape(shape_value, shape, &ndims, &size) != SUCCESS ||
        cpu_host_array_parse_dtype(dtype, &float32) != SUCCESS) {
        return FAILURE;
    }
    object_init_ex(return_value, php_tensors_host_array_ce);
    cpu_host_array_object *array = cpu_host_array_from_obj(Z_OBJ_P(return_value));
    array->values = size ? safe_emalloc(size, sizeof(double), 0) : NULL;
    array->size = size;
    array->ndims = ndims;
    array->float32 = float32;
    memcpy(array->shape, shape, ndims * sizeof(zend_long));
    double value = float32 ? (double)(float)fill : fill;
    for (size_t i = 0; i < size; i++) {
        array->values[i] = value;
    }
    if (php_tensors_backend_attach_registered(return_value, "cpu") != SUCCESS) {
        zval_ptr_dtor(return_value);
        ZVAL_UNDEF(return_value);
        return FAILURE;
    }
    return SUCCESS;
}

ZEND_METHOD(PhpTensorsHostArray, zeros)
{
    zval *shape;
    zend_string *dtype = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();
    if (cpu_host_array_filled(return_value, shape, 0.0, dtype) != SUCCESS) {
        RETURN_THROWS();
    }
}

ZEND_METHOD(PhpTensorsHostArray, ones)
{
    zval *shape;
    zend_string *dtype = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY(shape)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dtype)
    ZEND_PARSE_PARAMETERS_END();
    if (cpu_host_array_filled(return_value, shape, 1.0, dtype) != SUCCESS) {
        RETURN_THROWS();
    }
}

ZEND_METHOD(PhpTensorsHostArray, getShape)
{
    ZEND_PARSE_PARAMETERS_NONE();
    cpu_host_array_object *array = cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS));
    array_init_size(return_value, array->ndims);
    for (uint32_t i = 0; i < array->ndims; i++) {
        add_next_index_long(return_value, array->shape[i]);
    }
}

ZEND_METHOD(PhpTensorsHostArray, getNdims)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_LONG(cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS))->ndims);
}

ZEND_METHOD(PhpTensorsHostArray, getSize)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_LONG((zend_long)cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS))->size);
}

ZEND_METHOD(PhpTensorsHostArray, getElementSize)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_LONG(cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS))->float32 ?
                sizeof(float) : sizeof(double));
}

ZEND_METHOD(PhpTensorsHostArray, count)
{
    ZEND_PARSE_PARAMETERS_NONE();
    cpu_host_array_object *array = cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS));
    RETURN_LONG(array->shape[0]);
}

ZEND_METHOD(PhpTensorsHostArray, dtype)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_STRING(cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS))->float32 ? "float32" : "float64");
}

ZEND_METHOD(PhpTensorsHostArray, getDtype)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_STRING(cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS))->float32 ? "float32" : "float64");
}

ZEND_METHOD(PhpTensorsHostArray, getBackend)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_STRING("cpu");
}

static void cpu_host_array_to_nested(zval *result, cpu_host_array_object *array,
                                     uint32_t dimension, size_t *index)
{
    if (dimension == array->ndims) {
        ZVAL_DOUBLE(result, array->values[*index]);
        (*index)++;
        return;
    }
    array_init_size(result, (uint32_t)array->shape[dimension]);
    for (zend_long i = 0; i < array->shape[dimension]; i++) {
        zval child;
        ZVAL_UNDEF(&child);
        cpu_host_array_to_nested(&child, array, dimension + 1, index);
        add_next_index_zval(result, &child);
    }
}

ZEND_METHOD(PhpTensorsHostArray, toArray)
{
    ZEND_PARSE_PARAMETERS_NONE();
    cpu_host_array_object *array = cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS));
    size_t index = 0;
    cpu_host_array_to_nested(return_value, array, 0, &index);
}

static zend_result cpu_host_array_get_indices(cpu_host_array_object *array, zval *indices,
                                               uint32_t count, zval *return_value)
{
    if (count != array->ndims) {
        zend_throw_exception_ex(cuda_invalid_argument_exception_ce, 0,
                                "Expected %u indices for a rank-%u HostArray",
                                array->ndims, array->ndims);
        return FAILURE;
    }
    size_t flat_index = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (Z_TYPE(indices[i]) != IS_LONG || Z_LVAL(indices[i]) < 0 ||
            Z_LVAL(indices[i]) >= array->shape[i]) {
            zend_throw_exception_ex(cuda_invalid_argument_exception_ce, 0,
                                    "Index at dimension %u is out of bounds", i);
            return FAILURE;
        }
        flat_index = flat_index * (size_t)array->shape[i] + (size_t)Z_LVAL(indices[i]);
    }
    ZVAL_DOUBLE(return_value, array->values[flat_index]);
    return SUCCESS;
}

ZEND_METHOD(PhpTensorsHostArray, get)
{
    zval *indices;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(indices)
    ZEND_PARSE_PARAMETERS_END();
    if (!zend_array_is_list(Z_ARRVAL_P(indices))) {
        zend_throw_exception(cuda_invalid_argument_exception_ce,
                             "Indices must be a list of integers", 0);
        RETURN_THROWS();
    }
    uint32_t count = (uint32_t)zend_hash_num_elements(Z_ARRVAL_P(indices));
    if (count > CPU_ARRAY_MAX_DIMS) {
        zend_throw_exception_ex(cuda_invalid_argument_exception_ce, 0,
                                "HostArray indices cannot exceed %d dimensions",
                                CPU_ARRAY_MAX_DIMS);
        RETURN_THROWS();
    }
    zval *coordinate = count ? safe_emalloc(count, sizeof(zval), 0) : NULL;
    zval *value;
    uint32_t index = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(indices), value) {
        ZVAL_COPY(&coordinate[index++], value);
    } ZEND_HASH_FOREACH_END();
    zend_result status = cpu_host_array_get_indices(
        cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS)), coordinate, count, return_value);
    for (uint32_t i = 0; i < count; i++) zval_ptr_dtor(&coordinate[i]);
    if (coordinate) efree(coordinate);
    if (status != SUCCESS) RETURN_THROWS();
}

ZEND_METHOD(PhpTensorsHostArray, at)
{
    zval *indices;
    uint32_t count;
    ZEND_PARSE_PARAMETERS_START(1, -1)
        Z_PARAM_VARIADIC('+', indices, count)
    ZEND_PARSE_PARAMETERS_END();
    if (cpu_host_array_get_indices(cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS)),
                                   indices, count, return_value) != SUCCESS) {
        RETURN_THROWS();
    }
}

static zend_string *cpu_host_array_make_buffer(cpu_host_array_object *array)
{
    size_t element_size = array->float32 ? sizeof(float) : sizeof(double);
    if (array->size > ZSTR_MAX_LEN / element_size) {
        zend_throw_exception(cuda_invalid_argument_exception_ce,
                             "HostArray buffer exceeds the PHP string size limit", 0);
        return NULL;
    }
    zend_string *bytes = zend_string_alloc(array->size * element_size, 0);
    char *buffer = ZSTR_VAL(bytes);
    for (size_t i = 0; i < array->size; i++) {
        if (array->float32) {
            float value = (float)array->values[i];
            memcpy(buffer + i * sizeof(float), &value, sizeof(float));
        } else {
            memcpy(buffer + i * sizeof(double), &array->values[i], sizeof(double));
        }
    }
    ZSTR_VAL(bytes)[ZSTR_LEN(bytes)] = '\0';
    return bytes;
}

ZEND_METHOD(PhpTensorsHostArray, toBuffer)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zend_string *bytes = cpu_host_array_make_buffer(
        cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS)));
    if (!bytes) RETURN_THROWS();
    RETURN_STR(bytes);
}

ZEND_METHOD(PhpTensorsHostArray, __serialize)
{
    ZEND_PARSE_PARAMETERS_NONE();
    cpu_host_array_object *array = cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS));
    array_init_size(return_value, 3);
    zval shape;
    array_init_size(&shape, array->ndims);
    for (uint32_t i = 0; i < array->ndims; i++) {
        add_next_index_long(&shape, array->shape[i]);
    }
    add_assoc_zval(return_value, "shape", &shape);
    add_assoc_string(return_value, "dtype", array->float32 ? "float32" : "float64");
    zend_string *buffer = cpu_host_array_make_buffer(array);
    if (!buffer) {
        zval_ptr_dtor(return_value);
        ZVAL_UNDEF(return_value);
        RETURN_THROWS();
    }
    add_assoc_str(return_value, "buffer", buffer);
}

ZEND_METHOD(PhpTensorsHostArray, __unserialize)
{
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();

    zval *shape_value = zend_hash_str_find(data, "shape", sizeof("shape") - 1);
    zval *dtype_value = zend_hash_str_find(data, "dtype", sizeof("dtype") - 1);
    zval *buffer_value = zend_hash_str_find(data, "buffer", sizeof("buffer") - 1);
    if (!shape_value || Z_TYPE_P(shape_value) != IS_ARRAY ||
        !dtype_value || Z_TYPE_P(dtype_value) != IS_STRING ||
        !buffer_value || Z_TYPE_P(buffer_value) != IS_STRING) {
        zend_throw_exception(cuda_invalid_argument_exception_ce,
                             "Serialized HostArray data must contain shape, dtype, and buffer", 0);
        RETURN_THROWS();
    }
    zend_long shape[CPU_ARRAY_MAX_DIMS] = {0};
    uint32_t ndims = 0;
    size_t size = 0;
    zend_bool float32 = 0;
    if (cpu_host_array_parse_shape(shape_value, shape, &ndims, &size) != SUCCESS ||
        cpu_host_array_parse_dtype(Z_STR_P(dtype_value), &float32) != SUCCESS) {
        RETURN_THROWS();
    }
    cpu_host_array_object *array = cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS));
    if (array->values) {
        efree(array->values);
        array->values = NULL;
    }
    if (cpu_host_array_assign_buffer(array, Z_STR_P(buffer_value), shape, ndims,
                                     size, float32) != SUCCESS ||
        php_tensors_backend_attach_registered(ZEND_THIS, "cpu") != SUCCESS) {
        RETURN_THROWS();
    }
}

static void cpu_host_array_binary(INTERNAL_FUNCTION_PARAMETERS, char operation)
{
    zval *other;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other)
    ZEND_PARSE_PARAMETERS_END();

    cpu_host_array_object *left = cpu_host_array_from_obj(Z_OBJ_P(ZEND_THIS));
    cpu_host_array_object *right = NULL;
    double scalar = 0.0;
    if (Z_TYPE_P(other) == IS_OBJECT &&
        instanceof_function(Z_OBJCE_P(other), php_tensors_host_array_ce)) {
        right = cpu_host_array_from_obj(Z_OBJ_P(other));
        if (left->ndims != right->ndims ||
            memcmp(left->shape, right->shape, left->ndims * sizeof(zend_long)) != 0) {
            zend_throw_exception(cuda_invalid_argument_exception_ce,
                                 "CPU HostArray arithmetic requires identical shapes", 0);
            RETURN_THROWS();
        }
    } else if (Z_TYPE_P(other) == IS_LONG || Z_TYPE_P(other) == IS_DOUBLE ||
               Z_TYPE_P(other) == IS_TRUE || Z_TYPE_P(other) == IS_FALSE) {
        scalar = zval_get_double(other);
    } else {
        zend_throw_exception(cuda_invalid_argument_exception_ce,
                             "CPU HostArray arithmetic expects a scalar or HostArray", 0);
        RETURN_THROWS();
    }
    if (operation == '/' && (right == NULL ? scalar == 0.0 : 0)) {
        zend_throw_exception(cuda_invalid_argument_exception_ce,
                             "Division by zero is not supported by CPU HostArray", 0);
        RETURN_THROWS();
    }

    object_init_ex(return_value, php_tensors_host_array_ce);
    cpu_host_array_object *result = cpu_host_array_from_obj(Z_OBJ_P(return_value));
    result->values = left->size ? safe_emalloc(left->size, sizeof(double), 0) : NULL;
    result->size = left->size;
    result->ndims = left->ndims;
    result->float32 = left->float32;
    memcpy(result->shape, left->shape, left->ndims * sizeof(zend_long));
    for (size_t i = 0; i < left->size; i++) {
        double rhs = right ? right->values[i] : scalar;
        if (operation == '/' && rhs == 0.0) {
            zval_ptr_dtor(return_value);
            ZVAL_UNDEF(return_value);
            zend_throw_exception(cuda_invalid_argument_exception_ce,
                                 "Division by zero is not supported by CPU HostArray", 0);
            RETURN_THROWS();
        }
        double value = operation == '+' ? left->values[i] + rhs :
                       operation == '-' ? left->values[i] - rhs :
                       operation == '*' ? left->values[i] * rhs :
                                          left->values[i] / rhs;
        result->values[i] = result->float32 ? (double)(float)value : value;
    }
    if (php_tensors_backend_attach_registered(return_value, "cpu") != SUCCESS) {
        zval_ptr_dtor(return_value);
        ZVAL_UNDEF(return_value);
        RETURN_THROWS();
    }
}

ZEND_METHOD(PhpTensorsHostArray, add) { cpu_host_array_binary(INTERNAL_FUNCTION_PARAM_PASSTHRU, '+'); }
ZEND_METHOD(PhpTensorsHostArray, subtract) { cpu_host_array_binary(INTERNAL_FUNCTION_PARAM_PASSTHRU, '-'); }
ZEND_METHOD(PhpTensorsHostArray, multiply) { cpu_host_array_binary(INTERNAL_FUNCTION_PARAM_PASSTHRU, '*'); }
ZEND_METHOD(PhpTensorsHostArray, divide) { cpu_host_array_binary(INTERNAL_FUNCTION_PARAM_PASSTHRU, '/'); }

ZEND_METHOD(PhpTensorsHostArray, toGpu)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zend_throw_exception(cuda_runtime_exception_ce,
                         "CUDA backend is unavailable in the CPU-only build", 0);
    RETURN_THROWS();
}

static const zend_function_entry cpu_host_array_methods[] = {
    ZEND_ME(PhpTensorsHostArray, __construct, arginfo_cpu_host_array_construct, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, fromFlatArray, arginfo_cpu_host_array_factory, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(PhpTensorsHostArray, fromBuffer, arginfo_cpu_host_array_from_buffer, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(PhpTensorsHostArray, zeros, arginfo_cpu_host_array_fill, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(PhpTensorsHostArray, ones, arginfo_cpu_host_array_fill, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(PhpTensorsHostArray, isPinned, arginfo_cpu_host_array_is_pinned, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, get, arginfo_cpu_host_array_get, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, at, arginfo_cpu_host_array_at, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, getBackend, arginfo_cpu_host_array_backend, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, getShape, arginfo_cpu_host_array_shape, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, getNdims, arginfo_cpu_host_array_ndims, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, getSize, arginfo_cpu_host_array_size, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, dtype, arginfo_cpu_host_array_dtype, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, getDtype, arginfo_cpu_host_array_dtype, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, getElementSize, arginfo_cpu_host_array_element_size, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, count, arginfo_cpu_host_array_size, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, toArray, arginfo_cpu_host_array_to_array, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, toBuffer, arginfo_cpu_host_array_to_buffer, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, add, arginfo_cpu_host_array_binary, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, subtract, arginfo_cpu_host_array_binary, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, multiply, arginfo_cpu_host_array_binary, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, divide, arginfo_cpu_host_array_binary, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, toGpu, arginfo_cpu_host_array_to_gpu, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, __serialize, arginfo_cpu_host_array_serialize, ZEND_ACC_PUBLIC)
    ZEND_ME(PhpTensorsHostArray, __unserialize, arginfo_cpu_host_array_unserialize, ZEND_ACC_PUBLIC)
    ZEND_FE_END
};

int cpu_host_array_init(void)
{
    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "PhpTensors", "HostArray", cpu_host_array_methods);
    php_tensors_host_array_ce = zend_register_internal_class(&ce);
    if (!php_tensors_host_array_ce) {
        return FAILURE;
    }
    php_tensors_host_array_ce->ce_flags |= ZEND_ACC_FINAL;
    php_tensors_host_array_ce->create_object = cpu_host_array_create_object;
    zend_class_implements(php_tensors_host_array_ce, 1, php_tensors_tensor_ce);
    zend_register_class_alias("Cuda\\HostArray", php_tensors_host_array_ce);
    zend_register_class_alias("Cuda\\ContiguousArray", php_tensors_host_array_ce);

    memcpy(&cpu_host_array_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    cpu_host_array_handlers.offset = XtOffsetOf(cpu_host_array_object, std);
    cpu_host_array_handlers.free_obj = cpu_host_array_free_object;
    return SUCCESS;
}
