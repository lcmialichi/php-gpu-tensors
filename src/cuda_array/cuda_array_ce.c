#include "php.h"
#include "cuda_array_ce.h"
#include "ca_private.h"
#include "ca_arginfo.h"
#include "operations.h"
#include "tensor_factory.h"
#include "memory_pool.h"
#include "cuda.h"
#include "zend_smart_str.h"
#include "data_types.h"
#include "contiguous_array_ce.h"
#include "tensor_transfer.h"
#include "cuda_exceptions.h"
#include "concat_kernels.h"
#include "tensor_import.h"
#include "tensor_where.h"
#include "fusion.h"

zend_class_entry *cuda_array_ce;
static zend_object_handlers cuda_array_handlers;

static cuda_array_obj *php_cuda_array_fetch_object(zend_object *obj);
static cuda_array_obj *php_cuda_array_fetch_deferred_object(zend_object *obj);
static zend_object *cuda_array_create_object(zend_class_entry *class_type);
static void cuda_array_free_object(zend_object *object);
static zend_object *cuda_array_clone_obj(zend_object *old_object);
static int parse_slice_parameter(zval *param, slice_info_t *slice);
static zend_result cuda_array_do_operation(zend_uchar opcode, zval *result, zval *op1, zval *op2);
static zval *cuda_array_read_dimension(zend_object *object, zval *offset, int type, zval *rv);
static void cuda_array_write_dimension(zend_object *object, zval *offset, zval *value);
static tensor_t *cuda_tensor_concat(zval *tensors_array, int axis);
static void static_tensor_creator(INTERNAL_FUNCTION_PARAMETERS, const char *method_name, scalar_value_t scalar_value);
static void rand_tensor_creator(INTERNAL_FUNCTION_PARAMETERS, unsigned long long seed);

static void reduction_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, operation_type_t operation_type, int return_arg);
static void unary_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, operation_type_t operation_type);
static void binary_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, operation_type_t operation_type);

static void sync_php_object_shape(cuda_array_obj *obj, tensor_t *tensor);

static dtype_t parse_dtype_param(zend_string *dtype_str)
{
    if (!dtype_str || ZSTR_LEN(dtype_str) == 0)
    {
        return DTYPE_FLOAT32;
    }

    dtype_t dtype = dtype_from_string(ZSTR_VAL(dtype_str));
    if (dtype >= DTYPE_COUNT)
    {
        return DTYPE_UNKNOWN;
    }

    return dtype;
}

ZEND_METHOD(CudaArray, __construct)
{
    if (Z_CUDA_ARRAY_P(ZEND_THIS)->tensor_handle &&
        !fusion_check_tensor_mutation(Z_CUDA_ARRAY_P(ZEND_THIS)->tensor_handle))
        RETURN_THROWS();
    zval *data;
    zend_string *dtype_str = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ARRAY(data)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR(dtype_str)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    dtype_t dtype = parse_dtype_param(dtype_str);
    if (dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Invalid dtype: '%s'", ZSTR_VAL(dtype_str));
        RETURN_NULL();
    }

    tensor_t *tensor = create_tensor_from_php_array(data, dtype);

    if (!tensor)
    {
        RETURN_THROWS();
    }

    if (obj->tensor_handle) cuda_tensor_destroy(obj->tensor_handle);
    obj->tensor_handle = tensor;
    sync_php_object_shape(obj, tensor);
}

ZEND_METHOD(CudaArray, fromFlatArray)
{
    zval *values, *shape_array;
    zend_string *dtype_name = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
    Z_PARAM_ARRAY(values)
    Z_PARAM_ARRAY(shape_array)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR(dtype_name)
    ZEND_PARSE_PARAMETERS_END();

    dtype_t dtype = parse_dtype_param(dtype_name);
    if (dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Invalid dtype: '%s'", ZSTR_VAL(dtype_name));
        RETURN_THROWS();
    }
    int shape[MAX_DIMS];
    size_t elements;
    int ndims = tensor_import_shape(shape_array, shape, &elements);
    if (!ndims) RETURN_THROWS();
    tensor_t *tensor = cuda_tensor_create_from_flat_array(values, shape, ndims, dtype);
    if (!tensor) RETURN_THROWS();
    create_result_object(return_value, tensor);
}

ZEND_METHOD(CudaArray, fromBuffer)
{
    zend_string *bytes;
    zval *shape_array;
    zend_string *dtype_name = NULL;

    ZEND_PARSE_PARAMETERS_START(2, 3)
    Z_PARAM_STR(bytes)
    Z_PARAM_ARRAY(shape_array)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR(dtype_name)
    ZEND_PARSE_PARAMETERS_END();

    dtype_t dtype = parse_dtype_param(dtype_name);
    if (dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Invalid dtype: '%s'", ZSTR_VAL(dtype_name));
        RETURN_THROWS();
    }

    int shape[MAX_DIMS];
    size_t elements;
    int ndims = tensor_import_shape(shape_array, shape, &elements);
    if (!ndims) RETURN_THROWS();

    size_t element_size = dtype_size(dtype);
    if (!element_size || elements > SIZE_MAX / element_size ||
        ZSTR_LEN(bytes) != elements * element_size)
    {
        CUDA_THROW_INVALID("Buffer size does not match shape and dtype");
        RETURN_THROWS();
    }

    tensor_t *tensor = cuda_tensor_create_from_host_buffer(shape, ndims, dtype, ZSTR_VAL(bytes), ZSTR_LEN(bytes));
    if (!tensor) RETURN_THROWS();
    create_result_object(return_value, tensor);
}

ZEND_METHOD(CudaArray, fromFile)
{
    zend_string *path;
    zval *shape_array;
    zend_string *dtype_name = NULL;

    ZEND_PARSE_PARAMETERS_START(2, 3)
    Z_PARAM_STR(path)
    Z_PARAM_ARRAY(shape_array)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR(dtype_name)
    ZEND_PARSE_PARAMETERS_END();

    dtype_t dtype = parse_dtype_param(dtype_name);
    if (dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Invalid dtype: '%s'", ZSTR_VAL(dtype_name));
        RETURN_THROWS();
    }

    int shape[MAX_DIMS];
    size_t elements;
    int ndims = tensor_import_shape(shape_array, shape, &elements);
    if (!ndims) RETURN_THROWS();

    size_t element_size = dtype_size(dtype);
    if (!element_size || elements > SIZE_MAX / element_size)
    {
        CUDA_THROW_INVALID("Tensor byte size exceeds supported limits");
        RETURN_THROWS();
    }

    tensor_t *tensor = tensor_import_file(path, shape, ndims, dtype, elements * element_size);
    if (!tensor) RETURN_THROWS();
    create_result_object(return_value, tensor);
}

ZEND_METHOD(CudaArray, fromNpy)
{
    zend_string *path;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_STR(path)
    ZEND_PARSE_PARAMETERS_END();

    tensor_t *tensor = tensor_import_npy(path);
    if (!tensor) RETURN_THROWS();
    create_result_object(return_value, tensor);
}

ZEND_METHOD(CudaArray, where)
{
    zval *condition_value, *true_value, *false_value;
    ZEND_PARSE_PARAMETERS_START(3, 3)
    Z_PARAM_ZVAL(condition_value)
    Z_PARAM_ZVAL(true_value)
    Z_PARAM_ZVAL(false_value)
    ZEND_PARSE_PARAMETERS_END();

    zval *values[] = {condition_value, true_value, false_value};
    tensor_t *tensors[3];
    for (int index = 0; index < 3; index++)
    {
        if (Z_TYPE_P(values[index]) != IS_OBJECT ||
            !instanceof_function(Z_OBJCE_P(values[index]), cuda_array_ce))
        {
            CUDA_THROW_INVALID("where expects condition, x and y to be CudaArray objects");
            RETURN_THROWS();
        }
        cuda_array_obj *object = fusion_active()
            ? php_cuda_array_fetch_deferred_object(Z_OBJ_P(values[index]))
            : php_cuda_array_fetch_valid_object(Z_OBJ_P(values[index]));
        if (!object) RETURN_THROWS();
        tensors[index] = object->tensor_handle;
    }

    tensor_t *result = fusion_active() ? fusion_where(tensors[0], tensors[1], tensors[2])
        : cuda_tensor_where(tensors[0], tensors[1], tensors[2]);
    if (!result) RETURN_THROWS();
    create_result_object(return_value, result);
}

ZEND_METHOD(CudaArray, __serialize)
{
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    tensor_t *tensor = obj->tensor_handle;

    if (tensor->is_view)
    {
        CUDA_THROW_RUNTIME("Cannot serialize non-contiguous CudaArray views");
        RETURN_THROWS();
    }

    size_t data_size = tensor->total_size * tensor->element_size;
    char *host_data = emalloc(data_size);

    cudaError_t status = data_size ? cudaMemcpy(host_data, tensor->data, data_size, cudaMemcpyDeviceToHost) : cudaSuccess;
    if (status != cudaSuccess)
    {
        efree(host_data);
        CUDA_THROW_RUNTIME("CUDA error copying serialized data to host: %s", cudaGetErrorString(status));
        RETURN_THROWS();
    }

    array_init(return_value);
    add_assoc_stringl(return_value, "__cuda_array_v1", "1", 1);
    add_assoc_long(return_value, "ndims", tensor->ndims);
    add_assoc_string(return_value, "dtype", dtype_to_string(tensor->dtype));
    add_assoc_long(return_value, "total_elements", tensor->total_size);
    add_assoc_long(return_value, "element_size", tensor->element_size);

    zval shape_array;
    array_init(&shape_array);
    for (int i = 0; i < tensor->ndims; i++)
    {
        add_next_index_long(&shape_array, tensor->shape[i]);
    }
    add_assoc_zval(return_value, "shape", &shape_array);

    add_assoc_stringl(return_value, "data", host_data, data_size);
    efree(host_data);
}

ZEND_METHOD(CudaArray, __unserialize)
{
    if (!fusion_check_tensor_mutation(Z_CUDA_ARRAY_P(ZEND_THIS)->tensor_handle)) RETURN_THROWS();
    HashTable *data;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();

    zval *version = zend_hash_str_find(data, "__cuda_array_v1", sizeof("__cuda_array_v1") - 1);
    if (!version)
    {
        CUDA_THROW_INVALID("Invalid serialized CudaArray payload");
        RETURN_THROWS();
    }

    zval *ndims_zv = zend_hash_str_find(data, "ndims", sizeof("ndims") - 1);
    zval *dtype_zv = zend_hash_str_find(data, "dtype", sizeof("dtype") - 1);
    zval *shape_zv = zend_hash_str_find(data, "shape", sizeof("shape") - 1);
    zval *data_zv = zend_hash_str_find(data, "data", sizeof("data") - 1);

    if (!ndims_zv || Z_TYPE_P(ndims_zv) != IS_LONG ||
        !dtype_zv || Z_TYPE_P(dtype_zv) != IS_STRING ||
        !shape_zv || Z_TYPE_P(shape_zv) != IS_ARRAY ||
        !data_zv || Z_TYPE_P(data_zv) != IS_STRING)
    {
        CUDA_THROW_INVALID("Malformed serialized CudaArray payload");
        RETURN_THROWS();
    }

    int ndims = (int)Z_LVAL_P(ndims_zv);
    if (Z_LVAL_P(ndims_zv) < 0 || Z_LVAL_P(ndims_zv) > MAX_DIMS ||
        zend_hash_num_elements(Z_ARRVAL_P(shape_zv)) != (uint32_t)ndims)
    {
        CUDA_THROW_INVALID("Invalid serialized CudaArray shape");
        RETURN_THROWS();
    }

    dtype_t dtype = dtype_from_string(Z_STRVAL_P(dtype_zv));
    if (dtype == DTYPE_UNKNOWN || dtype >= DTYPE_COUNT)
    {
        CUDA_THROW_INVALID("Invalid serialized CudaArray dtype: %s", Z_STRVAL_P(dtype_zv));
        RETURN_THROWS();
    }

    int shape[MAX_DIMS] = {0};
    size_t total_elements = 1;
    int i = 0;
    zval *dim_zv;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_zv), dim_zv)
    {
        if (Z_TYPE_P(dim_zv) != IS_LONG || Z_LVAL_P(dim_zv) < 0 || Z_LVAL_P(dim_zv) > INT_MAX ||
            (Z_LVAL_P(dim_zv) && total_elements > SIZE_MAX / (size_t)Z_LVAL_P(dim_zv)))
        {
            CUDA_THROW_INVALID("Invalid serialized CudaArray dimension");
            RETURN_THROWS();
        }
        zend_long dim = Z_LVAL_P(dim_zv);
        shape[i++] = (int)dim;
        total_elements *= (size_t)dim;
    }
    ZEND_HASH_FOREACH_END();

    if (total_elements > SIZE_MAX / dtype_size(dtype))
    {
        CUDA_THROW_INVALID("Serialized CudaArray size overflow");
        RETURN_THROWS();
    }
    size_t expected_size = total_elements * dtype_size(dtype);
    if (Z_STRLEN_P(data_zv) != expected_size)
    {
        CUDA_THROW_INVALID("Serialized CudaArray data size mismatch: expected %zu bytes, got %zu", expected_size, Z_STRLEN_P(data_zv));
        RETURN_THROWS();
    }

    tensor_t *tensor = cuda_tensor_create_from_host_buffer(shape, ndims, dtype, Z_STRVAL_P(data_zv), Z_STRLEN_P(data_zv));
    if (!tensor)
    {
        RETURN_THROWS();
    }

    cuda_array_obj *obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    if (obj->tensor_handle)
    {
        cuda_tensor_destroy(obj->tensor_handle);
    }

    obj->tensor_handle = tensor;
    sync_php_object_shape(obj, tensor);
}

ZEND_METHOD(CudaArray, multiply)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Multiplication", OP_MUL);
}

ZEND_METHOD(CudaArray, divide)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Division", OP_DIV);
}

ZEND_METHOD(CudaArray, add)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Addition", OP_ADD);
}

ZEND_METHOD(CudaArray, subtract)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Subtraction", OP_SUB);
}

ZEND_METHOD(CudaArray, power)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Power", OP_POW);
}

ZEND_METHOD(CudaArray, gt)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Greater", OP_GT);
}

ZEND_METHOD(CudaArray, lt)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Less", OP_LT);
}

ZEND_METHOD(CudaArray, eq)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Equal", OP_EQ);
}

ZEND_METHOD(CudaArray, ne)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "NotEqual", OP_NE);
}

ZEND_METHOD(CudaArray, ge)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "GreaterEqual", OP_GE);
}

ZEND_METHOD(CudaArray, le)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "LessEqual", OP_LE);
}

ZEND_METHOD(CudaArray, zeros)
{
    scalar_value_t scalar_value;
    scalar_value.v.f32 = 0.0f;
    scalar_value.is_neg = 0;
    scalar_value.dtype = DTYPE_FLOAT32;

    static_tensor_creator(INTERNAL_FUNCTION_PARAM_PASSTHRU, "zeros", scalar_value);
}

ZEND_METHOD(CudaArray, ones)
{
    scalar_value_t scalar_value;
    scalar_value.v.f32 = 1.0f;
    scalar_value.is_neg = 0;
    scalar_value.dtype = DTYPE_FLOAT32;

    static_tensor_creator(INTERNAL_FUNCTION_PARAM_PASSTHRU, "ones", scalar_value);
}

ZEND_METHOD(CudaArray, rand)
{
    rand_tensor_creator(INTERNAL_FUNCTION_PARAM_PASSTHRU, 4242424242424242ULL);
}

ZEND_METHOD(CudaArray, transpose)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!this_obj) RETURN_THROWS();
    tensor_t *tensor = this_obj->tensor_handle;

    zval *dims_array = NULL;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_ARRAY_OR_NULL(dims_array)
    ZEND_PARSE_PARAMETERS_END();

    if (dims_array == NULL)
    {
        int default_axis[MAX_DIMS];
        for (int i = 0; i < tensor->ndims; i++)
        {
            default_axis[i] = tensor->ndims - 1 - i;
        }

        tensor_t *result_tensor = cuda_tensor_transpose(tensor, default_axis, tensor->ndims);
        if (result_tensor == NULL)
        {
            CUDA_THROW_RUNTIME("transpose failed");
            RETURN_NULL();
        }
        create_result_object(return_value, result_tensor);
        return;
    }

    int axis[MAX_DIMS] = {0};
    int naxis = tensor->ndims;
    int i = 0;

    for (i = 0; i < naxis; i++)
    {
        axis[i] = i;
    }

    zval *dim;
    int j = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(dims_array), dim)
    {
        if (j >= MAX_DIMS)
        {
            CUDA_THROW_INVALID("too many dimensions in transpose argument (max %d)", MAX_DIMS);
            RETURN_NULL();
        }

        if (Z_TYPE_P(dim) != IS_LONG)
        {
            CUDA_THROW_INVALID("invalid argument for 'transpose' - expected integer dimensions");
            RETURN_NULL();
        }

        axis[j++] = Z_LVAL_P(dim);
    }
    ZEND_HASH_FOREACH_END();

    if (tensor->ndims != j)
    {
        CUDA_THROW_INVALID("transpose expects %d dimensions, got %d", tensor->ndims, j);
        RETURN_NULL();
    }

    bool axis_used[MAX_DIMS] = {false};
    for (i = 0; i < naxis; i++)
    {
        if (axis[i] < 0 || axis[i] >= naxis)
        {
            CUDA_THROW_INVALID("invalid axis %d for tensor with %d dimensions", axis[i], naxis);
            RETURN_NULL();
        }
        if (axis_used[axis[i]])
        {
            CUDA_THROW_INVALID("duplicate axis %d in transpose", axis[i]);
            RETURN_NULL();
        }
        axis_used[axis[i]] = true;
    }

    tensor_t *result_tensor = cuda_tensor_transpose(tensor, axis, naxis);
    if (result_tensor == NULL)
    {
        CUDA_THROW_RUNTIME("transpose operation failed");
        RETURN_NULL();
    }

    create_result_object(return_value, result_tensor);
}

ZEND_METHOD(CudaArray, matmul)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!this_obj) RETURN_THROWS();
    tensor_t *tensor_a = this_obj->tensor_handle;

    zval *other_array = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_OBJECT(other_array)
    ZEND_PARSE_PARAMETERS_END();

    if (!instanceof_function(Z_OBJCE_P(other_array), cuda_array_ce))
    {
        CUDA_THROW_INVALID("Matrix multiplication operand must be a CudaArray");
        RETURN_THROWS();
    }
    cuda_array_obj *other_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(other_array));
    if (!other_obj)
    {
        RETURN_THROWS();
    }

    tensor_t *tensor_b = other_obj->tensor_handle;

    if (tensor_a == NULL || tensor_b == NULL)
    {
        CUDA_THROW_INVALID("Both operands must be valid tensor objects.");
        RETURN_NULL();
    }

    tensor_t *result_tensor = cuda_tensor_matmul(tensor_a, tensor_b);
    if (result_tensor == NULL)
    {
        if (!EG(exception))
            CUDA_THROW_INVALID("Matrix multiplication failed - incompatible dimensions");
        RETURN_THROWS();
    }

    create_result_object(return_value, result_tensor);
}

ZEND_METHOD(CudaArray, sqrt)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Sqrt", OP_SQRT);
}

ZEND_METHOD(CudaArray, floor)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Floor", OP_FLOOR);
}

ZEND_METHOD(CudaArray, ceil)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Ceil", OP_CEIL);
}

ZEND_METHOD(CudaArray, round)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Round", OP_ROUND);
}

ZEND_METHOD(CudaArray, exp)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Exp", OP_EXP);
}

ZEND_METHOD(CudaArray, log)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Log", OP_LOG);
}

ZEND_METHOD(CudaArray, sin)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Sin", OP_SIN);
}

ZEND_METHOD(CudaArray, cos)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Cos", OP_COS);
}

ZEND_METHOD(CudaArray, tan)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Tan", OP_TAN);
}

ZEND_METHOD(CudaArray, abs)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Abs", OP_ABS);
}

ZEND_METHOD(CudaArray, neg)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Neg", OP_NEG);
}

ZEND_METHOD(CudaArray, sum)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Sum Reduction", OP_REDUCE_SUM, 0);
}

ZEND_METHOD(CudaArray, mean)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Mean Reduction", OP_REDUCE_MEAN, 0);
}

ZEND_METHOD(CudaArray, max)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Max Reduction", OP_REDUCE_MAX, 0);
}

ZEND_METHOD(CudaArray, min)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Min Reduction", OP_REDUCE_MIN, 0);
}

ZEND_METHOD(CudaArray, prod)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Product Reduction", OP_REDUCE_PROD, 0);
}

ZEND_METHOD(CudaArray, argMax)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "ArgMax Reduction", OP_ARG_MAX, 1);
}

ZEND_METHOD(CudaArray, argMin)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "ArgMin Reduction", OP_ARG_MIN, 1);
}

ZEND_METHOD(CudaArray, full)
{
    zval *shape_array;
    zval *value;
    zend_string *dtype_str = NULL;

    ZEND_PARSE_PARAMETERS_START(2, 3)
    Z_PARAM_ARRAY(shape_array)
    Z_PARAM_ZVAL(value)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR(dtype_str)
    ZEND_PARSE_PARAMETERS_END();

    int shape[10] = {0};
    int ndims = 0;

    zval *dim;
    int i = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_array), dim)
    {
        if (i < 10 && Z_TYPE_P(dim) == IS_LONG)
        {
            shape[i++] = Z_LVAL_P(dim);
        }
    }
    ZEND_HASH_FOREACH_END();
    ndims = i;

    if (ndims == 0)
    {
        CUDA_THROW_INVALID("Invalid shape: must provide dimensions");
        RETURN_NULL();
    }

    dtype_t dtype = parse_dtype_param(dtype_str);
    scalar_value_t scalar_value;
    SCALAR_FROM_ZVAL(value, scalar_value);

    tensor_t *tensor = cuda_tensor_create_with_value(shape, ndims, scalar_value, dtype);
    if (!tensor)
    {
        CUDA_THROW_RUNTIME("Failed to create full tensor");
        RETURN_NULL();
    }

    create_result_object(return_value, tensor);
}

ZEND_METHOD(CudaArray, astype)
{
    zend_string *dtype_str = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_STR(dtype_str)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    if (!obj->tensor_handle)
    {
        CUDA_THROW_INVALID("Invalid tensor");
        RETURN_NULL();
    }

    if (!dtype_str || ZSTR_LEN(dtype_str) == 0)
    {
        CUDA_THROW_INVALID("Invalid dtype string");
        RETURN_NULL();
    }

    tensor_t *new_tensor = tensor_cast_string(obj->tensor_handle, ZSTR_VAL(dtype_str));
    if (!new_tensor)
    {
        CUDA_THROW_RUNTIME("Failed to cast tensor to %s", ZSTR_VAL(dtype_str));
        RETURN_NULL();
    }

    create_result_object(return_value, new_tensor);
}

ZEND_METHOD(CudaArray, dtype)
{
    cuda_array_obj *obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    if (!obj->tensor_handle)
    {
        CUDA_THROW_INVALID("Invalid tensor");
        RETURN_NULL();
    }

    const char *dtype_name = dtype_to_string(obj->tensor_handle->dtype);
    if (!dtype_name)
    {
        dtype_name = "unknown";
    }

    RETURN_STRING(dtype_name);
}

ZEND_METHOD(CudaArray, reshape)
{
    zval *new_shape_array;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY(new_shape_array)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *this_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!this_obj) RETURN_THROWS();
    int new_shape[10] = {0};
    int new_ndims = 0;

    zval *dim_val;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(new_shape_array), dim_val)
    {
        if (new_ndims >= 10)
        {
            CUDA_THROW_INVALID("Too many dimensions: maximum 10 supported");
            RETURN_NULL();
        }

        if (Z_TYPE_P(dim_val) == IS_LONG)
        {
            new_shape[new_ndims++] = Z_LVAL_P(dim_val);
        }
        else
        {
            CUDA_THROW_INVALID("Shape dimensions must be integers");
            RETURN_NULL();
        }
    }
    ZEND_HASH_FOREACH_END();

    if (new_ndims == 0)
    {
        CUDA_THROW_INVALID("Invalid shape: must provide at least one dimension");
        RETURN_NULL();
    }

    size_t new_total_size = 1;
    for (int i = 0; i < new_ndims; i++)
    {
        if (new_shape[i] < 0)
        {
            CUDA_THROW_INVALID("Invalid dimension size: %d", new_shape[i]);
            RETURN_NULL();
        }
        new_total_size *= new_shape[i];
    }

    size_t current_total_size = 1;
    for (int i = 0; i < this_obj->tensor_handle->ndims; i++)
    {
        current_total_size *= this_obj->tensor_handle->shape[i];
    }

    if (new_total_size != current_total_size)
    {
        CUDA_THROW_INVALID("Cannot reshape tensor of size %zu into size %zu",
                           current_total_size, new_total_size);
        RETURN_NULL();
    }

    tensor_t *reshaped_tensor = cuda_tensor_reshape(this_obj->tensor_handle, new_shape, new_ndims);

    if (reshaped_tensor == NULL)
    {
        CUDA_THROW_RUNTIME("Reshape operation failed");
        RETURN_NULL();
    }

    create_result_object(return_value, reshaped_tensor);
}

ZEND_METHOD(CudaArray, flatten)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!this_obj) RETURN_THROWS();
    size_t total_size = 1;
    for (int i = 0; i < this_obj->tensor_handle->ndims; i++)
    {
        total_size *= this_obj->tensor_handle->shape[i];
    }

    int flat_shape[] = {(int)total_size};

    tensor_t *flat_tensor = cuda_tensor_reshape(this_obj->tensor_handle, flat_shape, 1);

    if (flat_tensor == NULL)
    {
        CUDA_THROW_RUNTIME("Flatten operation failed");
        RETURN_NULL();
    }

    create_result_object(return_value, flat_tensor);
}

ZEND_METHOD(CudaArray, getShape)
{
    cuda_array_obj *obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    array_init_size(return_value, zend_array_count(obj->shape));

    zval *current;
    ZEND_HASH_FOREACH_VAL(obj->shape, current)
    {
        zval copy;
        ZVAL_COPY(&copy, current);
        zend_hash_next_index_insert(Z_ARRVAL_P(return_value), &copy);
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(CudaArray, getStrides)
{
    cuda_array_obj *obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    tensor_t *t = obj->tensor_handle;
    array_init_size(return_value, t->ndims);

    for (int i = 0; i < t->ndims; i++)
    {
        add_next_index_long(return_value, t->strides[i]);
    }
}

ZEND_METHOD(CudaArray, getNdims)
{
    cuda_array_obj *obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    tensor_t *t = obj->tensor_handle;
    RETURN_LONG(t->ndims);
}

ZEND_METHOD(CudaArray, getSize)
{
    cuda_array_obj *obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    tensor_t *t = obj->tensor_handle;
    if (!t->total_size)
    {
        RETURN_LONG(0);
    }

    RETURN_LONG((int)t->total_size);
}

ZEND_METHOD(CudaArray, concat)
{
    zval *this_ptr = ZEND_THIS;
    zval *input_tensors_array;
    zend_long axis_long = 0;

    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ARRAY(input_tensors_array)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(axis_long)
    ZEND_PARSE_PARAMETERS_END();

    int axis = (int)axis_long;

    zval full_tensors_list;
    array_init(&full_tensors_list);

    zval temp_zval;
    ZVAL_COPY(&temp_zval, this_ptr);
    zend_hash_next_index_insert(Z_ARRVAL(full_tensors_list), &temp_zval);

    HashTable *input_ht = Z_ARRVAL_P(input_tensors_array);
    zval *pzval;

    ZEND_HASH_FOREACH_VAL(input_ht, pzval)
    {
        zval temp_zval_arg;
        ZVAL_COPY(&temp_zval_arg, pzval);
        zend_hash_next_index_insert(Z_ARRVAL(full_tensors_list), &temp_zval_arg);
    }
    ZEND_HASH_FOREACH_END();

    tensor_t *new_tensor = cuda_tensor_concat(&full_tensors_list, axis);
    zend_array_destroy(Z_ARRVAL(full_tensors_list));

    if (!new_tensor)
    {
        RETURN_THROWS();
    }

    create_result_object(return_value, new_tensor);
}

ZEND_METHOD(CudaArray, toArray)
{
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    tensor_to_php_array(return_value, obj->tensor_handle);
}

ZEND_METHOD(CudaArray, toBuffer)
{
    ZEND_PARSE_PARAMETERS_NONE();
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    zend_string *bytes = tensor_to_buffer(obj->tensor_handle);
    if (!bytes) RETURN_THROWS();
    RETURN_STR(bytes);
}

ZEND_METHOD(CudaArray, toHost)
{
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    tensor_t *host_tensor = tensor_copy_to_host(obj->tensor_handle);
    if (!host_tensor)
    {
        RETURN_THROWS();
    }

    zend_object *host_obj = contiguous_array_from_tensor(host_tensor);
    if (!host_obj)
    {
        efree(host_tensor->data);
        if (host_tensor->shape)
            efree(host_tensor->shape);
        if (host_tensor->strides)
            efree(host_tensor->strides);
        efree(host_tensor);
        CUDA_THROW_RUNTIME("Failed to create ContiguousArray object");
        RETURN_NULL();
    }

    ZVAL_OBJ(return_value, host_obj);
}

typedef struct
{
    int index;
    int has_start, has_stop;
    zend_long start, stop, step;
} ca_slice_selector;

static void ca_slice_trim(const char **text, size_t *length)
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

static int ca_slice_integer(const char *text, size_t length, zend_long *value)
{
    ca_slice_trim(&text, &length);
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

static int ca_slice_text(const char *text, size_t length, ca_slice_selector *selector)
{
    memset(selector, 0, sizeof(*selector));
    selector->step = 1;
    ca_slice_trim(&text, &length);
    if (!memchr(text, ':', length))
    {
        selector->index = selector->has_start = 1;
        return ca_slice_integer(text, length, &selector->start);
    }
    size_t start = 0;
    int field = 0;
    for (size_t i = 0; i <= length; i++)
    {
        if (i != length && text[i] != ':') continue;
        if (field >= 3) return 0;
        const char *part = text + start;
        size_t part_length = i - start;
        ca_slice_trim(&part, &part_length);
        if (part_length)
        {
            zend_long number;
            if (!ca_slice_integer(part, part_length, &number)) return 0;
            if (field == 0) { selector->start = number; selector->has_start = 1; }
            else if (field == 1) { selector->stop = number; selector->has_stop = 1; }
            else selector->step = number;
        }
        field++;
        start = i + 1;
    }
    return 1;
}

static int ca_slice_value(zval *value, ca_slice_selector *selector)
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
        return ca_slice_text(Z_STRVAL_P(value), Z_STRLEN_P(value), selector);
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

static int ca_slice_bound(zend_long value, int size)
{
    if (value < 0) value += size;
    if (value < 0) return 0;
    if (value > size) return size;
    return (int)value;
}

ZEND_METHOD(CudaArray, slice)
{
    zval *selectors;
    int count;
    ZEND_PARSE_PARAMETERS_START(0, -1)
        Z_PARAM_VARIADIC('*', selectors, count)
    ZEND_PARSE_PARAMETERS_END();
    cuda_array_obj *object = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!object) RETURN_THROWS();
    tensor_t *source = object->tensor_handle;
    if (!fusion_active() && !fusion_materialize(source)) RETURN_THROWS();
    ca_slice_selector parsed[MAX_DIMS];
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
            if (parsed_count >= source->ndims ||
                !ca_slice_text(text + start, i - start, &parsed[parsed_count]))
            {
                CUDA_THROW_INVALID("Invalid slice expression at axis %d", parsed_count);
                RETURN_THROWS();
            }
            parsed_count++;
            start = i + 1;
        }
    }
    else
    {
        if (count > source->ndims)
        {
            CUDA_THROW_INVALID("Slice has %d selectors for %d axes", count, source->ndims);
            RETURN_THROWS();
        }
        for (; parsed_count < count; parsed_count++)
            if (!ca_slice_value(&selectors[parsed_count], &parsed[parsed_count]))
            {
                CUDA_THROW_INVALID("Invalid slice selector at axis %d", parsed_count);
                RETURN_THROWS();
            }
    }
    int shape[MAX_DIMS], axes[MAX_DIMS], starts[MAX_DIMS], steps[MAX_DIMS], ndims = 0;
    size_t strides[MAX_DIMS], offset = 0, total = 1;
    for (int d = 0; d < source->ndims; d++)
    {
        ca_slice_selector selector = { .step = 1 };
        if (d < parsed_count) selector = parsed[d];
        int size = source->shape[d];
        if (selector.step <= 0 || selector.step > INT_MAX)
        {
            CUDA_THROW_INVALID("Slice step must be between 1 and INT_MAX; reverse slices are not supported");
            RETURN_THROWS();
        }
        steps[d] = (int)selector.step;
        if (selector.index)
        {
            zend_long index = selector.start;
            if (index < 0) index += size;
            if (index < 0 || index >= size)
            {
                CUDA_THROW_INVALID("Slice index out of bounds at axis %d (size %d)", d, size);
                RETURN_THROWS();
            }
            starts[d] = (int)index;
        }
        else
        {
            starts[d] = selector.has_start ? ca_slice_bound(selector.start, size) : 0;
            int stop = selector.has_stop ? ca_slice_bound(selector.stop, size) : size;
            int length = stop > starts[d] ? 1 + (stop - starts[d] - 1) / steps[d] : 0;
            if (source->strides[d] > SIZE_MAX / (size_t)steps[d])
            {
                CUDA_THROW_INVALID("Slice stride overflow at axis %d", d);
                RETURN_THROWS();
            }
            axes[ndims] = d;
            shape[ndims] = length;
            strides[ndims++] = source->strides[d] * (size_t)steps[d];
            total *= length;
        }
        if (starts[d] && source->strides[d] > (SIZE_MAX - offset) / (size_t)starts[d])
        {
            CUDA_THROW_INVALID("Slice offset overflow");
            RETURN_THROWS();
        }
        offset += (size_t)starts[d] * source->strides[d];
    }
    tensor_t *view;
    if (fusion_active())
        view = fusion_slice(source, shape, strides, ndims, axes, starts, steps);
    else
    {
        view = cuda_tensor_create_view(source, shape, strides, ndims, total ? offset : 0, total);
        /* Storage pointers already include the slice displacement. */
        if (view) view->offset = 0;
    }
    if (!view) RETURN_THROWS();
    create_result_object(return_value, view);
}

ZEND_METHOD(CudaArray, __invoke)
{
    zval *slices;
    int slice_count;

    ZEND_PARSE_PARAMETERS_START(0, -1)
    Z_PARAM_VARIADIC('*', slices, slice_count)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    if (!this_obj) RETURN_THROWS();

    int ndim = this_obj->tensor_handle->ndims;
    slice_info_t *slice_info = (slice_info_t *)emalloc(ndim * sizeof(slice_info_t));

    for (int i = 0; i < ndim; i++)
    {
        if (i < slice_count)
        {
            if (!parse_slice_parameter(&slices[i], &slice_info[i]))
            {
                efree(slice_info);
                CUDA_THROW_INVALID("Invalid slice parameter at dimension %d", i + 1);
                RETURN_NULL();
            }
        }
        else
        {
            memset(&slice_info[i], 0, sizeof(slice_info_t));
            slice_info[i].type = SLICE_ALL;
        }
    }

    tensor_t *view_tensor = cuda_tensor_create_sliced_view(this_obj->tensor_handle, slice_info, ndim);
    efree(slice_info);

    if (!view_tensor)
    {
        CUDA_THROW_RUNTIME("Failed to create tensor view");
        RETURN_NULL();
    }

    create_result_object(return_value, view_tensor);
}

ZEND_METHOD(CudaArray, __debugInfo)
{
    cuda_array_obj *obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!obj) RETURN_THROWS();
    tensor_t *tensor = obj->tensor_handle;
    array_init(return_value);

    if (!tensor || tensor->ndims <= 0)
    {
        add_assoc_string(return_value, "Error", "CudaArray handle is NULL or has zero dimensions");
        return;
    }

    zval shape_array;
    array_init(&shape_array);

    for (int i = 0; i < tensor->ndims; i++)
    {
        add_next_index_long(&shape_array, (zend_long)tensor->shape[i]);
    }
    add_assoc_zval(return_value, "shape", &shape_array);

    const char *dtype_str = dtype_to_string(tensor->dtype);
    size_t element_size = dtype_size(tensor->dtype);

    add_assoc_string(return_value, "dtype", (char *)dtype_str);
    add_assoc_long(return_value, "elements", (zend_long)tensor->total_size);
}

static void sync_php_object_shape(cuda_array_obj *obj, tensor_t *tensor)
{
    if (obj->shape)
    {
        zend_array_destroy(obj->shape);
    }

    obj->shape = zend_new_array(tensor->ndims);

    for (int i = 0; i < tensor->ndims; i++)
    {
        zval dim;
        ZVAL_LONG(&dim, tensor->shape[i]);
        zend_hash_index_update(obj->shape, i, &dim);
    }
}

int cuda_array_init(size_t mb)
{
    if (!tensor_mem_init(mb))
    {
        php_error_docref(NULL, E_WARNING,
                         "Failed to initialize CUDA memory pool with %ld MB.",
                         mb);
        return 0;
    }

    memcpy(&cuda_array_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));

    cuda_array_handlers.offset = XtOffsetOf(cuda_array_obj, obj);
    cuda_array_handlers.free_obj = cuda_array_free_object;
    cuda_array_handlers.clone_obj = cuda_array_clone_obj;
    cuda_array_handlers.do_operation = cuda_array_do_operation;
    cuda_array_handlers.read_dimension = cuda_array_read_dimension;
    cuda_array_handlers.write_dimension = cuda_array_write_dimension;

    zend_class_entry *ce = register_cuda_array_class();
    ce->create_object = cuda_array_create_object;

    return 1;
}

void cuda_array_shutdown()
{
    tensor_mem_destroy();
}

static int parse_slice_parameter(zval *param, slice_info_t *slice)
{
    memset(slice, 0, sizeof(slice_info_t));

    if (Z_TYPE_P(param) == IS_NULL)
    {
        slice->type = SLICE_ALL;
        return 1;
    }

    if (Z_TYPE_P(param) == IS_LONG)
    {
        slice->type = SLICE_INDEX;
        slice->data.index = Z_LVAL_P(param);
        return 1;
    }

    if (Z_TYPE_P(param) == IS_ARRAY)
    {
        HashTable *ht = Z_ARRVAL_P(param);
        if (zend_array_count(ht) == 2)
        {
            zval *start_val = zend_hash_index_find(ht, 0);
            zval *end_val = zend_hash_index_find(ht, 1);

            if (start_val && end_val &&
                Z_TYPE_P(start_val) == IS_LONG &&
                Z_TYPE_P(end_val) == IS_LONG)
            {

                slice->type = SLICE_RANGE;
                slice->data.range.start = Z_LVAL_P(start_val);
                slice->data.range.end = Z_LVAL_P(end_val);
                return 1;
            }
        }
    }

    return 0;
}

static cuda_array_obj *php_cuda_array_fetch_object(zend_object *obj)
{
    return (cuda_array_obj *)((char *)obj - XtOffsetOf(cuda_array_obj, obj));
}

cuda_array_obj *php_cuda_array_fetch_valid_object(zend_object *obj)
{
    cuda_array_obj *object = php_cuda_array_fetch_deferred_object(obj);
    if (object && !fusion_materialize(object->tensor_handle)) return NULL;
    return object;
}

static cuda_array_obj *php_cuda_array_fetch_deferred_object(zend_object *obj)
{
    cuda_array_obj *this_obj = (cuda_array_obj *)((char *)obj - XtOffsetOf(cuda_array_obj, obj));

    if (!this_obj || this_obj->tensor_handle == NULL)
    {
        CUDA_THROW_RUNTIME("Attempting to access uninitialized tensor");
        return NULL;
    }
    if (this_obj->tensor_handle->fusion_failed)
    {
        CUDA_THROW_RUNTIME("Tensor belongs to an aborted Fusion capture");
        return NULL;
    }

    if (this_obj->shape == NULL)
    {
        CUDA_THROW_RUNTIME("Attempting to access tensor with no shape");
        return NULL;
    }

    if (this_obj->tensor_handle->is_view && !this_obj->tensor_handle->base_tensor)
    {
        CUDA_THROW_RUNTIME("Attempting to access a view with no base tensor");
        return NULL;
    }

    return this_obj;
}

static zend_object *cuda_array_create_object(zend_class_entry *class_type)
{
    cuda_array_obj *obj = (cuda_array_obj *)ecalloc(1, sizeof(cuda_array_obj));

    zend_object_std_init(&obj->obj, class_type);
    object_properties_init(&obj->obj, class_type);

    if (cuda_array_handlers.do_operation == NULL) {
        CUDA_THROW_RUNTIME("CudaArray operation handler is not initialized");
    }
    
    obj->obj.handlers = &cuda_array_handlers;
    obj->tensor_handle = NULL;
    obj->shape = NULL;

    return &obj->obj;
}

static zend_object *cuda_array_clone_obj(zend_object *old_object)
{
    cuda_array_obj *old_ca = php_cuda_array_fetch_valid_object(old_object);
    if (!old_ca) return NULL;
    zend_object *new_object = cuda_array_create_object(cuda_array_ce);

    cuda_array_obj *new_ca = php_cuda_array_fetch_object(new_object);
    if (!new_ca)
    {
        CUDA_THROW_RUNTIME("Internal error during object cloning: cannot fetch object data.");
        zend_object_release(new_object);
        return NULL;
    }

    zend_objects_clone_members(new_object, old_object);
    tensor_t *new_tensor = cuda_tensor_clone(old_ca->tensor_handle);
    if (new_tensor == NULL)
    {
        CUDA_THROW_RUNTIME("Failed to clone CUDA tensor data during object cloning.");
        zend_object_std_dtor(new_object);
        zend_object_release(new_object);
        return NULL;
    }

    new_ca->tensor_handle = new_tensor;
    sync_php_object_shape(new_ca, new_ca->tensor_handle);

    return new_object;
}

static void cuda_array_free_object(zend_object *object)
{
    cuda_array_obj *obj = php_cuda_array_fetch_object(object);

    if (obj->tensor_handle != NULL)
    {
        cuda_tensor_destroy(obj->tensor_handle);
        obj->tensor_handle = NULL;
    }

    if (obj->shape != NULL)
    {
        zend_array_destroy(obj->shape);
        obj->shape = NULL;
    }

    zend_object_std_dtor(&obj->obj);
}

void create_result_object(zval *return_value, tensor_t *result_tensor)
{
    object_init_ex(return_value, cuda_array_ce);
    cuda_array_obj *result_obj = php_cuda_array_fetch_object(Z_OBJ_P(return_value));

    result_obj->tensor_handle = result_tensor;

    sync_php_object_shape(result_obj, result_tensor);
}

static void unary_operation_handler(INTERNAL_FUNCTION_PARAMETERS,
                                    const char *operation_name,
                                    operation_type_t operation_type)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!this_obj) RETURN_THROWS();

    if (this_obj->tensor_handle == NULL)
    {
        CUDA_THROW_RUNTIME("CudaArray not initialized");
        RETURN_NULL();
    }

    tensor_t *result_tensor = cuda_unary_op(this_obj->tensor_handle, operation_type);

    if (result_tensor == NULL)
    {
        CUDA_THROW_RUNTIME("%s failed", operation_name);
        RETURN_NULL();
    }

    create_result_object(return_value, result_tensor);
}

static void reduction_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, operation_type_t operation_type, int return_arg)
{
    zend_long axis_zv = REDUCE_GLOBAL_FLAG;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(axis_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *this_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!this_obj) RETURN_THROWS();

    if (this_obj->tensor_handle == NULL)
    {
        CUDA_THROW_RUNTIME("Input tensor not initialized.");
        RETURN_NULL();
    }

    tensor_t *input_tensor = this_obj->tensor_handle;
    int axis = (int)axis_zv;

    if (fusion_active() && axis == REDUCE_GLOBAL_FLAG)
    {
        tensor_t *result_tensor = fusion_reduce(input_tensor, -1, operation_type, return_arg);
        if (!result_tensor) RETURN_THROWS();
        create_result_object(return_value, result_tensor);
        return;
    }

    if (axis == REDUCE_GLOBAL_FLAG)
    {
        size_t total_size = 1;
        for (int i = 0; i < this_obj->tensor_handle->ndims; i++)
        {
            total_size *= this_obj->tensor_handle->shape[i];
        }

        int flat_shape[] = {(int)total_size};
        input_tensor = cuda_tensor_reshape(this_obj->tensor_handle, flat_shape, 1);
        if (!input_tensor)
        {
            CUDA_THROW_RUNTIME("Failed to flatten tensor for reduction");
            RETURN_THROWS();
        }
        axis = 0;
    }

    axis = axis >= 0 ? axis : input_tensor->ndims + axis;

    if ((axis < 0 || axis >= input_tensor->ndims) && axis != REDUCE_GLOBAL_FLAG)
    {
        CUDA_THROW_INVALID("Axis %d out of bounds for tensor with %d dimensions.", axis, input_tensor->ndims);
        if (input_tensor != this_obj->tensor_handle) cuda_tensor_destroy(input_tensor);
        RETURN_NULL();
    }

    tensor_t *result_tensor = (return_arg == 1)
                                  ? cuda_tensor_reduce_arg(input_tensor, axis, operation_type)
                                  : cuda_tensor_reduce(input_tensor, axis, operation_type);
    if (input_tensor != this_obj->tensor_handle) cuda_tensor_destroy(input_tensor);

    if (result_tensor == NULL)
    {
        if (!EG(exception)) CUDA_THROW_RUNTIME("%s failed", operation_name);
        RETURN_THROWS();
    }

    create_result_object(return_value, result_tensor);
}

static void binary_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, operation_type_t operation_type)
{
    zval *other_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *this_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(ZEND_THIS));
    if (!this_obj) RETURN_THROWS();
    tensor_t *result_tensor = NULL;

    if (Z_TYPE_P(other_zv) == IS_OBJECT && instanceof_function(Z_OBJCE_P(other_zv), cuda_array_ce))
    {
        cuda_array_obj *other_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(other_zv));
        if (!other_obj) RETURN_THROWS();

        if (other_obj->tensor_handle == NULL)
        {
            CUDA_THROW_RUNTIME("Other tensor not initialized");
            RETURN_NULL();
        }

        result_tensor = cuda_tensor_op(this_obj->tensor_handle, other_obj->tensor_handle, operation_type);
    }
    else
    {
        scalar_value_t scalar_value;
        SCALAR_FROM_ZVAL(other_zv, scalar_value);
        if (scalar_value.dtype == DTYPE_UNKNOWN)
        {
            CUDA_THROW_INVALID("Invalid dtype: '%s' for scalar operation", dtype_to_string(scalar_value.dtype));
            RETURN_NULL();
        }

        result_tensor = cuda_scalar_op(this_obj->tensor_handle, scalar_value, operation_type);
    }

    if (result_tensor == NULL)
    {
        CUDA_THROW_RUNTIME("%s failed", operation_name);
        RETURN_NULL();
    }

    create_result_object(return_value, result_tensor);
}

static zend_result cuda_array_do_operation(zend_uchar opcode, zval *result, zval *op1, zval *op2)
{
    if (fusion_active() && (result == op1 || result == op2 ||
        opcode == ZEND_PRE_INC || opcode == ZEND_POST_INC ||
        opcode == ZEND_PRE_DEC || opcode == ZEND_POST_DEC))
    {
        fusion_check_mutation();
        return FAILURE;
    }
    zend_bool define_value = 0;
    float op_value = 0.0f;
    const char *operation_name = NULL;
    int operation_type = 0;

    switch (opcode)
    {
    case ZEND_ADD:
        operation_name = "Addition (+)";
        operation_type = OP_ADD;
        break;
    case ZEND_SUB:
        operation_name = "Subtraction (-)";
        operation_type = OP_SUB;
        break;
    case ZEND_MUL:
        operation_name = "Multiplication (*)";
        operation_type = OP_MUL;
        break;
    case ZEND_DIV:
        operation_name = "Division (/)";
        operation_type = OP_DIV;
        break;
    case ZEND_POW:
        operation_name = "Power (**)";
        operation_type = OP_POW;
        break;
    case ZEND_PRE_INC:
    case ZEND_POST_INC:
        operation_name = "Increment (++)";
        operation_type = OP_ADD;
        break;
    case ZEND_PRE_DEC:
    case ZEND_POST_DEC:
        operation_name = "Decrement (--)";
        operation_type = OP_SUB;
        break;
    default:
        return FAILURE;
    }

    tensor_t *result_tensor = NULL;

    if (Z_TYPE_P(op1) == IS_OBJECT && instanceof_function(Z_OBJCE_P(op1), cuda_array_ce))
    {
        cuda_array_obj *this_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(op1));
        if (!this_obj || this_obj->tensor_handle == NULL)
        {
            return FAILURE;
        }

        if (Z_TYPE_P(op2) == IS_OBJECT && instanceof_function(Z_OBJCE_P(op2), cuda_array_ce))
        {
            cuda_array_obj *other_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(op2));
            if (!other_obj) return FAILURE;
            result_tensor = cuda_tensor_op(this_obj->tensor_handle, other_obj->tensor_handle, operation_type);
        }
        else
        {
            scalar_value_t scalar_value;
            SCALAR_FROM_ZVAL(op2, scalar_value);
            if (scalar_value.dtype == DTYPE_UNKNOWN)
            {
                CUDA_THROW_INVALID("Invalid dtype: '%s' for scalar operation", dtype_to_string(scalar_value.dtype));
                return FAILURE;
            }

            result_tensor = cuda_scalar_op(this_obj->tensor_handle, scalar_value, operation_type);
        }
    }
    else if (Z_TYPE_P(op2) == IS_OBJECT && Z_OBJCE_P(op2) == cuda_array_ce)
    {
        cuda_array_obj *this_obj = php_cuda_array_fetch_deferred_object(Z_OBJ_P(op2));
        if (!this_obj || this_obj->tensor_handle == NULL)
        {
            return FAILURE;
        }

        scalar_value_t scalar_value;

        SCALAR_FROM_ZVAL(op1, scalar_value);
        if (scalar_value.dtype == DTYPE_UNKNOWN)
        {
            CUDA_THROW_INVALID("Invalid dtype: '%s' for scalar operation", dtype_to_string(scalar_value.dtype));
            return FAILURE;
        }

        result_tensor = cuda_inv_scalar_op(this_obj->tensor_handle, scalar_value, operation_type);
    }
    else
    {
        return FAILURE;
    }

    if (result_tensor == NULL)
    {
        CUDA_THROW_RUNTIME("CudaArray operation %s failed (incompatible shapes or internal error)", operation_name);
        return FAILURE;
    }

    create_result_object(result, result_tensor);
    return SUCCESS;
}

static zval *cuda_array_read_dimension(zend_object *object, zval *offset, int type, zval *rv)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(object);
    if (!this_obj) return &EG(uninitialized_zval);
    if (!offset)
    {
        CUDA_THROW_INVALID("An index is required to read a CudaArray");
        return &EG(uninitialized_zval);
    }
    tensor_t *base_tensor = this_obj->tensor_handle;
    int ndim = base_tensor->ndims;

    if (ndim == 0)
    {
        CUDA_THROW_INVALID("Cannot slice a zero-dimensional tensor (scalar).");
        return &EG(uninitialized_zval);
    }

    slice_info_t slice_info_array[MAX_DIMS];

    if (!parse_slice_parameter(offset, &slice_info_array[0]))
    {
        CUDA_THROW_INVALID("Invalid dimension access: key must be NULL, integer, or [start, end] array.");
        return &EG(uninitialized_zval);
    }

    if (base_tensor->ndims == 1 && slice_info_array[0].type == SLICE_INDEX)
    {
        float result_val;
        if (cuda_tensor_get_scalar_value(base_tensor, &result_val, slice_info_array[0].data.index) != SUCCESS)
        {
            if (!EG(exception)) CUDA_THROW_RUNTIME("Failed to extract scalar value from GPU.");
            return &EG(uninitialized_zval);
        }

        ZVAL_DOUBLE(rv, (double)result_val);
        return rv;
    }

    for (int i = 1; i < ndim; i++)
    {
        slice_info_array[i].type = SLICE_ALL;
    }

    tensor_t *view_tensor = cuda_tensor_create_dim_view(
        base_tensor,
        slice_info_array,
        ndim);

    if (view_tensor == NULL)
    {
        if (!EG(exception)) CUDA_THROW_RUNTIME("Failed to create tensor view during array access.");
        return &EG(uninitialized_zval);
    }

    create_result_object(rv, view_tensor);
    return rv;
}

static void cuda_array_write_dimension(zend_object *object, zval *offset, zval *value)
{
    if (!fusion_check_tensor_mutation(((cuda_array_obj *)((char *)object - XtOffsetOf(cuda_array_obj, obj)))->tensor_handle))
        return;
    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(object);
    if (!this_obj) return;
    if (offset == NULL)
    {
        CUDA_THROW_INVALID("It is not permitted to append (operator []) to a CudaArray.");
        return;
    }

    slice_info_t slice_info;
    if (!parse_slice_parameter(offset, &slice_info))
    {
        CUDA_THROW_INVALID("Invalid tensor index parameter.");
        return;
    }

    tensor_t *base_tensor = this_obj->tensor_handle;

    if (slice_info.type == SLICE_INDEX)
    {
        int index = slice_info.data.index;
        size_t element_offset = (size_t)index * base_tensor->strides[0];

        if (base_tensor->ndims == 0 || index < 0 || index >= base_tensor->shape[0])
        {
            CUDA_THROW_INVALID("Index out of bounds for write operation (Dim 0).");
            return;
        }

        if (Z_TYPE_P(value) == IS_DOUBLE || Z_TYPE_P(value) == IS_LONG)
        {
            scalar_value_t scalar_value;
            SCALAR_FROM_ZVAL(value, scalar_value);
            if (cuda_tensor_set_scalar(base_tensor, element_offset, scalar_value) != SUCCESS)
            {
                if (!EG(exception)) CUDA_THROW_RUNTIME("Failed to write scalar value to GPU memory.");
            }
        }
        else if (Z_TYPE_P(value) == IS_OBJECT && instanceof_function(Z_OBJCE_P(value), cuda_array_ce))
        {
            cuda_array_obj *src_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(value));
            if (!src_obj) return;
            tensor_t *src_tensor = src_obj->tensor_handle;
            int dest_ndims = base_tensor->ndims - 1;

            if (src_tensor->ndims != dest_ndims)
            {
                CUDA_THROW_INVALID("CudaArray assignment requires a source with %d dimensions, but %d given.", dest_ndims, src_tensor->ndims);
                return;
            }
            for (int i = 0; i < dest_ndims; i++)
            {
                if (src_tensor->shape[i] != base_tensor->shape[i + 1])
                {
                    CUDA_THROW_INVALID("Shape mismatch for tensor assignment at dimension %d.", i + 1);
                    return;
                }
            }

            if (cuda_tensor_set_tensor(base_tensor, element_offset, src_tensor) != SUCCESS)
            {
                if (!EG(exception)) CUDA_THROW_RUNTIME("Failed GPU memory copy during tensor assignment");
            }
        }
        else
        {
            CUDA_THROW_INVALID("Only scalar or CudaArray assignment is supported for single index.");
        }
    }
    else if (slice_info.type == SLICE_RANGE)
    {
        CUDA_THROW_RUNTIME("SLICE_RANGE not implemented yet.");
    }
    else
    {
        CUDA_THROW_INVALID("Only single index, array index list, or range slice assignment is supported in this context for now.");
    }
}

static void rand_tensor_creator(INTERNAL_FUNCTION_PARAMETERS, unsigned long long seed)
{
    zval *shape_array;
    zend_string *dtype_str = NULL;

    zval *z_min = NULL;
    zval *z_max = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 4)
    Z_PARAM_ARRAY(shape_array)
    Z_PARAM_OPTIONAL
    Z_PARAM_ZVAL_OR_NULL(z_min)
    Z_PARAM_ZVAL_OR_NULL(z_max)
    Z_PARAM_STR_OR_NULL(dtype_str)
    ZEND_PARSE_PARAMETERS_END();

    int shape[10] = {0};
    int ndims = 0;

    zval *dim;
    int i = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_array), dim)
    {
        if (i < 10 && Z_TYPE_P(dim) == IS_LONG)
        {
            shape[i++] = Z_LVAL_P(dim);
        }
    }
    ZEND_HASH_FOREACH_END();
    ndims = i;

    if (ndims == 0)
    {
        CUDA_THROW_INVALID("Invalid shape: must provide dimensions");
        RETURN_NULL();
    }

    scalar_value_t min_v = {.v.f32 = 0.00f, .dtype = DTYPE_FLOAT32, .is_neg = 0};
    scalar_value_t max_v = {.v.f32 = 100.00f, .dtype = DTYPE_FLOAT32, .is_neg = 0};

    if (z_min != NULL)
    {
        SCALAR_FROM_ZVAL(z_min, min_v);
    }

    if (z_max != NULL)
    {
        SCALAR_FROM_ZVAL(z_max, max_v);
    }

    dtype_t dtype = parse_dtype_param(dtype_str);

    tensor_t *tensor = cuda_tensor_create_rand(shape, ndims, min_v, max_v, dtype, seed);

    if (!tensor)
    {
        CUDA_THROW_RUNTIME("Failed to create random tensor");
        RETURN_NULL();
    }

    create_result_object(return_value, tensor);
}

static void static_tensor_creator(INTERNAL_FUNCTION_PARAMETERS, const char *method_name, scalar_value_t scalar_value)
{
    if (scalar_value.dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Invalid dtype: '%s' for scalar operation", dtype_to_string(scalar_value.dtype));
        RETURN_NULL();
    }

    zval *shape_array;
    zend_string *dtype_str = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ARRAY(shape_array)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR(dtype_str)
    ZEND_PARSE_PARAMETERS_END();

    int shape[10] = {0};
    int ndims = 0;

    zval *dim;
    int i = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_array), dim)
    {
        if (i < 10 && Z_TYPE_P(dim) == IS_LONG)
        {
            shape[i++] = Z_LVAL_P(dim);
        }
    }
    ZEND_HASH_FOREACH_END();
    ndims = i;

    if (ndims == 0)
    {
        CUDA_THROW_INVALID("Invalid shape: must provide dimensions");
        RETURN_NULL();
    }
    
    dtype_t dtype = parse_dtype_param(dtype_str);
    tensor_t *tensor = cuda_tensor_create_with_value(shape, ndims, scalar_value, dtype);

    if (!tensor)
    {
        CUDA_THROW_RUNTIME("Failed to create %s tensor", method_name);
        RETURN_NULL();
    }

    create_result_object(return_value, tensor);
}

static tensor_t *cuda_tensor_concat(zval *tensors_array, int axis)
{
    HashTable *ht = Z_ARRVAL_P(tensors_array);
    zval *pzval;
    int i = 0;

    int list_count = zend_hash_num_elements(ht);

    if (list_count == 0)
    {
        CUDA_THROW_INVALID("Concat requires at least one tensor.");
        return NULL;
    }

    if (list_count > MAX_CONCAT_TENSORS)
    {
        CUDA_THROW_INVALID("Too many tensors to concatenate. Maximum is %d.", MAX_CONCAT_TENSORS);
        return NULL;
    }

    tensor_t **tensor_list = (tensor_t **)emalloc(sizeof(tensor_t *) * list_count);
    size_t total_length_on_axis = 0;
    int first_ndims = -1;

    ZEND_HASH_FOREACH_VAL(ht, pzval)
    {
        if (Z_TYPE_P(pzval) != IS_OBJECT || !instanceof_function(Z_OBJCE_P(pzval), cuda_array_ce))
        {
            CUDA_THROW_INVALID("All elements must be CudaArray objects.");
            efree(tensor_list);
            return NULL;
        }

        cuda_array_obj *other_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(pzval));
        if (!other_obj)
        {
            efree(tensor_list);
            return NULL;
        }
        tensor_t *current_tensor = other_obj->tensor_handle;

        tensor_list[i] = current_tensor;

        if (i == 0)
        {
            first_ndims = current_tensor->ndims;
            if (axis < 0 || axis >= first_ndims)
            {
                CUDA_THROW_INVALID("Axis %d is out of bounds for the first tensor (dims: %d).", axis, first_ndims);
                efree(tensor_list);
                return NULL;
            }
        }
        else
        {
            if (current_tensor->ndims != first_ndims)
            {
                CUDA_THROW_INVALID("All tensors must have the same number of dimensions (%d != %d).",
                                 current_tensor->ndims, first_ndims);
                efree(tensor_list);
                return NULL;
            }
            for (int d = 0; d < first_ndims; d++)
            {
                if (d != axis && current_tensor->shape[d] != tensor_list[0]->shape[d])
                {
                    CUDA_THROW_INVALID("Shapes must match along non-concatenated axis %d.", d);
                    efree(tensor_list);
                    return NULL;
                }
            }
        }

        total_length_on_axis += current_tensor->shape[axis];
        i++;
    }
    ZEND_HASH_FOREACH_END();

    int *new_shape = (int *)emalloc(sizeof(int) * first_ndims);
    memcpy(new_shape, tensor_list[0]->shape, sizeof(int) * first_ndims);
    new_shape[axis] = (int)total_length_on_axis;
    tensor_t *new_tensor = cuda_tensor_create_empty(new_shape, first_ndims);

    efree(new_shape);

    if (!new_tensor)
    {
        CUDA_THROW_OOM("Failed to allocate memory for concatenated tensor.");
        efree(tensor_list);
        return NULL;
    }

    size_t outer_dims = 1;
    for (int d = 0; d < axis; d++)
    {
        outer_dims *= new_tensor->shape[d];
    }

    size_t inner_dims = 1;
    for (int d = axis + 1; d < new_tensor->ndims; d++)
    {
        inner_dims *= new_tensor->shape[d];
    }

    int result = launch_concat_kernel_host(
        tensor_list,
        list_count,
        new_tensor,
        axis,
        outer_dims,
        inner_dims,
        (int)total_length_on_axis);

    efree(tensor_list);

    if (result != SUCCESS)
    {
        cuda_tensor_destroy(new_tensor);
        CUDA_THROW_RUNTIME("CUDA concat kernel failed.");
        return NULL;
    }

    return new_tensor;
}