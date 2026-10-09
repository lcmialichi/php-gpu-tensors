#include "optimizer.h"

#include "cuda_array_ce.h"
#include "cuda_exceptions.h"

#include <math.h>
#include <string.h>

static zend_class_entry *optimizer_ce;

ZEND_BEGIN_ARG_INFO_EX(optimizer_private_args, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(optimizer_adamw_args, 0, 0, Cuda\\Optimizer, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, learningRate, IS_DOUBLE, 0, "0.001")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, beta1, IS_DOUBLE, 0, "0.9")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, beta2, IS_DOUBLE, 0, "0.999")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, epsilon, IS_DOUBLE, 0, "1.0e-8")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, weightDecay, IS_DOUBLE, 0, "0.01")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(optimizer_sgd_args, 0, 0, Cuda\\Optimizer, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, learningRate, IS_DOUBLE, 0, "0.01")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, momentum, IS_DOUBLE, 0, "0.0")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, weightDecay, IS_DOUBLE, 0, "0.0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(optimizer_parameters_args, 0, 1, IS_ARRAY, 0)
ZEND_ARG_TYPE_INFO(0, parameters, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(optimizer_zero_grad_args, 0, 1, IS_VOID, 0)
ZEND_ARG_TYPE_INFO(0, parameters, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(optimizer_step_args, 0, 3, IS_ARRAY, 0)
ZEND_ARG_TYPE_INFO(0, parameters, IS_ARRAY, 0)
ZEND_ARG_TYPE_INFO(0, gradients, IS_ARRAY, 0)
ZEND_ARG_TYPE_INFO(0, state, IS_ARRAY, 0)
ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, learningRate, Cuda\\CudaArray, 1, "null")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, weightDecayMask, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()

static zend_result optimizer_invoke(zval *object, const char *method, zval *arguments,
                                    uint32_t argument_count, zval *result)
{
    zend_class_entry *scope = object ? Z_OBJCE_P(object) : cuda_array_ce;
    ZVAL_UNDEF(result);
    zend_function *function = zend_hash_str_find_ptr(&scope->function_table, method, strlen(method));
    if (!function)
    {
        CUDA_THROW_RUNTIME("Optimizer could not find method %s", method);
        return FAILURE;
    }

    zend_call_known_function(function, object ? Z_OBJ_P(object) : NULL, scope,
                             result, argument_count, arguments, NULL);
    if (EG(exception))
    {
        if (Z_TYPE_P(result) != IS_UNDEF)
            zval_ptr_dtor(result);
        ZVAL_UNDEF(result);
        return FAILURE;
    }
    return Z_TYPE_P(result) == IS_UNDEF ? FAILURE : SUCCESS;
}

static zend_result optimizer_array_unary(zval *array, const char *method, zval *result)
{
    if (Z_TYPE_P(array) != IS_OBJECT || !instanceof_function(Z_OBJCE_P(array), cuda_array_ce))
    {
        CUDA_THROW_INVALID("Optimizer state and values must be Cuda\\CudaArray instances");
        return FAILURE;
    }
    return optimizer_invoke(array, method, NULL, 0, result);
}

static zend_result optimizer_array_binary(zval *array, const char *method,
                                          zval *operand, zval *result)
{
    if (Z_TYPE_P(array) != IS_OBJECT || !instanceof_function(Z_OBJCE_P(array), cuda_array_ce))
    {
        CUDA_THROW_INVALID("Optimizer state and values must be Cuda\\CudaArray instances");
        return FAILURE;
    }
    zval argument;
    ZVAL_COPY(&argument, operand);
    zend_result status = optimizer_invoke(array, method, &argument, 1, result);
    zval_ptr_dtor(&argument);
    return status;
}

static zend_result optimizer_static_array_call(const char *method, zval *arguments,
                                               uint32_t argument_count, zval *result)
{
    return optimizer_invoke(NULL, method, arguments, argument_count, result);
}

static int optimizer_is_list(zval *value, const char *name)
{
    if (Z_TYPE_P(value) != IS_ARRAY || !zend_array_is_list(Z_ARRVAL_P(value)))
    {
        CUDA_THROW_INVALID("%s must be a list", name);
        return 0;
    }
    return 1;
}

static int optimizer_validate_parameters(zval *parameters, const char *name)
{
    if (!optimizer_is_list(parameters, name))
        return 0;
    if (zend_hash_num_elements(Z_ARRVAL_P(parameters)) == 0)
    {
        CUDA_THROW_INVALID("%s cannot be empty", name);
        return 0;
    }
    zval *value;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(parameters), value)
    {
        ZVAL_DEREF(value);
        if (Z_TYPE_P(value) != IS_OBJECT || !instanceof_function(Z_OBJCE_P(value), cuda_array_ce))
        {
            CUDA_THROW_INVALID("Every entry in %s must be a Cuda\\CudaArray", name);
            return 0;
        }
        zval dtype;
        if (optimizer_array_unary(value, "dtype", &dtype) == FAILURE)
            return 0;
        int supported = Z_TYPE(dtype) == IS_STRING &&
            (zend_string_equals_literal(Z_STR(dtype), "float32") ||
             zend_string_equals_literal(Z_STR(dtype), "float64"));
        zval_ptr_dtor(&dtype);
        if (!supported)
        {
            CUDA_THROW_INVALID("Optimizer %s must use float32 or float64 tensors", name);
            return 0;
        }
    }
    ZEND_HASH_FOREACH_END();
    return 1;
}

static int optimizer_valid_nonnegative(double value, const char *name, int allow_zero)
{
    if (!isfinite(value) || value < 0.0 || (!allow_zero && value == 0.0))
    {
        CUDA_THROW_INVALID("%s must be a finite %s value", name,
                           allow_zero ? "non-negative" : "positive");
        return 0;
    }
    return 1;
}

static void optimizer_create(zval *result, const char *algorithm, double learning_rate,
                             double momentum, double beta1, double beta2, double epsilon,
                             double weight_decay)
{
    object_init_ex(result, optimizer_ce);
    zend_update_property_string(optimizer_ce, Z_OBJ_P(result), "algorithm",
                                sizeof("algorithm") - 1, algorithm);
    zend_update_property_double(optimizer_ce, Z_OBJ_P(result), "learningRate",
                                sizeof("learningRate") - 1, learning_rate);
    zend_update_property_double(optimizer_ce, Z_OBJ_P(result), "momentum",
                                sizeof("momentum") - 1, momentum);
    zend_update_property_double(optimizer_ce, Z_OBJ_P(result), "beta1",
                                sizeof("beta1") - 1, beta1);
    zend_update_property_double(optimizer_ce, Z_OBJ_P(result), "beta2",
                                sizeof("beta2") - 1, beta2);
    zend_update_property_double(optimizer_ce, Z_OBJ_P(result), "epsilon",
                                sizeof("epsilon") - 1, epsilon);
    zend_update_property_double(optimizer_ce, Z_OBJ_P(result), "weightDecay",
                                sizeof("weightDecay") - 1, weight_decay);
}

ZEND_METHOD(Optimizer, __construct)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zend_throw_error(NULL, "Create an optimizer with Cuda\\Optimizer::adamW() or ::sgd()");
    RETURN_THROWS();
}

ZEND_METHOD(Optimizer, adamW)
{
    double learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999;
    double epsilon = 1e-8, weight_decay = 0.01;
    ZEND_PARSE_PARAMETERS_START(0, 5)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(learning_rate)
        Z_PARAM_DOUBLE(beta1)
        Z_PARAM_DOUBLE(beta2)
        Z_PARAM_DOUBLE(epsilon)
        Z_PARAM_DOUBLE(weight_decay)
    ZEND_PARSE_PARAMETERS_END();

    if (!optimizer_valid_nonnegative(learning_rate, "learningRate", 0) ||
        !isfinite(beta1) || beta1 < 0.0 || beta1 >= 1.0 ||
        !isfinite(beta2) || beta2 < 0.0 || beta2 >= 1.0 ||
        !optimizer_valid_nonnegative(epsilon, "epsilon", 0) ||
        !optimizer_valid_nonnegative(weight_decay, "weightDecay", 1))
    {
        if (!EG(exception))
            CUDA_THROW_INVALID("beta1 and beta2 must be finite values in [0, 1)");
        RETURN_THROWS();
    }
    optimizer_create(return_value, "adamW", learning_rate, 0.0, beta1, beta2,
                     epsilon, weight_decay);
}

ZEND_METHOD(Optimizer, sgd)
{
    double learning_rate = 0.01, momentum = 0.0, weight_decay = 0.0;
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(learning_rate)
        Z_PARAM_DOUBLE(momentum)
        Z_PARAM_DOUBLE(weight_decay)
    ZEND_PARSE_PARAMETERS_END();

    if (!optimizer_valid_nonnegative(learning_rate, "learningRate", 0) ||
        !isfinite(momentum) || momentum < 0.0 || momentum >= 1.0 ||
        !optimizer_valid_nonnegative(weight_decay, "weightDecay", 1))
    {
        if (!EG(exception))
            CUDA_THROW_INVALID("momentum must be a finite value in [0, 1)");
        RETURN_THROWS();
    }
    optimizer_create(return_value, "sgd", learning_rate, momentum, 0.0, 0.0,
                     1e-8, weight_decay);
}

static zend_result optimizer_create_zeros(zval *parameters, zval *result)
{
    array_init_size(result, zend_hash_num_elements(Z_ARRVAL_P(parameters)));
    zval *parameter;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(parameters), parameter)
    {
        zval shape, dtype, zero, arguments[2];
        if (optimizer_array_unary(parameter, "getshape", &shape) == FAILURE)
            return FAILURE;
        if (optimizer_array_unary(parameter, "dtype", &dtype) == FAILURE)
        {
            zval_ptr_dtor(&shape);
            return FAILURE;
        }
        ZVAL_COPY(&arguments[0], &shape);
        ZVAL_COPY(&arguments[1], &dtype);
        zend_result status = optimizer_static_array_call("zeros", arguments, 2, &zero);
        zval_ptr_dtor(&arguments[0]);
        zval_ptr_dtor(&arguments[1]);
        zval_ptr_dtor(&shape);
        zval_ptr_dtor(&dtype);
        if (status == FAILURE)
            return FAILURE;
        add_next_index_zval(result, &zero);
    }
    ZEND_HASH_FOREACH_END();
    return SUCCESS;
}

static zend_result optimizer_create_one(zval *parameter, zval *result)
{
    zval shape, dtype, arguments[2];
    array_init_size(&shape, 1);
    add_next_index_long(&shape, 1);
    if (optimizer_array_unary(parameter, "dtype", &dtype) == FAILURE)
    {
        zval_ptr_dtor(&shape);
        return FAILURE;
    }
    ZVAL_COPY(&arguments[0], &shape);
    ZVAL_COPY(&arguments[1], &dtype);
    zend_result status = optimizer_static_array_call("ones", arguments, 2, result);
    zval_ptr_dtor(&arguments[0]);
    zval_ptr_dtor(&arguments[1]);
    zval_ptr_dtor(&shape);
    zval_ptr_dtor(&dtype);
    return status;
}

ZEND_METHOD(Optimizer, zeroGrad)
{
    zval *parameters;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(parameters)
    ZEND_PARSE_PARAMETERS_END();
    if (!optimizer_validate_parameters(parameters, "parameters"))
        RETURN_THROWS();

    zval *parameter;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(parameters), parameter)
    {
        zval ignored;
        if (optimizer_array_unary(parameter, "zerograd", &ignored) == FAILURE)
        {
            if (Z_TYPE(ignored) != IS_UNDEF)
                zval_ptr_dtor(&ignored);
            RETURN_THROWS();
        }
        zval_ptr_dtor(&ignored);
    }
    ZEND_HASH_FOREACH_END();
    RETURN_NULL();
}

static int optimizer_same_shape(zval *left, zval *right)
{
    zval left_shape, right_shape;
    ZVAL_UNDEF(&left_shape);
    ZVAL_UNDEF(&right_shape);
    if (optimizer_array_unary(left, "getshape", &left_shape) == FAILURE ||
        optimizer_array_unary(right, "getshape", &right_shape) == FAILURE)
    {
        if (Z_TYPE(left_shape) != IS_UNDEF) zval_ptr_dtor(&left_shape);
        if (Z_TYPE(right_shape) != IS_UNDEF) zval_ptr_dtor(&right_shape);
        return -1;
    }
    int equal = zend_hash_num_elements(Z_ARRVAL(left_shape)) == zend_hash_num_elements(Z_ARRVAL(right_shape));
    if (equal)
    {
        for (uint32_t axis = 0; axis < zend_hash_num_elements(Z_ARRVAL(left_shape)); axis++)
        {
            zval *left_dimension = zend_hash_index_find(Z_ARRVAL(left_shape), axis);
            zval *right_dimension = zend_hash_index_find(Z_ARRVAL(right_shape), axis);
            if (!left_dimension || !right_dimension || Z_TYPE_P(left_dimension) != IS_LONG ||
                Z_TYPE_P(right_dimension) != IS_LONG || Z_LVAL_P(left_dimension) != Z_LVAL_P(right_dimension))
            {
                equal = 0;
                break;
            }
        }
    }
    zval_ptr_dtor(&left_shape);
    zval_ptr_dtor(&right_shape);
    return equal;
}

static int optimizer_same_dtype(zval *left, zval *right)
{
    zval left_dtype, right_dtype;
    ZVAL_UNDEF(&left_dtype);
    ZVAL_UNDEF(&right_dtype);
    if (optimizer_array_unary(left, "dtype", &left_dtype) == FAILURE ||
        optimizer_array_unary(right, "dtype", &right_dtype) == FAILURE)
    {
        if (Z_TYPE(left_dtype) != IS_UNDEF) zval_ptr_dtor(&left_dtype);
        if (Z_TYPE(right_dtype) != IS_UNDEF) zval_ptr_dtor(&right_dtype);
        return -1;
    }
    int equal = Z_TYPE(left_dtype) == IS_STRING && Z_TYPE(right_dtype) == IS_STRING &&
        zend_string_equals(Z_STR(left_dtype), Z_STR(right_dtype));
    zval_ptr_dtor(&left_dtype);
    zval_ptr_dtor(&right_dtype);
    return equal;
}

static zend_result optimizer_validate_gradient_pairs(zval *parameters, zval *gradients)
{
    uint32_t index = 0;
    zval *parameter;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(parameters), parameter)
    {
        zval *gradient = zend_hash_index_find(Z_ARRVAL_P(gradients), index);
        int same_shape = optimizer_same_shape(parameter, gradient);
        if (same_shape < 0)
            return FAILURE;
        if (!same_shape)
        {
            CUDA_THROW_INVALID("Each gradient must have the same shape as its parameter");
            return FAILURE;
        }
        int same_dtype = optimizer_same_dtype(parameter, gradient);
        if (same_dtype < 0)
            return FAILURE;
        if (!same_dtype)
        {
            CUDA_THROW_INVALID("Each gradient must have the same dtype as its parameter");
            return FAILURE;
        }
        index++;
    }
    ZEND_HASH_FOREACH_END();
    return SUCCESS;
}

static zend_result optimizer_get_state_list(zval *state, const char *key, zval *parameters,
                                            zval **result)
{
    zval *value = zend_hash_str_find(Z_ARRVAL_P(state), key, strlen(key));
    if (!value || Z_TYPE_P(value) != IS_ARRAY || !zend_array_is_list(Z_ARRVAL_P(value)) ||
        zend_hash_num_elements(Z_ARRVAL_P(value)) != zend_hash_num_elements(Z_ARRVAL_P(parameters)))
    {
        CUDA_THROW_INVALID("Optimizer state '%s' must be a list matching the parameter count", key);
        return FAILURE;
    }
    zval *parameter, *tensor;
    uint32_t index = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(parameters), parameter)
    {
        tensor = zend_hash_index_find(Z_ARRVAL_P(value), index);
        if (!tensor || Z_TYPE_P(tensor) != IS_OBJECT || !instanceof_function(Z_OBJCE_P(tensor), cuda_array_ce))
        {
            CUDA_THROW_INVALID("Optimizer state '%s' entries must be Cuda\\CudaArray tensors", key);
            return FAILURE;
        }
        int same_shape = optimizer_same_shape(parameter, tensor);
        if (same_shape < 0)
            return FAILURE;
        if (!same_shape)
        {
            CUDA_THROW_INVALID("Optimizer state '%s' tensor shapes must match parameters", key);
            return FAILURE;
        }
        int same_dtype = optimizer_same_dtype(parameter, tensor);
        if (same_dtype < 0)
            return FAILURE;
        if (!same_dtype)
        {
            CUDA_THROW_INVALID("Optimizer state '%s' tensor dtypes must match parameters", key);
            return FAILURE;
        }
        index++;
    }
    ZEND_HASH_FOREACH_END();
    *result = value;
    return SUCCESS;
}

static int optimizer_decay_enabled(zval *mask, uint32_t index)
{
    if (!mask || Z_TYPE_P(mask) == IS_NULL)
        return 1;
    zval *value = zend_hash_index_find(Z_ARRVAL_P(mask), index);
    return value && Z_TYPE_P(value) == IS_TRUE;
}

static void optimizer_clear_values(zval *values, uint32_t count)
{
    for (uint32_t index = 0; index < count; index++)
        if (Z_TYPE(values[index]) != IS_UNDEF)
            zval_ptr_dtor(&values[index]);
}

static zend_result optimizer_where(zval *condition, zval *when_true, zval *when_false, zval *result)
{
    zval arguments[3];
    ZVAL_COPY(&arguments[0], condition);
    ZVAL_COPY(&arguments[1], when_true);
    ZVAL_COPY(&arguments[2], when_false);
    zend_result status = optimizer_static_array_call("where", arguments, 3, result);
    for (uint32_t index = 0; index < 3; index++)
        zval_ptr_dtor(&arguments[index]);
    return status;
}

static zend_result optimizer_append_selected(zval *gradient, zval *updated,
                                             zval *original, zval *output, int detach)
{
    zval condition, selected, detached;
    ZVAL_UNDEF(&condition);
    ZVAL_UNDEF(&selected);
    ZVAL_UNDEF(&detached);
    if (optimizer_array_binary(gradient, "eq", gradient, &condition) == FAILURE ||
        optimizer_where(&condition, updated, original, &selected) == FAILURE)
    {
        if (Z_TYPE(condition) != IS_UNDEF) zval_ptr_dtor(&condition);
        if (Z_TYPE(selected) != IS_UNDEF) zval_ptr_dtor(&selected);
        return FAILURE;
    }
    zval_ptr_dtor(&condition);
    if (detach && optimizer_array_unary(&selected, "detach", &detached) == FAILURE)
    {
        zval_ptr_dtor(&selected);
        return FAILURE;
    }
    if (detach)
    {
        zval_ptr_dtor(&selected);
        ZVAL_COPY_VALUE(&selected, &detached);
    }
    add_next_index_zval(output, &selected);
    return SUCCESS;
}

static zend_result optimizer_append_selected_state(zval *gradient, zval *updated,
                                                    zval *original, zval *output)
{
    zval condition, selected;
    ZVAL_UNDEF(&condition);
    ZVAL_UNDEF(&selected);
    if (optimizer_array_binary(gradient, "eq", gradient, &condition) == FAILURE ||
        optimizer_where(&condition, updated, original, &selected) == FAILURE)
    {
        if (Z_TYPE(condition) != IS_UNDEF) zval_ptr_dtor(&condition);
        if (Z_TYPE(selected) != IS_UNDEF) zval_ptr_dtor(&selected);
        return FAILURE;
    }
    zval_ptr_dtor(&condition);
    add_next_index_zval(output, &selected);
    return SUCCESS;
}

static zend_result optimizer_adamw_step(zval *optimizer, zval *parameters, zval *gradients,
                                       zval *state, zval *learning_rate, zval *decay_mask,
                                       zval *outputs, zval *next_state)
{
    uint32_t count = zend_hash_num_elements(Z_ARRVAL_P(parameters));
    zval *first_moments, *second_moments;
    zval *beta1_power = zend_hash_str_find(Z_ARRVAL_P(state), "beta1Power", sizeof("beta1Power") - 1);
    zval *beta2_power = zend_hash_str_find(Z_ARRVAL_P(state), "beta2Power", sizeof("beta2Power") - 1);
    if (!beta1_power || !beta2_power ||
        optimizer_get_state_list(state, "firstMoment", parameters, &first_moments) == FAILURE ||
        optimizer_get_state_list(state, "secondMoment", parameters, &second_moments) == FAILURE)
    {
        if (!EG(exception))
            CUDA_THROW_INVALID("AdamW state must be created by initState()");
        return FAILURE;
    }

    double beta1 = Z_DVAL_P(zend_read_property(optimizer_ce, Z_OBJ_P(optimizer), "beta1", sizeof("beta1") - 1, 0, NULL));
    double beta2 = Z_DVAL_P(zend_read_property(optimizer_ce, Z_OBJ_P(optimizer), "beta2", sizeof("beta2") - 1, 0, NULL));
    double epsilon = Z_DVAL_P(zend_read_property(optimizer_ce, Z_OBJ_P(optimizer), "epsilon", sizeof("epsilon") - 1, 0, NULL));
    double weight_decay = Z_DVAL_P(zend_read_property(optimizer_ce, Z_OBJ_P(optimizer), "weightDecay", sizeof("weightDecay") - 1, 0, NULL));
    zval beta1_value, beta2_value, one_minus_beta1, one_minus_beta2, epsilon_value;
    zval one, negative_one, decay_value;
    ZVAL_DOUBLE(&beta1_value, beta1);
    ZVAL_DOUBLE(&beta2_value, beta2);
    ZVAL_DOUBLE(&one_minus_beta1, 1.0 - beta1);
    ZVAL_DOUBLE(&one_minus_beta2, 1.0 - beta2);
    ZVAL_DOUBLE(&epsilon_value, epsilon);
    ZVAL_DOUBLE(&one, 1.0);
    ZVAL_DOUBLE(&negative_one, -1.0);
    ZVAL_DOUBLE(&decay_value, weight_decay);

    zval shared[6];
    for (uint32_t index = 0; index < 6; index++)
        ZVAL_UNDEF(&shared[index]);
    if (optimizer_array_binary(beta1_power, "multiply", &beta1_value, &shared[0]) == FAILURE ||
        optimizer_array_binary(beta2_power, "multiply", &beta2_value, &shared[1]) == FAILURE ||
        optimizer_array_binary(&shared[0], "subtract", &one, &shared[2]) == FAILURE ||
        optimizer_array_binary(&shared[1], "subtract", &one, &shared[3]) == FAILURE ||
        optimizer_array_binary(&shared[2], "multiply", &negative_one, &shared[4]) == FAILURE ||
        optimizer_array_binary(&shared[3], "multiply", &negative_one, &shared[5]) == FAILURE)
    {
        optimizer_clear_values(shared, 6);
        return FAILURE;
    }

    array_init_size(outputs, count);
    zval next_first, next_second, temporaries[18];
    ZVAL_UNDEF(&next_first);
    ZVAL_UNDEF(&next_second);
    array_init_size(&next_first, count);
    array_init_size(&next_second, count);
    for (uint32_t index = 0; index < 18; index++)
        ZVAL_UNDEF(&temporaries[index]);
    zval *parameter;
    uint32_t parameter_index = 0;
    zend_result status = SUCCESS;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(parameters), parameter)
    {
        zval *gradient = zend_hash_index_find(Z_ARRVAL_P(gradients), parameter_index);
        zval *old_first = zend_hash_index_find(Z_ARRVAL_P(first_moments), parameter_index);
        zval *old_second = zend_hash_index_find(Z_ARRVAL_P(second_moments), parameter_index);
        optimizer_clear_values(temporaries, 18);
        for (uint32_t index = 0; index < 18; index++)
            ZVAL_UNDEF(&temporaries[index]);

        if (optimizer_array_binary(old_first, "multiply", &beta1_value, &temporaries[0]) == FAILURE ||
            optimizer_array_binary(gradient, "multiply", &one_minus_beta1, &temporaries[1]) == FAILURE ||
            optimizer_array_binary(&temporaries[0], "add", &temporaries[1], &temporaries[2]) == FAILURE ||
            optimizer_array_binary(gradient, "multiply", gradient, &temporaries[3]) == FAILURE ||
            optimizer_array_binary(old_second, "multiply", &beta2_value, &temporaries[4]) == FAILURE ||
            optimizer_array_binary(&temporaries[3], "multiply", &one_minus_beta2, &temporaries[5]) == FAILURE ||
            optimizer_array_binary(&temporaries[4], "add", &temporaries[5], &temporaries[6]) == FAILURE ||
            optimizer_array_binary(&temporaries[2], "divide", &shared[4], &temporaries[7]) == FAILURE ||
            optimizer_array_binary(&temporaries[6], "divide", &shared[5], &temporaries[8]) == FAILURE ||
            optimizer_array_unary(&temporaries[8], "sqrt", &temporaries[9]) == FAILURE ||
            optimizer_array_binary(&temporaries[9], "add", &epsilon_value, &temporaries[10]) == FAILURE ||
            optimizer_array_binary(&temporaries[7], "divide", &temporaries[10], &temporaries[11]) == FAILURE ||
            optimizer_array_binary(&temporaries[11], "multiply", learning_rate, &temporaries[12]) == FAILURE)
        {
            status = FAILURE;
            break;
        }

        if (weight_decay > 0.0 && optimizer_decay_enabled(decay_mask, parameter_index))
        {
            if (Z_TYPE_P(learning_rate) == IS_OBJECT)
            {
                if (optimizer_array_binary(learning_rate, "multiply", &decay_value, &temporaries[13]) == FAILURE ||
                    optimizer_array_binary(&temporaries[13], "subtract", &one, &temporaries[14]) == FAILURE ||
                    optimizer_array_binary(&temporaries[14], "multiply", &negative_one, &temporaries[15]) == FAILURE)
                    status = FAILURE;
            }
            else
            {
                ZVAL_DOUBLE(&temporaries[15], 1.0 - Z_DVAL_P(learning_rate) * weight_decay);
            }
            if (status == SUCCESS &&
                (optimizer_array_binary(parameter, "multiply", &temporaries[15], &temporaries[16]) == FAILURE ||
                optimizer_array_binary(&temporaries[16], "subtract", &temporaries[12], &temporaries[17]) == FAILURE)
            )
                status = FAILURE;
        }
        else if (optimizer_array_binary(parameter, "subtract", &temporaries[12], &temporaries[17]) == FAILURE)
        {
            status = FAILURE;
        }

        if (status == SUCCESS &&
            (optimizer_append_selected(gradient, &temporaries[17], parameter, outputs, 1) == FAILURE ||
             optimizer_append_selected_state(gradient, &temporaries[2],
                 zend_hash_index_find(Z_ARRVAL_P(first_moments), parameter_index), &next_first) == FAILURE ||
             optimizer_append_selected_state(gradient, &temporaries[6],
                 zend_hash_index_find(Z_ARRVAL_P(second_moments), parameter_index), &next_second) == FAILURE))
            status = FAILURE;
        if (status == FAILURE)
            break;
        parameter_index++;
    }
    ZEND_HASH_FOREACH_END();
    optimizer_clear_values(temporaries, 18);
    if (status == FAILURE)
    {
        zval_ptr_dtor(outputs);
        zval_ptr_dtor(&next_first);
        zval_ptr_dtor(&next_second);
        optimizer_clear_values(shared, 6);
        return FAILURE;
    }
    add_assoc_zval(next_state, "firstMoment", &next_first);
    add_assoc_zval(next_state, "secondMoment", &next_second);
    add_assoc_zval(next_state, "beta1Power", &shared[0]);
    add_assoc_zval(next_state, "beta2Power", &shared[1]);
    zval_ptr_dtor(&shared[2]); zval_ptr_dtor(&shared[3]);
    zval_ptr_dtor(&shared[4]); zval_ptr_dtor(&shared[5]);
    return SUCCESS;
}

static zend_result optimizer_sgd_step(zval *optimizer, zval *parameters, zval *gradients,
                                     zval *state, zval *learning_rate, zval *decay_mask,
                                     zval *outputs, zval *next_state)
{
    uint32_t count = zend_hash_num_elements(Z_ARRVAL_P(parameters));
    zval *velocities;
    if (optimizer_get_state_list(state, "velocity", parameters, &velocities) == FAILURE)
        return FAILURE;
    double momentum = Z_DVAL_P(zend_read_property(optimizer_ce, Z_OBJ_P(optimizer), "momentum", sizeof("momentum") - 1, 0, NULL));
    double weight_decay = Z_DVAL_P(zend_read_property(optimizer_ce, Z_OBJ_P(optimizer), "weightDecay", sizeof("weightDecay") - 1, 0, NULL));
    zval momentum_value, decay_value;
    ZVAL_DOUBLE(&momentum_value, momentum);
    ZVAL_DOUBLE(&decay_value, weight_decay);

    array_init_size(outputs, count);
    zval velocity_outputs, temporaries[7];
    array_init_size(&velocity_outputs, count);
    for (uint32_t index = 0; index < 7; index++)
        ZVAL_UNDEF(&temporaries[index]);
    zval *parameter;
    uint32_t parameter_index = 0;
    zend_result status = SUCCESS;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(parameters), parameter)
    {
        zval *gradient = zend_hash_index_find(Z_ARRVAL_P(gradients), parameter_index);
        zval *velocity = zend_hash_index_find(Z_ARRVAL_P(velocities), parameter_index);
        optimizer_clear_values(temporaries, 7);
        for (uint32_t index = 0; index < 7; index++)
            ZVAL_UNDEF(&temporaries[index]);
        if (optimizer_array_binary(velocity, "multiply", &momentum_value, &temporaries[0]) == FAILURE ||
            optimizer_array_binary(gradient, "add", &temporaries[0], &temporaries[1]) == FAILURE)
        {
            status = FAILURE;
            break;
        }
        uint32_t direction = 1;
        if (weight_decay > 0.0 && optimizer_decay_enabled(decay_mask, parameter_index))
        {
            if (optimizer_array_binary(parameter, "multiply", &decay_value, &temporaries[2]) == FAILURE ||
                optimizer_array_binary(&temporaries[1], "add", &temporaries[2], &temporaries[3]) == FAILURE)
            {
                status = FAILURE;
                break;
            }
            direction = 3;
        }
        if (optimizer_array_binary(&temporaries[direction], "multiply", learning_rate, &temporaries[4]) == FAILURE ||
            optimizer_array_binary(parameter, "subtract", &temporaries[4], &temporaries[5]) == FAILURE ||
            optimizer_append_selected(gradient, &temporaries[5], parameter, outputs, 1) == FAILURE ||
            optimizer_append_selected_state(gradient, &temporaries[1], velocity, &velocity_outputs) == FAILURE)
        {
            status = FAILURE;
            break;
        }
        parameter_index++;
    }
    ZEND_HASH_FOREACH_END();
    optimizer_clear_values(temporaries, 7);
    if (status == FAILURE)
    {
        zval_ptr_dtor(outputs);
        zval_ptr_dtor(&velocity_outputs);
        return FAILURE;
    }
    add_assoc_zval(next_state, "velocity", &velocity_outputs);
    return SUCCESS;
}

ZEND_METHOD(Optimizer, initState)
{
    zval *parameters;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(parameters)
    ZEND_PARSE_PARAMETERS_END();
    if (!optimizer_validate_parameters(parameters, "parameters"))
        RETURN_THROWS();
    zval *algorithm = zend_read_property(optimizer_ce, Z_OBJ_P(ZEND_THIS),
                                         "algorithm", sizeof("algorithm") - 1, 0, NULL);
    zval state, first, second;
    ZVAL_UNDEF(&first);
    ZVAL_UNDEF(&second);
    array_init(&state);
    if (zend_string_equals_literal(Z_STR_P(algorithm), "adamW"))
    {
        zend_result status = optimizer_create_zeros(parameters, &first);
        if (status == SUCCESS)
            status = optimizer_create_zeros(parameters, &second);
        zval beta1_power, beta2_power;
        ZVAL_UNDEF(&beta1_power);
        ZVAL_UNDEF(&beta2_power);
        zval *first_parameter = zend_hash_index_find(Z_ARRVAL_P(parameters), 0);
        if (status == SUCCESS)
            status = optimizer_create_one(first_parameter, &beta1_power);
        if (status == SUCCESS)
            status = optimizer_create_one(first_parameter, &beta2_power);
        if (status == FAILURE)
        {
            if (Z_TYPE(beta1_power) != IS_UNDEF) zval_ptr_dtor(&beta1_power);
            if (Z_TYPE(beta2_power) != IS_UNDEF) zval_ptr_dtor(&beta2_power);
            zval_ptr_dtor(&state);
            if (Z_TYPE(first) != IS_UNDEF) zval_ptr_dtor(&first);
            if (Z_TYPE(second) != IS_UNDEF) zval_ptr_dtor(&second);
            RETURN_THROWS();
        }
        add_assoc_zval(&state, "firstMoment", &first);
        add_assoc_zval(&state, "secondMoment", &second);
        add_assoc_zval(&state, "beta1Power", &beta1_power);
        add_assoc_zval(&state, "beta2Power", &beta2_power);
    }
    else
    {
        if (optimizer_create_zeros(parameters, &first) == FAILURE)
        {
            zval_ptr_dtor(&state);
            if (Z_TYPE(first) != IS_UNDEF) zval_ptr_dtor(&first);
            if (Z_TYPE(second) != IS_UNDEF) zval_ptr_dtor(&second);
            RETURN_THROWS();
        }
        add_assoc_zval(&state, "velocity", &first);
    }
    RETURN_ZVAL(&state, 0, 1);
}

ZEND_METHOD(Optimizer, step)
{
    zval *parameters, *gradients, *state, *learning_rate = NULL, *decay_mask = NULL;
    ZEND_PARSE_PARAMETERS_START(3, 5)
        Z_PARAM_ARRAY(parameters)
        Z_PARAM_ARRAY(gradients)
        Z_PARAM_ARRAY(state)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJECT_OF_CLASS_OR_NULL(learning_rate, cuda_array_ce)
        Z_PARAM_ARRAY_OR_NULL(decay_mask)
    ZEND_PARSE_PARAMETERS_END();

    if (!optimizer_validate_parameters(parameters, "parameters") ||
        !optimizer_validate_parameters(gradients, "gradients") ||
        Z_TYPE_P(state) != IS_ARRAY ||
        zend_hash_num_elements(Z_ARRVAL_P(parameters)) != zend_hash_num_elements(Z_ARRVAL_P(gradients)))
    {
        if (!EG(exception))
            CUDA_THROW_INVALID("parameters and gradients must be equally sized CudaArray lists");
        RETURN_THROWS();
    }
    if (optimizer_validate_gradient_pairs(parameters, gradients) == FAILURE)
        RETURN_THROWS();
    if (decay_mask && Z_TYPE_P(decay_mask) != IS_NULL)
    {
        if (!optimizer_is_list(decay_mask, "weightDecayMask") ||
            zend_hash_num_elements(Z_ARRVAL_P(decay_mask)) != zend_hash_num_elements(Z_ARRVAL_P(parameters)))
        {
            if (!EG(exception))
                CUDA_THROW_INVALID("weightDecayMask must match the parameter count");
            RETURN_THROWS();
        }
        zval *mask_value;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(decay_mask), mask_value)
        {
            if (Z_TYPE_P(mask_value) != IS_TRUE && Z_TYPE_P(mask_value) != IS_FALSE)
            {
                CUDA_THROW_INVALID("weightDecayMask entries must be booleans");
                RETURN_THROWS();
            }
        }
        ZEND_HASH_FOREACH_END();
    }

    zval rate;
    if (learning_rate)
        ZVAL_COPY(&rate, learning_rate);
    else
        ZVAL_COPY(&rate, zend_read_property(
            optimizer_ce, Z_OBJ_P(ZEND_THIS), "learningRate", sizeof("learningRate") - 1, 0, NULL));

    zval outputs, next_state;
    ZVAL_UNDEF(&outputs);
    array_init(&next_state);
    zval *algorithm = zend_read_property(optimizer_ce, Z_OBJ_P(ZEND_THIS),
                                         "algorithm", sizeof("algorithm") - 1, 0, NULL);
    zend_result status = zend_string_equals_literal(Z_STR_P(algorithm), "adamW")
        ? optimizer_adamw_step(ZEND_THIS, parameters, gradients, state, &rate,
                               decay_mask, &outputs, &next_state)
        : optimizer_sgd_step(ZEND_THIS, parameters, gradients, state, &rate,
                             decay_mask, &outputs, &next_state);
    zval_ptr_dtor(&rate);
    if (status == FAILURE)
    {
        if (Z_TYPE(outputs) != IS_UNDEF)
            zval_ptr_dtor(&outputs);
        zval_ptr_dtor(&next_state);
        RETURN_THROWS();
    }
    array_init(return_value);
    add_assoc_zval(return_value, "parameters", &outputs);
    add_assoc_zval(return_value, "state", &next_state);
}

static const zend_function_entry optimizer_methods[] = {
    ZEND_ME(Optimizer, __construct, optimizer_private_args, ZEND_ACC_PRIVATE)
    ZEND_ME(Optimizer, adamW, optimizer_adamw_args, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(Optimizer, sgd, optimizer_sgd_args, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(Optimizer, initState, optimizer_parameters_args, ZEND_ACC_PUBLIC)
    ZEND_ME(Optimizer, zeroGrad, optimizer_zero_grad_args, ZEND_ACC_PUBLIC)
    ZEND_ME(Optimizer, step, optimizer_step_args, ZEND_ACC_PUBLIC)
    ZEND_FE_END
};

int cuda_optimizer_init(void)
{
    zend_class_entry entry;
    INIT_NS_CLASS_ENTRY(entry, "Cuda", "Optimizer", optimizer_methods);
    optimizer_ce = zend_register_internal_class(&entry);
    optimizer_ce->ce_flags |= ZEND_ACC_FINAL;
    zend_declare_property_string(optimizer_ce, "algorithm", sizeof("algorithm") - 1, "", ZEND_ACC_PRIVATE);
    zend_declare_property_double(optimizer_ce, "learningRate", sizeof("learningRate") - 1, 0.0, ZEND_ACC_PRIVATE);
    zend_declare_property_double(optimizer_ce, "momentum", sizeof("momentum") - 1, 0.0, ZEND_ACC_PRIVATE);
    zend_declare_property_double(optimizer_ce, "beta1", sizeof("beta1") - 1, 0.0, ZEND_ACC_PRIVATE);
    zend_declare_property_double(optimizer_ce, "beta2", sizeof("beta2") - 1, 0.0, ZEND_ACC_PRIVATE);
    zend_declare_property_double(optimizer_ce, "epsilon", sizeof("epsilon") - 1, 0.0, ZEND_ACC_PRIVATE);
    zend_declare_property_double(optimizer_ce, "weightDecay", sizeof("weightDecay") - 1, 0.0, ZEND_ACC_PRIVATE);
    return SUCCESS;
}
