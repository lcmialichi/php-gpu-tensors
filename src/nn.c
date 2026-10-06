#include "nn.h"
#include "nn_backend.h"
#include "cuda_array_ce.h"
#include "ca_private.h"
#include "tensor_factory.h"
#include "fusion.h"
#include "cuda_exceptions.h"
#include <limits.h>
#include <stdint.h>

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(nn_available_args, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(nn_conv_args, 0, 2, Cuda\\CudaArray, 0)
ZEND_ARG_OBJ_INFO(0, input, Cuda\\CudaArray, 0)
ZEND_ARG_OBJ_INFO(0, weights, Cuda\\CudaArray, 0)
ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, bias, Cuda\\CudaArray, 1, "null")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, stride, IS_ARRAY, 0, "[1, 1]")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, padding, IS_ARRAY, 0, "[0, 0]")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dilation, IS_ARRAY, 0, "[1, 1]")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, groups, IS_LONG, 0, "1")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(nn_pool_args, 0, 2, Cuda\\CudaArray, 0)
ZEND_ARG_OBJ_INFO(0, input, Cuda\\CudaArray, 0)
ZEND_ARG_TYPE_INFO(0, window, IS_ARRAY, 0)
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, stride, IS_ARRAY, 0, "[2, 2]")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, padding, IS_ARRAY, 0, "[0, 0]")
ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, mode, IS_STRING, 0, "\"max\"")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(nn_softmax_args, 0, 1, Cuda\\CudaArray, 0)
ZEND_ARG_OBJ_INFO(0, input, Cuda\\CudaArray, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(nn_private_args, 0, 0, 0)
ZEND_END_ARG_INFO()

static tensor_t *nn_input(zval *value, int ndims)
{
    if (fusion_active()) {
        CUDA_THROW_RUNTIME("Cuda\\NN inference is not supported inside Fusion capture");
        return NULL;
    }
    cuda_array_obj *object = php_cuda_array_fetch_valid_object(Z_OBJ_P(value));
    if (!object) return NULL;
    tensor_t *tensor = object->tensor_handle;
    if (tensor->dtype != DTYPE_FLOAT32 || tensor->ndims != ndims || !is_contiguous(tensor)) {
        CUDA_THROW_INVALID("Cuda\\NN expects contiguous float32 tensors with %d dimensions", ndims);
        return NULL;
    }
    for (int d = 0; d < ndims; d++)
        if (!tensor->shape[d]) {
            CUDA_THROW_INVALID("Cuda\\NN does not accept empty axes");
            return NULL;
        }
    if (tensor->total_size > INT_MAX) {
        CUDA_THROW_INVALID("Cuda\\NN tensor exceeds INT_MAX elements");
        return NULL;
    }
    return tensor;
}

static int nn_pair(zval *value, int output[2], int minimum, const char *name)
{
    if (!value) return 1;
    if (zend_hash_num_elements(Z_ARRVAL_P(value)) != 2 || !zend_array_is_list(Z_ARRVAL_P(value))) {
        CUDA_THROW_INVALID("%s must be a list of two integers", name);
        return 0;
    }
    int index = 0;
    zval *item;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(value), item) {
        ZVAL_DEREF(item);
        if (Z_TYPE_P(item) != IS_LONG || Z_LVAL_P(item) < minimum || Z_LVAL_P(item) > INT_MAX) {
            CUDA_THROW_INVALID("%s values must be integers between %d and INT_MAX", name, minimum);
            return 0;
        }
        output[index++] = (int)Z_LVAL_P(item);
    } ZEND_HASH_FOREACH_END();
    return 1;
}

static int nn_shape(tensor_t *input, int channels, const int kernel[2], const int stride[2],
                     const int padding[2], const int dilation[2], int shape[4])
{
    shape[0] = input->shape[0]; shape[1] = channels;
    size_t elements = (size_t)shape[0] * shape[1];
    for (int d = 0; d < 2; d++) {
        int64_t numerator = (int64_t)input->shape[d+2] + 2LL * padding[d] -
            ((int64_t)dilation[d] * (kernel[d] - 1) + 1);
        int64_t dimension = numerator < 0 ? 0 : numerator / stride[d] + 1;
        if (dimension <= 0 || dimension > INT_MAX || elements > INT_MAX / (size_t)dimension) {
            CUDA_THROW_INVALID("Cuda\\NN output shape is empty or exceeds supported limits");
            return 0;
        }
        shape[d+2] = (int)dimension;
        elements *= (size_t)dimension;
    }
    return 1;
}

static void nn_result(INTERNAL_FUNCTION_PARAMETERS, tensor_t *result, int ok)
{
    if (!ok) {
        cudaStreamSynchronize(NULL);
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("cuDNN inference failed: %s", cuda_nn_error());
        return;
    }
    create_result_object(return_value, result);
}

ZEND_METHOD(NN, __construct) { ZEND_PARSE_PARAMETERS_NONE(); }
ZEND_METHOD(NN, isAvailable) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_BOOL(cuda_nn_available()); }
ZEND_METHOD(NN, conv2d)
{
    zval *x, *w, *bias = NULL, *stride_value = NULL, *padding_value = NULL, *dilation_value = NULL;
    zend_long groups = 1;
    ZEND_PARSE_PARAMETERS_START(2, 7)
    Z_PARAM_OBJECT_OF_CLASS(x, cuda_array_ce)
    Z_PARAM_OBJECT_OF_CLASS(w, cuda_array_ce)
    Z_PARAM_OPTIONAL
    Z_PARAM_OBJECT_OF_CLASS_OR_NULL(bias, cuda_array_ce)
    Z_PARAM_ARRAY(stride_value)
    Z_PARAM_ARRAY(padding_value)
    Z_PARAM_ARRAY(dilation_value)
    Z_PARAM_LONG(groups)
    ZEND_PARSE_PARAMETERS_END();
    tensor_t *input = nn_input(x, 4), *weights;
    if (!input) RETURN_THROWS();
    weights = nn_input(w, 4);
    if (!weights) RETURN_THROWS();
    int stride[] = {1,1}, padding[] = {0,0}, dilation[] = {1,1};
    if (!nn_pair(stride_value, stride, 1, "stride") || !nn_pair(padding_value, padding, 0, "padding") ||
        !nn_pair(dilation_value, dilation, 1, "dilation")) RETURN_THROWS();
    if (groups < 1 || groups > INT_MAX || input->shape[1] % groups || weights->shape[0] % groups ||
        weights->shape[1] != input->shape[1] / groups) {
        CUDA_THROW_INVALID("Invalid convolution groups or input/filter channels");
        RETURN_THROWS();
    }
    tensor_t *bias_tensor = NULL;
    if (bias && Z_TYPE_P(bias) != IS_NULL) {
        bias_tensor = nn_input(bias, 1);
        if (!bias_tensor) RETURN_THROWS();
        if (bias_tensor->shape[0] != weights->shape[0]) {
            CUDA_THROW_INVALID("Convolution bias must match output channels");
            RETURN_THROWS();
        }
    }
    int shape[4], kernel[] = {weights->shape[2], weights->shape[3]};
    if (!nn_shape(input, weights->shape[0], kernel, stride, padding, dilation, shape)) RETURN_THROWS();
    if (!cuda_nn_available()) { CUDA_THROW_RUNTIME("%s", cuda_nn_error()); RETURN_THROWS(); }
    tensor_t *result = cuda_tensor_create_empty_dtype(shape, 4, DTYPE_FLOAT32);
    if (!result) RETURN_THROWS();
    nn_result(INTERNAL_FUNCTION_PARAM_PASSTHRU, result, cuda_nn_conv(input->data, weights->data,
        bias_tensor ? bias_tensor->data : NULL, result->data, input->shape, weights->shape, shape,
        stride, padding, dilation, (int)groups));
}
ZEND_METHOD(NN, pool2d)
{
    zval *x, *window_value, *stride_value = NULL, *padding_value = NULL;
    zend_string *mode = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 5)
    Z_PARAM_OBJECT_OF_CLASS(x, cuda_array_ce)
    Z_PARAM_ARRAY(window_value)
    Z_PARAM_OPTIONAL
    Z_PARAM_ARRAY(stride_value)
    Z_PARAM_ARRAY(padding_value)
    Z_PARAM_STR(mode)
    ZEND_PARSE_PARAMETERS_END();
    tensor_t *input = nn_input(x, 4);
    if (!input) RETURN_THROWS();
    int window[2], stride[] = {2,2}, padding[] = {0,0}, dilation[] = {1,1}, shape[4];
    if (!nn_pair(window_value, window, 1, "window") || !nn_pair(stride_value, stride, 1, "stride") ||
        !nn_pair(padding_value, padding, 0, "padding")) RETURN_THROWS();
    int average = mode && zend_string_equals_literal(mode, "average");
    if ((mode && !average && !zend_string_equals_literal(mode, "max")) ||
        padding[0] >= window[0] || padding[1] >= window[1]) {
        CUDA_THROW_INVALID("Pooling mode must be max or average; padding must be smaller than window");
        RETURN_THROWS();
    }
    if (!nn_shape(input, input->shape[1], window, stride, padding, dilation, shape)) RETURN_THROWS();
    if (!cuda_nn_available()) { CUDA_THROW_RUNTIME("%s", cuda_nn_error()); RETURN_THROWS(); }
    tensor_t *result = cuda_tensor_create_empty_dtype(shape, 4, DTYPE_FLOAT32);
    if (!result) RETURN_THROWS();
    nn_result(INTERNAL_FUNCTION_PARAM_PASSTHRU, result, cuda_nn_pool(input->data, result->data, input->shape,
        shape, window, stride, padding, average));
}
ZEND_METHOD(NN, softmax)
{
    zval *x;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_OBJECT_OF_CLASS(x, cuda_array_ce)
    ZEND_PARSE_PARAMETERS_END();
    tensor_t *input = nn_input(x, 4);
    if (!input) RETURN_THROWS();
    if (!cuda_nn_available()) { CUDA_THROW_RUNTIME("%s", cuda_nn_error()); RETURN_THROWS(); }
    tensor_t *result = cuda_tensor_create_empty_dtype(input->shape, 4, DTYPE_FLOAT32);
    if (!result) RETURN_THROWS();
    nn_result(INTERNAL_FUNCTION_PARAM_PASSTHRU, result, cuda_nn_softmax(input->data, result->data, input->shape));
}
static const zend_function_entry nn_methods[] = {
    ZEND_ME(NN, __construct, nn_private_args, ZEND_ACC_PRIVATE)
    ZEND_ME(NN, isAvailable, nn_available_args, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NN, conv2d, nn_conv_args, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NN, pool2d, nn_pool_args, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(NN, softmax, nn_softmax_args, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_FE_END
};
int cuda_nn_init(void)
{
    zend_class_entry entry;
    INIT_NS_CLASS_ENTRY(entry, "Cuda", "NN", nn_methods);
    zend_class_entry *ce = zend_register_internal_class(&entry);
    ce->ce_flags |= ZEND_ACC_FINAL;
    return SUCCESS;
}
