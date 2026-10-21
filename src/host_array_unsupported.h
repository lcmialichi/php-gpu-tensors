#ifndef PHPTENSORS_HOST_ARRAY_UNSUPPORTED_H
#define PHPTENSORS_HOST_ARRAY_UNSUPPORTED_H

#include "php.h"
#include "api_arginfo.h"

ZEND_FUNCTION(php_tensors_host_array_unsupported);

#define HOST_ARRAY_UNSUPPORTED_METHOD(name, arginfo, flags) \
    ZEND_NAMED_ME(name, zif_php_tensors_host_array_unsupported, arginfo, flags)

#define HOST_ARRAY_UNSUPPORTED_METHODS \
    HOST_ARRAY_UNSUPPORTED_METHOD(fromFile, arginfo_api_PhpTensors_HostArray_fromFile, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(fromNpy, arginfo_api_PhpTensors_HostArray_fromNpy, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(__invoke, arginfo_api_PhpTensors_HostArray___invoke, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(concat, arginfo_api_PhpTensors_HostArray_concat, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(maximum, arginfo_api_PhpTensors_HostArray_maximum, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(minimum, arginfo_api_PhpTensors_HostArray_minimum, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(clamp, arginfo_api_PhpTensors_HostArray_clamp, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(swapaxes, arginfo_api_PhpTensors_HostArray_swapaxes, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(moveaxis, arginfo_api_PhpTensors_HostArray_moveaxis, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(diagonal, arginfo_api_PhpTensors_HostArray_diagonal, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(repeat, arginfo_api_PhpTensors_HostArray_repeat, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(tile, arginfo_api_PhpTensors_HostArray_tile, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(flip, arginfo_api_PhpTensors_HostArray_flip, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(roll, arginfo_api_PhpTensors_HostArray_roll, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(power, arginfo_api_PhpTensors_HostArray_power, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(prod, arginfo_api_PhpTensors_HostArray_prod, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(cumsum, arginfo_api_PhpTensors_HostArray_cumsum, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(cumprod, arginfo_api_PhpTensors_HostArray_cumprod, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(argMax, arginfo_api_PhpTensors_HostArray_argMax, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(argMin, arginfo_api_PhpTensors_HostArray_argMin, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(all, arginfo_api_PhpTensors_HostArray_all, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(any, arginfo_api_PhpTensors_HostArray_any, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(var, arginfo_api_PhpTensors_HostArray_var, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(std, arginfo_api_PhpTensors_HostArray_std, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(softmax, arginfo_api_PhpTensors_HostArray_softmax, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(layerNorm, arginfo_api_PhpTensors_HostArray_layerNorm, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(gather, arginfo_api_PhpTensors_HostArray_gather, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(scatterAdd, arginfo_api_PhpTensors_HostArray_scatterAdd, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(squeeze, arginfo_api_PhpTensors_HostArray_squeeze, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(unsqueeze, arginfo_api_PhpTensors_HostArray_unsqueeze, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(flatten, arginfo_api_PhpTensors_HostArray_flatten, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(eye, arginfo_api_PhpTensors_HostArray_eye, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(identity, arginfo_api_PhpTensors_HostArray_identity, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(arange, arginfo_api_PhpTensors_HostArray_arange, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(linspace, arginfo_api_PhpTensors_HostArray_linspace, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(rand, arginfo_api_PhpTensors_HostArray_rand, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(randn, arginfo_api_PhpTensors_HostArray_randn, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(requiresGrad, arginfo_api_PhpTensors_HostArray_requiresGrad, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(backward, arginfo_api_PhpTensors_HostArray_backward, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(grad, arginfo_api_PhpTensors_HostArray_grad, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(zeroGrad, arginfo_api_PhpTensors_HostArray_zeroGrad, ZEND_ACC_PUBLIC) \
    HOST_ARRAY_UNSUPPORTED_METHOD(detach, arginfo_api_PhpTensors_HostArray_detach, ZEND_ACC_PUBLIC)

#endif
