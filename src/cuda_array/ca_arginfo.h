#ifndef CUDA_ARRAY_ARGINFO_H
#define CUDA_ARRAY_ARGINFO_H

#include "php.h"
#include "../api_arginfo.h"

#define CA_CLASS_NAME "PhpTensors\\CudaArray"

static zend_function_entry cuda_array_methods[] = {
    ZEND_ME(CudaArray, __construct, arginfo_api_PhpTensors_CudaArray___construct, ZEND_ACC_PUBLIC | ZEND_ACC_CTOR)
    ZEND_ME(CudaArray, fromBuffer, arginfo_api_PhpTensors_CudaArray_fromBuffer, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, fromFlatArray, arginfo_api_PhpTensors_CudaArray_fromFlatArray, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, fromFile, arginfo_api_PhpTensors_CudaArray_fromFile, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, fromNpy, arginfo_api_PhpTensors_CudaArray_fromNpy, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, where, arginfo_api_PhpTensors_CudaArray_where, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(CudaArray, __serialize, arginfo_api_PhpTensors_CudaArray___serialize, ZEND_ACC_PUBLIC)
            ZEND_ME(CudaArray, __unserialize, arginfo_api_PhpTensors_CudaArray___unserialize, ZEND_ACC_PUBLIC)
        ZEND_ME(CudaArray, __invoke, arginfo_api_PhpTensors_CudaArray___invoke, ZEND_ACC_PUBLIC)
        ZEND_ME(CudaArray, slice, arginfo_api_PhpTensors_CudaArray_slice, ZEND_ACC_PUBLIC)
            ZEND_ME(CudaArray, __debugInfo, arginfo_api_PhpTensors_CudaArray___debugInfo, ZEND_ACC_PUBLIC)

                ZEND_ME(CudaArray, concat, arginfo_api_PhpTensors_CudaArray_concat, ZEND_ACC_PUBLIC)
                    ZEND_ME(CudaArray, multiply, arginfo_api_PhpTensors_CudaArray_multiply, ZEND_ACC_PUBLIC)
                        ZEND_ME(CudaArray, divide, arginfo_api_PhpTensors_CudaArray_divide, ZEND_ACC_PUBLIC)
                            ZEND_ME(CudaArray, add, arginfo_api_PhpTensors_CudaArray_add, ZEND_ACC_PUBLIC)
                                ZEND_ME(CudaArray, subtract, arginfo_api_PhpTensors_CudaArray_subtract, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, maximum, arginfo_api_PhpTensors_CudaArray_maximum, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, minimum, arginfo_api_PhpTensors_CudaArray_minimum, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, clamp, arginfo_api_PhpTensors_CudaArray_clamp, ZEND_ACC_PUBLIC)
                                    ZEND_ME(CudaArray, matmul, arginfo_api_PhpTensors_CudaArray_matmul, ZEND_ACC_PUBLIC)
                                        ZEND_ME(CudaArray, transpose, arginfo_api_PhpTensors_CudaArray_transpose, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, copy, arginfo_api_PhpTensors_CudaArray_copy, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, swapaxes, arginfo_api_PhpTensors_CudaArray_swapaxes, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, moveaxis, arginfo_api_PhpTensors_CudaArray_moveaxis, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, diagonal, arginfo_api_PhpTensors_CudaArray_diagonal, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, repeat, arginfo_api_PhpTensors_CudaArray_repeat, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, tile, arginfo_api_PhpTensors_CudaArray_tile, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, flip, arginfo_api_PhpTensors_CudaArray_flip, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, roll, arginfo_api_PhpTensors_CudaArray_roll, ZEND_ACC_PUBLIC)
                                            ZEND_ME(CudaArray, power, arginfo_api_PhpTensors_CudaArray_power, ZEND_ACC_PUBLIC)
                                                ZEND_ME(CudaArray, sqrt, arginfo_api_PhpTensors_CudaArray_sqrt, ZEND_ACC_PUBLIC)
                                                    ZEND_ME(CudaArray, exp, arginfo_api_PhpTensors_CudaArray_exp, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, expm1, arginfo_api_PhpTensors_CudaArray_expm1, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, exp2, arginfo_api_PhpTensors_CudaArray_exp2, ZEND_ACC_PUBLIC)
                                                        ZEND_ME(CudaArray, log, arginfo_api_PhpTensors_CudaArray_log, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, log1p, arginfo_api_PhpTensors_CudaArray_log1p, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, log2, arginfo_api_PhpTensors_CudaArray_log2, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, log10, arginfo_api_PhpTensors_CudaArray_log10, ZEND_ACC_PUBLIC)
                                                            ZEND_ME(CudaArray, sin, arginfo_api_PhpTensors_CudaArray_sin, ZEND_ACC_PUBLIC)
                                                                ZEND_ME(CudaArray, cos, arginfo_api_PhpTensors_CudaArray_cos, ZEND_ACC_PUBLIC)
                                                                    ZEND_ME(CudaArray, tan, arginfo_api_PhpTensors_CudaArray_tan, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, asin, arginfo_api_PhpTensors_CudaArray_asin, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, acos, arginfo_api_PhpTensors_CudaArray_acos, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, atan, arginfo_api_PhpTensors_CudaArray_atan, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, sinh, arginfo_api_PhpTensors_CudaArray_sinh, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, cosh, arginfo_api_PhpTensors_CudaArray_cosh, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, tanh, arginfo_api_PhpTensors_CudaArray_tanh, ZEND_ACC_PUBLIC)
                                                                        ZEND_ME(CudaArray, abs, arginfo_api_PhpTensors_CudaArray_abs, ZEND_ACC_PUBLIC)
                                                                            ZEND_ME(CudaArray, neg, arginfo_api_PhpTensors_CudaArray_neg, ZEND_ACC_PUBLIC)
                                                                                ZEND_ME(CudaArray, floor, arginfo_api_PhpTensors_CudaArray_floor, ZEND_ACC_PUBLIC)
                                                                                    ZEND_ME(CudaArray, ceil, arginfo_api_PhpTensors_CudaArray_ceil, ZEND_ACC_PUBLIC)
                                                                                        ZEND_ME(CudaArray, round, arginfo_api_PhpTensors_CudaArray_round, ZEND_ACC_PUBLIC)
                                                                                            ZEND_ME(CudaArray, gt, arginfo_api_PhpTensors_CudaArray_gt, ZEND_ACC_PUBLIC)
                                                                                                ZEND_ME(CudaArray, lt, arginfo_api_PhpTensors_CudaArray_lt, ZEND_ACC_PUBLIC)
                                                                                                    ZEND_ME(CudaArray, eq, arginfo_api_PhpTensors_CudaArray_eq, ZEND_ACC_PUBLIC)
                                                                                                        ZEND_ME(CudaArray, ne, arginfo_api_PhpTensors_CudaArray_ne, ZEND_ACC_PUBLIC)
                                                                                                            ZEND_ME(CudaArray, ge, arginfo_api_PhpTensors_CudaArray_ge, ZEND_ACC_PUBLIC)
                                                                                                                ZEND_ME(CudaArray, le, arginfo_api_PhpTensors_CudaArray_le, ZEND_ACC_PUBLIC)

                                                                                                                    ZEND_ME(CudaArray, sum, arginfo_api_PhpTensors_CudaArray_sum, ZEND_ACC_PUBLIC)
                                                                                                                        ZEND_ME(CudaArray, mean, arginfo_api_PhpTensors_CudaArray_mean, ZEND_ACC_PUBLIC)
                                                                                                                            ZEND_ME(CudaArray, max, arginfo_api_PhpTensors_CudaArray_max, ZEND_ACC_PUBLIC)
                                                                                                                                ZEND_ME(CudaArray, min, arginfo_api_PhpTensors_CudaArray_min, ZEND_ACC_PUBLIC)
                                                                                                                                    ZEND_ME(CudaArray, prod, arginfo_api_PhpTensors_CudaArray_prod, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, cumsum, arginfo_api_PhpTensors_CudaArray_cumsum, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, cumprod, arginfo_api_PhpTensors_CudaArray_cumprod, ZEND_ACC_PUBLIC)
                                                                                                                                        ZEND_ME(CudaArray, argMax, arginfo_api_PhpTensors_CudaArray_argMax, ZEND_ACC_PUBLIC)
                                                                                                                                            ZEND_ME(CudaArray, argMin, arginfo_api_PhpTensors_CudaArray_argMin, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, all, arginfo_api_PhpTensors_CudaArray_all, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, any, arginfo_api_PhpTensors_CudaArray_any, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, var, arginfo_api_PhpTensors_CudaArray_var, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, std, arginfo_api_PhpTensors_CudaArray_std, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, softmax, arginfo_api_PhpTensors_CudaArray_softmax, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, layerNorm, arginfo_api_PhpTensors_CudaArray_layerNorm, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, item, arginfo_api_PhpTensors_CudaArray_item, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, gather, arginfo_api_PhpTensors_CudaArray_gather, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, scatterAdd, arginfo_api_PhpTensors_CudaArray_scatterAdd, ZEND_ACC_PUBLIC)

                                                                                                                                                ZEND_ME(CudaArray, getShape, arginfo_api_PhpTensors_CudaArray_getShape, ZEND_ACC_PUBLIC)
                                                                                                                                                    ZEND_ME(CudaArray, getStrides, arginfo_api_PhpTensors_CudaArray_getStrides, ZEND_ACC_PUBLIC)
                                                                                                                                                        ZEND_ME(CudaArray, getNdims, arginfo_api_PhpTensors_CudaArray_getNdims, ZEND_ACC_PUBLIC)
                                                                                                                                                            ZEND_ME(CudaArray, getSize, arginfo_api_PhpTensors_CudaArray_getSize, ZEND_ACC_PUBLIC)

                                                                                                                                                                ZEND_ME(CudaArray, toArray, arginfo_api_PhpTensors_CudaArray_toArray, ZEND_ACC_PUBLIC)
                                                                                                                                                                    ZEND_ME(CudaArray, toHost, arginfo_api_PhpTensors_CudaArray_toHost, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, toBuffer, arginfo_api_PhpTensors_CudaArray_toBuffer, ZEND_ACC_PUBLIC)
                                                                                                                                                                        ZEND_ME(CudaArray, reshape, arginfo_api_PhpTensors_CudaArray_reshape, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, squeeze, arginfo_api_PhpTensors_CudaArray_squeeze, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, unsqueeze, arginfo_api_PhpTensors_CudaArray_unsqueeze, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaArray, broadcastTo, arginfo_api_PhpTensors_CudaArray_broadcastTo, ZEND_ACC_PUBLIC)
                                                                                                                                                                            ZEND_ME(CudaArray, flatten, arginfo_api_PhpTensors_CudaArray_flatten, ZEND_ACC_PUBLIC)

                                                                                                                                                                                ZEND_ME(CudaArray, zeros, arginfo_api_PhpTensors_CudaArray_zeros, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
                                                                                                                                                                                    ZEND_ME(CudaArray, ones, arginfo_api_PhpTensors_CudaArray_ones, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
                                                                                                                                                                                        ZEND_ME(CudaArray, full, arginfo_api_PhpTensors_CudaArray_full, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
                                                                                                                                                                                            ZEND_ME(CudaArray, rand, arginfo_api_PhpTensors_CudaArray_rand, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, zerosLike, arginfo_api_PhpTensors_CudaArray_zerosLike, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, onesLike, arginfo_api_PhpTensors_CudaArray_onesLike, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, fullLike, arginfo_api_PhpTensors_CudaArray_fullLike, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, eye, arginfo_api_PhpTensors_CudaArray_eye, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, identity, arginfo_api_PhpTensors_CudaArray_identity, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, randn, arginfo_api_PhpTensors_CudaArray_randn, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, arange, arginfo_api_PhpTensors_CudaArray_arange, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaArray, linspace, arginfo_api_PhpTensors_CudaArray_linspace, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

                                                                                                                                                                                                ZEND_ME(CudaArray, astype, arginfo_api_PhpTensors_CudaArray_astype, ZEND_ACC_PUBLIC)
                                                                                                                                                                                                    ZEND_ME(CudaArray, dtype, arginfo_api_PhpTensors_CudaArray_dtype, ZEND_ACC_PUBLIC)
                                                                                                                                                                                                    ZEND_ME(CudaArray, getBackend, arginfo_api_PhpTensors_CudaArray_getBackend, ZEND_ACC_PUBLIC)
                                                                                                                                                                                                        ZEND_ME(CudaArray, requiresGrad, arginfo_api_PhpTensors_CudaArray_requiresGrad, ZEND_ACC_PUBLIC)
                                                                                                                                                                                                            ZEND_ME(CudaArray, backward, arginfo_api_PhpTensors_CudaArray_backward, ZEND_ACC_PUBLIC)
                                                                                                                                                                                                                ZEND_ME(CudaArray, grad, arginfo_api_PhpTensors_CudaArray_grad, ZEND_ACC_PUBLIC)
                                                                                                                                                                                                                    ZEND_ME(CudaArray, zeroGrad, arginfo_api_PhpTensors_CudaArray_zeroGrad, ZEND_ACC_PUBLIC)
                                                                                                                                                                                                                        ZEND_ME(CudaArray, detach, arginfo_api_PhpTensors_CudaArray_detach, ZEND_ACC_PUBLIC)

                                                                                                                                                                                                        PHP_FE_END};

static zend_class_entry *register_cuda_array_class(void)
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, CA_CLASS_NAME, cuda_array_methods);
    cuda_array_ce = zend_register_internal_class(&ce);
    cuda_array_ce->ce_flags |= ZEND_ACC_FINAL;

    return cuda_array_ce;
}

#endif