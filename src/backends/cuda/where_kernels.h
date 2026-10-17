#ifndef WHERE_KERNELS_H
#define WHERE_KERNELS_H

#include "../../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t launch_where_kernel(const tensor_t *condition, const tensor_t *on_true,
                                const tensor_t *on_false, tensor_t *output,
                                const size_t *condition_strides, const size_t *true_strides,
                                const size_t *false_strides, int fast_path);

#ifdef __cplusplus
}
#endif

#endif