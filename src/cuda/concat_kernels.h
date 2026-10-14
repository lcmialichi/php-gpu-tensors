#ifndef CONCAT_KERNELS_H
#define CONCAT_KERNELS_H

#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

int launch_concat_kernel_host(tensor_t **input_tensors, int num_tensors,
                              tensor_t *output_tensor, int axis);

#ifdef __cplusplus
}
#endif

#endif