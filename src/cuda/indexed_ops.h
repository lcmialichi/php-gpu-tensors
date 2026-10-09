#ifndef CUDA_INDEXED_OPS_H
#define CUDA_INDEXED_OPS_H

#include <cuda_runtime.h>
#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cuda_validate_indices(const int *indices, const int *shape,
                                  const size_t *strides, int ndims, int axis,
                                  int axis_size, size_t total);
cudaError_t cuda_launch_gather(const void *input, const int *indices, void *output,
                               dtype_t dtype, const int *input_shape,
                               const size_t *input_strides, const int *index_shape,
                               const size_t *index_strides, int ndims, int axis,
                               size_t total);
cudaError_t cuda_launch_scatter_copy(const void *input, void *output, dtype_t dtype,
                                     const int *shape, const size_t *strides,
                                     int ndims, size_t total);
cudaError_t cuda_launch_scatter_add(const int *indices, const void *updates,
                                    void *output, dtype_t dtype,
                                    const int *index_shape, const size_t *output_strides,
                                    const size_t *index_strides, const size_t *update_strides,
                                    int ndims, int axis,
                                    int axis_size, size_t total);

#ifdef __cplusplus
}
#endif

#endif
