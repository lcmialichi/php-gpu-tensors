#ifndef MATMUL_KERNELS_H
#define MATMUL_KERNELS_H

#include <stddef.h>
#include <cuda_runtime_api.h>
#include "../../data_types.h"

enum {
    CUDA_MATMUL_PRECISION_FP32_STRICT = 0,
    CUDA_MATMUL_PRECISION_TF32 = 1
};

#define CUDA_MATMUL_TUNING_MAX_CANDIDATES 8

typedef struct {
    int algorithm_id;
    size_t workspace_bytes;
    double median_ms;
} cuda_matmul_tuning_candidate;

typedef struct {
    int candidate_count;
    int selected_candidate;
    cuda_matmul_tuning_candidate candidates[CUDA_MATMUL_TUNING_MAX_CANDIDATES];
} cuda_matmul_tuning_result;

#ifdef __cplusplus
extern "C" {
#endif

void cuda_blas_shutdown(void);
int cuda_blas_set_precision(int precision);
const char *cuda_blas_precision_name(void);
int cuda_blas_tune_matmul(float *a, float *b, float *c,
                          int rows, int inner, int cols,
                          size_t a_stride0, size_t a_stride1,
                          size_t b_stride0, size_t b_stride1,
                          cudaStream_t stream, cuda_matmul_tuning_result *result);

int cuda_matmul_launcher(float *a, float *b, float *c,
                        int m, int n, int k,
                        size_t a_stride0, size_t a_stride1,
                        size_t b_stride0, size_t b_stride1,
                        size_t c_stride0, size_t c_stride1, cudaStream_t stream);

/* Shape and stride pointers refer to host metadata. */
int cuda_batched_matmul_nd_launcher(
    float *a, float *b, float *c,
    int *shape_a, size_t *stride_a, int nd_a,
    int *shape_b, size_t *stride_b, int nd_b,
    int *shape_c, size_t *stride_c, int nd_c, cudaStream_t stream);

int cuda_low_precision_matmul_nd_launcher(
    void *a, void *b, void *c, dtype_t dtype,
    int *shape_a, size_t *stride_a, int nd_a,
    int *shape_b, size_t *stride_b, int nd_b,
    int *shape_c, size_t *stride_c, int nd_c, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif