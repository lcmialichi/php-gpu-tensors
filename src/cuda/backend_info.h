#ifndef CUDA_BACKEND_INFO_H
#define CUDA_BACKEND_INFO_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int cublas, cublas_lt, cudnn;
    unsigned long long blas_calls, lt_calls, builtin_matmul_calls;
    unsigned long long cub_calls, axis_calls, generic_reduce_calls;
    const char *last_matmul;
    const char *precision;
    int last_blas_status;
} cuda_backend_info;
void cuda_blas_info(cuda_backend_info *info);
void cuda_reduction_info(cuda_backend_info *info);
#ifdef __cplusplus
}
#endif
#endif
