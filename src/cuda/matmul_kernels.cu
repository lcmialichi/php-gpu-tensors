#include "matmul_kernels.h"
#include "launch_config.cuh"
#include "../tensor.h"
#include <cuda_runtime.h>
#include <string.h>
#ifdef HAVE_CUBLAS
#include <cublas_v2.h>
#include <limits.h>
#endif

#define TILE_SIZE 32

struct MatMulParamsND
{
    float *A, *B, *C;
    int shapeA[MAX_DIMS], shapeB[MAX_DIMS], shapeC[MAX_DIMS];
    size_t strideA[MAX_DIMS], strideB[MAX_DIMS], strideC[MAX_DIMS];
    int ndA, ndB, ndC;
    int M, N, K;
    int total_batches;
};

#ifdef HAVE_CUBLAS
struct BlasContext
{
    cublasHandle_t handle;
    int device;

    BlasContext() : handle(nullptr), device(-1) {}
    ~BlasContext() { if (handle) cublasDestroy(handle); }
};

static thread_local BlasContext blas_context;

extern "C" void cuda_blas_shutdown(void)
{
    if (blas_context.handle)
    {
        cublasDestroy(blas_context.handle);
        blas_context.handle = nullptr;
        blas_context.device = -1;
    }
}

static cublasHandle_t get_blas_handle()
{
    int device;
    if (cudaGetDevice(&device) != cudaSuccess)
        return nullptr;
    if (blas_context.handle && blas_context.device != device)
    {
        cublasDestroy(blas_context.handle);
        blas_context.handle = nullptr;
    }
    if (!blas_context.handle)
    {
        if (cublasCreate(&blas_context.handle) != CUBLAS_STATUS_SUCCESS)
            return nullptr;
        blas_context.device = device;
    }
    return blas_context.handle;
}

static int blas_matrix_layout(int rows, int cols, size_t stride_row, size_t stride_col,
                              cublasOperation_t *operation, int *leading_dimension)
{
    if (stride_col == 1 && stride_row == (size_t)cols)
    {
        *operation = CUBLAS_OP_N;
        *leading_dimension = cols;
        return 1;
    }
    if (stride_row == 1 && stride_col == (size_t)rows)
    {
        *operation = CUBLAS_OP_T;
        *leading_dimension = rows;
        return 1;
    }
    return 0;
}

static int blas_matmul_2d(float *a, float *b, float *c,
                          int rows, int inner, int cols,
                          size_t a_row, size_t a_col, size_t b_row, size_t b_col,
                          size_t c_row, size_t c_col, cudaStream_t stream)
{
    if ((double)rows * inner * cols < 500000 || c_col != 1 || c_row != (size_t)cols)
        return 0;
    cublasOperation_t op_a, op_b;
    int lda, ldb;
    if (!blas_matrix_layout(rows, inner, a_row, a_col, &op_a, &lda) ||
        !blas_matrix_layout(inner, cols, b_row, b_col, &op_b, &ldb))
        return 0;

    cublasHandle_t handle = get_blas_handle();
    if (!handle) return 0;
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) return 0;
    cublasStatus_t status;
    if (rows == 1 && cols == 1 && inner >= 4096 && a_col <= INT_MAX && b_row <= INT_MAX)
    {
        if (cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) != CUBLAS_STATUS_SUCCESS)
            return 0;
        status = cublasSdot(handle, inner, a, (int)a_col, b, (int)b_row, c);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    }
    else
    {
        const float alpha = 1.0f, beta = 0.0f;
        status = cublasSgemm(handle, op_b, op_a, cols, rows, inner,
                             &alpha, b, ldb, a, lda, &beta, c, cols);
    }
    if (status == CUBLAS_STATUS_SUCCESS) return 1;
    cublasDestroy(blas_context.handle);
    blas_context.handle = nullptr;
    return 0;
}

static int blas_batch_stride(const int *shape, const size_t *strides, int ndims,
                             const MatMulParamsND *params, size_t *batch_stride)
{
    if (ndims == 2)
    {
        *batch_stride = 0;
        return 1;
    }
    if (ndims != params->ndC)
        return 0;

    int broadcast = 1, dense = 1;
    for (int axis = 0; axis < ndims - 2; axis++)
    {
        broadcast &= shape[axis] == 1;
        dense &= shape[axis] == params->shapeC[axis];
    }
    if (broadcast)
    {
        *batch_stride = 0;
        return 1;
    }
    if (!dense)
        return 0;

    size_t stride = strides[ndims - 3];
    if (stride == 0)
        return 0;
    for (int axis = ndims - 4; axis >= 0; axis--)
    {
        if ((size_t)shape[axis + 1] > SIZE_MAX / stride ||
            strides[axis] != stride * (size_t)shape[axis + 1])
            return 0;
        stride = strides[axis];
    }
    *batch_stride = strides[ndims - 3];
    return 1;
}

static int blas_matmul_batched(const MatMulParamsND *params, cudaStream_t stream)
{
    if (params->ndC < 3 || (double)params->M * params->N * params->K * params->total_batches < 500000)
        return 0;

    cublasOperation_t op_a, op_b;
    int lda, ldb;
    if (!blas_matrix_layout(params->M, params->K,
                            params->strideA[params->ndA - 2], params->strideA[params->ndA - 1], &op_a, &lda) ||
        !blas_matrix_layout(params->K, params->N,
                            params->strideB[params->ndB - 2], params->strideB[params->ndB - 1], &op_b, &ldb))
        return 0;

    size_t stride_a, stride_b;
    size_t stride_c = params->strideC[params->ndC - 3];
    if (!blas_batch_stride(params->shapeA, params->strideA, params->ndA, params, &stride_a) ||
        !blas_batch_stride(params->shapeB, params->strideB, params->ndB, params, &stride_b) ||
        stride_c != (size_t)params->M * params->N ||
        stride_a > LLONG_MAX || stride_b > LLONG_MAX || stride_c > LLONG_MAX)
        return 0;

    cublasHandle_t handle = get_blas_handle();
    if (!handle) return 0;
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasSgemmStridedBatched(handle, op_b, op_a,
        params->N, params->M, params->K,
        &alpha, params->B, ldb, (long long)stride_b,
        params->A, lda, (long long)stride_a,
        &beta, params->C, params->N, (long long)stride_c, params->total_batches);
    if (status == CUBLAS_STATUS_SUCCESS) return 1;
    cublasDestroy(blas_context.handle);
    blas_context.handle = nullptr;
    return 0;
}
#endif

#ifndef HAVE_CUBLAS
extern "C" void cuda_blas_shutdown(void) {}
#endif

static __global__ void matmul_kernel(float *a, float *b, float *c,
                                     int m, int n, int k,
                                     size_t a_stride0, size_t a_stride1,
                                     size_t b_stride0, size_t b_stride1,
                                     size_t c_stride0, size_t c_stride1)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
        {
            size_t a_idx = row * a_stride0 + i * a_stride1;
            size_t b_idx = i * b_stride0 + col * b_stride1;
            sum += a[a_idx] * b[b_idx];
        }

        size_t c_idx = row * c_stride0 + col * c_stride1;
        c[c_idx] = sum;
    }
}

static __global__ void matmul_nd_tiled_kernel(MatMulParamsND params)
{
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int batch_id = blockIdx.z;
    int global_row = blockIdx.y * TILE_SIZE + ty;
    int global_col = blockIdx.x * TILE_SIZE + tx;

    if (batch_id >= params.total_batches)
        return;

    size_t batch_offset_a = 0;
    size_t batch_offset_b = 0;
    size_t batch_offset_c = 0;
    int batch_dims = params.ndC - 2;
    int remaining = batch_id;

    for (int i = batch_dims - 1; i >= 0; i--)
    {
        int coord = remaining % params.shapeC[i];
        remaining /= params.shapeC[i];
        batch_offset_c += (size_t)coord * params.strideC[i];
        if (params.ndA > i + 2 && params.shapeA[i] > 1)
            batch_offset_a += (size_t)coord * params.strideA[i];
        if (params.ndB > i + 2 && params.shapeB[i] > 1)
            batch_offset_b += (size_t)coord * params.strideB[i];
    }

    float sum = 0.0f;
    int stride_a_row = params.strideA[params.ndA - 2];
    int stride_a_col = params.strideA[params.ndA - 1];
    int stride_b_row = params.strideB[params.ndB - 2];
    int stride_b_col = params.strideB[params.ndB - 1];

    for (int k_offset = 0; k_offset < params.K; k_offset += TILE_SIZE)
    {
        int a_col = k_offset + tx;
        int b_row = k_offset + ty;

        tile_a[ty][tx] = (global_row < params.M && a_col < params.K)
            ? params.A[batch_offset_a + global_row * stride_a_row + a_col * stride_a_col] : 0.0f;
        tile_b[ty][tx] = (b_row < params.K && global_col < params.N)
            ? params.B[batch_offset_b + b_row * stride_b_row + global_col * stride_b_col] : 0.0f;

        __syncthreads();
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++)
            sum += tile_a[ty][i] * tile_b[i][tx];
        __syncthreads();
    }

    if (global_row < params.M && global_col < params.N)
    {
        size_t output_index = batch_offset_c + global_row * params.strideC[params.ndC - 2]
                                           + global_col * params.strideC[params.ndC - 1];
        params.C[output_index] = sum;
    }
}

extern "C" int cuda_batched_matmul_nd_launcher(
    float *a, float *b, float *c,
    int *shape_a, size_t *stride_a, int nd_a,
    int *shape_b, size_t *stride_b, int nd_b,
    int *shape_c, size_t *stride_c, int nd_c, cudaStream_t stream)
{
    if (nd_a < 2 || nd_b < 2 || nd_c < 2 ||
        nd_a > MAX_DIMS || nd_b > MAX_DIMS || nd_c > MAX_DIMS)
        return 0;

    MatMulParamsND params = {};
    memcpy(params.shapeA, shape_a, nd_a * sizeof(int));
    memcpy(params.strideA, stride_a, nd_a * sizeof(size_t));
    memcpy(params.shapeB, shape_b, nd_b * sizeof(int));
    memcpy(params.strideB, stride_b, nd_b * sizeof(size_t));
    memcpy(params.shapeC, shape_c, nd_c * sizeof(int));
    memcpy(params.strideC, stride_c, nd_c * sizeof(size_t));

    if (params.shapeA[nd_a - 1] != params.shapeB[nd_b - 2])
        return 0;

    int rows = params.shapeA[nd_a - 2];
    int cols = params.shapeB[nd_b - 1];
    int inner = params.shapeA[nd_a - 1];
    if (rows < 0 || cols < 0 || inner < 0 ||
        params.shapeC[nd_c - 2] != rows || params.shapeC[nd_c - 1] != cols)
        return 0;
    if (!rows || !cols) return 1;

    int batches = 1;
    for (int i = 0; i < nd_c - 2; i++)
    {
        if (!params.shapeC[i]) return 1;
        if (params.shapeC[i] < 0 || batches > 65535 / params.shapeC[i])
            return 0;
        batches *= params.shapeC[i];
    }

    params.A = a;
    params.B = b;
    params.C = c;
    params.ndA = nd_a;
    params.ndB = nd_b;
    params.ndC = nd_c;
    params.M = rows;
    params.N = cols;
    params.K = inner;
    params.total_batches = batches;

#ifdef HAVE_CUBLAS
    if (inner && blas_matmul_batched(&params, stream))
        return 1;
#endif

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid = cuda_grid_2d(cols, rows, TILE_SIZE, TILE_SIZE, batches);
    matmul_nd_tiled_kernel<<<grid, block, 0, stream>>>(params);
    return cuda_launch_status() == cudaSuccess;
}

extern "C" int cuda_matmul_launcher(float *a, float *b, float *c,
                                     int m, int n, int k,
                                     size_t a_stride0, size_t a_stride1,
                                     size_t b_stride0, size_t b_stride1,
                                     size_t c_stride0, size_t c_stride1, cudaStream_t stream)
{
    if (m < 0 || n < 0 || k < 0)
        return 0;
    if (!m || !k) return 1;

#ifdef HAVE_CUBLAS
    if (n && blas_matmul_2d(a, b, c, m, n, k,
                       a_stride0, a_stride1, b_stride0, b_stride1,
                       c_stride0, c_stride1, stream))
        return 1;
#endif

    dim3 block(32, 32);
    dim3 grid = cuda_grid_2d(k, m, 32, 32);
    matmul_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k,
                                   a_stride0, a_stride1, b_stride0, b_stride1,
                                   c_stride0, c_stride1);
    return cuda_launch_status() == cudaSuccess;
}