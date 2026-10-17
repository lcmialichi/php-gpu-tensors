#include "matmul_kernels.h"
#include "launch_config.cuh"
#include "../../tensor.h"
#include <cuda_runtime.h>
#include <string.h>
#include <stdint.h>
#include <algorithm>
#include <stdio.h>
#include <unistd.h>
#include "backend_info.h"
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#ifdef HAVE_CUBLASLT
#include <cublasLt.h>
#endif
#ifdef HAVE_CUBLAS
#include <cublas_v2.h>
#include <limits.h>
#endif

#define TILE_SIZE 32
static constexpr double MATMUL_BLAS_MIN_WORK = 200000.0;
static constexpr double MATMUL_LT_MIN_WORK = 10000000.0;
static constexpr double MATMUL_LT_MAX_STRICT_WORK = 16000000000.0;
static constexpr int MATMUL_SKINNY_MIN_INNER = 256;
static constexpr size_t MATMUL_SKINNY_MAX_OUTPUTS = 1024;
static thread_local cuda_backend_info backend_info = {};
static thread_local int matmul_precision = CUDA_MATMUL_PRECISION_FP32_STRICT;

extern "C" int cuda_blas_set_precision(int precision)
{
    if (precision != CUDA_MATMUL_PRECISION_FP32_STRICT &&
        precision != CUDA_MATMUL_PRECISION_TF32)
        return 0;
    matmul_precision = precision;
    return 1;
}

extern "C" const char *cuda_blas_precision_name(void)
{
    return matmul_precision == CUDA_MATMUL_PRECISION_TF32 ? "tf32" : "fp32-strict";
}

extern "C" void cuda_blas_info(cuda_backend_info *info)
{
    info->blas_calls = backend_info.blas_calls;
    info->lt_calls = backend_info.lt_calls;
    info->builtin_matmul_calls = backend_info.builtin_matmul_calls;
    info->last_matmul = backend_info.last_matmul ? backend_info.last_matmul : "none";
    info->last_blas_status = backend_info.last_blas_status;
    info->precision = cuda_blas_precision_name();
#ifdef HAVE_CUBLAS
    info->cublas = 1;
#endif
#ifdef HAVE_CUBLASLT
    info->cublas_lt = 1;
#endif
}

struct MatMulParamsND
{
    float *A, *B, *C;
    int shapeA[MAX_DIMS], shapeB[MAX_DIMS], shapeC[MAX_DIMS];
    size_t strideA[MAX_DIMS], strideB[MAX_DIMS], strideC[MAX_DIMS];
    int ndA, ndB, ndC;
    int M, N, K;
    int total_batches;
};

template <typename T>
struct LowMatMulParams
{
    const T *A, *B;
    T *C;
    int shapeA[MAX_DIMS], shapeB[MAX_DIMS], shapeC[MAX_DIMS];
    size_t strideA[MAX_DIMS], strideB[MAX_DIMS], strideC[MAX_DIMS];
    int ndA, ndB, ndC;
    int M, N, K, total_batches;
};

template <typename T>
__device__ float low_precision_to_float(T value);

template <>
__device__ float low_precision_to_float<__half>(__half value)
{
    return __half2float(value);
}

template <>
__device__ float low_precision_to_float<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T>
__device__ T low_precision_from_float(float value);

template <>
__device__ __half low_precision_from_float<__half>(float value)
{
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 low_precision_from_float<__nv_bfloat16>(float value)
{
    return __float2bfloat16_rn(value);
}

#ifdef HAVE_CUBLAS
struct BlasContext
{
    cublasHandle_t handle;
    int device;
    int precision;

    BlasContext() : handle(nullptr), device(-1), precision(-1) {}
};

static thread_local BlasContext blas_context;

#ifdef HAVE_CUBLASLT
struct LtPlan {
    int rows, inner, cols, lda, ldb;
    int precision;
    uint32_t alignment_a, alignment_b, alignment_c;
    cublasOperation_t op_a, op_b;
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulAlgo_t algorithm;
    cublasLtMatmulHeuristicResult_t candidates[CUDA_MATMUL_TUNING_MAX_CANDIDATES];
    int candidate_count, selected_candidate, tuned;
};
static thread_local cublasLtHandle_t lt_handle = nullptr;
static thread_local std::vector<LtPlan> lt_plans;

static void lt_shutdown()
{
    for (auto &plan : lt_plans) {
        cublasLtMatmulDescDestroy(plan.operation);
        cublasLtMatrixLayoutDestroy(plan.a);
        cublasLtMatrixLayoutDestroy(plan.b);
        cublasLtMatrixLayoutDestroy(plan.c);
    }
    lt_plans.clear();
    if (lt_handle) cublasLtDestroy(lt_handle);
    lt_handle = nullptr;
}

static int lt_plan_for(float *a, float *b, float *c, int rows, int inner, int cols,
                       int lda, int ldb, cublasOperation_t op_a, cublasOperation_t op_b,
                       int create, LtPlan **selected)
{
    auto pointer_alignment = [](const void *pointer) -> uint32_t {
        uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
        return address % 16 == 0 ? 16 : (address % 8 == 0 ? 8 : 4);
    };
    uint32_t alignment_a = pointer_alignment(b), alignment_b = pointer_alignment(a), alignment_c = pointer_alignment(c);
    if (!lt_handle) {
        if (!create) return 0;
        cublasStatus_t status = cublasLtCreate(&lt_handle);
        if (status != CUBLAS_STATUS_SUCCESS) { backend_info.last_blas_status = status; return -1; }
    }
    *selected = nullptr;
    for (auto &plan : lt_plans)
        if (plan.rows == rows && plan.inner == inner && plan.cols == cols &&
            plan.lda == lda && plan.ldb == ldb && plan.op_a == op_a && plan.op_b == op_b &&
            plan.precision == matmul_precision &&
            plan.alignment_a == alignment_a && plan.alignment_b == alignment_b && plan.alignment_c == alignment_c) {
            *selected = &plan;
            break;
        }
    if (*selected) return 1;
    if (!create || lt_plans.size() >= 64) return 0;
    {
        LtPlan plan = {};
        plan.rows = rows; plan.inner = inner; plan.cols = cols;
        plan.lda = lda; plan.ldb = ldb; plan.op_a = op_a; plan.op_b = op_b;
        plan.precision = matmul_precision;
        plan.alignment_a = alignment_a; plan.alignment_b = alignment_b; plan.alignment_c = alignment_c;
        cublasLtMatmulPreference_t preference = nullptr;
        cublasComputeType_t compute_type = matmul_precision == CUDA_MATMUL_PRECISION_TF32
            ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F_PEDANTIC;
        cublasStatus_t status = cublasLtMatmulDescCreate(&plan.operation, compute_type, CUDA_R_32F);
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_TRANSA, &op_b, sizeof(op_b));
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_TRANSB, &op_a, sizeof(op_a));
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_32F, op_b == CUBLAS_OP_N ? cols : inner,
                                                op_b == CUBLAS_OP_N ? inner : cols, ldb);
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_32F, op_a == CUBLAS_OP_N ? inner : rows,
                                                op_a == CUBLAS_OP_N ? rows : inner, lda);
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_32F, cols, rows, cols);
        if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulPreferenceCreate(&preference);
        /* No shared scratch: concurrent Fusion streams may replay the same cached algorithm. */
        size_t workspace = 0;
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace));
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignment_a, sizeof(alignment_a));
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignment_b, sizeof(alignment_b));
        for (auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
            if (status == CUBLAS_STATUS_SUCCESS)
                status = cublasLtMatmulPreferenceSetAttribute(preference, attribute, &alignment_c, sizeof(alignment_c));
        int count = 0;
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatmulAlgoGetHeuristic(lt_handle, plan.operation, plan.a, plan.b, plan.c, plan.c,
                                                    preference, CUDA_MATMUL_TUNING_MAX_CANDIDATES,
                                                    plan.candidates, &count);
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (status != CUBLAS_STATUS_SUCCESS || !count) {
            if (plan.operation) cublasLtMatmulDescDestroy(plan.operation);
            if (plan.a) cublasLtMatrixLayoutDestroy(plan.a);
            if (plan.b) cublasLtMatrixLayoutDestroy(plan.b);
            if (plan.c) cublasLtMatrixLayoutDestroy(plan.c);
            if (status == CUBLAS_STATUS_NOT_SUPPORTED || (status == CUBLAS_STATUS_SUCCESS && !count)) return 0;
            backend_info.last_blas_status = status;
            return -1;
        }
        int valid_count = 0;
        for (int i = 0; i < count; i++)
            if (plan.candidates[i].state == CUBLAS_STATUS_SUCCESS)
                plan.candidates[valid_count++] = plan.candidates[i];
        if (!valid_count) {
            if (plan.operation) cublasLtMatmulDescDestroy(plan.operation);
            if (plan.a) cublasLtMatrixLayoutDestroy(plan.a);
            if (plan.b) cublasLtMatrixLayoutDestroy(plan.b);
            if (plan.c) cublasLtMatrixLayoutDestroy(plan.c);
            return 0;
        }
        plan.candidate_count = valid_count;
        plan.algorithm = plan.candidates[0].algo;
        lt_plans.push_back(plan);
        *selected = &lt_plans.back();
    }
    return 1;
}

static int lt_matmul(float *a, float *b, float *c, int rows, int inner, int cols,
                      int lda, int ldb, cublasOperation_t op_a, cublasOperation_t op_b,
                      cudaStream_t stream, int create, int tuned_only)
{
    LtPlan *selected = nullptr;
    int plan_status = lt_plan_for(a, b, c, rows, inner, cols, lda, ldb, op_a, op_b,
                                  create, &selected);
    if (plan_status <= 0) return plan_status;
    if (tuned_only && !selected->tuned) return 0;
    const float alpha = 1, beta = 0;
    cublasStatus_t status = cublasLtMatmul(lt_handle, selected->operation,
        &alpha, b, selected->a, a, selected->b, &beta, c, selected->c, c, selected->c,
        &selected->algorithm, nullptr, 0, stream);
    backend_info.last_blas_status = status;
    if (status != CUBLAS_STATUS_SUCCESS) return -1;
    backend_info.lt_calls++;
    backend_info.last_matmul = "cublasLt";
    return 1;
}
#endif

extern "C" void cuda_blas_shutdown(void)
{
#ifdef HAVE_CUBLASLT
    lt_shutdown();
#endif
    if (blas_context.handle)
    {
        cublasDestroy(blas_context.handle);
        blas_context.handle = nullptr;
        blas_context.device = -1;
        blas_context.precision = -1;
    }
    backend_info = {};
}

static cublasHandle_t get_blas_handle()
{
    int device;
    if (cudaGetDevice(&device) != cudaSuccess)
        return nullptr;
    if (blas_context.handle && blas_context.device != device)
    {
        cuda_blas_shutdown();
    }
    if (!blas_context.handle)
    {
        cublasStatus_t status = cublasCreate(&blas_context.handle);
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasSetMathMode(blas_context.handle,
                matmul_precision == CUDA_MATMUL_PRECISION_TF32
                    ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH);
        backend_info.last_blas_status = status;
        if (status != CUBLAS_STATUS_SUCCESS) {
            if (blas_context.handle) cublasDestroy(blas_context.handle);
            blas_context.handle = nullptr;
            return nullptr;
        }
        blas_context.device = device;
        blas_context.precision = matmul_precision;
    }
    else if (blas_context.precision != matmul_precision)
    {
        cublasStatus_t status = cublasSetMathMode(blas_context.handle,
            matmul_precision == CUDA_MATMUL_PRECISION_TF32
                ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH);
        backend_info.last_blas_status = status;
        if (status != CUBLAS_STATUS_SUCCESS) return nullptr;
        blas_context.precision = matmul_precision;
    }
    return blas_context.handle;
}

static int blas_matrix_layout(int rows, int cols, size_t stride_row, size_t stride_col,
                              cublasOperation_t *operation, int *leading_dimension)
{
    if (stride_col == 1 && stride_row >= (size_t)cols && stride_row <= INT_MAX)
    {
        *operation = CUBLAS_OP_N;
        *leading_dimension = (int)stride_row;
        return 1;
    }
    if (stride_row == 1 && stride_col >= (size_t)rows && stride_col <= INT_MAX)
    {
        *operation = CUBLAS_OP_T;
        *leading_dimension = (int)stride_col;
        return 1;
    }
    return 0;
}

extern "C" int cuda_blas_tune_matmul(float *a, float *b, float *c,
                                      int rows, int inner, int cols,
                                      size_t a_stride0, size_t a_stride1,
                                      size_t b_stride0, size_t b_stride1,
                                      cudaStream_t stream, cuda_matmul_tuning_result *result)
{
#ifdef HAVE_CUBLASLT
    if (!a || !b || !c || !result || rows <= 0 || inner <= 0 || cols <= 0 ||
        (double)rows * inner * cols < MATMUL_LT_MIN_WORK)
        return 0;

    cublasOperation_t op_a, op_b;
    int lda, ldb;
    if (!blas_matrix_layout(rows, inner, a_stride0, a_stride1, &op_a, &lda) ||
        !blas_matrix_layout(inner, cols, b_stride0, b_stride1, &op_b, &ldb))
        return 0;

    cublasHandle_t blas = get_blas_handle();
    if (!blas) return -1;
    cublasStatus_t blas_status = cublasSetStream(blas, stream);
    backend_info.last_blas_status = blas_status;
    if (blas_status != CUBLAS_STATUS_SUCCESS) return -1;

    LtPlan *plan = nullptr;
    int plan_status = lt_plan_for(a, b, c, rows, inner, cols, lda, ldb, op_a, op_b, 1, &plan);
    if (plan_status <= 0) return plan_status;

    cudaEvent_t start = nullptr, stop = nullptr;
    cudaError_t cuda_status = cudaEventCreate(&start);
    if (cuda_status == cudaSuccess) cuda_status = cudaEventCreate(&stop);
    if (cuda_status != cudaSuccess) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        return -1;
    }

    cuda_matmul_tuning_result measured = {};
    constexpr int warmups = 2;
    constexpr int samples = 5;
    const float alpha = 1.0f, beta = 0.0f;
    double best_ms = 0.0;
    int selected_candidate = -1;
    for (int i = 0; i < plan->candidate_count; i++) {
        cublasLtMatmulHeuristicResult_t *candidate = &plan->candidates[i];
        if (candidate->workspaceSize != 0) continue;

        int algorithm_id = -1;
        size_t written = 0;
        cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(
            &candidate->algo, CUBLASLT_ALGO_CONFIG_ID, &algorithm_id, sizeof(algorithm_id), &written);
        if (status != CUBLAS_STATUS_SUCCESS) {
            backend_info.last_blas_status = status;
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return -1;
        }
        cuda_status = cudaStreamSynchronize(stream);
        if (cuda_status != cudaSuccess) {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return -1;
        }

        float timings[samples];
        for (int sample = -warmups; sample < samples; sample++) {
            if (sample >= 0) {
                cuda_status = cudaEventRecord(start, stream);
                if (cuda_status != cudaSuccess) break;
            }
            status = cublasLtMatmul(lt_handle, plan->operation,
                &alpha, b, plan->a, a, plan->b, &beta, c, plan->c, c, plan->c,
                &candidate->algo, nullptr, 0, stream);
            if (status != CUBLAS_STATUS_SUCCESS) {
                backend_info.last_blas_status = status;
                cudaEventDestroy(stop);
                cudaEventDestroy(start);
                return -1;
            }
            backend_info.lt_calls++;
            if (sample >= 0) {
                cuda_status = cudaEventRecord(stop, stream);
                if (cuda_status == cudaSuccess) cuda_status = cudaEventSynchronize(stop);
                if (cuda_status == cudaSuccess)
                    cuda_status = cudaEventElapsedTime(&timings[sample], start, stop);
                if (cuda_status != cudaSuccess) break;
            }
        }
        if (cuda_status != cudaSuccess) {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return -1;
        }
        std::sort(timings, timings + samples);
        double median_ms = timings[samples / 2];
        int output_index = measured.candidate_count++;
        measured.candidates[output_index].algorithm_id = algorithm_id;
        measured.candidates[output_index].workspace_bytes = candidate->workspaceSize;
        measured.candidates[output_index].median_ms = median_ms;
        if (selected_candidate < 0 || median_ms < best_ms) {
            best_ms = median_ms;
            selected_candidate = i;
            measured.selected_candidate = output_index;
        }
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    if (selected_candidate < 0) return 0;

    plan->selected_candidate = selected_candidate;
    plan->algorithm = plan->candidates[selected_candidate].algo;
    plan->tuned = 1;
    *result = measured;
    backend_info.last_matmul = "cublasLt";
    return 1;
#else
    (void)a; (void)b; (void)c; (void)rows; (void)inner; (void)cols;
    (void)a_stride0; (void)a_stride1; (void)b_stride0; (void)b_stride1;
    (void)stream; (void)result;
    return 0;
#endif
}

static int blas_matmul_2d(float *a, float *b, float *c,
                          int rows, int inner, int cols,
                          size_t a_row, size_t a_col, size_t b_row, size_t b_col,
                          size_t c_row, size_t c_col, cudaStream_t stream)
{
    bool dot = rows == 1 && cols == 1 && inner >= 4096 && a_col <= INT_MAX && b_row <= INT_MAX;
    bool underutilized = inner >= MATMUL_SKINNY_MIN_INNER &&
                         (size_t)rows * (size_t)cols <= MATMUL_SKINNY_MAX_OUTPUTS;
    double work = (double)rows * inner * cols;
    if ((!dot && !underutilized && work < MATMUL_BLAS_MIN_WORK) ||
        c_col != 1 || c_row != (size_t)cols)
        return 0;
    cublasOperation_t op_a, op_b;
    int lda, ldb;
    if (!dot && (!blas_matrix_layout(rows, inner, a_row, a_col, &op_a, &lda) ||
        !blas_matrix_layout(inner, cols, b_row, b_col, &op_b, &ldb))
        ) return 0;

    cublasHandle_t handle = get_blas_handle();
    if (!handle) return -1;
    cublasStatus_t status = cublasSetStream(handle, stream);
    backend_info.last_blas_status = status;
    if (status != CUBLAS_STATUS_SUCCESS) return -1;
    if (dot)
    {
        status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        if (status != CUBLAS_STATUS_SUCCESS) { backend_info.last_blas_status = status; return -1; }
        status = cublasSdot(handle, inner, a, (int)a_col, b, (int)b_row, c);
        cublasStatus_t restored = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        if (status == CUBLAS_STATUS_SUCCESS) status = restored;
    }
    else
    {
#ifdef HAVE_CUBLASLT
        bool prefer_lt = matmul_precision == CUDA_MATMUL_PRECISION_TF32 ||
                         work < MATMUL_LT_MAX_STRICT_WORK;
        if (work >= MATMUL_LT_MIN_WORK) {
            int lt = lt_matmul(a, b, c, rows, inner, cols, lda, ldb, op_a, op_b,
                               stream, prefer_lt, !prefer_lt);
            if (lt) return lt;
        }
#endif
        const float alpha = 1.0f, beta = 0.0f;
        status = cublasSgemm(handle, op_b, op_a, cols, rows, inner,
                             &alpha, b, ldb, a, lda, &beta, c, cols);
    }
    backend_info.last_blas_status = status;
    if (status != CUBLAS_STATUS_SUCCESS) return -1;
    backend_info.blas_calls++;
    backend_info.last_matmul = dot ? "cublasDot" : "cublas";
    return 1;
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
    if (params->ndC < 3 ||
        (double)params->M * params->N * params->K * params->total_batches < MATMUL_BLAS_MIN_WORK)
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
    if (!handle) return -1;
    cublasStatus_t status = cublasSetStream(handle, stream);
    backend_info.last_blas_status = status;
    if (status != CUBLAS_STATUS_SUCCESS) return -1;
    const float alpha = 1.0f, beta = 0.0f;
    status = cublasSgemmStridedBatched(handle, op_b, op_a,
        params->N, params->M, params->K,
        &alpha, params->B, ldb, (long long)stride_b,
        params->A, lda, (long long)stride_a,
        &beta, params->C, params->N, (long long)stride_c, params->total_batches);
    backend_info.last_blas_status = status;
    if (status != CUBLAS_STATUS_SUCCESS) return -1;
    backend_info.blas_calls++;
    backend_info.last_matmul = "cublasBatched";
    return 1;
}
#endif

#ifndef HAVE_CUBLAS
extern "C" void cuda_blas_shutdown(void) { backend_info = {}; }
#endif

static __global__ void matmul_small_kernel(float *a, float *b, float *c,
                                     int m, int n, int k,
                                     size_t a_stride0, size_t a_stride1,
                                     size_t b_stride0, size_t b_stride1,
                                     size_t c_stride0, size_t c_stride1)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0;
        for (int i = 0; i < n; i++)
            sum += a[(size_t)row * a_stride0 + (size_t)i * a_stride1] *
                   b[(size_t)i * b_stride0 + (size_t)col * b_stride1];
        c[(size_t)row * c_stride0 + (size_t)col * c_stride1] = sum;
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
    if (inner) {
        int selected = blas_matmul_batched(&params, stream);
        if (selected) return selected > 0;
    }
#endif

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid = cuda_grid_2d(cols, rows, TILE_SIZE, TILE_SIZE, batches);
    matmul_nd_tiled_kernel<<<grid, block, 0, stream>>>(params);
    backend_info.builtin_matmul_calls++;
    backend_info.last_matmul = "builtin";
    return cuda_launch_status() == cudaSuccess;
}

template <typename T>
static __global__ void low_precision_matmul_nd_kernel(LowMatMulParams<T> params)
{
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int batch_id = blockIdx.z;
    int global_row = blockIdx.y * TILE_SIZE + ty;
    int global_col = blockIdx.x * TILE_SIZE + tx;
    if (batch_id >= params.total_batches) return;

    size_t batch_offset_a = 0, batch_offset_b = 0, batch_offset_c = 0;
    int remaining = batch_id;
    for (int i = params.ndC - 3; i >= 0; i--)
    {
        int coord = remaining % params.shapeC[i];
        remaining /= params.shapeC[i];
        batch_offset_c += (size_t)coord * params.strideC[i];
        if (params.ndA > i + 2 && params.shapeA[i] > 1)
            batch_offset_a += (size_t)coord * params.strideA[i];
        if (params.ndB > i + 2 && params.shapeB[i] > 1)
            batch_offset_b += (size_t)coord * params.strideB[i];
    }

    size_t a_row_stride = params.strideA[params.ndA - 2];
    size_t a_col_stride = params.strideA[params.ndA - 1];
    size_t b_row_stride = params.strideB[params.ndB - 2];
    size_t b_col_stride = params.strideB[params.ndB - 1];
    float sum = 0.0f;
    for (int k_offset = 0; k_offset < params.K; k_offset += TILE_SIZE)
    {
        int a_col = k_offset + tx;
        int b_row = k_offset + ty;
        tile_a[ty][tx] = global_row < params.M && a_col < params.K
            ? low_precision_to_float(params.A[batch_offset_a + (size_t)global_row * a_row_stride +
                                             (size_t)a_col * a_col_stride]) : 0.0f;
        tile_b[ty][tx] = b_row < params.K && global_col < params.N
            ? low_precision_to_float(params.B[batch_offset_b + (size_t)b_row * b_row_stride +
                                             (size_t)global_col * b_col_stride]) : 0.0f;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++)
            sum += tile_a[ty][i] * tile_b[i][tx];
        __syncthreads();
    }
    if (global_row < params.M && global_col < params.N)
    {
        size_t output_index = batch_offset_c +
            (size_t)global_row * params.strideC[params.ndC - 2] +
            (size_t)global_col * params.strideC[params.ndC - 1];
        params.C[output_index] = low_precision_from_float<T>(sum);
    }
}

template <typename T>
static int launch_low_precision_tiled(LowMatMulParams<T> &params, cudaStream_t stream)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid = cuda_grid_2d(params.N, params.M, TILE_SIZE, TILE_SIZE, params.total_batches);
    low_precision_matmul_nd_kernel<T><<<grid, block, 0, stream>>>(params);
    backend_info.builtin_matmul_calls++;
    backend_info.last_matmul = "builtin-low-precision";
    return cuda_launch_status() == cudaSuccess;
}

template <typename T>
static int launch_low_precision_matmul(LowMatMulParams<T> &params, dtype_t dtype,
                                       cudaStream_t stream)
{
#ifdef HAVE_CUBLAS
    double work = (double)params.M * params.N * params.K;
    if (params.ndC == 2 && work >= MATMUL_BLAS_MIN_WORK &&
        params.strideC[1] == 1 && params.strideC[0] == (size_t)params.N)
    {
        cublasOperation_t op_a, op_b;
        int lda, ldb;
        if (blas_matrix_layout(params.M, params.K, params.strideA[params.ndA - 2],
                               params.strideA[params.ndA - 1], &op_a, &lda) &&
            blas_matrix_layout(params.K, params.N, params.strideB[params.ndB - 2],
                               params.strideB[params.ndB - 1], &op_b, &ldb))
        {
            cublasHandle_t handle = get_blas_handle();
            if (!handle) return 0;
            cublasStatus_t status = cublasSetStream(handle, stream);
            backend_info.last_blas_status = status;
            if (status != CUBLAS_STATUS_SUCCESS) return 0;
            const float alpha = 1.0f, beta = 0.0f;
            cudaDataType_t data_type = dtype == DTYPE_FLOAT16 ? CUDA_R_16F : CUDA_R_16BF;
            status = cublasGemmEx(handle, op_b, op_a, params.N, params.M, params.K,
                &alpha, params.B, data_type, ldb, params.A, data_type, lda,
                &beta, params.C, data_type, params.N, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            backend_info.last_blas_status = status;
            if (status == CUBLAS_STATUS_NOT_SUPPORTED ||
                status == CUBLAS_STATUS_ARCH_MISMATCH)
            {
                backend_info.last_blas_status = CUBLAS_STATUS_SUCCESS;
                return launch_low_precision_tiled(params, stream);
            }
            if (status != CUBLAS_STATUS_SUCCESS) return 0;
            backend_info.blas_calls++;
            backend_info.last_matmul = "cublas-low-precision";
            return 1;
        }
    }
#else
    (void)dtype;
#endif
    return launch_low_precision_tiled(params, stream);
}

extern "C" int cuda_low_precision_matmul_nd_launcher(
    void *a, void *b, void *c, dtype_t dtype,
    int *shape_a, size_t *stride_a, int nd_a,
    int *shape_b, size_t *stride_b, int nd_b,
    int *shape_c, size_t *stride_c, int nd_c, cudaStream_t stream)
{
    if ((dtype != DTYPE_FLOAT16 && dtype != DTYPE_BFLOAT16) ||
        nd_a < 2 || nd_b < 2 || nd_c < 2 ||
        nd_a > MAX_DIMS || nd_b > MAX_DIMS || nd_c > MAX_DIMS)
        return 0;

    int rows = shape_a[nd_a - 2], inner = shape_a[nd_a - 1];
    int cols = shape_b[nd_b - 1];
    if (inner != shape_b[nd_b - 2] ||
        shape_c[nd_c - 2] != rows || shape_c[nd_c - 1] != cols ||
        rows < 0 || inner < 0 || cols < 0)
        return 0;

    int batches = 1;
    for (int i = 0; i < nd_c - 2; i++)
    {
        if (shape_c[i] < 0 || (shape_c[i] && batches > 65535 / shape_c[i]))
            return 0;
        batches *= shape_c[i];
    }
    if (!rows || !cols || !batches) return 1;

    if (dtype == DTYPE_FLOAT16)
    {
        LowMatMulParams<__half> params = {};
        params.A = (const __half *)a; params.B = (const __half *)b; params.C = (__half *)c;
        memcpy(params.shapeA, shape_a, nd_a * sizeof(int));
        memcpy(params.strideA, stride_a, nd_a * sizeof(size_t));
        memcpy(params.shapeB, shape_b, nd_b * sizeof(int));
        memcpy(params.strideB, stride_b, nd_b * sizeof(size_t));
        memcpy(params.shapeC, shape_c, nd_c * sizeof(int));
        memcpy(params.strideC, stride_c, nd_c * sizeof(size_t));
        params.ndA = nd_a; params.ndB = nd_b; params.ndC = nd_c;
        params.M = rows; params.N = cols; params.K = inner; params.total_batches = batches;
        return launch_low_precision_matmul(params, dtype, stream);
    }

    LowMatMulParams<__nv_bfloat16> params = {};
    params.A = (const __nv_bfloat16 *)a; params.B = (const __nv_bfloat16 *)b;
    params.C = (__nv_bfloat16 *)c;
    memcpy(params.shapeA, shape_a, nd_a * sizeof(int));
    memcpy(params.strideA, stride_a, nd_a * sizeof(size_t));
    memcpy(params.shapeB, shape_b, nd_b * sizeof(int));
    memcpy(params.strideB, stride_b, nd_b * sizeof(size_t));
    memcpy(params.shapeC, shape_c, nd_c * sizeof(int));
    memcpy(params.strideC, stride_c, nd_c * sizeof(size_t));
    params.ndA = nd_a; params.ndB = nd_b; params.ndC = nd_c;
    params.M = rows; params.N = cols; params.K = inner; params.total_batches = batches;
    return launch_low_precision_matmul(params, dtype, stream);
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
    if (n) {
        int selected = blas_matmul_2d(a, b, c, m, n, k,
                       a_stride0, a_stride1, b_stride0, b_stride1,
                       c_stride0, c_stride1, stream);
        if (selected) return selected > 0;
    }
#endif

    /* Small products need more independent blocks, not shared-memory tile barriers. */
    if (n <= 128 && (double)m * n * k < MATMUL_BLAS_MIN_WORK) {
        dim3 block(16, 16), grid = cuda_grid_2d(k, m, 16, 16);
        matmul_small_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k,
            a_stride0, a_stride1, b_stride0, b_stride1, c_stride0, c_stride1);
        backend_info.builtin_matmul_calls++;
        backend_info.last_matmul = "builtin";
        return cuda_launch_status() == cudaSuccess;
    }

    int shape_a[] = {m, n}, shape_b[] = {n, k}, shape_c[] = {m, k};
    size_t stride_a[] = {a_stride0, a_stride1}, stride_b[] = {b_stride0, b_stride1};
    size_t stride_c[] = {c_stride0, c_stride1};
    return cuda_batched_matmul_nd_launcher(a, b, c, shape_a, stride_a, 2, shape_b, stride_b, 2,
                                           shape_c, stride_c, 2, stream);
}