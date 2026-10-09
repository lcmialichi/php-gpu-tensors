#include <cuda_runtime.h>
#include "reduction_ops.cuh"
#include "reduction_ops.h"
#include "dispatcher.h"
#include <stdio.h>
#include <vector>

thread_local cudaError_t reduction_error = cudaSuccess;
thread_local cuda_backend_info reduction_counters = {};
struct ReductionWorkspace { int device; cudaStream_t stream; void *data; size_t bytes; };
static thread_local std::vector<ReductionWorkspace> reduction_workspaces;

cudaError_t reduction_workspace(cudaStream_t stream, size_t bytes, void **storage)
{
    int device;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    for (auto &entry : reduction_workspaces)
        if (entry.device == device && entry.stream == stream) {
            if (entry.bytes < bytes) {
                error = cudaStreamSynchronize(stream);
                if (error != cudaSuccess) return error;
                error = cudaFree(entry.data);
                if (error != cudaSuccess) return error;
                entry.data = nullptr;
                entry.bytes = 0;
                error = cudaMalloc(&entry.data, bytes);
                if (error != cudaSuccess) return error;
                entry.bytes = bytes;
            }
            *storage = entry.data;
            return cudaSuccess;
        }
    /* Scratch is stream-local; reuse never races concurrent Fusion executions. */
    if (reduction_workspaces.size() >= 64) return cudaErrorMemoryAllocation;
    void *data;
    error = cudaMalloc(&data, bytes);
    if (error != cudaSuccess) return error;
    reduction_workspaces.push_back({device, stream, data, bytes});
    *storage = data;
    return cudaSuccess;
}

extern "C" cudaError_t cuda_reduction_status(void) { return reduction_error; }
extern "C" cudaError_t cuda_reduction_release_stream(cudaStream_t stream)
{
    int device;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    for (auto entry = reduction_workspaces.begin(); entry != reduction_workspaces.end(); ++entry)
        if (entry->device == device && entry->stream == stream) {
            error = cudaFree(entry->data);
            if (error != cudaSuccess) return error;
            reduction_workspaces.erase(entry);
            break;
        }
    return cudaSuccess;
}
extern "C" void cuda_reduction_info(cuda_backend_info *info)
{
    info->cub_calls = reduction_counters.cub_calls;
    info->axis_calls = reduction_counters.axis_calls;
    info->generic_reduce_calls = reduction_counters.generic_reduce_calls;
}
extern "C" void cuda_reduction_shutdown(void)
{
    int original;
    if (cudaGetDevice(&original) != cudaSuccess) return;
    for (auto &entry : reduction_workspaces) {
        cudaSetDevice(entry.device);
        /* Streams may have been destroyed by Fusion shutdown; cudaFree waits for outstanding use. */
        cudaFree(entry.data);
    }
    cudaSetDevice(original);
    reduction_workspaces.clear();
    reduction_counters = {};
}

extern "C" void launch_reduction(
    void *input, void *output,
    dtype_t dtype, operation_type_t op_type,
    int *input_shape, int input_ndims,
    int *result_shape,
    size_t *input_strides,
    int result_ndims, int axis,
    size_t total_elements_out, size_t input_base_offset, cudaStream_t stream)
{
    reduction_error = cudaSuccess;
    if (op_type == OP_REDUCE_MEAN)
    {
        if (dtype != DTYPE_FLOAT32)
        {
            DISPATCH_DTYPE(dtype, {
                launch_reduce_op_kernel<scalar_t, double, AddOpT<double>>(
                    (scalar_t *)input,
                    (double *)output,
                    input_shape,
                    input_ndims,
                    result_shape,
                    input_strides,
                    result_ndims,
                    axis,
                    total_elements_out,
                    input_base_offset,
                    input_shape[axis], stream);
            });
        }
        else
        {
            DISPATCH_DTYPE(dtype, {
                launch_reduce_op_kernel<scalar_t, float, AddOpT<float>>(
                    (scalar_t *)input,
                    (float *)output,
                    input_shape,
                    input_ndims,
                    result_shape,
                    input_strides,
                    result_ndims,
                    axis,
                    total_elements_out,
                    input_base_offset,
                    input_shape[axis], stream);
            });
        }
        return;
    }

    if (op_type == OP_REDUCE_ALL || op_type == OP_REDUCE_ANY)
    {
        DISPATCH_DTYPE(dtype, {
            if (op_type == OP_REDUCE_ALL)
            {
                typedef AllOpT<scalar_t> reduce_op_t;
                launch_reduce_op_kernel<scalar_t, bool, reduce_op_t>(
                    (scalar_t *)input, (bool *)output, input_shape, input_ndims,
                    result_shape, input_strides, result_ndims, axis,
                    total_elements_out, input_base_offset, 1, stream);
            }
            else
            {
                typedef AnyOpT<scalar_t> reduce_op_t;
                launch_reduce_op_kernel<scalar_t, bool, reduce_op_t>(
                    (scalar_t *)input, (bool *)output, input_shape, input_ndims,
                    result_shape, input_strides, result_ndims, axis,
                    total_elements_out, input_base_offset, 1, stream);
            }
        });
        return;
    }

    DISPATCH_DTYPE(dtype, {
        DISPATCH_OP_REDUCTION(op_type, {
            launch_reduce_op_kernel<scalar_t, scalar_t, bin_op_t>(
                (scalar_t *)input,
                (scalar_t *)output,
                input_shape,
                input_ndims,
                result_shape,
                input_strides,
                result_ndims,
                axis,
                total_elements_out,
                input_base_offset,
                1, stream);
        });
    });
}

extern "C" void launch_arg_reduction(
    void *input, int *output,
    dtype_t dtype, operation_type_t op_type,
    int *input_shape, int input_ndims,
    int *result_shape,
    size_t *input_strides,
    int result_ndims, int axis,
    size_t total_elements_out, size_t input_base_offset, cudaStream_t stream)
{
    reduction_error = cudaSuccess;
    DISPATCH_DTYPE(dtype, {
        DISPATCH_OP_ARG_REDUCTION(op_type, {
            launch_arg_reduce_kernel<scalar_t, bin_op_t>(
                (scalar_t *)input,
                output,
                input_shape,
                input_ndims,
                input_strides,
                axis,
                total_elements_out,
                input_base_offset, stream);
        });
    });
}
