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

template <typename InputT, typename AccumT>
static cudaError_t launch_moments_typed(
    const InputT *input,
    AccumT *output,
    const int *input_shape,
    int input_ndims,
    const size_t *input_strides,
    int axis,
    size_t reduce_size,
    size_t output_size,
    size_t denominator,
    int standard_deviation,
    cudaStream_t stream)
{
    if (reduce_size >= 4096 && (axis < 0 || output_size == 1))
    {
        WelfordInputTransform<InputT, AccumT> transform = {};
        transform.input = input;
        transform.ndims = input_ndims;
        transform.axis = axis;
        memcpy(transform.shape, input_shape, input_ndims * sizeof(int));
        memcpy(transform.strides, input_strides, input_ndims * sizeof(size_t));

        using State = WelfordState<AccumT>;
        using Counter = cub::CountingInputIterator<size_t>;
        using Iterator = cub::TransformInputIterator<State, WelfordInputTransform<InputT, AccumT>, Counter>;
        Counter counter(0);
        Iterator iterator(counter, transform);
        WelfordCombine<AccumT> combine;
        State identity = {0, static_cast<AccumT>(0), static_cast<AccumT>(0)};
        size_t temporary_bytes = 0;
        State *query_output = reinterpret_cast<State *>(output);
        cudaError_t error = cub::DeviceReduce::Reduce(
            nullptr, temporary_bytes, iterator, query_output, reduce_size, combine, identity, stream);
        if (error != cudaSuccess) return error;

        size_t alignment = alignof(State);
        size_t state_offset = (temporary_bytes + alignment - 1) / alignment * alignment;
        if (state_offset < temporary_bytes ||
            state_offset > std::numeric_limits<size_t>::max() - sizeof(State))
            return cudaErrorMemoryAllocation;

        void *storage = nullptr;
        error = reduction_workspace(stream, state_offset + sizeof(State), &storage);
        if (error != cudaSuccess) return error;
        State *state = reinterpret_cast<State *>(static_cast<char *>(storage) + state_offset);
        error = cub::DeviceReduce::Reduce(
            storage, temporary_bytes, iterator, state, reduce_size, combine, identity, stream);
        if (error != cudaSuccess) return error;

        finalize_moments_kernel<<<1, 1, 0, stream>>>(
            state, output, denominator, standard_deviation);
        return cudaGetLastError();
    }

    MomentsParams params = {};
    params.ndims = input_ndims;
    params.axis = axis;
    params.reduce_size = reduce_size;
    params.output_size = output_size;
    params.denominator = denominator;
    params.standard_deviation = standard_deviation;
    memcpy(params.shape, input_shape, input_ndims * sizeof(int));
    memcpy(params.strides, input_strides, input_ndims * sizeof(size_t));
    moments_axis_kernel<<<output_size, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        input, output, params);
    return cudaGetLastError();
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

extern "C" void launch_moments_reduction(
    const void *input, void *output,
    dtype_t input_dtype, dtype_t result_dtype,
    const int *input_shape, int input_ndims,
    const size_t *input_strides,
    int axis, size_t total_elements_out,
    int correction, int standard_deviation, cudaStream_t stream)
{
    reduction_error = cudaSuccess;
    if (!total_elements_out) return;

    size_t reduce_size = axis < 0 ? 1 : static_cast<size_t>(input_shape[axis]);
    if (axis < 0)
    {
        for (int dim = 0; dim < input_ndims; dim++)
            reduce_size *= static_cast<size_t>(input_shape[dim]);
    }
    size_t denominator = reduce_size - static_cast<size_t>(correction);
    if (!denominator)
    {
        reduction_error = cudaErrorInvalidValue;
        return;
    }

    if (result_dtype == DTYPE_FLOAT64)
    {
        DISPATCH_DTYPE(input_dtype, {
            reduction_error = launch_moments_typed<scalar_t, double>(
                static_cast<const scalar_t *>(input), static_cast<double *>(output),
                input_shape, input_ndims, input_strides, axis,
                reduce_size, total_elements_out, denominator,
                standard_deviation, stream);
        });
    }
    else if (result_dtype == DTYPE_FLOAT32)
    {
        DISPATCH_DTYPE(input_dtype, {
            reduction_error = launch_moments_typed<scalar_t, float>(
                static_cast<const scalar_t *>(input), static_cast<float *>(output),
                input_shape, input_ndims, input_strides, axis,
                reduce_size, total_elements_out, denominator,
                standard_deviation, stream);
        });
    }
    else
    {
        reduction_error = cudaErrorInvalidValue;
    }
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
