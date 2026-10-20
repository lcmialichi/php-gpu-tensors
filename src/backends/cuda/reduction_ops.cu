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
    moments_axis_kernel<<<cuda_grid_blocks(output_size), REDUCTION_BLOCK_SIZE, 0, stream>>>(
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

template <typename T>
struct CumulativeAdd
{
    __host__ __device__ T operator()(T left, T right) const
    {
        return left + right;
    }
};

template <typename T>
struct CumulativeMultiply
{
    __host__ __device__ T operator()(T left, T right) const
    {
        return left * right;
    }
};

template <typename InputT, typename OutputT>
struct CumulativeInput
{
    const InputT *input;

    __host__ __device__ OutputT operator()(size_t index) const
    {
        return static_cast<OutputT>(input[index]);
    }
};

template <typename T>
struct CumulativeSegmentValue
{
    size_t segment;
    T value;
};

template <typename InputT, typename OutputT>
struct CumulativeSegmentInput
{
    const InputT *input;
    size_t segment_length;

    __host__ __device__ CumulativeSegmentValue<OutputT> operator()(size_t index) const
    {
        return {index / segment_length,
                static_cast<OutputT>(input[index])};
    }
};

template <typename T, typename ScanOp>
struct CumulativeSegmentCombine
{
    __host__ __device__ CumulativeSegmentValue<T> operator()(
        CumulativeSegmentValue<T> left, CumulativeSegmentValue<T> right) const
    {
        return {right.segment,
                left.segment == right.segment
                    ? ScanOp()(left.value, right.value) : right.value};
    }
};

template <typename T>
__global__ static void cumulative_segment_extract(
    const CumulativeSegmentValue<T> *__restrict__ input,
    T *__restrict__ output, size_t total_items)
{
    for (size_t index = cuda_linear_index(blockIdx.x, blockDim.x, threadIdx.x);
         index < total_items; index += cuda_grid_stride(gridDim.x, blockDim.x))
        output[index] = input[index].value;
}

template <typename InputT, typename OutputT, typename ScanOp>
static cudaError_t launch_cumulative_typed(
    const InputT *input, OutputT *output, size_t total_items,
    size_t segment_length, size_t segment_count, cudaStream_t stream)
{
    using Counter = cub::CountingInputIterator<size_t>;
    size_t temporary_bytes = 0;
    cudaError_t error;

    if (segment_count == 1)
    {
        using InputIterator = cub::TransformInputIterator<
            OutputT, CumulativeInput<InputT, OutputT>, Counter>;
        InputIterator input_iterator(Counter(0), CumulativeInput<InputT, OutputT>{input});
        error = cub::DeviceScan::InclusiveScan(
            nullptr, temporary_bytes, input_iterator, output, ScanOp(),
            total_items, stream);
        if (error != cudaSuccess) return error;
        void *temporary = nullptr;
        if (temporary_bytes)
        {
            error = reduction_workspace(stream, temporary_bytes, &temporary);
            if (error != cudaSuccess) return error;
        }
        return cub::DeviceScan::InclusiveScan(
            temporary, temporary_bytes, input_iterator, output, ScanOp(),
            total_items, stream);
    }

    using SegmentValue = CumulativeSegmentValue<OutputT>;
    using SegmentInput = cub::TransformInputIterator<
        SegmentValue, CumulativeSegmentInput<InputT, OutputT>, Counter>;
    SegmentInput input_iterator(Counter(0),
        CumulativeSegmentInput<InputT, OutputT>{input, segment_length});
    using SegmentOp = CumulativeSegmentCombine<OutputT, ScanOp>;
    SegmentOp segment_op = {};
    SegmentValue *query_output = nullptr;
    error = cub::DeviceScan::InclusiveScan(
        nullptr, temporary_bytes, input_iterator, query_output, segment_op,
        total_items, stream);
    if (error != cudaSuccess) return error;

    if (total_items > SIZE_MAX / sizeof(SegmentValue))
        return cudaErrorMemoryAllocation;
    size_t values_bytes = total_items * sizeof(SegmentValue);
    size_t values_aligned = (values_bytes + alignof(SegmentValue) - 1) /
        alignof(SegmentValue) * alignof(SegmentValue);
    if (values_aligned < values_bytes ||
        values_aligned > SIZE_MAX - temporary_bytes)
        return cudaErrorMemoryAllocation;
    void *storage = nullptr;
    error = reduction_workspace(stream, values_aligned + temporary_bytes, &storage);
    if (error != cudaSuccess) return error;
    SegmentValue *segment_output = static_cast<SegmentValue *>(storage);
    void *temporary = static_cast<char *>(storage) + values_aligned;
    error = cub::DeviceScan::InclusiveScan(
        temporary, temporary_bytes, input_iterator, segment_output,
        segment_op, total_items, stream);
    if (error != cudaSuccess) return error;

    size_t blocks = (total_items - 1) / 256 + 1;
    if (blocks > 65535) blocks = 65535;
    cumulative_segment_extract<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
        segment_output, output, total_items);
    return cudaGetLastError();
}

template <typename InputT>
static cudaError_t dispatch_cumulative_output(
    const void *input, void *output, dtype_t output_dtype,
    size_t total_items, size_t segment_length, size_t segment_count,
    operation_type_t operation, cudaStream_t stream)
{
#define CUMULATIVE_OUTPUT_CASE(DTYPE, TYPE)                                      \
    case DTYPE:                                                                  \
        return operation == OP_CUMSUM                                           \
            ? launch_cumulative_typed<InputT, TYPE, CumulativeAdd<TYPE>>(       \
                  static_cast<const InputT *>(input),                            \
                  static_cast<TYPE *>(output), total_items, segment_length,     \
                  segment_count, stream)                                        \
            : launch_cumulative_typed<InputT, TYPE, CumulativeMultiply<TYPE>>(  \
                  static_cast<const InputT *>(input),                            \
                  static_cast<TYPE *>(output), total_items, segment_length,     \
                  segment_count, stream)
    switch (output_dtype)
    {
        CUMULATIVE_OUTPUT_CASE(DTYPE_FLOAT32, float);
        CUMULATIVE_OUTPUT_CASE(DTYPE_FLOAT64, double);
        CUMULATIVE_OUTPUT_CASE(DTYPE_INT64, int64_t);
        CUMULATIVE_OUTPUT_CASE(DTYPE_UINT64, uint64_t);
        default: return cudaErrorInvalidValue;
    }
#undef CUMULATIVE_OUTPUT_CASE
}

extern "C" cudaError_t launch_cumulative_scan(
    const void *input, void *output, dtype_t input_dtype,
    dtype_t output_dtype, size_t total_items, size_t segment_length,
    size_t segment_count, operation_type_t operation, cudaStream_t stream)
{
    if (!total_items) return cudaSuccess;
    if (!input || !output || !segment_length || !segment_count ||
        segment_count > static_cast<size_t>(INT_MAX) ||
        (operation != OP_CUMSUM && operation != OP_CUMPROD))
        return cudaErrorInvalidValue;

#define CUMULATIVE_INPUT_CASE(DTYPE, TYPE)                                       \
    case DTYPE:                                                                  \
        return dispatch_cumulative_output<TYPE>(input, output, output_dtype,    \
            total_items, segment_length, segment_count, operation, stream)
    switch (input_dtype)
    {
        CUMULATIVE_INPUT_CASE(DTYPE_FLOAT32, float);
        CUMULATIVE_INPUT_CASE(DTYPE_FLOAT64, double);
        CUMULATIVE_INPUT_CASE(DTYPE_INT8, int8_t);
        CUMULATIVE_INPUT_CASE(DTYPE_INT16, int16_t);
        CUMULATIVE_INPUT_CASE(DTYPE_INT32, int32_t);
        CUMULATIVE_INPUT_CASE(DTYPE_INT64, int64_t);
        CUMULATIVE_INPUT_CASE(DTYPE_UINT8, uint8_t);
        CUMULATIVE_INPUT_CASE(DTYPE_UINT16, uint16_t);
        CUMULATIVE_INPUT_CASE(DTYPE_UINT32, uint32_t);
        CUMULATIVE_INPUT_CASE(DTYPE_UINT64, uint64_t);
        CUMULATIVE_INPUT_CASE(DTYPE_BOOL, bool);
        default: return cudaErrorInvalidValue;
    }
#undef CUMULATIVE_INPUT_CASE
}

struct CumulativeReorderParams
{
    int ndims;
    int axis;
    int shape[MAX_DIMS];
};

template <typename T>
__global__ static void cumulative_reorder_kernel(
    const T *__restrict__ input, T *__restrict__ output,
    CumulativeReorderParams params, size_t total_items)
{
    for (size_t index = cuda_linear_index(blockIdx.x, blockDim.x, threadIdx.x);
         index < total_items; index += cuda_grid_stride(gridDim.x, blockDim.x))
    {
        size_t remaining = index;
        size_t coordinates[MAX_DIMS] = {};
        for (int axis = params.ndims - 1; axis >= 0; axis--)
        {
            size_t extent = static_cast<size_t>(params.shape[axis]);
            coordinates[axis] = remaining % extent;
            remaining /= extent;
        }

        size_t source_index = 0;
        for (int permuted_axis = 0; permuted_axis < params.ndims; permuted_axis++)
        {
            int source_axis = permuted_axis < params.axis
                ? permuted_axis
                : (permuted_axis == params.ndims - 1
                    ? params.axis : permuted_axis + 1);
            source_index = source_index *
                static_cast<size_t>(params.shape[source_axis]) +
                coordinates[source_axis];
        }
        output[index] = input[source_index];
    }
}

extern "C" cudaError_t launch_cumulative_reorder(
    const void *input, void *output, dtype_t dtype,
    const int *shape, int ndims, int axis, size_t total_items,
    cudaStream_t stream)
{
    if (!total_items) return cudaSuccess;
    if (!input || !output || !shape || ndims < 1 || ndims > MAX_DIMS ||
        axis < 0 || axis >= ndims)
        return cudaErrorInvalidValue;

    CumulativeReorderParams params = {};
    params.ndims = ndims;
    params.axis = axis;
    memcpy(params.shape, shape, ndims * sizeof(int));
    size_t blocks = (total_items - 1) / 256 + 1;
    if (blocks > 65535) blocks = 65535;
#define CUMULATIVE_REORDER_CASE(DTYPE, TYPE)                                     \
    case DTYPE:                                                                  \
        cumulative_reorder_kernel<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>( \
            static_cast<const TYPE *>(input), static_cast<TYPE *>(output),       \
            params, total_items);                                                \
        break
    switch (dtype)
    {
        CUMULATIVE_REORDER_CASE(DTYPE_FLOAT32, float);
        CUMULATIVE_REORDER_CASE(DTYPE_FLOAT64, double);
        CUMULATIVE_REORDER_CASE(DTYPE_INT64, int64_t);
        CUMULATIVE_REORDER_CASE(DTYPE_UINT64, uint64_t);
        default: return cudaErrorInvalidValue;
    }
#undef CUMULATIVE_REORDER_CASE
    return cudaGetLastError();
}

template <typename T>
struct CumulativeGradientValue
{
    T multiplier;
    T value;
    int head;
};

template <typename T>
struct CumulativeProductGradientInput
{
    const T *input;
    const T *upstream;
    int input_ndims;
    int axis;
    int input_shape[MAX_DIMS];
    size_t input_strides[MAX_DIMS];
    size_t upstream_strides[MAX_DIMS];
    size_t total_items;
    size_t axis_length;
    size_t inner_size;

    __host__ __device__ size_t input_offset(size_t logical_index) const
    {
        size_t offset = 0;
        for (int axis = input_ndims - 1; axis >= 0; axis--)
        {
            size_t coordinate =
                logical_index % static_cast<size_t>(input_shape[axis]);
            logical_index /= static_cast<size_t>(input_shape[axis]);
            offset += coordinate * input_strides[axis];
        }
        return offset;
    }

    __host__ __device__ size_t upstream_offset(size_t logical_index) const
    {
        if (axis < 0)
            return logical_index * upstream_strides[0];
        size_t offset = 0;
        for (int axis = input_ndims - 1; axis >= 0; axis--)
        {
            size_t coordinate =
                logical_index % static_cast<size_t>(input_shape[axis]);
            logical_index /= static_cast<size_t>(input_shape[axis]);
            offset += coordinate * upstream_strides[axis];
        }
        return offset;
    }

    __host__ __device__ CumulativeGradientValue<T> operator()(
        size_t reversed_index) const
    {
        size_t logical_index;
        size_t coordinate;
        if (axis < 0)
        {
            logical_index = total_items - 1 - reversed_index;
            coordinate = logical_index;
        }
        else
        {
            size_t segment = reversed_index / axis_length;
            size_t reversed_axis = reversed_index % axis_length;
            size_t outer = segment / inner_size;
            size_t inner = segment % inner_size;
            coordinate = axis_length - 1 - reversed_axis;
            logical_index = outer * axis_length * inner_size +
                coordinate * inner_size + inner;
        }

        size_t next_index = logical_index;
        int has_next = axis < 0
            ? logical_index + 1 < total_items
            : coordinate + 1 < axis_length;
        if (has_next)
            next_index += axis < 0 ? 1 : inner_size;
        T multiplier = has_next
            ? input[input_offset(next_index)] : static_cast<T>(0);
        return {multiplier, upstream[upstream_offset(logical_index)],
                axis < 0 ? reversed_index == 0
                         : reversed_index % axis_length == 0};
    }
};

template <typename T>
struct CumulativeGradientCombine
{
    __host__ __device__ CumulativeGradientValue<T> operator()(
        CumulativeGradientValue<T> left,
        CumulativeGradientValue<T> right) const
    {
        if (right.head) return right;
        return {right.multiplier * left.multiplier,
                right.value + right.multiplier * left.value,
                left.head};
    }
};

struct CumulativeGradientOutputParams
{
    int input_ndims;
    int axis;
    int input_shape[MAX_DIMS];
    size_t total_items;
    size_t axis_length;
    size_t inner_size;
};

template <typename T>
__global__ static void cumulative_product_gradient_extract(
    const CumulativeGradientValue<T> *__restrict__ scanned,
    const T *__restrict__ prefix_product,
    T *__restrict__ input_gradient,
    CumulativeGradientOutputParams params)
{
    size_t total_items = params.total_items;
    for (size_t reversed_index = cuda_linear_index(blockIdx.x, blockDim.x, threadIdx.x);
         reversed_index < total_items;
         reversed_index += cuda_grid_stride(gridDim.x, blockDim.x))
    {
        size_t logical_index;
        size_t coordinate;
        size_t previous_index;
        if (params.axis < 0)
        {
            logical_index = total_items - 1 - reversed_index;
            coordinate = logical_index;
            previous_index = logical_index ? logical_index - 1 : 0;
        }
        else
        {
            size_t segment = reversed_index / params.axis_length;
            size_t reversed_axis = reversed_index % params.axis_length;
            size_t outer = segment / params.inner_size;
            size_t inner = segment % params.inner_size;
            coordinate = params.axis_length - 1 - reversed_axis;
            logical_index = outer * params.axis_length * params.inner_size +
                coordinate * params.inner_size + inner;
            previous_index = logical_index >= params.inner_size
                ? logical_index - params.inner_size : 0;
        }

        T prefix = coordinate
            ? prefix_product[previous_index] : static_cast<T>(1);
        input_gradient[logical_index] = scanned[reversed_index].value * prefix;
    }
}

template <typename T>
static cudaError_t launch_cumulative_product_gradient_typed(
    const T *input, const int *input_shape, const size_t *input_strides,
    int input_ndims, const T *upstream, const size_t *upstream_strides,
    const T *prefix_product, T *input_gradient, size_t total_items,
    int axis, cudaStream_t stream)
{
    size_t axis_length = axis < 0
        ? total_items : static_cast<size_t>(input_shape[axis]);
    size_t inner_size = 1;
    if (axis >= 0)
        for (int dimension = axis + 1; dimension < input_ndims; dimension++)
            inner_size *= static_cast<size_t>(input_shape[dimension]);

    CumulativeProductGradientInput<T> transform = {};
    transform.input = input;
    transform.upstream = upstream;
    transform.input_ndims = input_ndims;
    transform.axis = axis;
    transform.total_items = total_items;
    transform.axis_length = axis_length;
    transform.inner_size = inner_size;
    memcpy(transform.input_shape, input_shape, input_ndims * sizeof(int));
    memcpy(transform.input_strides, input_strides,
           input_ndims * sizeof(size_t));
    if (axis < 0)
        transform.upstream_strides[0] = upstream_strides[0];
    else
        memcpy(transform.upstream_strides, upstream_strides,
               input_ndims * sizeof(size_t));

    using Counter = cub::CountingInputIterator<size_t>;
    using Iterator = cub::TransformInputIterator<
        CumulativeGradientValue<T>, CumulativeProductGradientInput<T>, Counter>;
    Iterator input_iterator(Counter(0), transform);
    size_t temporary_bytes = 0;
    CumulativeGradientValue<T> *query_output = nullptr;
    cudaError_t error = cub::DeviceScan::InclusiveScan(
        nullptr, temporary_bytes, input_iterator, query_output,
        CumulativeGradientCombine<T>(), total_items, stream);
    if (error != cudaSuccess) return error;
    if (total_items > SIZE_MAX / sizeof(CumulativeGradientValue<T>))
        return cudaErrorMemoryAllocation;
    size_t values_bytes = total_items * sizeof(CumulativeGradientValue<T>);
    size_t values_aligned =
        (values_bytes + alignof(CumulativeGradientValue<T>) - 1) /
        alignof(CumulativeGradientValue<T>) *
        alignof(CumulativeGradientValue<T>);
    if (values_aligned < values_bytes ||
        values_aligned > SIZE_MAX - temporary_bytes)
        return cudaErrorMemoryAllocation;
    void *storage = nullptr;
    error = reduction_workspace(stream, values_aligned + temporary_bytes,
                                &storage);
    if (error != cudaSuccess) return error;
    auto *scanned = static_cast<CumulativeGradientValue<T> *>(storage);
    void *temporary = static_cast<char *>(storage) + values_aligned;
    error = cub::DeviceScan::InclusiveScan(
        temporary, temporary_bytes, input_iterator, scanned,
        CumulativeGradientCombine<T>(), total_items, stream);
    if (error != cudaSuccess) return error;

    CumulativeGradientOutputParams params = {};
    params.input_ndims = input_ndims;
    params.axis = axis;
    params.total_items = total_items;
    params.axis_length = axis_length;
    params.inner_size = inner_size;
    memcpy(params.input_shape, input_shape, input_ndims * sizeof(int));
    size_t blocks = (total_items - 1) / 256 + 1;
    if (blocks > 65535) blocks = 65535;
    cumulative_product_gradient_extract<<<
        static_cast<unsigned int>(blocks), 256, 0, stream>>>(
            scanned, prefix_product, input_gradient, params);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_cumulative_product_gradient(
    const void *input, dtype_t dtype, const int *input_shape,
    const size_t *input_strides, int input_ndims,
    const void *upstream, const size_t *upstream_strides,
    const void *prefix_product, void *input_gradient,
    size_t total_items, int axis, cudaStream_t stream)
{
    if (!total_items) return cudaSuccess;
    if (!input || !input_shape || !input_strides || !upstream ||
        !upstream_strides || !prefix_product || !input_gradient ||
        input_ndims < 1 || input_ndims > MAX_DIMS ||
        axis < -1 || axis >= input_ndims)
        return cudaErrorInvalidValue;

    switch (dtype)
    {
        case DTYPE_FLOAT32:
            return launch_cumulative_product_gradient_typed<float>(
                static_cast<const float *>(input), input_shape, input_strides,
                input_ndims, static_cast<const float *>(upstream),
                upstream_strides, static_cast<const float *>(prefix_product),
                static_cast<float *>(input_gradient), total_items, axis, stream);
        case DTYPE_FLOAT64:
            return launch_cumulative_product_gradient_typed<double>(
                static_cast<const double *>(input), input_shape, input_strides,
                input_ndims, static_cast<const double *>(upstream),
                upstream_strides, static_cast<const double *>(prefix_product),
                static_cast<double *>(input_gradient), total_items, axis, stream);
        default:
            return cudaErrorInvalidValue;
    }
}
