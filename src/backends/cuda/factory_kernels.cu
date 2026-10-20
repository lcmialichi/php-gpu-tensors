#include <cuda_runtime.h>
#include "factory_kernels.cuh"
#include "../../tensor_transform.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "launch_config.cuh"
#include "dispatcher.h"
#include "../../data_types.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

struct transfer_layout
{
    int shape[10];
    size_t strides[10];
    int ndims;
};

template <typename T>
__global__ void pack_strided_kernel(const T *source, T *destination, size_t elements, transfer_layout layout)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < elements; i += (size_t)blockDim.x * gridDim.x)
    {
        size_t remaining = i, offset = 0;
        for (int d = layout.ndims - 1; d >= 0; d--)
        {
            offset += (remaining % layout.shape[d]) * layout.strides[d];
            remaining /= layout.shape[d];
        }
        destination[i] = source[offset];
    }
}

extern "C" cudaError_t launch_pack_strided(
    const void *source, void *destination, size_t elements, size_t element_size,
    const int *shape, const size_t *strides, int ndims)
{
    if (!elements) return cudaSuccess;
    transfer_layout layout;
    if (ndims < 1 || ndims > (int)(sizeof(layout.shape) / sizeof(layout.shape[0])))
        return cudaErrorInvalidValue;
    layout.ndims = ndims;
    for (int d = 0; d < ndims; d++)
    {
        layout.shape[d] = shape[d];
        layout.strides[d] = strides[d];
    }
    dim3 grid = cuda_grid_1d(elements);
    switch (element_size)
    {
        case 1: pack_strided_kernel<<<grid, 256>>>((const uint8_t *)source, (uint8_t *)destination, elements, layout); break;
        case 2: pack_strided_kernel<<<grid, 256>>>((const uint16_t *)source, (uint16_t *)destination, elements, layout); break;
        case 4: pack_strided_kernel<<<grid, 256>>>((const uint32_t *)source, (uint32_t *)destination, elements, layout); break;
        case 8: pack_strided_kernel<<<grid, 256>>>((const uint64_t *)source, (uint64_t *)destination, elements, layout); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

__global__ void bernoulli_kernel(
    const float *values,
    bool *output_data,
    size_t size,
    float p)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output_data[idx] = values[idx] <= p;
    }
}

extern "C" void launch_bernoulli_kernel(
    float *values,
    bool *base,
    size_t total_elements,
    float p)
{
    if (total_elements == 0)
        return;

    bernoulli_kernel<<<cuda_grid_1d(total_elements), 256>>>(values, base, total_elements, p);
}

__global__ static void fill_low_precision_kernel(uint16_t *base, uint16_t value, size_t count)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)blockDim.x * gridDim.x)
        base[i] = value;
}

__global__ static void scale_low_precision_kernel(
    const float *values, uint16_t *base, dtype_t dtype,
    float min_value, float max_value, size_t count)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)blockDim.x * gridDim.x)
    {
        float value = (1.0f - values[i]) * min_value + values[i] * max_value;
        base[i] = dtype == DTYPE_FLOAT16
            ? __half_as_ushort(__float2half_rn(value))
            : __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
}

__device__ static double read_cast_source(const void *source, dtype_t dtype, size_t index)
{
    switch (dtype)
    {
    case DTYPE_FLOAT16: return (double)__half2float(__ushort_as_half(((const uint16_t *)source)[index]));
    case DTYPE_BFLOAT16: return (double)__bfloat162float(__ushort_as_bfloat16(((const uint16_t *)source)[index]));
    case DTYPE_FLOAT32: return ((const float *)source)[index];
    case DTYPE_FLOAT64: return ((const double *)source)[index];
    case DTYPE_INT8: return (double)((const int8_t *)source)[index];
    case DTYPE_INT16: return (double)((const int16_t *)source)[index];
    case DTYPE_INT32: return (double)((const int32_t *)source)[index];
    case DTYPE_INT64: return (double)((const int64_t *)source)[index];
    case DTYPE_UINT8: return (double)((const uint8_t *)source)[index];
    case DTYPE_UINT16: return (double)((const uint16_t *)source)[index];
    case DTYPE_UINT32: return (double)((const uint32_t *)source)[index];
    case DTYPE_UINT64: return (double)((const uint64_t *)source)[index];
    case DTYPE_BOOL: return ((const bool *)source)[index] ? 1.0 : 0.0;
    default: return 0.0;
    }
}

__global__ static void cast_tensor_kernel(
    const void *source, void *destination,
    dtype_t source_dtype, dtype_t destination_dtype, size_t count)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (size_t)blockDim.x * gridDim.x)
    {
        double value = read_cast_source(source, source_dtype, i);
        if (destination_dtype == DTYPE_FLOAT16)
            ((uint16_t *)destination)[i] = __half_as_ushort(__float2half_rn((float)value));
        else if (destination_dtype == DTYPE_BFLOAT16)
            ((uint16_t *)destination)[i] = __bfloat16_as_ushort(__float2bfloat16_rn((float)value));
        else if (destination_dtype == DTYPE_FLOAT32)
            ((float *)destination)[i] = (float)value;
        else if (destination_dtype == DTYPE_FLOAT64)
            ((double *)destination)[i] = (double)value;
    }
}

extern "C" cudaError_t launch_tensor_cast(
    const void *source, void *destination,
    dtype_t source_dtype, dtype_t destination_dtype,
    size_t total_elements)
{
    if (!total_elements) return cudaSuccess;
    int source_supported = source_dtype == DTYPE_FLOAT16 || source_dtype == DTYPE_BFLOAT16 ||
        source_dtype == DTYPE_FLOAT32 || source_dtype == DTYPE_FLOAT64;
    int destination_supported = destination_dtype == DTYPE_FLOAT16 ||
        destination_dtype == DTYPE_BFLOAT16 || destination_dtype == DTYPE_FLOAT32 ||
        destination_dtype == DTYPE_FLOAT64;
    if (!source || !destination || !source_supported || !destination_supported)
        return cudaErrorInvalidValue;
    cast_tensor_kernel<<<cuda_grid_1d(total_elements), 256>>>(
        source, destination, source_dtype, destination_dtype, total_elements);
    return cudaGetLastError();
}

extern "C" void launch_assign_scalar_val_kernel(
    void *base,
    dtype_t dtype,
    scalar_value_t value,
    size_t total_elements)
{
    if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16)
    {
        if (total_elements)
            fill_low_precision_kernel<<<cuda_grid_1d(total_elements), 256>>>(
                (uint16_t *)base, value.v.u16, total_elements);
        return;
    }

    DISPATCH_DTYPE(dtype, {
        scalar_t scalar = get_scalar_value<scalar_t>(value);

        launch_fill_kernel_with_scalar<scalar_t>(
            (scalar_t *)base,
            scalar,
            total_elements);
    });
}

extern "C" void launch_scale_range_kernel(
    float *values,
    void *base,
    dtype_t dtype,
    scalar_value_t min,
    scalar_value_t max,
    size_t total_elements)
{
    if (total_elements == 0)
        return;

    if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16)
    {
        float low_min = dtype == DTYPE_FLOAT16
            ? float16_bits_to_float(min.v.u16)
            : bfloat16_bits_to_float(min.v.u16);
        float low_max = dtype == DTYPE_FLOAT16
            ? float16_bits_to_float(max.v.u16)
            : bfloat16_bits_to_float(max.v.u16);
        scale_low_precision_kernel<<<cuda_grid_1d(total_elements), 256>>>(
            values, (uint16_t *)base, dtype, low_min, low_max, total_elements);
        return;
    }

    DISPATCH_DTYPE(dtype, {
        scalar_t s_min = get_scalar_value<scalar_t>(min);
        scalar_t s_max = get_scalar_value<scalar_t>(max);

        scale_kernel<scalar_t><<<cuda_grid_1d(total_elements), 256>>>(
            values,
            (scalar_t *)base,
            total_elements,
            s_min,
            s_max);
    });
}

template <typename T>
__global__ static void eye_kernel(T *output, size_t total, int columns, int diagonal)
{
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)blockDim.x * gridDim.x)
    {
        int row = (int)(index / (size_t)columns);
        int column = (int)(index % (size_t)columns);
        output[index] = (int64_t)column - row == diagonal ? (T)1 : (T)0;
    }
}

__global__ static void eye_low_precision_kernel(uint16_t *output, size_t total,
                                                 int columns, int diagonal,
                                                 uint16_t one)
{
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += (size_t)blockDim.x * gridDim.x)
    {
        int row = (int)(index / (size_t)columns);
        int column = (int)(index % (size_t)columns);
        output[index] = (int64_t)column - row == diagonal ? one : 0;
    }
}

extern "C" cudaError_t launch_eye(void *output, dtype_t dtype, int columns,
                                    int diagonal, size_t total_elements)
{
    if (!total_elements) return cudaSuccess;
    if (!output || columns <= 0) return cudaErrorInvalidValue;
    dim3 grid = cuda_grid_1d(total_elements);
    if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16)
    {
        uint16_t one = dtype == DTYPE_FLOAT16 ? 0x3c00 : 0x3f80;
        eye_low_precision_kernel<<<grid, 256>>>(
            (uint16_t *)output, total_elements, columns, diagonal, one);
    }
    else
    {
        switch (dtype)
        {
            case DTYPE_FLOAT32: eye_kernel<<<grid, 256>>>((float *)output, total_elements, columns, diagonal); break;
            case DTYPE_FLOAT64: eye_kernel<<<grid, 256>>>((double *)output, total_elements, columns, diagonal); break;
            case DTYPE_INT8: eye_kernel<<<grid, 256>>>((int8_t *)output, total_elements, columns, diagonal); break;
            case DTYPE_INT16: eye_kernel<<<grid, 256>>>((int16_t *)output, total_elements, columns, diagonal); break;
            case DTYPE_INT32: eye_kernel<<<grid, 256>>>((int32_t *)output, total_elements, columns, diagonal); break;
            case DTYPE_INT64: eye_kernel<<<grid, 256>>>((int64_t *)output, total_elements, columns, diagonal); break;
            case DTYPE_UINT8: eye_kernel<<<grid, 256>>>((uint8_t *)output, total_elements, columns, diagonal); break;
            case DTYPE_UINT16: eye_kernel<<<grid, 256>>>((uint16_t *)output, total_elements, columns, diagonal); break;
            case DTYPE_UINT32: eye_kernel<<<grid, 256>>>((uint32_t *)output, total_elements, columns, diagonal); break;
            case DTYPE_UINT64: eye_kernel<<<grid, 256>>>((uint64_t *)output, total_elements, columns, diagonal); break;
            case DTYPE_BOOL: eye_kernel<<<grid, 256>>>((bool *)output, total_elements, columns, diagonal); break;
            default: return cudaErrorInvalidValue;
        }
    }
    return cudaGetLastError();
}

__device__ static void store_generated_value(void *output, dtype_t dtype,
                                              size_t index, double value)
{
    switch (dtype)
    {
        case DTYPE_FLOAT16:
            ((uint16_t *)output)[index] =
                __half_as_ushort(__float2half_rn((float)value));
            break;
        case DTYPE_BFLOAT16:
            ((uint16_t *)output)[index] =
                __bfloat16_as_ushort(__float2bfloat16_rn((float)value));
            break;
        case DTYPE_FLOAT32: ((float *)output)[index] = (float)value; break;
        case DTYPE_FLOAT64: ((double *)output)[index] = value; break;
        case DTYPE_INT8: ((int8_t *)output)[index] = (int8_t)value; break;
        case DTYPE_INT16: ((int16_t *)output)[index] = (int16_t)value; break;
        case DTYPE_INT32: ((int32_t *)output)[index] = (int32_t)value; break;
        case DTYPE_INT64: ((int64_t *)output)[index] = (int64_t)value; break;
        case DTYPE_UINT8: ((uint8_t *)output)[index] = (uint8_t)value; break;
        case DTYPE_UINT16: ((uint16_t *)output)[index] = (uint16_t)value; break;
        case DTYPE_UINT32: ((uint32_t *)output)[index] = (uint32_t)value; break;
        case DTYPE_UINT64: ((uint64_t *)output)[index] = (uint64_t)value; break;
        case DTYPE_BOOL: ((bool *)output)[index] = value != 0.0; break;
        default: break;
    }
}

__device__ static void store_generated_integer(void *output, dtype_t dtype,
                                                size_t index, int64_t value)
{
    if (dtype == DTYPE_INT64)
        ((int64_t *)output)[index] = value;
    else if (dtype == DTYPE_UINT64)
        ((uint64_t *)output)[index] = (uint64_t)value;
    else
        store_generated_value(output, dtype, index, (double)value);
}

__global__ static void arange_kernel(void *output, dtype_t dtype,
                                      size_t total_elements, double start,
                                      double step, int exact_integer,
                                      int64_t integer_start, int64_t integer_step)
{
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements; index += (size_t)blockDim.x * gridDim.x)
    {
        if (exact_integer)
        {
            uint64_t bits = (uint64_t)integer_start +
                (uint64_t)index * (uint64_t)integer_step;
            store_generated_integer(output, dtype, index, (int64_t)bits);
        }
        else
        {
            store_generated_value(output, dtype, index,
                                  fma((double)index, step, start));
        }
    }
}

extern "C" cudaError_t launch_arange(void *output, dtype_t dtype,
                                      size_t total_elements, double start,
                                      double step, int exact_integer,
                                      int64_t integer_start, int64_t integer_step)
{
    if (dtype < 0 || dtype >= DTYPE_COUNT || dtype == DTYPE_UNKNOWN ||
        total_elements > INT_MAX)
        return cudaErrorInvalidValue;
    if (!total_elements) return cudaSuccess;
    if (!output) return cudaErrorInvalidValue;
    arange_kernel<<<cuda_grid_1d(total_elements), 256>>>(
        output, dtype, total_elements, start, step, exact_integer,
        integer_start, integer_step);
    return cudaGetLastError();
}

__global__ static void linspace_kernel(void *output, dtype_t dtype,
                                        size_t total_elements, double start,
                                        double stop, int endpoint, int exact_integer,
                                        int64_t integer_start, int64_t integer_stop)
{
    const double denominator = endpoint
        ? (total_elements > 1 ? (double)(total_elements - 1) : 1.0)
        : (double)total_elements;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements; index += (size_t)blockDim.x * gridDim.x)
    {
        if (exact_integer)
        {
            uint64_t divisor = endpoint
                ? (total_elements > 1 ? total_elements - 1 : 1) : total_elements;
            int ascending = integer_stop >= integer_start;
            uint64_t distance = ascending
                ? (uint64_t)integer_stop - (uint64_t)integer_start
                : (uint64_t)integer_start - (uint64_t)integer_stop;
            // Split the product to cover the full int64 span without overflowing.
            uint64_t remainder_product = (distance % divisor) * index;
            uint64_t offset = (distance / divisor) * index + remainder_product / divisor;
            int64_t value = (int64_t)(ascending
                ? (uint64_t)integer_start + offset : (uint64_t)integer_start - offset);
            if (remainder_product % divisor)
            {
                if (ascending && value < 0) value++;
                else if (!ascending && value > 0) value--;
            }
            store_generated_integer(output, dtype, index, value);
            continue;
        }
        double ratio = (double)index / denominator;
        double value = total_elements == 1 || start == stop
            ? start
            : ((start < 0) == (stop < 0)
                ? fma(ratio, stop - start, start)
                : start * (1.0 - ratio) + stop * ratio);
        if (endpoint && total_elements > 1 && index + 1 == total_elements)
            value = stop;
        store_generated_value(output, dtype, index, value);
    }
}

extern "C" cudaError_t launch_linspace(void *output, dtype_t dtype,
                                        size_t total_elements, double start,
                                        double stop, int endpoint, int exact_integer,
                                        int64_t integer_start, int64_t integer_stop)
{
    if (dtype < 0 || dtype >= DTYPE_COUNT || dtype == DTYPE_UNKNOWN ||
        total_elements > INT_MAX)
        return cudaErrorInvalidValue;
    if (!total_elements) return cudaSuccess;
    if (!output) return cudaErrorInvalidValue;
    linspace_kernel<<<cuda_grid_1d(total_elements), 256>>>(
        output, dtype, total_elements, start, stop, endpoint, exact_integer,
        integer_start, integer_stop);
    return cudaGetLastError();
}

__device__ static void tensor_transform_source(
    const tensor_transform_spec_t &spec, size_t output_index,
    int *input_coordinates, size_t *storage_offset, size_t *logical_index)
{
    int output_coordinates[MAX_DIMS] = {0};
    size_t remaining = output_index;
    for (int axis = spec.output_ndims - 1; axis >= 0; axis--)
    {
        int extent = spec.output_shape[axis];
        output_coordinates[axis] = (int)(remaining % (size_t)extent);
        remaining /= (size_t)extent;
    }

    for (int axis = 0; axis < spec.input_ndims; axis++)
        input_coordinates[axis] = 0;

    if (spec.op == OP_DIAGONAL)
    {
        int diagonal_index = output_coordinates[spec.output_ndims - 1];
        input_coordinates[spec.axis1] =
            diagonal_index + (spec.offset < 0 ? -spec.offset : 0);
        input_coordinates[spec.axis2] =
            diagonal_index + (spec.offset > 0 ? spec.offset : 0);
        int output_axis = 0;
        for (int input_axis = 0; input_axis < spec.input_ndims; input_axis++)
        {
            if (input_axis == spec.axis1 || input_axis == spec.axis2)
                continue;
            input_coordinates[input_axis] = output_coordinates[output_axis++];
        }
    }
    else if (spec.op == OP_REPEAT && spec.axis < 0)
    {
        size_t input_index = output_index / (size_t)spec.repeats;
        for (int axis = spec.input_ndims - 1; axis >= 0; axis--)
        {
            input_coordinates[axis] =
                (int)(input_index % (size_t)spec.input_shape[axis]);
            input_index /= (size_t)spec.input_shape[axis];
        }
    }
    else if (spec.op == OP_TILE)
    {
        int leading_axes = spec.output_ndims - spec.input_ndims;
        for (int axis = 0; axis < spec.input_ndims; axis++)
        {
            int output_axis = axis + leading_axes;
            input_coordinates[axis] =
                output_coordinates[output_axis] % spec.input_shape[axis];
        }
    }
    else if (spec.op == OP_ROLL && spec.axis < 0)
    {
        size_t total = 1;
        for (int axis = 0; axis < spec.input_ndims; axis++)
            total *= (size_t)spec.input_shape[axis];
        size_t shift = (size_t)spec.shifts[0];
        size_t source_index = output_index >= (size_t)shift
            ? output_index - (size_t)shift
            : total - ((size_t)shift - output_index);
        for (int axis = spec.input_ndims - 1; axis >= 0; axis--)
        {
            input_coordinates[axis] =
                (int)(source_index % (size_t)spec.input_shape[axis]);
            source_index /= (size_t)spec.input_shape[axis];
        }
    }
    else
    {
        for (int axis = 0; axis < spec.input_ndims; axis++)
        {
            int coordinate = output_coordinates[axis];
            if (spec.op == OP_REPEAT && axis == spec.axis)
                coordinate /= spec.repeats;
            else if (spec.op == OP_FLIP && spec.axes[axis])
                coordinate = spec.input_shape[axis] - 1 - coordinate;
            else if (spec.op == OP_ROLL && spec.shifts[axis])
            {
                coordinate -= (int)spec.shifts[axis];
                if (coordinate < 0) coordinate += spec.input_shape[axis];
            }
            input_coordinates[axis] = coordinate;
        }
    }

    *storage_offset = 0;
    *logical_index = 0;
    for (int axis = 0; axis < spec.input_ndims; axis++)
    {
        *storage_offset += (size_t)input_coordinates[axis] *
            spec.input_strides[axis];
        *logical_index = *logical_index * (size_t)spec.input_shape[axis] +
            (size_t)input_coordinates[axis];
    }
}

template <typename ElementT>
__global__ static void tensor_transform_kernel(
    const ElementT *input, ElementT *output,
    tensor_transform_spec_t spec, size_t total_elements)
{
    for (size_t output_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         output_index < total_elements;
         output_index += (size_t)blockDim.x * gridDim.x)
    {
        int input_coordinates[MAX_DIMS];
        size_t source_offset, logical_index;
        tensor_transform_source(spec, output_index, input_coordinates,
                                &source_offset, &logical_index);
        output[output_index] = input[source_offset];
    }
}

extern "C" cudaError_t launch_tensor_transform(
    const void *input, void *output, dtype_t dtype,
    const tensor_transform_spec_t *spec, size_t total_elements)
{
    if (!total_elements) return cudaSuccess;
    if (!input || !output || !spec || spec->input_ndims < 1 ||
        spec->input_ndims > MAX_DIMS || spec->output_ndims < 1 ||
        spec->output_ndims > MAX_DIMS)
        return cudaErrorInvalidValue;

    size_t element_size;
    switch (dtype)
    {
        case DTYPE_BOOL:
        case DTYPE_INT8:
        case DTYPE_UINT8: element_size = 1; break;
        case DTYPE_FLOAT16:
        case DTYPE_BFLOAT16:
        case DTYPE_INT16:
        case DTYPE_UINT16: element_size = 2; break;
        case DTYPE_FLOAT32:
        case DTYPE_INT32:
        case DTYPE_UINT32: element_size = 4; break;
        case DTYPE_FLOAT64:
        case DTYPE_INT64:
        case DTYPE_UINT64: element_size = 8; break;
        default: return cudaErrorInvalidValue;
    }

    dim3 grid = cuda_grid_1d(total_elements);
    switch (element_size)
    {
        case 1: tensor_transform_kernel<<<grid, 256>>>(
            (const uint8_t *)input, (uint8_t *)output, *spec, total_elements); break;
        case 2: tensor_transform_kernel<<<grid, 256>>>(
            (const uint16_t *)input, (uint16_t *)output, *spec, total_elements); break;
        case 4: tensor_transform_kernel<<<grid, 256>>>(
            (const uint32_t *)input, (uint32_t *)output, *spec, total_elements); break;
        case 8: tensor_transform_kernel<<<grid, 256>>>(
            (const uint64_t *)input, (uint64_t *)output, *spec, total_elements); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

struct RepeatCountParams
{
    int input_ndims;
    int output_ndims;
    int axis;
    int input_shape[MAX_DIMS];
    size_t input_strides[MAX_DIMS];
    int output_shape[MAX_DIMS];
    size_t count;
};

template <typename ElementT>
__global__ static void tensor_repeat_counts_kernel(
    const ElementT *__restrict__ input, ElementT *__restrict__ output,
    const int *__restrict__ prefix_counts, RepeatCountParams params,
    size_t output_elements)
{
    for (size_t output_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         output_index < output_elements;
         output_index += (size_t)blockDim.x * gridDim.x)
    {
        size_t repeated_coordinate;
        size_t logical_input_index;
        size_t source_offset = 0;
        if (params.axis < 0)
        {
            repeated_coordinate = output_index;
            size_t low = 0;
            size_t high = params.count;
            while (low < high)
            {
                size_t middle = low + (high - low) / 2;
                if ((size_t)prefix_counts[middle] <= repeated_coordinate)
                    low = middle + 1;
                else
                    high = middle;
            }
            logical_input_index = low;
            for (int axis = params.input_ndims - 1; axis >= 0; axis--)
            {
                size_t coordinate = logical_input_index %
                    (size_t)params.input_shape[axis];
                logical_input_index /= (size_t)params.input_shape[axis];
                source_offset += coordinate * params.input_strides[axis];
            }
        }
        else
        {
            size_t remaining = output_index;
            size_t coordinates[MAX_DIMS] = {};
            for (int axis = params.output_ndims - 1; axis >= 0; axis--)
            {
                size_t extent = (size_t)params.output_shape[axis];
                coordinates[axis] = remaining % extent;
                remaining /= extent;
            }
            repeated_coordinate = coordinates[params.axis];
            size_t low = 0;
            size_t high = params.count;
            while (low < high)
            {
                size_t middle = low + (high - low) / 2;
                if ((size_t)prefix_counts[middle] <= repeated_coordinate)
                    low = middle + 1;
                else
                    high = middle;
            }
            for (int axis = 0; axis < params.input_ndims; axis++)
            {
                size_t coordinate = axis == params.axis
                    ? low : coordinates[axis];
                source_offset += coordinate * params.input_strides[axis];
            }
        }
        output[output_index] = input[source_offset];
    }
}

extern "C" cudaError_t launch_tensor_repeat_counts(
    const void *input, void *output, size_t element_size,
    const int *prefix_counts, size_t count, int axis,
    const int *input_shape, const size_t *input_strides, int input_ndims,
    const int *output_shape, int output_ndims, size_t output_elements)
{
    if (!output_elements) return cudaSuccess;
    if (!input || !output || !prefix_counts || !count ||
        !input_shape || !input_strides || !output_shape ||
        input_ndims < 1 || input_ndims > MAX_DIMS ||
        output_ndims < 1 || output_ndims > MAX_DIMS ||
        axis < -1 ||
        (axis >= 0 && axis >= input_ndims) ||
        (axis < 0 && output_ndims != 1) ||
        count > SIZE_MAX / sizeof(int) ||
        (size_t)prefix_counts[count - 1] !=
            (size_t)output_shape[axis < 0 ? 0 : axis])
        return cudaErrorInvalidValue;

    RepeatCountParams params = {};
    params.input_ndims = input_ndims;
    params.output_ndims = output_ndims;
    params.axis = axis;
    params.count = count;
    memcpy(params.input_shape, input_shape, input_ndims * sizeof(int));
    memcpy(params.input_strides, input_strides, input_ndims * sizeof(size_t));
    memcpy(params.output_shape, output_shape, output_ndims * sizeof(int));

    int *device_prefix = nullptr;
    cudaError_t error = cudaMalloc(&device_prefix, count * sizeof(int));
    if (error != cudaSuccess) return error;
    error = cudaMemcpy(device_prefix, prefix_counts, count * sizeof(int),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cudaFree(device_prefix);
        return error;
    }

    dim3 grid = cuda_grid_1d(output_elements);
    switch (element_size)
    {
        case 1: tensor_repeat_counts_kernel<<<grid, 256>>>(
            (const uint8_t *)input, (uint8_t *)output, device_prefix,
            params, output_elements); break;
        case 2: tensor_repeat_counts_kernel<<<grid, 256>>>(
            (const uint16_t *)input, (uint16_t *)output, device_prefix,
            params, output_elements); break;
        case 4: tensor_repeat_counts_kernel<<<grid, 256>>>(
            (const uint32_t *)input, (uint32_t *)output, device_prefix,
            params, output_elements); break;
        case 8: tensor_repeat_counts_kernel<<<grid, 256>>>(
            (const uint64_t *)input, (uint64_t *)output, device_prefix,
            params, output_elements); break;
        default:
            cudaFree(device_prefix);
            return cudaErrorInvalidValue;
    }
    error = cudaGetLastError();
    cudaError_t free_error = cudaFree(device_prefix);
    return error == cudaSuccess ? free_error : error;
}

template <typename GradientT>
__global__ static void tensor_repeat_counts_gradient_kernel(
    const GradientT *__restrict__ upstream,
    GradientT *__restrict__ input_gradient,
    const int *__restrict__ prefix_counts,
    RepeatCountParams params, size_t input_elements)
{
    for (size_t input_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         input_index < input_elements;
         input_index += (size_t)blockDim.x * gridDim.x)
    {
        size_t coordinates[MAX_DIMS] = {};
        size_t remaining = input_index;
        for (int axis = params.input_ndims - 1; axis >= 0; axis--)
        {
            coordinates[axis] = remaining % (size_t)params.input_shape[axis];
            remaining /= (size_t)params.input_shape[axis];
        }

        size_t count_index = params.axis < 0
            ? input_index : coordinates[params.axis];
        size_t first_repeat = count_index
            ? (size_t)prefix_counts[count_index - 1] : 0;
        size_t end_repeat = (size_t)prefix_counts[count_index];
        GradientT sum = static_cast<GradientT>(0);
        for (size_t repeat = first_repeat; repeat < end_repeat; repeat++)
        {
            size_t upstream_offset = 0;
            if (params.axis < 0)
                upstream_offset = repeat;
            else
                for (int axis = 0; axis < params.input_ndims; axis++)
                    upstream_offset += (axis == params.axis
                        ? repeat : coordinates[axis]) *
                        params.input_strides[axis];
            sum += upstream[upstream_offset];
        }
        input_gradient[input_index] = sum;
    }
}

extern "C" cudaError_t launch_tensor_repeat_counts_gradient(
    const void *upstream, void *input_gradient, dtype_t dtype,
    const int *input_shape, int input_ndims,
    const size_t *upstream_strides,
    const int *prefix_counts, size_t count, int axis,
    size_t input_elements)
{
    if (!input_elements) return cudaSuccess;
    if (!upstream || !input_gradient || !prefix_counts ||
        !input_shape || !upstream_strides ||
        input_ndims < 1 || input_ndims > MAX_DIMS ||
        axis < -1 || (axis >= 0 && axis >= input_ndims) ||
        (axis < 0 && count != input_elements) ||
        (axis >= 0 && count != (size_t)input_shape[axis]) ||
        !count || count > SIZE_MAX / sizeof(int) ||
        (dtype != DTYPE_FLOAT32 && dtype != DTYPE_FLOAT64))
        return cudaErrorInvalidValue;

    RepeatCountParams params = {};
    params.input_ndims = input_ndims;
    params.axis = axis;
    params.count = count;
    memcpy(params.input_shape, input_shape, input_ndims * sizeof(int));
    if (axis >= 0)
        memcpy(params.input_strides, upstream_strides,
               input_ndims * sizeof(size_t));

    int *device_prefix = nullptr;
    cudaError_t error = cudaMalloc(&device_prefix, count * sizeof(int));
    if (error != cudaSuccess) return error;
    error = cudaMemcpy(device_prefix, prefix_counts, count * sizeof(int),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cudaFree(device_prefix);
        return error;
    }

    dim3 grid = cuda_grid_1d(input_elements);
    if (dtype == DTYPE_FLOAT32)
        tensor_repeat_counts_gradient_kernel<<<grid, 256>>>(
            (const float *)upstream, (float *)input_gradient,
            device_prefix, params, input_elements);
    else
        tensor_repeat_counts_gradient_kernel<<<grid, 256>>>(
            (const double *)upstream, (double *)input_gradient,
            device_prefix, params, input_elements);
    error = cudaGetLastError();
    cudaError_t free_error = cudaFree(device_prefix);
    return error == cudaSuccess ? free_error : error;
}

template <typename GradientT>
__device__ static void tensor_transform_atomic_add(GradientT *address,
                                                    GradientT value);

template <>
__device__ void tensor_transform_atomic_add<float>(float *address, float value)
{
    atomicAdd(address, value);
}

template <>
__device__ void tensor_transform_atomic_add<double>(double *address, double value)
{
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, value);
#else
    unsigned long long *bits = (unsigned long long *)address;
    unsigned long long old = *bits, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(bits, assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

template <typename GradientT>
__global__ static void tensor_transform_gradient_kernel(
    const GradientT *upstream, GradientT *input_gradient,
    tensor_transform_spec_t spec, size_t total_elements)
{
    for (size_t output_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         output_index < total_elements;
         output_index += (size_t)blockDim.x * gridDim.x)
    {
        int input_coordinates[MAX_DIMS], output_coordinates[MAX_DIMS] = {0};
        size_t source_offset, logical_index, remaining = output_index;
        for (int axis = spec.output_ndims - 1; axis >= 0; axis--)
        {
            int extent = spec.output_shape[axis];
            output_coordinates[axis] = (int)(remaining % (size_t)extent);
            remaining /= (size_t)extent;
        }
        size_t upstream_offset = 0;
        for (int axis = 0; axis < spec.output_ndims; axis++)
            upstream_offset += (size_t)output_coordinates[axis] *
                spec.upstream_strides[axis];
        tensor_transform_source(spec, output_index, input_coordinates,
                                &source_offset, &logical_index);
        tensor_transform_atomic_add(input_gradient + logical_index,
                                    upstream[upstream_offset]);
    }
}

extern "C" cudaError_t launch_tensor_transform_gradient(
    const void *upstream, void *input_gradient, dtype_t dtype,
    const tensor_transform_spec_t *spec, size_t total_elements,
    cudaStream_t stream)
{
    if (!total_elements) return cudaSuccess;
    if (!upstream || !input_gradient || !spec ||
        (dtype != DTYPE_FLOAT32 && dtype != DTYPE_FLOAT64))
        return cudaErrorInvalidValue;

    dim3 grid = cuda_grid_1d(total_elements);
    if (dtype == DTYPE_FLOAT32)
        tensor_transform_gradient_kernel<<<grid, 256, 0, stream>>>(
            (const float *)upstream, (float *)input_gradient, *spec,
            total_elements);
    else
        tensor_transform_gradient_kernel<<<grid, 256, 0, stream>>>(
            (const double *)upstream, (double *)input_gradient, *spec,
            total_elements);
    return cudaGetLastError();
}

template <typename RandomT>
__global__ static void normal_scale_kernel(const RandomT *values, void *output,
                                            dtype_t dtype, double mean,
                                            double standard_deviation,
                                            size_t total_elements)
{
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements; index += (size_t)blockDim.x * gridDim.x)
    {
        double value = (double)values[index] * standard_deviation + mean;
        switch (dtype)
        {
            case DTYPE_FLOAT16:
                ((uint16_t *)output)[index] =
                    __half_as_ushort(__float2half_rn((float)value));
                break;
            case DTYPE_BFLOAT16:
                ((uint16_t *)output)[index] =
                    __bfloat16_as_ushort(__float2bfloat16_rn((float)value));
                break;
            case DTYPE_FLOAT32: ((float *)output)[index] = (float)value; break;
            case DTYPE_FLOAT64: ((double *)output)[index] = value; break;
            default: break;
        }
    }
}

extern "C" cudaError_t launch_scale_normal_kernel(const float *values, void *output,
                                                    dtype_t dtype, double mean,
                                                    double standard_deviation,
                                                    size_t total_elements)
{
    if (!total_elements) return cudaSuccess;
    if (!values || !output || (dtype != DTYPE_FLOAT16 && dtype != DTYPE_BFLOAT16 &&
        dtype != DTYPE_FLOAT32 && dtype != DTYPE_FLOAT64))
        return cudaErrorInvalidValue;
    normal_scale_kernel<<<cuda_grid_1d(total_elements), 256>>>(
        values, output, dtype, mean, standard_deviation, total_elements);
    return cudaGetLastError();
}

extern "C" cudaError_t launch_scale_normal_double_kernel(const double *values, void *output,
                                                           double mean,
                                                           double standard_deviation,
                                                           size_t total_elements)
{
    if (!total_elements) return cudaSuccess;
    if (!values || !output) return cudaErrorInvalidValue;
    normal_scale_kernel<<<cuda_grid_1d(total_elements), 256>>>(
        values, output, DTYPE_FLOAT64, mean, standard_deviation, total_elements);
    return cudaGetLastError();
}
