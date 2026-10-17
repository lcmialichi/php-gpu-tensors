#ifndef FACTORY_KERNELS_H
#define FACTORY_KERNELS_H
#include "../../data_types.h"
#include "../../tensor_transform.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

    cudaError_t launch_pack_strided(
        const void *source, void *destination, size_t elements, size_t element_size,
        const int *shape, const size_t *strides, int ndims);

    void launch_bernoulli_kernel(
        float *values,
        bool *base,
        size_t total_elements,
        float p);
        
    void launch_assign_scalar_val_kernel(
        void *base,
        dtype_t dtype,
        scalar_value_t value,
        size_t total_elements);

    cudaError_t launch_tensor_cast(
        const void *source, void *destination,
        dtype_t source_dtype, dtype_t destination_dtype,
        size_t total_elements);

    void launch_scale_range_kernel(
        float *values,
        void *base,
        dtype_t dtype,
        scalar_value_t min,
        scalar_value_t max,
        size_t total_elements);

    cudaError_t launch_eye(void *output, dtype_t dtype, int columns,
                           int diagonal, size_t total_elements);
    cudaError_t launch_arange(void *output, dtype_t dtype, size_t total_elements,
                              double start, double step, int exact_integer,
                              int64_t integer_start, int64_t integer_step);
    cudaError_t launch_linspace(void *output, dtype_t dtype, size_t total_elements,
                                double start, double stop, int endpoint);
    cudaError_t launch_tensor_transform(const void *input, void *output,
                                        dtype_t dtype,
                                        const tensor_transform_spec_t *spec,
                                        size_t total_elements);
    cudaError_t launch_tensor_repeat_counts(
        const void *input, void *output, size_t element_size,
        const int *prefix_counts, size_t count, int axis,
        const int *input_shape, const size_t *input_strides, int input_ndims,
        const int *output_shape, int output_ndims, size_t output_elements);
    cudaError_t launch_tensor_repeat_counts_gradient(
        const void *upstream, void *input_gradient, dtype_t dtype,
        const int *input_shape, int input_ndims,
        const size_t *upstream_strides,
        const int *prefix_counts, size_t count, int axis,
        size_t input_elements);
    cudaError_t launch_tensor_transform_gradient(
        const void *upstream, void *input_gradient, dtype_t dtype,
        const tensor_transform_spec_t *spec, size_t total_elements,
        cudaStream_t stream);

    cudaError_t launch_scale_normal_kernel(const float *values, void *output,
                                           dtype_t dtype, double mean,
                                           double standard_deviation,
                                           size_t total_elements);
    cudaError_t launch_scale_normal_double_kernel(const double *values, void *output,
                                                  double mean,
                                                  double standard_deviation,
                                                  size_t total_elements);

#ifdef __cplusplus
}
#endif

#endif