#ifndef CUDA_NN_BACKEND_H
#define CUDA_NN_BACKEND_H
#ifdef __cplusplus
extern "C" {
#endif
int cuda_nn_available(void);
void cuda_nn_shutdown(void);
const char *cuda_nn_error(void);
int cuda_nn_conv(const float *input, const float *weights, const float *bias, float *output,
                 const int *input_shape, const int *filter_shape, const int *output_shape,
                 const int *stride, const int *padding, const int *dilation, int groups);
int cuda_nn_pool(const float *input, float *output, const int *input_shape, const int *output_shape,
                 const int *window, const int *stride, const int *padding, int average);
int cuda_nn_softmax(const float *input, float *output, const int *shape);
#ifdef __cplusplus
}
#endif
#endif
