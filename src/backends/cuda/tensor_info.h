#ifndef TENSOR_INFO_H
#define TENSOR_INFO_H
#include <stdint.h>

typedef struct {
    float *data;
    int32_t *shape;
    int32_t *strides;

    int32_t ndims;
    int32_t offset;
} TensorInfo; 

#endif