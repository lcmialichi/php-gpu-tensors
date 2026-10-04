#ifndef CUDA_FUSION_H
#define CUDA_FUSION_H

#include "php.h"
#include "tensor.h"
#include "operations.h"

struct fusion_scope;
struct fusion_node;

int fusion_init(void);
int fusion_active(void);
int fusion_check_mutation(void);
int fusion_check_tensor_mutation(tensor_t *tensor);
int fusion_materialize(tensor_t *tensor);
void fusion_release_node(tensor_t *tensor);
tensor_t *fusion_binary(tensor_t *a, tensor_t *b, operation_type_t op);
tensor_t *fusion_scalar(tensor_t *a, scalar_value_t scalar, operation_type_t op, int inverse);
tensor_t *fusion_unary(tensor_t *a, operation_type_t op);
tensor_t *fusion_cast(tensor_t *a, dtype_t dtype);
tensor_t *fusion_cast_eager(tensor_t *a, dtype_t dtype);
tensor_t *fusion_where(tensor_t *condition, tensor_t *x, tensor_t *y);
tensor_t *fusion_reduce(tensor_t *a, int axis, operation_type_t op, int arg);
tensor_t *fusion_matmul(tensor_t *a, tensor_t *b);
tensor_t *fusion_view(tensor_t *a, operation_type_t op, int *shape, size_t *strides, int ndims, int *axes);
tensor_t *fusion_slice(tensor_t *a, int *shape, size_t *strides, int ndims,
                       int *axes, int *starts, int *steps);
void fusion_request_shutdown(void);

#endif
