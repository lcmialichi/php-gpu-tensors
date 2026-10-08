#ifndef CUDA_AUTOGRAD_H
#define CUDA_AUTOGRAD_H

#include "tensor.h"
#include "operations_strctures.h"

typedef enum
{
    AUTOGRAD_BINARY,
    AUTOGRAD_SCALAR,
    AUTOGRAD_UNARY,
    AUTOGRAD_REDUCE,
    AUTOGRAD_MATMUL,
    AUTOGRAD_VIEW,
    AUTOGRAD_CAST,
    AUTOGRAD_WHERE
} autograd_kind_t;

typedef struct autograd_node
{
    autograd_kind_t kind;
    tensor_t *a;
    tensor_t *b;
    tensor_t *c;
    operation_type_t op;
    scalar_value_t scalar;
    int parameter;
    int axes[MAX_DIMS];
    int ndims;
} autograd_node_t;

void autograd_release_node(tensor_t *tensor);
void autograd_clear_gradient(tensor_t *tensor);
int autograd_record_binary(tensor_t *result, tensor_t *a, tensor_t *b, operation_type_t op);
int autograd_record_scalar(tensor_t *result, tensor_t *a, scalar_value_t scalar, operation_type_t op, int inverse);
int autograd_record_unary(tensor_t *result, tensor_t *a, operation_type_t op);
int autograd_record_reduce(tensor_t *result, tensor_t *a, operation_type_t op, int axis);
int autograd_record_matmul(tensor_t *result, tensor_t *a, tensor_t *b);
int autograd_record_view(tensor_t *result, tensor_t *a, operation_type_t op, const int *axes);
int autograd_record_cast(tensor_t *result, tensor_t *a);
int autograd_record_where(tensor_t *result, tensor_t *condition, tensor_t *x, tensor_t *y);
int autograd_backward(tensor_t *tensor, tensor_t *gradient);
int autograd_set_requires_grad(tensor_t *tensor, int requires_grad);
tensor_t *autograd_detach(tensor_t *tensor);

#endif
