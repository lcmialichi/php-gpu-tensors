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
    AUTOGRAD_WHERE,
    AUTOGRAD_GATHER,
    AUTOGRAD_SCATTER_ADD,
    AUTOGRAD_UNSUPPORTED,
    AUTOGRAD_CONCAT,
    AUTOGRAD_SLICE
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
    uint64_t result_version;
    uint64_t a_version;
    uint64_t b_version;
    uint64_t c_version;
    tensor_t *parents[MAX_CONCAT_TENSORS];
    uint64_t parent_versions[MAX_CONCAT_TENSORS];
    size_t parent_count;
    int input_shape[MAX_DIMS];
    int input_ndims;
    int starts[MAX_DIMS];
    int steps[MAX_DIMS];
} autograd_node_t;

void autograd_release_node(tensor_t *tensor);
void autograd_clear_gradient(tensor_t *tensor);
int autograd_record_binary(tensor_t *result, tensor_t *a, tensor_t *b, operation_type_t op);
int autograd_record_scalar(tensor_t *result, tensor_t *a, scalar_value_t scalar, operation_type_t op, int inverse);
int autograd_record_unary(tensor_t *result, tensor_t *a, operation_type_t op);
int autograd_record_reduce(tensor_t *result, tensor_t *a, operation_type_t op, int axis);
int autograd_record_matmul(tensor_t *result, tensor_t *a, tensor_t *b);
int autograd_record_view(tensor_t *result, tensor_t *a, operation_type_t op, const int *axes);
int autograd_record_slice(tensor_t *result, tensor_t *a, const int *axes,
                          const int *starts, const int *steps);
int autograd_record_concat(tensor_t *result, tensor_t **parents,
                           size_t parent_count, int axis);
int autograd_record_gather(tensor_t *result, tensor_t *input, tensor_t *indices, int axis);
int autograd_record_scatter_add(tensor_t *result, tensor_t *input, tensor_t *indices,
                                tensor_t *updates, int axis);
int autograd_record_cast(tensor_t *result, tensor_t *a);
int autograd_record_where(tensor_t *result, tensor_t *condition, tensor_t *x, tensor_t *y);
int autograd_backward(tensor_t *tensor, tensor_t *gradient, int retain_graph);
uint64_t autograd_tensor_version(const tensor_t *tensor);
void autograd_mark_mutated(tensor_t *tensor);
int autograd_set_requires_grad(tensor_t *tensor, int requires_grad);
tensor_t *autograd_detach(tensor_t *tensor);

#endif
