#include "fusion_internal.h"
int fusion_active(void)
{
    return CUDA_G(fusion_scope) != NULL;
}

int fusion_check_mutation(void)
{
    if (fusion_active() || CUDA_G(fusion_pending))
    {
        CUDA_THROW_RUNTIME("Mutation, custom kernel launches and device changes are not allowed during Fusion capture or pending execution");
        return 0;
    }
    return 1;
}

int fusion_check_tensor_mutation(tensor_t *tensor)
{
    if (!fusion_check_mutation()) return 0;
    while (tensor && tensor->base_tensor) tensor = tensor->base_tensor;
    if (tensor && tensor->fusion_readers)
    {
        CUDA_THROW_RUNTIME("Tensor storage is in use by an asynchronous Fusion execution");
        return 0;
    }
    return 1;
}

static tensor_t *fusion_metadata(const int *shape, int ndims, dtype_t dtype)
{
    if (ndims < 0 || ndims > MAX_DIMS || !dtype_is_numeric_or_bool(dtype))
    {
        CUDA_THROW_INVALID("Invalid fusion tensor metadata");
        return NULL;
    }
    tensor_t *tensor = ecalloc(1, sizeof(tensor_t));
    tensor->dtype = dtype;
    tensor->element_size = dtype_size(dtype);
    tensor->ndims = ndims;
    tensor->ref_count = 1;
    tensor->is_on_gpu = 1;
    tensor->is_contiguous_cached = 1;
    tensor->total_size = 1;
    if (ndims)
    {
        tensor->shape = emalloc(ndims * sizeof(int));
        tensor->strides = emalloc(ndims * sizeof(size_t));
    }
    for (int i = ndims - 1; i >= 0; i--)
    {
        if (shape[i] < 0 || (shape[i] && tensor->total_size > SIZE_MAX / (size_t)shape[i]))
        {
            cuda_tensor_destroy(tensor);
            CUDA_THROW_INVALID("Fusion shape size overflow");
            return NULL;
        }
        tensor->shape[i] = shape[i];
        tensor->strides[i] = tensor->total_size;
        tensor->total_size *= shape[i];
    }
    if (tensor->total_size > SIZE_MAX / tensor->element_size)
    {
        cuda_tensor_destroy(tensor);
        CUDA_THROW_INVALID("Fusion allocation size overflow");
        return NULL;
    }
    return tensor;
}

tensor_t *fusion_record(fusion_kind kind, tensor_t *a, tensor_t *b,
                              operation_type_t op, const int *shape, int ndims, dtype_t dtype)
{
    fusion_scope *scope = CUDA_G(fusion_scope);
    if (kind == FUSION_BINARY || kind == FUSION_UNARY || kind == FUSION_CAST)
    {
        for (size_t i = 0; i < scope->count; i++)
        {
            tensor_t *existing = scope->nodes[i];
            fusion_node *node = existing->fusion;
            if (node && node->kind == kind && node->a == a && node->b == b &&
                node->op == op && existing->dtype == dtype)
            {
                existing->ref_count++;
                return existing;
            }
        }
    }
    if (scope->count == FUSION_MAX_NODES)
    {
        CUDA_THROW_INVALID("Fusion capture exceeds the limit of %d nodes", FUSION_MAX_NODES);
        return NULL;
    }
    tensor_t *tensor = fusion_metadata(shape, ndims, dtype);
    if (!tensor) return NULL;
    tensor->fusion = ecalloc(1, sizeof(fusion_node));
    tensor->fusion->kind = kind;
    tensor->fusion->a = a;
    tensor->fusion->b = b;
    tensor->fusion->op = op;
    if (a) a->ref_count++;
    if (b) b->ref_count++;
    if (scope->count == scope->capacity)
    {
        scope->capacity = scope->capacity ? scope->capacity * 2 : 16;
        scope->nodes = erealloc(scope->nodes, scope->capacity * sizeof(tensor_t *));
    }
    tensor->ref_count++;
    scope->nodes[scope->count++] = tensor;
    return tensor;
}

void fusion_release_node(tensor_t *tensor)
{
    fusion_node *node = tensor->fusion;
    if (!node) return;
    tensor->fusion = NULL;
    cuda_tensor_destroy(node->a);
    cuda_tensor_destroy(node->b);
    cuda_tensor_destroy(node->c);
    efree(node);
}

void fusion_scope_free(fusion_scope *scope, int invalidate)
{
    if (!scope) return;
    if (invalidate)
    {
        for (size_t i = 0; i < scope->count; i++)
        {
            tensor_t *tensor = scope->nodes[i];
            if (tensor->fusion)
            {
                tensor->fusion_failed = 1;
                fusion_release_node(tensor);
            }
        }
    }
    for (size_t i = 0; i < scope->count; i++)
        cuda_tensor_destroy(scope->nodes[i]);
    if (scope->nodes) efree(scope->nodes);
    efree(scope);
}

void fusion_request_shutdown(void)
{
    fusion_scope *scope = CUDA_G(fusion_scope);
    CUDA_G(fusion_scope) = NULL;
    fusion_scope_free(scope, 1);
    fusion_cache_shutdown();
}

tensor_t *fusion_binary(tensor_t *a, tensor_t *b, operation_type_t op)
{
    int shape[MAX_DIMS], ndims, a_strides[MAX_DIMS], b_strides[MAX_DIMS];
    size_t total;
    if (!prepare_broadcast_operation(a, b, shape, &ndims, a_strides, b_strides, &total))
    {
        CUDA_THROW_INVALID("Broadcast failed: operand shapes are incompatible");
        return NULL;
    }
    dtype_t dtype = promote_types_for_arithmetic(a->dtype, b->dtype, op);
    if (!can_safely_cast_to(a->dtype, dtype) || !can_safely_cast_to(b->dtype, dtype))
    {
        CUDA_THROW_INVALID("Cannot safely promote fusion operands %s and %s to %s",
                           dtype_to_string(a->dtype), dtype_to_string(b->dtype), dtype_to_string(dtype));
        return NULL;
    }
    return fusion_record(FUSION_BINARY, a, b, op, shape, ndims, dtype);
}

tensor_t *fusion_scalar(tensor_t *a, scalar_value_t scalar, operation_type_t op, int inverse)
{
    dtype_t dtype = promote_scalar_for_arithmetic(a->dtype, scalar.dtype, op, scalar.is_neg);
    tensor_t *tensor = fusion_record(FUSION_SCALAR, a, NULL, op, a->shape, a->ndims, dtype);
    if (tensor)
    {
        tensor->fusion->scalar = scalar;
        tensor->fusion->parameter = inverse;
    }
    return tensor;
}

tensor_t *fusion_unary(tensor_t *a, operation_type_t op)
{
    return fusion_record(FUSION_UNARY, a, NULL, op, a->shape, a->ndims, a->dtype);
}

tensor_t *fusion_cast(tensor_t *a, dtype_t dtype)
{
    return fusion_record(FUSION_CAST, a, NULL, OP_ADD, a->shape, a->ndims, dtype);
}

tensor_t *fusion_where(tensor_t *condition, tensor_t *x, tensor_t *y)
{
    if (x->dtype != y->dtype)
    {
        CUDA_THROW_INVALID("where requires x and y to have the same dtype");
        return NULL;
    }
    int shape[MAX_DIMS];
    int ndims = condition->ndims;
    if (x->ndims > ndims) ndims = x->ndims;
    if (y->ndims > ndims) ndims = y->ndims;
    if (condition->ndims < 1 || x->ndims < 1 || y->ndims < 1)
    {
        CUDA_THROW_INVALID("where received unsupported tensor dimensions");
        return NULL;
    }
    tensor_t *operands[] = {condition, x, y};
    for (int d = 0; d < ndims; d++)
    {
        int dimension = 1;
        for (int i = 0; i < 3; i++)
        {
            int axis = d - (ndims - operands[i]->ndims);
            int size = axis < 0 ? 1 : operands[i]->shape[axis];
            if (size < 0 || (size != 1 && dimension != 1 && size != dimension))
            {
                CUDA_THROW_INVALID("where operands have incompatible shapes");
                return NULL;
            }
            if (dimension == 1) dimension = size;
        }
        shape[d] = dimension;
    }
    tensor_t *tensor = fusion_record(FUSION_WHERE, condition, x, OP_SELECT, shape, ndims, x->dtype);
    if (tensor)
    {
        tensor->fusion->c = y;
        y->ref_count++;
    }
    return tensor;
}

tensor_t *fusion_reduce(tensor_t *a, int axis, operation_type_t op, int arg)
{
    int shape[MAX_DIMS];
    size_t total;
    int ndims;
    if (axis == -1)
    {
        shape[0] = 1;
        ndims = 1;
    }
    else ndims = calculate_reduction_shape(a, axis, shape, &total);
    if (ndims <= 0) return NULL;
    size_t output_size = 1;
    for (int d = 0; d < ndims; d++) output_size *= shape[d];
    if (output_size && (axis == -1 ? !a->total_size : a->shape[axis] == 0) &&
        (arg || op == OP_REDUCE_MAX || op == OP_REDUCE_MIN))
    {
        CUDA_THROW_INVALID("Min/max and arg reductions have no identity for an empty axis");
        return NULL;
    }
    dtype_t dtype = arg ? DTYPE_INT32 :
        op == OP_REDUCE_MEAN ?
            (a->dtype == DTYPE_FLOAT64 || dtype_is_integer(a->dtype) || a->dtype == DTYPE_BOOL
                ? DTYPE_FLOAT64 : DTYPE_FLOAT32) : a->dtype;
    tensor_t *tensor = fusion_record(arg ? FUSION_ARG_REDUCE : FUSION_REDUCE,
                                    a, NULL, op, shape, ndims, dtype);
    if (tensor) tensor->fusion->parameter = axis;
    return tensor;
}

tensor_t *fusion_matmul(tensor_t *a, tensor_t *b)
{
    int shape[MAX_DIMS], ndims;
    if (a->ndims < 2 || b->ndims < 2 ||
        !prepare_matmul_result_shape(a->ndims, a->shape, b->ndims, b->shape, &ndims, shape))
    {
        CUDA_THROW_INVALID("Matrix multiplication failed - incompatible dimensions");
        return NULL;
    }
    if (a->dtype != DTYPE_FLOAT32 || b->dtype != DTYPE_FLOAT32)
    {
        CUDA_THROW_INVALID("Fusion matmul currently requires float32 operands");
        return NULL;
    }
    return fusion_record(FUSION_MATMUL, a, b, OP_MATMUL, shape, ndims, DTYPE_FLOAT32);
}

tensor_t *fusion_view(tensor_t *a, operation_type_t op, int *shape, size_t *strides, int ndims, int *axes)
{
    tensor_t *tensor = fusion_record(FUSION_VIEW, a, NULL, op, shape, ndims, a->dtype);
    if (tensor)
    {
        memcpy(tensor->strides, strides, ndims * sizeof(size_t));
        tensor->is_contiguous_cached = -1;
        if (axes) memcpy(tensor->fusion->axes, axes, ndims * sizeof(int));
    }
    return tensor;
}

tensor_t *fusion_slice(tensor_t *a, int *shape, size_t *strides, int ndims,
                       int *axes, int *starts, int *steps)
{
    tensor_t *tensor = fusion_view(a, OP_SLICE, shape, strides, ndims, axes);
    if (!tensor) return NULL;
    fusion_node *node = tensor->fusion;
    size_t logical_stride = 1;
    for (int d = a->ndims - 1; d >= 0; d--)
    {
        node->slice_starts[d] = starts[d];
        node->slice_steps[d] = steps[d];
        node->slice_offset += (size_t)starts[d] * logical_stride;
        for (int j = 0; j < ndims; j++)
            if (axes[j] == d) node->slice_strides[j] = logical_stride * (size_t)steps[d];
        logical_stride *= a->shape[d];
    }
    return tensor;
}
