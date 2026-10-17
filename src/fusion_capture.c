#include "fusion_internal.h"
#include "autograd.h"
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
    if (node->repeat_prefix_counts) efree(node->repeat_prefix_counts);
    efree(node);
}

void fusion_scope_free(fusion_scope *scope, int invalidate)
{
    if (!scope) return;
    for (size_t i = 0; i < scope->count; i++)
        autograd_clear_gradient(scope->nodes[i]);
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
    tensor_t *tensor = fusion_record(FUSION_BINARY, a, b, op, shape, ndims, dtype);
    if (tensor) autograd_record_binary(tensor, a, b, op);
    return tensor;
}

tensor_t *fusion_scalar(tensor_t *a, scalar_value_t scalar, operation_type_t op, int inverse)
{
    dtype_t dtype = promote_scalar_for_arithmetic(a->dtype, scalar.dtype, op, scalar.is_neg);
    tensor_t *tensor = fusion_record(FUSION_SCALAR, a, NULL, op, a->shape, a->ndims, dtype);
    if (tensor)
    {
        tensor->fusion->scalar = scalar;
        tensor->fusion->parameter = inverse;
        autograd_record_scalar(tensor, a, scalar, op, inverse);
    }
    return tensor;
}

tensor_t *fusion_unary(tensor_t *a, operation_type_t op)
{
    tensor_t *tensor = fusion_record(FUSION_UNARY, a, NULL, op, a->shape, a->ndims, a->dtype);
    if (tensor) autograd_record_unary(tensor, a, op);
    return tensor;
}

tensor_t *fusion_cast(tensor_t *a, dtype_t dtype)
{
    int low_precision_source = a->dtype == DTYPE_FLOAT16 || a->dtype == DTYPE_BFLOAT16;
    int low_precision_target = dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16;
    if ((low_precision_source && dtype != DTYPE_FLOAT32) || low_precision_target)
    {
        CUDA_THROW_INVALID("Fusion supports widening float16 or bfloat16 to float32 only");
        return NULL;
    }
    tensor_t *tensor = fusion_record(FUSION_CAST, a, NULL, OP_ADD, a->shape, a->ndims, dtype);
    if (tensor && !autograd_record_cast(tensor, a))
    {
        cuda_tensor_destroy(tensor);
        return NULL;
    }
    return tensor;
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
        autograd_record_where(tensor, condition, x, y);
    }
    return tensor;
}

tensor_t *fusion_reduce(tensor_t *a, int axis, operation_type_t op, int arg)
{
    if (a->dtype == DTYPE_FLOAT16 || a->dtype == DTYPE_BFLOAT16)
    {
        tensor_t *promoted = fusion_cast(a, DTYPE_FLOAT32);
        tensor_t *result = promoted
            ? fusion_reduce(promoted, axis, op, arg) : NULL;
        if (promoted) cuda_tensor_destroy(promoted);
        return result;
    }
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
        (op == OP_REDUCE_ALL || op == OP_REDUCE_ANY) ? DTYPE_BOOL :
        op == OP_REDUCE_MEAN ?
            (a->dtype == DTYPE_FLOAT64 || dtype_is_integer(a->dtype) || a->dtype == DTYPE_BOOL
                ? DTYPE_FLOAT64 : DTYPE_FLOAT32) : a->dtype;
    tensor_t *tensor = fusion_record(arg ? FUSION_ARG_REDUCE : FUSION_REDUCE,
                                    a, NULL, op, shape, ndims, dtype);
    if (tensor)
    {
        tensor->fusion->parameter = axis;
        autograd_record_reduce(tensor, a, op, axis);
    }
    return tensor;
}

tensor_t *fusion_scan(tensor_t *a, int axis, operation_type_t op)
{
    if (!a || (op != OP_CUMSUM && op != OP_CUMPROD) ||
        axis < -1 || axis >= a->ndims)
    {
        CUDA_THROW_INVALID("Invalid Fusion cumulative scan operation or axis");
        return NULL;
    }
    if (a->dtype == DTYPE_FLOAT16 || a->dtype == DTYPE_BFLOAT16)
    {
        tensor_t *promoted = fusion_cast(a, DTYPE_FLOAT32);
        tensor_t *result = promoted
            ? fusion_scan(promoted, axis, op) : NULL;
        if (promoted) cuda_tensor_destroy(promoted);
        return result;
    }
    dtype_t dtype = cuda_cumulative_result_dtype(a->dtype);
    if (dtype == DTYPE_UNKNOWN)
    {
        CUDA_THROW_INVALID("Cumulative scans do not support dtype %s",
                           dtype_to_string(a->dtype));
        return NULL;
    }

    int shape[MAX_DIMS];
    int ndims;
    if (axis == -1)
    {
        if (a->total_size > INT_MAX)
        {
            CUDA_THROW_INVALID("Flattened cumulative scan exceeds the supported dimension size");
            return NULL;
        }
        shape[0] = (int)a->total_size;
        ndims = 1;
    }
    else
    {
        memcpy(shape, a->shape, a->ndims * sizeof(int));
        ndims = a->ndims;
    }
    tensor_t *tensor = fusion_record(FUSION_SCAN, a, NULL, op, shape, ndims, dtype);
    if (tensor)
    {
        tensor->fusion->parameter = axis;
        autograd_record_scan(tensor, a, op, axis);
    }
    return tensor;
}

tensor_t *fusion_scan_backward(tensor_t *input, tensor_t *upstream, int axis)
{
    if (!input || !upstream || axis < -1 || axis >= input->ndims ||
        input->dtype != upstream->dtype ||
        (input->dtype != DTYPE_FLOAT32 && input->dtype != DTYPE_FLOAT64))
    {
        CUDA_THROW_INVALID("Fusion cumulative product gradient requires matching float32/float64 inputs and a valid axis");
        return NULL;
    }
    tensor_t *tensor = fusion_record(FUSION_SCAN_BACKWARD, input, upstream,
                                     OP_CUMPROD, input->shape, input->ndims,
                                     upstream->dtype);
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
    if (a->dtype != b->dtype ||
        (a->dtype != DTYPE_FLOAT32 && a->dtype != DTYPE_FLOAT16 &&
         a->dtype != DTYPE_BFLOAT16))
    {
        CUDA_THROW_INVALID("Fusion matmul requires matching float32, float16 or bfloat16 operands");
        return NULL;
    }
    tensor_t *tensor = fusion_record(FUSION_MATMUL, a, b, OP_MATMUL, shape, ndims, a->dtype);
    if (tensor) autograd_record_matmul(tensor, a, b);
    return tensor;
}

tensor_t *fusion_indexed_gather(tensor_t *input, tensor_t *indices, int axis)
{
    if (!cuda_tensor_validate_indexed_shapes(input, indices, NULL, axis, "gather"))
        return NULL;

    tensor_t *tensor = fusion_record(FUSION_GATHER, input, indices, OP_ADD,
                                     indices->shape, indices->ndims, input->dtype);
    if (tensor)
    {
        tensor->fusion->parameter = axis;
        autograd_record_gather(tensor, input, indices, axis);
    }
    return tensor;
}

tensor_t *fusion_indexed_scatter_add(tensor_t *input, tensor_t *indices,
                                     tensor_t *updates, int axis)
{
    if (!cuda_tensor_validate_indexed_shapes(input, indices, updates, axis, "scatterAdd"))
        return NULL;
    if (input->dtype == DTYPE_FLOAT16 || input->dtype == DTYPE_BFLOAT16)
    {
        tensor_t *promoted_input = fusion_cast(input, DTYPE_FLOAT32);
        tensor_t *promoted_updates = promoted_input
            ? fusion_cast(updates, DTYPE_FLOAT32) : NULL;
        tensor_t *result = promoted_updates
            ? fusion_indexed_scatter_add(
                  promoted_input, indices, promoted_updates, axis)
            : NULL;
        if (promoted_input) cuda_tensor_destroy(promoted_input);
        if (promoted_updates) cuda_tensor_destroy(promoted_updates);
        return result;
    }
    if (input->dtype != DTYPE_FLOAT32 && input->dtype != DTYPE_FLOAT64)
    {
        CUDA_THROW_INVALID("scatterAdd currently supports float32 and float64 tensors");
        return NULL;
    }

    tensor_t *tensor = fusion_record(FUSION_SCATTER_ADD, input, indices, OP_ADD,
                                     input->shape, input->ndims, input->dtype);
    if (tensor)
    {
        tensor->fusion->c = updates;
        updates->ref_count++;
        tensor->fusion->parameter = axis;
        autograd_record_scatter_add(tensor, input, indices, updates, axis);
    }
    return tensor;
}

tensor_t *fusion_concat(tensor_t **inputs, size_t input_count, int axis)
{
    if (!inputs || !input_count || input_count > MAX_CONCAT_TENSORS ||
        !inputs[0] || axis < 0 || axis >= inputs[0]->ndims)
    {
        CUDA_THROW_INVALID("concat requires tensors and a valid axis");
        return NULL;
    }

    int output_shape[MAX_DIMS];
    memcpy(output_shape, inputs[0]->shape, inputs[0]->ndims * sizeof(int));
    size_t axis_size = 0;
    for (size_t i = 0; i < input_count; i++)
    {
        tensor_t *input = inputs[i];
        if (!input || input->ndims != inputs[0]->ndims ||
            input->dtype != inputs[0]->dtype)
        {
            CUDA_THROW_INVALID("concat inputs must have the same rank and dtype");
            return NULL;
        }
        for (int dimension = 0; dimension < input->ndims; dimension++)
            if (dimension != axis &&
                input->shape[dimension] != inputs[0]->shape[dimension])
            {
                CUDA_THROW_INVALID("concat input shapes must match outside the concatenated axis");
                return NULL;
            }
        if (input->shape[axis] < 0 ||
            axis_size > (size_t)INT_MAX - (size_t)input->shape[axis])
        {
            CUDA_THROW_INVALID("Concatenated axis exceeds the supported dimension size");
            return NULL;
        }
        axis_size += (size_t)input->shape[axis];
    }
    output_shape[axis] = (int)axis_size;

    tensor_t *current = inputs[0];
    int current_owned = 0;
    size_t first_input = 1;
    if (input_count == 1)
    {
        tensor_t *copy = fusion_record(FUSION_CONCAT, current, NULL, OP_ADD,
                                       output_shape, current->ndims, current->dtype);
        if (!copy) return NULL;
        copy->fusion->parameter = axis;
        autograd_record_concat(copy, inputs, 1, axis);
        return copy;
    }

    for (size_t i = first_input; i < input_count; i++)
    {
        int shape[MAX_DIMS];
        memcpy(shape, current->shape, current->ndims * sizeof(int));
        if (inputs[i]->shape[axis] > INT_MAX - shape[axis])
        {
            if (current_owned) cuda_tensor_destroy(current);
            CUDA_THROW_INVALID("Concatenated axis exceeds the supported dimension size");
            return NULL;
        }
        shape[axis] += inputs[i]->shape[axis];
        tensor_t *concat = fusion_record(FUSION_CONCAT, current, inputs[i], OP_ADD,
                                         shape, current->ndims, current->dtype);
        if (!concat)
        {
            if (current_owned) cuda_tensor_destroy(current);
            return NULL;
        }
        concat->fusion->parameter = axis;
        tensor_t *parents[] = {current, inputs[i]};
        autograd_record_concat(concat, parents, 2, axis);
        if (current_owned) cuda_tensor_destroy(current);
        current = concat;
        current_owned = 1;
    }
    return current;
}

tensor_t *fusion_transform(tensor_t *a, const tensor_transform_spec_t *spec)
{
    if (!a || !spec || spec->input_ndims != a->ndims ||
        spec->output_ndims < 1 || spec->output_ndims > MAX_DIMS ||
        spec->op < OP_DIAGONAL || spec->op > OP_ROLL)
    {
        CUDA_THROW_INVALID("Invalid Fusion tensor transform metadata");
        return NULL;
    }

    tensor_t *tensor = fusion_record(FUSION_TRANSFORM, a, NULL, spec->op,
                                     spec->output_shape, spec->output_ndims,
                                     a->dtype);
    if (tensor)
    {
        tensor->fusion->transform = *spec;
        if (!autograd_record_index_transform(tensor, a, spec))
        {
            cuda_tensor_destroy(tensor);
            return NULL;
        }
    }
    return tensor;
}

tensor_t *fusion_transform_backward(tensor_t *upstream,
                                    const tensor_transform_spec_t *spec)
{
    if (!upstream || !spec ||
        (upstream->dtype != DTYPE_FLOAT32 && upstream->dtype != DTYPE_FLOAT64) ||
        upstream->ndims != spec->output_ndims)
    {
        CUDA_THROW_INVALID("Invalid Fusion tensor transform gradient metadata");
        return NULL;
    }
    for (int axis = 0; axis < upstream->ndims; axis++)
        if (upstream->shape[axis] != spec->output_shape[axis])
        {
            CUDA_THROW_INVALID("Fusion tensor transform gradient shape mismatch");
            return NULL;
        }

    tensor_t *tensor = fusion_record(FUSION_TRANSFORM_BACKWARD, upstream, NULL,
                                    spec->op, spec->input_shape,
                                    spec->input_ndims, upstream->dtype);
    if (tensor) tensor->fusion->transform = *spec;
    return tensor;
}

tensor_t *fusion_repeat_counts(tensor_t *a, const int *prefix_counts,
                               size_t count, int axis,
                               const int *output_shape, int output_ndims)
{
    if (!a || (count && !prefix_counts) || !output_shape ||
        count > (size_t)INT_MAX || count > SIZE_MAX / sizeof(int) ||
        output_ndims < 1 ||
        output_ndims > MAX_DIMS || axis < -1 || axis >= a->ndims)
    {
        CUDA_THROW_INVALID("Invalid Fusion repeat-count metadata");
        return NULL;
    }
    dtype_t dtype = a->dtype;
    tensor_t *tensor = fusion_record(FUSION_REPEAT_COUNTS, a, NULL, OP_REPEAT,
                                     (int *)output_shape, output_ndims, dtype);
    if (!tensor) return NULL;
    fusion_node *node = tensor->fusion;
    node->parameter = axis;
    node->repeat_count = count;
    if (count)
    {
        node->repeat_prefix_counts = emalloc(count * sizeof(int));
        memcpy(node->repeat_prefix_counts, prefix_counts, count * sizeof(int));
    }
    autograd_record_repeat_counts(tensor, a, prefix_counts, count, axis);
    return tensor;
}

tensor_t *fusion_repeat_counts_backward(tensor_t *upstream,
                                        const int *prefix_counts, size_t count,
                                        int axis, const int *input_shape,
                                        int input_ndims)
{
    if (!upstream || (count && !prefix_counts) || !input_shape ||
        input_ndims < 1 || input_ndims > MAX_DIMS ||
        axis < -1 || axis >= input_ndims ||
        count > (size_t)INT_MAX || count > SIZE_MAX / sizeof(int))
    {
        CUDA_THROW_INVALID("Invalid Fusion repeat-count gradient metadata");
        return NULL;
    }
    tensor_t *tensor = fusion_record(FUSION_REPEAT_COUNTS_BACKWARD, upstream,
                                     NULL, OP_REPEAT, (int *)input_shape,
                                     input_ndims, upstream->dtype);
    if (!tensor) return NULL;
    fusion_node *node = tensor->fusion;
    node->parameter = axis;
    node->repeat_count = count;
    if (count)
    {
        node->repeat_prefix_counts = emalloc(count * sizeof(int));
        memcpy(node->repeat_prefix_counts, prefix_counts, count * sizeof(int));
    }
    return tensor;
}

tensor_t *fusion_view(tensor_t *a, operation_type_t op, int *shape, size_t *strides, int ndims, int *axes)
{
    tensor_t *tensor = fusion_record(FUSION_VIEW, a, NULL, op, shape, ndims, a->dtype);
    if (tensor)
    {
        memcpy(tensor->strides, strides, ndims * sizeof(size_t));
        tensor->is_contiguous_cached = -1;
        if (axes) memcpy(tensor->fusion->axes, axes, ndims * sizeof(int));
        if (op != OP_SLICE)
            autograd_record_view(tensor, a, op, axes);
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
    autograd_record_slice(tensor, a, axes, starts, steps);
    return tensor;
}

tensor_t *fusion_slice_scatter(tensor_t *updates, const int *input_shape,
                               int input_ndims, const int *axes,
                               const int *starts, const int *steps)
{
    if (!updates || input_ndims < 0 || input_ndims > MAX_DIMS ||
        updates->ndims < 0 || updates->ndims > input_ndims ||
        (input_ndims > 0 && (!input_shape || !starts || !steps)) ||
        (updates->ndims > 0 && (!updates->shape || !axes)))
    {
        CUDA_THROW_INVALID("Invalid Fusion slice-gradient metadata");
        return NULL;
    }
    int used[MAX_DIMS] = {0};
    for (int d = 0; d < input_ndims; d++)
    {
        if (input_shape[d] < 0 || starts[d] < 0 || steps[d] <= 0 ||
            (input_shape[d] > 0 && starts[d] > input_shape[d]))
        {
            CUDA_THROW_INVALID("Invalid Fusion slice-gradient bounds");
            return NULL;
        }
    }
    for (int d = 0; d < updates->ndims; d++)
    {
        if (axes[d] < 0 || axes[d] >= input_ndims || used[axes[d]] ||
            updates->shape[d] < 0)
        {
            CUDA_THROW_INVALID("Invalid Fusion slice-gradient axis mapping");
            return NULL;
        }
        used[axes[d]] = 1;
    }

    tensor_t *tensor = fusion_record(FUSION_SLICE_SCATTER, updates, NULL,
                                    OP_SLICE, input_shape, input_ndims,
                                    updates->dtype);
    if (!tensor)
        return NULL;
    for (int d = 0; d < input_ndims; d++)
    {
        tensor->fusion->slice_starts[d] = starts[d];
        tensor->fusion->slice_steps[d] = steps[d];
    }
    for (int d = 0; d < updates->ndims; d++)
        tensor->fusion->axes[d] = axes[d];
    return tensor;
}
