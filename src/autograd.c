#include "autograd.h"
#include "ca_private.h"
#include "fusion_internal.h"
#include "tensor_factory.h"
#include "tensor_where.h"
#include "cuda_exceptions.h"
#include <string.h>

typedef struct
{
    tensor_t **items;
    size_t count;
    size_t capacity;
} tensor_list_t;

typedef struct
{
    tensor_t *tensor;
    tensor_t *gradient;
} gradient_entry_t;

ZEND_TLS int autograd_suppressed;

static int autograd_is_differentiable_type(dtype_t dtype)
{
    return dtype_is_floating(dtype);
}

static int autograd_node_add(tensor_t *result, autograd_kind_t kind,
                             tensor_t *a, tensor_t *b, tensor_t *c,
                             operation_type_t op)
{
    if (autograd_suppressed || !result || !autograd_is_differentiable_type(result->dtype))
        return 1;

    if (result->grad_fn)
        return 1;

    int requires_gradient = (a && a->requires_grad) || (b && b->requires_grad) ||
                            (c && c->requires_grad);
    if (!requires_gradient)
        return 1;

    autograd_node_t *node = ecalloc(1, sizeof(*node));
    node->kind = kind;
    node->a = a;
    node->b = b;
    node->c = c;
    node->op = op;
    if (a) a->ref_count++;
    if (b) b->ref_count++;
    if (c) c->ref_count++;
    result->grad_fn = node;
    result->requires_grad = 1;
    return 1;
}

void autograd_release_node(tensor_t *tensor)
{
    if (!tensor || !tensor->grad_fn)
        return;

    autograd_node_t *node = tensor->grad_fn;
    tensor->grad_fn = NULL;
    if (node->a) cuda_tensor_destroy(node->a);
    if (node->b) cuda_tensor_destroy(node->b);
    if (node->c) cuda_tensor_destroy(node->c);
    efree(node);
}

void autograd_clear_gradient(tensor_t *tensor)
{
    if (!tensor || !tensor->grad)
        return;

    tensor_t *gradient = tensor->grad;
    tensor->grad = NULL;
    cuda_tensor_destroy(gradient);
}

int autograd_record_binary(tensor_t *result, tensor_t *a, tensor_t *b, operation_type_t op)
{
    switch (op)
    {
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        case OP_POW:
            return autograd_node_add(result, AUTOGRAD_BINARY, a, b, NULL, op);
        default:
            return 1;
    }
}

int autograd_record_scalar(tensor_t *result, tensor_t *a, scalar_value_t scalar,
                           operation_type_t op, int inverse)
{
    if (!autograd_node_add(result, AUTOGRAD_SCALAR, a, NULL, NULL, op))
        return 0;
    if (result->grad_fn && result->grad_fn->a == a)
    {
        result->grad_fn->scalar = scalar;
        result->grad_fn->parameter = inverse;
    }
    return 1;
}

int autograd_record_unary(tensor_t *result, tensor_t *a, operation_type_t op)
{
    switch (op)
    {
        case OP_EXP:
        case OP_SQRT:
        case OP_LOG:
        case OP_SIN:
        case OP_COS:
        case OP_TAN:
        case OP_ABS:
        case OP_NEG:
        case OP_CEIL:
        case OP_FLOOR:
        case OP_ROUND:
            return autograd_node_add(result, AUTOGRAD_UNARY, a, NULL, NULL, op);
        default:
            return 1;
    }
}

int autograd_record_reduce(tensor_t *result, tensor_t *a, operation_type_t op, int axis)
{
    if (op != OP_REDUCE_SUM && op != OP_REDUCE_MEAN &&
        op != OP_REDUCE_MAX && op != OP_REDUCE_MIN && op != OP_REDUCE_PROD)
        return 1;
    if (!autograd_node_add(result, AUTOGRAD_REDUCE, a, NULL, NULL, op))
        return 0;
    if (result->grad_fn && result->grad_fn->a == a)
        result->grad_fn->parameter = axis;
    return 1;
}

int autograd_record_matmul(tensor_t *result, tensor_t *a, tensor_t *b)
{
    return autograd_node_add(result, AUTOGRAD_MATMUL, a, b, NULL, OP_MATMUL);
}

int autograd_record_view(tensor_t *result, tensor_t *a, operation_type_t op, const int *axes)
{
    if (!autograd_node_add(result, AUTOGRAD_VIEW, a, NULL, NULL, op))
        return 0;
    if (result->grad_fn && result->grad_fn->a == a && axes)
    {
        result->grad_fn->ndims = result->ndims;
        memcpy(result->grad_fn->axes, axes, result->ndims * sizeof(int));
    }
    return 1;
}

int autograd_record_cast(tensor_t *result, tensor_t *a)
{
    if (!autograd_is_differentiable_type(a->dtype) ||
        !autograd_is_differentiable_type(result->dtype))
        return 1;
    return autograd_node_add(result, AUTOGRAD_CAST, a, NULL, NULL, OP_ADD);
}

int autograd_record_where(tensor_t *result, tensor_t *condition, tensor_t *x, tensor_t *y)
{
    if (autograd_suppressed || !result || !autograd_is_differentiable_type(result->dtype) ||
        (!x->requires_grad && !y->requires_grad))
        return 1;

    autograd_node_t *node = ecalloc(1, sizeof(*node));
    node->kind = AUTOGRAD_WHERE;
    node->a = condition;
    node->b = x;
    node->c = y;
    node->op = OP_SELECT;
    if (condition) condition->ref_count++;
    x->ref_count++;
    y->ref_count++;
    result->grad_fn = node;
    result->requires_grad = 1;
    return 1;
}

int autograd_set_requires_grad(tensor_t *tensor, int requires_grad)
{
    if (!tensor)
    {
        CUDA_THROW_INVALID("Cannot configure gradient tracking on an uninitialized tensor");
        return 0;
    }
    if (requires_grad && !autograd_is_differentiable_type(tensor->dtype))
    {
        CUDA_THROW_INVALID("Gradient tracking is only supported for floating-point tensors");
        return 0;
    }
    if (tensor->grad_fn ||
        (tensor->fusion && tensor->fusion->kind != FUSION_INPUT))
    {
        CUDA_THROW_INVALID("requiresGrad() can only be changed on leaf tensors");
        return 0;
    }
    tensor->requires_grad = requires_grad;
    if (!requires_grad)
        autograd_clear_gradient(tensor);
    return 1;
}

tensor_t *autograd_detach(tensor_t *tensor)
{
    if (!tensor)
    {
        CUDA_THROW_INVALID("Cannot detach an uninitialized tensor");
        return NULL;
    }
    if (!fusion_active() && !fusion_materialize(tensor))
        return NULL;

    autograd_suppressed++;
    tensor_t *detached = fusion_active()
        ? fusion_view(tensor, OP_RESHAPE, tensor->shape, tensor->strides,
                      tensor->ndims, NULL)
        : cuda_tensor_create_view(tensor, tensor->shape, tensor->strides,
                                  tensor->ndims, 0, tensor->total_size);
    autograd_suppressed--;
    if (detached)
        detached->requires_grad = 0;
    return detached;
}

static int tensor_list_contains(const tensor_list_t *list, tensor_t *tensor)
{
    for (size_t i = 0; i < list->count; i++)
        if (list->items[i] == tensor)
            return 1;
    return 0;
}

static int tensor_list_append(tensor_list_t *list, tensor_t *tensor)
{
    if (list->count == list->capacity)
    {
        size_t capacity = list->capacity ? list->capacity * 2 : 32;
        list->items = list->items
            ? erealloc(list->items, capacity * sizeof(tensor_t *))
            : emalloc(capacity * sizeof(tensor_t *));
        list->capacity = capacity;
    }
    list->items[list->count++] = tensor;
    return 1;
}

static int autograd_visit(tensor_t *tensor, tensor_list_t *topology)
{
    if (!tensor || !tensor->requires_grad || tensor_list_contains(topology, tensor))
        return 1;

    autograd_node_t *node = tensor->grad_fn;
    if (node)
    {
        if (!autograd_visit(node->a, topology) ||
            !autograd_visit(node->b, topology) ||
            !autograd_visit(node->c, topology))
            return 0;
    }
    return tensor_list_append(topology, tensor);
}

static size_t gradient_find(const gradient_entry_t *entries, size_t count, tensor_t *tensor)
{
    for (size_t i = 0; i < count; i++)
        if (entries[i].tensor == tensor)
            return i;
    return SIZE_MAX;
}

static int gradient_add(gradient_entry_t **entries, size_t *count, size_t *capacity,
                        tensor_t *tensor, tensor_t *gradient)
{
    if (!tensor || !tensor->requires_grad || !gradient)
    {
        if (gradient) cuda_tensor_destroy(gradient);
        return 1;
    }

    size_t index = gradient_find(*entries, *count, tensor);
    if (index != SIZE_MAX)
    {
        tensor_t *sum = cuda_tensor_op((*entries)[index].gradient, gradient, OP_ADD);
        if (!sum)
        {
            cuda_tensor_destroy(gradient);
            return 0;
        }
        cuda_tensor_destroy((*entries)[index].gradient);
        cuda_tensor_destroy(gradient);
        (*entries)[index].gradient = sum;
        return 1;
    }

    if (*count == *capacity)
    {
        size_t new_capacity = *capacity ? *capacity * 2 : 32;
        *entries = *entries
            ? erealloc(*entries, new_capacity * sizeof(gradient_entry_t))
            : emalloc(new_capacity * sizeof(gradient_entry_t));
        *capacity = new_capacity;
    }
    (*entries)[*count].tensor = tensor;
    (*entries)[*count].gradient = gradient;
    (*count)++;
    return 1;
}

static tensor_t *gradient_get(const gradient_entry_t *entries, size_t count, tensor_t *tensor)
{
    size_t index = gradient_find(entries, count, tensor);
    return index == SIZE_MAX ? NULL : entries[index].gradient;
}

static tensor_t *gradient_scale(tensor_t *tensor, double value)
{
    scalar_value_t scalar = {0};
    scalar.dtype = tensor->dtype;
    if (scalar.dtype == DTYPE_FLOAT64) scalar.v.f64 = value;
    else scalar.v.f32 = (float)value;
    scalar.is_neg = value < 0;
    return cuda_scalar_op(tensor, scalar, OP_MUL);
}

static tensor_t *gradient_negate(tensor_t *tensor)
{
    return cuda_unary_op(tensor, OP_NEG);
}

static tensor_t *gradient_unbroadcast(tensor_t *gradient, const tensor_t *target)
{
    if (gradient->total_size == target->total_size)
    {
        int target_shape[MAX_DIMS];
        memcpy(target_shape, target->shape, target->ndims * sizeof(int));
        tensor_t *reshaped = cuda_tensor_reshape(gradient, target_shape, target->ndims);
        cuda_tensor_destroy(gradient);
        return reshaped;
    }

    if (gradient->ndims < target->ndims)
    {
        CUDA_THROW_INVALID("Gradient rank is smaller than the operand rank");
        cuda_tensor_destroy(gradient);
        return NULL;
    }

    tensor_t *current = gradient;
    int leading_axes = current->ndims - target->ndims;
    for (int i = 0; i < leading_axes; i++)
    {
        tensor_t *reduced = cuda_tensor_reduce(current, 0, OP_REDUCE_SUM);
        cuda_tensor_destroy(current);
        if (!reduced)
            return NULL;
        current = reduced;
    }

    for (int axis = current->ndims - 1; axis >= 0; axis--)
    {
        if (target->shape[axis] != 1 || current->shape[axis] == 1)
        {
            if (current->shape[axis] != target->shape[axis])
            {
                CUDA_THROW_INVALID("Gradient shape cannot be reduced to its operand shape at axis %d (%d versus %d)",
                                   axis, current->shape[axis], target->shape[axis]);
                cuda_tensor_destroy(current);
                return NULL;
            }
            continue;
        }

        tensor_t *reduced = cuda_tensor_reduce(current, axis, OP_REDUCE_SUM);
        cuda_tensor_destroy(current);
        if (!reduced)
            return NULL;
        current = reduced;
    }

    int target_shape[MAX_DIMS];
    memcpy(target_shape, target->shape, target->ndims * sizeof(int));
    tensor_t *reshaped = cuda_tensor_reshape(current, target_shape, target->ndims);
    cuda_tensor_destroy(current);
    return reshaped;
}

static tensor_t *gradient_expand_reduction(tensor_t *gradient, const tensor_t *input, int axis)
{
    int shape[MAX_DIMS];
    if (axis < 0)
    {
        for (int i = 0; i < input->ndims; i++)
            shape[i] = 1;
    }
    else
    {
        for (int i = 0; i < input->ndims; i++)
            shape[i] = i == axis ? 1 : input->shape[i];
    }
    tensor_t *expanded = cuda_tensor_reshape(gradient, shape, input->ndims);
    cuda_tensor_destroy(gradient);
    return expanded;
}

static tensor_t *gradient_broadcast_reduction(tensor_t *gradient, tensor_t *input)
{
    tensor_t *broadcast = fusion_active()
        ? fusion_where(input, gradient, gradient)
        : cuda_tensor_where(input, gradient, gradient);
    cuda_tensor_destroy(gradient);
    return broadcast;
}

static tensor_t *gradient_for_reduction(tensor_t *output, tensor_t *upstream,
                                        autograd_node_t *node)
{
    tensor_t *input = node->a;
    int axis = node->parameter;
    if (node->op != OP_REDUCE_SUM && node->op != OP_REDUCE_MEAN &&
        node->op != OP_REDUCE_MAX && node->op != OP_REDUCE_MIN)
    {
        CUDA_THROW_INVALID("Backward is not implemented for this reduction");
        cuda_tensor_destroy(upstream);
        return NULL;
    }

    tensor_t *expanded_gradient = gradient_expand_reduction(upstream, input, axis);
    if (!expanded_gradient)
        return NULL;

    if (node->op == OP_REDUCE_SUM)
        return gradient_broadcast_reduction(expanded_gradient, input);

    if (node->op == OP_REDUCE_MEAN)
    {
        size_t divisor = axis < 0 ? input->total_size : (size_t)input->shape[axis];
        if (!divisor)
        {
            CUDA_THROW_INVALID("Cannot differentiate a mean over an empty axis");
            cuda_tensor_destroy(expanded_gradient);
            return NULL;
        }
        scalar_value_t scalar = {0};
        scalar.dtype = expanded_gradient->dtype;
        if (scalar.dtype == DTYPE_FLOAT64) scalar.v.f64 = (double)divisor;
        else scalar.v.f32 = (float)divisor;
        tensor_t *result = cuda_scalar_op(expanded_gradient, scalar, OP_DIV);
        if (!result)
        {
            cuda_tensor_destroy(expanded_gradient);
            return NULL;
        }
        cuda_tensor_destroy(expanded_gradient);
        return gradient_broadcast_reduction(result, input);
    }

    output->ref_count++;
    tensor_t *expanded_output = gradient_expand_reduction(output, input, axis);
    if (!expanded_output)
    {
        cuda_tensor_destroy(expanded_gradient);
        return NULL;
    }
    tensor_t *mask = cuda_tensor_op(input, expanded_output, OP_EQ);
    cuda_tensor_destroy(expanded_output);
    if (!mask)
    {
        cuda_tensor_destroy(expanded_gradient);
        return NULL;
    }
    tensor_t *mask_float = tensor_cast(mask, expanded_gradient->dtype);
    cuda_tensor_destroy(mask);
    if (!mask_float)
    {
        cuda_tensor_destroy(expanded_gradient);
        return NULL;
    }
    tensor_t *counts = cuda_tensor_reduce(mask_float, axis, OP_REDUCE_SUM);
    if (!counts)
    {
        cuda_tensor_destroy(mask_float);
        cuda_tensor_destroy(expanded_gradient);
        return NULL;
    }
    tensor_t *expanded_counts = gradient_expand_reduction(counts, input, axis);
    if (!expanded_counts)
    {
        cuda_tensor_destroy(mask_float);
        cuda_tensor_destroy(expanded_gradient);
        return NULL;
    }
    tensor_t *share = cuda_tensor_op(expanded_gradient, expanded_counts, OP_DIV);
    cuda_tensor_destroy(expanded_gradient);
    cuda_tensor_destroy(expanded_counts);
    if (!share)
    {
        cuda_tensor_destroy(mask_float);
        return NULL;
    }

    scalar_value_t zero = {0};
    zero.dtype = share->dtype;
    tensor_t *zeros = cuda_scalar_op(share, zero, OP_MUL);
    if (!zeros)
    {
        cuda_tensor_destroy(mask_float);
        cuda_tensor_destroy(share);
        return NULL;
    }
    tensor_t *result = fusion_active()
        ? fusion_where(mask_float, share, zeros)
        : cuda_tensor_where(mask_float, share, zeros);
    cuda_tensor_destroy(mask_float);
    cuda_tensor_destroy(share);
    cuda_tensor_destroy(zeros);
    return result;
}

static int autograd_propagate(tensor_t *output, tensor_t *upstream,
                              gradient_entry_t **entries, size_t *count, size_t *capacity)
{
    autograd_node_t *node = output->grad_fn;
    if (!node)
    {
        if (output->requires_grad)
            return gradient_add(entries, count, capacity, output, upstream);
        cuda_tensor_destroy(upstream);
        return 1;
    }

    tensor_t *ga = NULL;
    tensor_t *gb = NULL;
    tensor_t *gc = NULL;
    tensor_t *temporary = NULL;
    switch (node->kind)
    {
        case AUTOGRAD_BINARY:
            switch (node->op)
            {
                case OP_ADD:
                    ga = upstream; upstream = NULL;
                    gb = ga ? ga : NULL;
                    if (gb) gb->ref_count++;
                    break;
                case OP_SUB:
                    ga = upstream; upstream = NULL;
                    gb = gradient_negate(ga);
                    break;
                case OP_MUL:
                    ga = cuda_tensor_op(upstream, node->b, OP_MUL);
                    gb = cuda_tensor_op(upstream, node->a, OP_MUL);
                    break;
                case OP_DIV:
                    ga = cuda_tensor_op(upstream, node->b, OP_DIV);
                    temporary = cuda_tensor_op(node->b, node->b, OP_MUL);
                    if (temporary)
                    {
                        tensor_t *numerator = cuda_tensor_op(upstream, node->a, OP_MUL);
                        if (numerator)
                        {
                            tensor_t *quotient = cuda_tensor_op(numerator, temporary, OP_DIV);
                            cuda_tensor_destroy(numerator);
                            if (quotient) gb = gradient_negate(quotient);
                            if (quotient) cuda_tensor_destroy(quotient);
                        }
                        cuda_tensor_destroy(temporary);
                    }
                    break;
                case OP_POW:
                    temporary = cuda_tensor_op(node->b, node->a, OP_POW);
                    if (temporary)
                    {
                        tensor_t *exponent_minus_one = cuda_scalar_op(node->b, (scalar_value_t){.v.f32 = 1.0f, .dtype = DTYPE_FLOAT32}, OP_SUB);
                        tensor_t *base_power = exponent_minus_one
                            ? cuda_tensor_op(node->a, exponent_minus_one, OP_POW) : NULL;
                        if (exponent_minus_one) cuda_tensor_destroy(exponent_minus_one);
                        if (base_power)
                        {
                            tensor_t *factor = cuda_tensor_op(node->b, base_power, OP_MUL);
                            if (factor) ga = cuda_tensor_op(upstream, factor, OP_MUL);
                            if (factor) cuda_tensor_destroy(factor);
                            cuda_tensor_destroy(base_power);
                        }
                        tensor_t *log_base = cuda_unary_op(node->a, OP_LOG);
                        tensor_t *factor_y = log_base ? cuda_tensor_op(temporary, log_base, OP_MUL) : NULL;
                        if (factor_y) gb = cuda_tensor_op(upstream, factor_y, OP_MUL);
                        if (factor_y) cuda_tensor_destroy(factor_y);
                        if (log_base) cuda_tensor_destroy(log_base);
                        cuda_tensor_destroy(temporary);
                    }
                    break;
                default:
                    CUDA_THROW_INVALID("Backward is not implemented for this binary operation");
                    cuda_tensor_destroy(upstream);
                    return 0;
            }
            break;

        case AUTOGRAD_SCALAR:
            if (node->parameter)
            {
                if (node->op == OP_SUB)
                    ga = gradient_negate(upstream);
                else if (node->op == OP_DIV)
                {
                    tensor_t *square = cuda_tensor_op(node->a, node->a, OP_MUL);
                    scalar_value_t c = node->scalar;
                    tensor_t *scaled = cuda_scalar_op(upstream, c, OP_MUL);
                    tensor_t *quotient = square && scaled ? cuda_tensor_op(scaled, square, OP_DIV) : NULL;
                    if (square) cuda_tensor_destroy(square);
                    if (scaled) cuda_tensor_destroy(scaled);
                    if (quotient) ga = gradient_negate(quotient);
                    if (quotient) cuda_tensor_destroy(quotient);
                }
                else
                {
                    CUDA_THROW_INVALID("Backward is not implemented for this inverse scalar operation");
                    cuda_tensor_destroy(upstream);
                    return 0;
                }
            }
            else if (node->op == OP_ADD || node->op == OP_SUB)
            {
                ga = upstream; upstream = NULL;
            }
            else if (node->op == OP_MUL)
                ga = cuda_scalar_op(upstream, node->scalar, OP_MUL);
            else if (node->op == OP_DIV)
                ga = cuda_scalar_op(upstream, node->scalar, OP_DIV);
            else if (node->op == OP_POW)
            {
                scalar_value_t exponent = node->scalar;
                double value = exponent.dtype == DTYPE_FLOAT64 ? exponent.v.f64 :
                    exponent.dtype == DTYPE_INT64 ? (double)exponent.v.i64 : exponent.v.f32;
                scalar_value_t lowered = {0};
                lowered.dtype = exponent.dtype;
                if (lowered.dtype == DTYPE_FLOAT64) lowered.v.f64 = value - 1.0;
                else lowered.v.f32 = (float)(value - 1.0);
                temporary = cuda_scalar_op(node->a, lowered, OP_POW);
                if (temporary)
                {
                    tensor_t *scaled = cuda_scalar_op(temporary, exponent, OP_MUL);
                    if (scaled) ga = cuda_tensor_op(upstream, scaled, OP_MUL);
                    if (scaled) cuda_tensor_destroy(scaled);
                    cuda_tensor_destroy(temporary);
                }
            }
            else
            {
                CUDA_THROW_INVALID("Backward is not implemented for this scalar operation");
                cuda_tensor_destroy(upstream);
                return 0;
            }
            break;

        case AUTOGRAD_UNARY:
            switch (node->op)
            {
                case OP_EXP:
                    ga = cuda_tensor_op(upstream, output, OP_MUL);
                    break;
                case OP_LOG:
                    ga = cuda_tensor_op(upstream, node->a, OP_DIV);
                    break;
                case OP_SQRT:
                    temporary = cuda_scalar_op(output, (scalar_value_t){.v.f32 = 2.0f, .dtype = DTYPE_FLOAT32}, OP_MUL);
                    if (temporary) ga = cuda_tensor_op(upstream, temporary, OP_DIV);
                    if (temporary) cuda_tensor_destroy(temporary);
                    break;
                case OP_SIN:
                    temporary = cuda_unary_op(node->a, OP_COS);
                    if (temporary) ga = cuda_tensor_op(upstream, temporary, OP_MUL);
                    if (temporary) cuda_tensor_destroy(temporary);
                    break;
                case OP_COS:
                    temporary = cuda_unary_op(node->a, OP_SIN);
                    if (temporary)
                    {
                        tensor_t *negative = gradient_negate(temporary);
                        if (negative) ga = cuda_tensor_op(upstream, negative, OP_MUL);
                        if (negative) cuda_tensor_destroy(negative);
                        cuda_tensor_destroy(temporary);
                    }
                    break;
                case OP_TAN:
                    temporary = cuda_unary_op(node->a, OP_COS);
                    if (temporary)
                    {
                        tensor_t *square = cuda_tensor_op(temporary, temporary, OP_MUL);
                        if (square) ga = cuda_tensor_op(upstream, square, OP_DIV);
                        if (square) cuda_tensor_destroy(square);
                        cuda_tensor_destroy(temporary);
                    }
                    break;
                case OP_ABS:
                    {
                        scalar_value_t zero = {0};
                        zero.dtype = node->a->dtype;
                        tensor_t *positive = cuda_scalar_op(node->a, zero, OP_GT);
                        tensor_t *negative = cuda_scalar_op(node->a, zero, OP_LT);
                        tensor_t *negative_gradient = gradient_negate(upstream);
                        tensor_t *zeros = gradient_scale(upstream, 0.0);
                        tensor_t *negative_branch = negative && negative_gradient && zeros
                            ? (fusion_active()
                                ? fusion_where(negative, negative_gradient, zeros)
                                : cuda_tensor_where(negative, negative_gradient, zeros))
                            : NULL;
                        if (positive && negative_branch)
                            ga = fusion_active()
                                ? fusion_where(positive, upstream, negative_branch)
                                : cuda_tensor_where(positive, upstream, negative_branch);
                        if (positive) cuda_tensor_destroy(positive);
                        if (negative) cuda_tensor_destroy(negative);
                        if (negative_gradient) cuda_tensor_destroy(negative_gradient);
                        if (zeros) cuda_tensor_destroy(zeros);
                        if (negative_branch) cuda_tensor_destroy(negative_branch);
                    }
                    break;
                case OP_NEG:
                    ga = gradient_negate(upstream);
                    break;
                default:
                    CUDA_THROW_INVALID("Backward is not implemented for this unary operation");
                    cuda_tensor_destroy(upstream);
                    return 0;
            }
            break;

        case AUTOGRAD_REDUCE:
            ga = gradient_for_reduction(output, upstream, node);
            upstream = NULL;
            break;

        case AUTOGRAD_MATMUL:
            if (node->a->ndims != 2 || node->b->ndims != 2)
            {
                CUDA_THROW_INVALID("Autograd matmul currently supports 2D tensors");
                cuda_tensor_destroy(upstream);
                return 0;
            }
            {
                int axes[] = {1, 0};
                tensor_t *b_transposed = cuda_tensor_transpose(node->b, axes, 2);
                if (b_transposed)
                {
                    ga = cuda_tensor_matmul(upstream, b_transposed);
                    cuda_tensor_destroy(b_transposed);
                }
                tensor_t *a_transposed = cuda_tensor_transpose(node->a, axes, 2);
                if (a_transposed)
                {
                    gb = cuda_tensor_matmul(a_transposed, upstream);
                    cuda_tensor_destroy(a_transposed);
                }
            }
            break;

        case AUTOGRAD_VIEW:
            if (node->op == OP_RESHAPE)
            {
                int shape[MAX_DIMS];
                memcpy(shape, node->a->shape, node->a->ndims * sizeof(int));
                ga = cuda_tensor_reshape(upstream, shape, node->a->ndims);
                upstream = NULL;
            }
            else if (node->op == OP_TRANSPOSE)
            {
                int inverse[MAX_DIMS];
                for (int i = 0; i < node->ndims; i++)
                    inverse[node->axes[i]] = i;
                ga = cuda_tensor_transpose(upstream, inverse, node->ndims);
            }
            else
            {
                CUDA_THROW_INVALID("Backward is not implemented for this tensor view");
                cuda_tensor_destroy(upstream);
                return 0;
            }
            break;

        case AUTOGRAD_CAST:
            ga = tensor_cast(upstream, node->a->dtype);
            break;

        case AUTOGRAD_WHERE:
            {
                tensor_t *zeros = gradient_scale(upstream, 0.0);
                if (zeros)
                {
                    tensor_t *x_gradient = fusion_active()
                        ? fusion_where(node->a, upstream, zeros)
                        : cuda_tensor_where(node->a, upstream, zeros);
                    tensor_t *y_gradient = fusion_active()
                        ? fusion_where(node->a, zeros, upstream)
                        : cuda_tensor_where(node->a, zeros, upstream);
                    if (x_gradient) gb = gradient_unbroadcast(x_gradient, node->b);
                    if (y_gradient) gc = gradient_unbroadcast(y_gradient, node->c);
                    cuda_tensor_destroy(zeros);
                }
            }
            break;
    }

    if (upstream) cuda_tensor_destroy(upstream);

    if ((node->kind != AUTOGRAD_WHERE && node->a && node->a->requires_grad && !ga) ||
        (node->b && node->b->requires_grad && !gb) ||
        (node->c && node->c->requires_grad && !gc))
    {
        if (ga) cuda_tensor_destroy(ga);
        if (gb) cuda_tensor_destroy(gb);
        if (gc) cuda_tensor_destroy(gc);
        if (!EG(exception))
            CUDA_THROW_RUNTIME("Failed to construct an autograd derivative");
        return 0;
    }

    if (ga) ga = gradient_unbroadcast(ga, node->a);
    if (gb) gb = gradient_unbroadcast(gb, node->b);
    if (gc) gc = gradient_unbroadcast(gc, node->c);
    if ((node->kind != AUTOGRAD_WHERE && node->a && node->a->requires_grad && !ga) ||
        (node->b && node->b->requires_grad && !gb) ||
        (node->c && node->c->requires_grad && !gc))
    {
        if (ga) cuda_tensor_destroy(ga);
        if (gb) cuda_tensor_destroy(gb);
        if (gc) cuda_tensor_destroy(gc);
        return 0;
    }
    return gradient_add(entries, count, capacity, node->a, ga) &&
           gradient_add(entries, count, capacity, node->b, gb) &&
           gradient_add(entries, count, capacity, node->c, gc);
}

int autograd_backward(tensor_t *tensor, tensor_t *gradient)
{
    if (!tensor || !tensor->requires_grad)
    {
        CUDA_THROW_INVALID("backward() requires a tensor marked with requiresGrad()");
        return 0;
    }
    if (!autograd_is_differentiable_type(tensor->dtype))
    {
        CUDA_THROW_INVALID("backward() is only supported for floating-point tensors");
        return 0;
    }
    if (!gradient && tensor->total_size != 1)
    {
        CUDA_THROW_INVALID("A gradient seed is required when backward() is called on a non-scalar tensor");
        return 0;
    }
    if (gradient && (gradient->ndims != tensor->ndims ||
        memcmp(gradient->shape, tensor->shape, tensor->ndims * sizeof(int)) != 0 ||
        gradient->dtype != tensor->dtype))
    {
        CUDA_THROW_INVALID("The backward gradient seed must match the tensor shape and dtype");
        return 0;
    }

    tensor_list_t topology = {0};
    gradient_entry_t *entries = NULL;
    size_t count = 0, capacity = 0;
    int success = 0;
    if (!autograd_visit(tensor, &topology))
        goto cleanup;

    if (!gradient)
    {
        scalar_value_t one = {0};
        one.dtype = tensor->dtype;
        if (one.dtype == DTYPE_FLOAT64) one.v.f64 = 1.0;
        else one.v.f32 = 1.0f;
        int shape[MAX_DIMS];
        memcpy(shape, tensor->shape, tensor->ndims * sizeof(int));
        gradient = cuda_tensor_create_with_value(shape, tensor->ndims, one, tensor->dtype);
        if (!gradient)
            goto cleanup;
    }
    else
        gradient->ref_count++;

    autograd_suppressed++;
    if (!gradient_add(&entries, &count, &capacity, tensor, gradient))
    {
        gradient = NULL;
        autograd_suppressed--;
        goto cleanup;
    }
    gradient = NULL;

    for (size_t i = topology.count; i > 0; i--)
    {
        tensor_t *current = topology.items[i - 1];
        if (!current->grad_fn)
            continue;
        tensor_t *upstream = gradient_get(entries, count, current);
        if (!upstream)
            continue;
        size_t entry_index = gradient_find(entries, count, current);
        tensor_t *owned_upstream = entries[entry_index].gradient;
        entries[entry_index].gradient = NULL;
        if (!autograd_propagate(current, owned_upstream, &entries, &count, &capacity))
        {
            autograd_suppressed--;
            goto cleanup;
        }
    }
    autograd_suppressed--;

    for (size_t i = 0; i < count; i++)
    {
        tensor_t *leaf = entries[i].tensor;
        if (leaf->grad_fn || !leaf->requires_grad)
            continue;
        if (leaf->grad)
        {
            tensor_t *sum = cuda_tensor_op(leaf->grad, entries[i].gradient, OP_ADD);
            if (!sum)
                goto cleanup;
            autograd_clear_gradient(leaf);
            leaf->grad = sum;
        }
        else
        {
            leaf->grad = entries[i].gradient;
            entries[i].gradient = NULL;
        }
    }
    success = 1;

cleanup:
    if (gradient) cuda_tensor_destroy(gradient);
    for (size_t i = 0; i < count; i++)
        if (entries[i].gradient) cuda_tensor_destroy(entries[i].gradient);
    if (entries) efree(entries);
    if (topology.items) efree(topology.items);
    if (!success && !EG(exception))
        CUDA_THROW_RUNTIME("Autograd backward failed");
    return success;
}
