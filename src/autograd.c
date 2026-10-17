#include "autograd.h"
#include "ca_private.h"
#include "fusion_internal.h"
#include "tensor_factory.h"
#include "tensor_where.h"
#include "factory_kernels.h"
#include "cuda_exceptions.h"
#include <math.h>
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

typedef struct
{
    tensor_t **keys;
    size_t *values;
    size_t capacity;
    size_t count;
} tensor_index_t;

ZEND_TLS int autograd_suppressed;

static size_t tensor_pointer_hash(const tensor_t *tensor)
{
    uint64_t value = (uint64_t)(uintptr_t)tensor;
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return (size_t)value;
}

static void tensor_index_place(tensor_index_t *index, tensor_t *tensor, size_t value)
{
    size_t slot = tensor_pointer_hash(tensor) & (index->capacity - 1);
    while (index->keys[slot])
        slot = (slot + 1) & (index->capacity - 1);
    index->keys[slot] = tensor;
    index->values[slot] = value;
    index->count++;
}

static int tensor_index_resize(tensor_index_t *index, size_t capacity)
{
    tensor_t **old_keys = index->keys;
    size_t *old_values = index->values;
    size_t old_capacity = index->capacity;
    index->keys = ecalloc(capacity, sizeof(tensor_t *));
    index->values = ecalloc(capacity, sizeof(size_t));
    index->capacity = capacity;
    index->count = 0;
    for (size_t i = 0; i < old_capacity; i++)
        if (old_keys[i])
            tensor_index_place(index, old_keys[i], old_values[i]);
    if (old_keys) efree(old_keys);
    if (old_values) efree(old_values);
    return 1;
}

static int tensor_index_set(tensor_index_t *index, tensor_t *tensor, size_t value)
{
    if (!tensor)
        return 1;
    if (!index->capacity && !tensor_index_resize(index, 64))
        return 0;
    if ((index->count + 1) * 10 >= index->capacity * 7)
    {
        if (index->capacity > SIZE_MAX / 2 ||
            !tensor_index_resize(index, index->capacity * 2))
            return 0;
    }
    size_t slot = tensor_pointer_hash(tensor) & (index->capacity - 1);
    while (index->keys[slot])
    {
        if (index->keys[slot] == tensor)
        {
            index->values[slot] = value;
            return 1;
        }
        slot = (slot + 1) & (index->capacity - 1);
    }
    index->keys[slot] = tensor;
    index->values[slot] = value;
    index->count++;
    return 1;
}

static int tensor_index_get(const tensor_index_t *index, tensor_t *tensor, size_t *value)
{
    if (!tensor || !index->capacity)
        return 0;
    size_t slot = tensor_pointer_hash(tensor) & (index->capacity - 1);
    while (index->keys[slot])
    {
        if (index->keys[slot] == tensor)
        {
            if (value) *value = index->values[slot];
            return 1;
        }
        slot = (slot + 1) & (index->capacity - 1);
    }
    return 0;
}

static void tensor_index_free(tensor_index_t *index)
{
    if (index->keys) efree(index->keys);
    if (index->values) efree(index->values);
    memset(index, 0, sizeof(*index));
}

static tensor_t *autograd_storage_root(tensor_t *tensor)
{
    while (tensor && tensor->base_tensor)
        tensor = tensor->base_tensor;
    return tensor;
}

uint64_t autograd_tensor_version(const tensor_t *tensor)
{
    const tensor_t *root = tensor;
    while (root && root->base_tensor)
        root = root->base_tensor;
    return root ? root->mutation_version : 0;
}

void autograd_mark_mutated(tensor_t *tensor)
{
    tensor_t *root = autograd_storage_root(tensor);
    if (root)
        root->mutation_version++;
}

static int autograd_is_differentiable_type(dtype_t dtype)
{
    return dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT64;
}

static int autograd_is_low_precision_type(dtype_t dtype)
{
    return dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16;
}

static int autograd_is_matmul_differentiable_type(dtype_t dtype)
{
    return autograd_is_differentiable_type(dtype) ||
        autograd_is_low_precision_type(dtype);
}

static void autograd_save_versions(autograd_node_t *node, tensor_t *result)
{
    node->result_version = autograd_tensor_version(result);
    node->a_version = autograd_tensor_version(node->a);
    node->b_version = autograd_tensor_version(node->b);
    node->c_version = autograd_tensor_version(node->c);
}

static int autograd_node_add(tensor_t *result, autograd_kind_t kind,
                             tensor_t *a, tensor_t *b, tensor_t *c,
                             operation_type_t op)
{
    int supported_type = result &&
        (autograd_is_differentiable_type(result->dtype) ||
         ((kind == AUTOGRAD_MATMUL || kind == AUTOGRAD_VIEW ||
           kind == AUTOGRAD_BINARY || kind == AUTOGRAD_SCALAR ||
           kind == AUTOGRAD_UNARY || kind == AUTOGRAD_INDEX_TRANSFORM ||
           kind == AUTOGRAD_REPEAT_COUNTS || kind == AUTOGRAD_GATHER) &&
          autograd_is_low_precision_type(result->dtype)));
    if (autograd_suppressed || !supported_type)
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
    autograd_save_versions(node, result);
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
    for (size_t i = 0; i < node->parent_count; i++)
        if (node->parents[i]) cuda_tensor_destroy(node->parents[i]);
    if (node->repeat_prefix_counts) efree(node->repeat_prefix_counts);
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
        case OP_MAXIMUM:
        case OP_MINIMUM:
            return autograd_node_add(result, AUTOGRAD_BINARY, a, b, NULL, op);
        default:
            return autograd_node_add(result, AUTOGRAD_UNSUPPORTED, a, b, NULL, op);
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
        case OP_EXPM1:
        case OP_EXP2:
        case OP_SQRT:
        case OP_LOG:
        case OP_LOG1P:
        case OP_LOG2:
        case OP_LOG10:
        case OP_SIN:
        case OP_COS:
        case OP_TAN:
        case OP_ASIN:
        case OP_ACOS:
        case OP_ATAN:
        case OP_SINH:
        case OP_COSH:
        case OP_TANH:
        case OP_ABS:
        case OP_NEG:
        case OP_CEIL:
        case OP_FLOOR:
        case OP_ROUND:
            return autograd_node_add(result, AUTOGRAD_UNARY, a, NULL, NULL, op);
        default:
            return autograd_node_add(result, AUTOGRAD_UNSUPPORTED, a, NULL, NULL, op);
    }
}

int autograd_record_reduce(tensor_t *result, tensor_t *a, operation_type_t op, int axis)
{
    autograd_kind_t kind = op == OP_REDUCE_SUM || op == OP_REDUCE_MEAN ||
        op == OP_REDUCE_MAX || op == OP_REDUCE_MIN || op == OP_REDUCE_PROD
        ? AUTOGRAD_REDUCE : AUTOGRAD_UNSUPPORTED;
    if (!autograd_node_add(result, kind, a, NULL, NULL, op))
        return 0;
    if (result->grad_fn && result->grad_fn->a == a)
        result->grad_fn->parameter = axis;
    return 1;
}

int autograd_record_scan(tensor_t *result, tensor_t *a, operation_type_t op, int axis)
{
    autograd_kind_t kind = op == OP_CUMSUM || op == OP_CUMPROD
        ? AUTOGRAD_SCAN : AUTOGRAD_UNSUPPORTED;
    if (!autograd_node_add(result, kind, a, NULL, NULL, op))
        return 0;
    if (result->grad_fn && result->grad_fn->a == a)
        result->grad_fn->parameter = axis;
    return 1;
}

int autograd_record_repeat_counts(tensor_t *result, tensor_t *a,
                                  const int *prefix_counts, size_t count,
                                  int axis)
{
    if (!autograd_node_add(result, AUTOGRAD_REPEAT_COUNTS, a, NULL, NULL,
                           OP_REPEAT))
        return 0;
    autograd_node_t *node = result ? result->grad_fn : NULL;
    if (!node || node->kind != AUTOGRAD_REPEAT_COUNTS || node->a != a)
        return 1;
    if ((count && !prefix_counts) || count > SIZE_MAX / sizeof(int))
    {
        CUDA_THROW_INVALID("Invalid per-element repeat gradient metadata");
        return 0;
    }
    node->parameter = axis;
    node->repeat_count = count;
    if (count)
    {
        node->repeat_prefix_counts = emalloc(count * sizeof(int));
        memcpy(node->repeat_prefix_counts, prefix_counts, count * sizeof(int));
    }
    return 1;
}

int autograd_record_matmul(tensor_t *result, tensor_t *a, tensor_t *b)
{
    return autograd_node_add(result, AUTOGRAD_MATMUL, a, b, NULL, OP_MATMUL);
}

int autograd_record_view(tensor_t *result, tensor_t *a, operation_type_t op, const int *axes)
{
    autograd_kind_t kind = op == OP_RESHAPE || op == OP_TRANSPOSE || op == OP_BROADCAST
        ? AUTOGRAD_VIEW : AUTOGRAD_UNSUPPORTED;
    if (!autograd_node_add(result, kind, a, NULL, NULL, op))
        return 0;
    if (result->grad_fn && result->grad_fn->a == a && axes)
    {
        result->grad_fn->ndims = result->ndims;
        memcpy(result->grad_fn->axes, axes, result->ndims * sizeof(int));
    }
    return 1;
}

int autograd_record_slice(tensor_t *result, tensor_t *a, const int *axes,
                          const int *starts, const int *steps)
{
    if (!autograd_node_add(result, AUTOGRAD_SLICE, a, NULL, NULL, OP_SLICE))
        return 0;
    autograd_node_t *node = result ? result->grad_fn : NULL;
    if (node && node->kind == AUTOGRAD_SLICE && node->a == a)
    {
        node->ndims = result->ndims;
        node->input_ndims = a->ndims;
        memcpy(node->input_shape, a->shape, a->ndims * sizeof(int));
        memcpy(node->axes, axes, result->ndims * sizeof(int));
        memcpy(node->starts, starts, a->ndims * sizeof(int));
        memcpy(node->steps, steps, a->ndims * sizeof(int));
    }
    return 1;
}

int autograd_record_index_transform(tensor_t *result, tensor_t *a,
                                    const tensor_transform_spec_t *spec)
{
    if (!spec ||
        !autograd_node_add(result, AUTOGRAD_INDEX_TRANSFORM, a, NULL, NULL,
                           spec->op))
        return 0;
    if (result->grad_fn && result->grad_fn->a == a)
        result->grad_fn->transform = *spec;
    return 1;
}

int autograd_record_concat(tensor_t *result, tensor_t **parents,
                           size_t parent_count, int axis)
{
    if (!result || !parents || !parent_count || parent_count > MAX_CONCAT_TENSORS ||
        autograd_suppressed || !autograd_is_differentiable_type(result->dtype))
        return 1;

    int requires_gradient = 0;
    for (size_t i = 0; i < parent_count; i++)
        requires_gradient |= parents[i] && parents[i]->requires_grad;
    if (!requires_gradient || result->grad_fn)
        return 1;

    autograd_node_t *node = ecalloc(1, sizeof(*node));
    node->kind = AUTOGRAD_CONCAT;
    node->op = OP_ADD;
    node->parameter = axis;
    node->parent_count = parent_count;
    autograd_save_versions(node, result);
    for (size_t i = 0; i < parent_count; i++)
    {
        node->parents[i] = parents[i];
        node->parent_versions[i] = autograd_tensor_version(parents[i]);
        parents[i]->ref_count++;
    }
    result->grad_fn = node;
    result->requires_grad = 1;
    return 1;
}

int autograd_record_gather(tensor_t *result, tensor_t *input, tensor_t *indices, int axis)
{
    if (!autograd_node_add(result, AUTOGRAD_GATHER, input, indices, NULL, OP_ADD))
        return 0;
    if (result->grad_fn && result->grad_fn->a == input)
        result->grad_fn->parameter = axis;
    return 1;
}

int autograd_record_scatter_add(tensor_t *result, tensor_t *input, tensor_t *indices,
                                tensor_t *updates, int axis)
{
    if (!autograd_node_add(result, AUTOGRAD_SCATTER_ADD, input, indices, updates, OP_ADD))
        return 0;
    if (result->grad_fn && result->grad_fn->a == input)
        result->grad_fn->parameter = axis;
    return 1;
}

int autograd_record_cast(tensor_t *result, tensor_t *a)
{
    if (autograd_suppressed)
        return 1;
    if (a->requires_grad && autograd_is_differentiable_type(a->dtype) &&
        !autograd_is_differentiable_type(result->dtype))
    {
        CUDA_THROW_INVALID("Cannot cast a gradient-tracked tensor to a non-floating dtype; detach it first");
        return 0;
    }
    if ((!autograd_is_differentiable_type(a->dtype) &&
         !autograd_is_low_precision_type(a->dtype)) ||
        !autograd_is_differentiable_type(result->dtype))
        return 1;
    return autograd_node_add(result, AUTOGRAD_CAST, a, NULL, NULL, OP_ADD);
}

int autograd_record_where(tensor_t *result, tensor_t *condition, tensor_t *x, tensor_t *y)
{
    if (autograd_suppressed || !result ||
        (!autograd_is_differentiable_type(result->dtype) &&
         !autograd_is_low_precision_type(result->dtype)) ||
        (!x->requires_grad && !y->requires_grad))
        return 1;

    autograd_node_t *node = ecalloc(1, sizeof(*node));
    node->kind = AUTOGRAD_WHERE;
    node->a = condition;
    node->b = x;
    node->c = y;
    node->op = OP_SELECT;
    autograd_save_versions(node, result);
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
    if (requires_grad && !autograd_is_matmul_differentiable_type(tensor->dtype))
    {
        CUDA_THROW_INVALID("Gradient tracking supports float32, float64, float16 and bfloat16 tensors");
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

static int autograd_visit(tensor_t *tensor, tensor_list_t *topology,
                          tensor_index_t *visited)
{
    if (!tensor || !tensor->requires_grad || tensor_index_get(visited, tensor, NULL))
        return 1;
    if (tensor->autograd_consumed)
    {
        CUDA_THROW_INVALID("This tensor's autograd graph was already freed; use backward(retainGraph: true) to retain it");
        return 0;
    }

    autograd_node_t *node = tensor->grad_fn;
    if (node)
    {
        if (node->result_version != autograd_tensor_version(tensor) ||
            node->a_version != autograd_tensor_version(node->a) ||
            node->b_version != autograd_tensor_version(node->b) ||
            node->c_version != autograd_tensor_version(node->c))
        {
            CUDA_THROW_INVALID("A tensor saved for backward was modified in place");
            return 0;
        }
        for (size_t i = 0; i < node->parent_count; i++)
        {
            if (node->parent_versions[i] != autograd_tensor_version(node->parents[i]))
            {
                CUDA_THROW_INVALID("A tensor saved for backward was modified in place");
                return 0;
            }
        }
        if (!autograd_visit(node->a, topology, visited) ||
            !autograd_visit(node->b, topology, visited) ||
            !autograd_visit(node->c, topology, visited))
            return 0;
        for (size_t i = 0; i < node->parent_count; i++)
            if (!autograd_visit(node->parents[i], topology, visited))
                return 0;
    }
    size_t index = topology->count;
    if (!tensor_list_append(topology, tensor))
        return 0;
    return tensor_index_set(visited, tensor, index);
}

static size_t gradient_find(const tensor_index_t *index, tensor_t *tensor)
{
    size_t position;
    return tensor_index_get(index, tensor, &position) ? position : SIZE_MAX;
}

static int gradient_add(gradient_entry_t **entries, size_t *count, size_t *capacity,
                        tensor_index_t *index, tensor_t *tensor, tensor_t *gradient)
{
    if (!tensor || !tensor->requires_grad || !gradient)
    {
        if (gradient) cuda_tensor_destroy(gradient);
        return 1;
    }

    size_t position = gradient_find(index, tensor);
    if (position != SIZE_MAX)
    {
        tensor_t *sum = cuda_tensor_op((*entries)[position].gradient, gradient, OP_ADD);
        if (!sum)
        {
            cuda_tensor_destroy(gradient);
            return 0;
        }
        cuda_tensor_destroy((*entries)[position].gradient);
        cuda_tensor_destroy(gradient);
        (*entries)[position].gradient = sum;
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
    size_t new_position = *count;
    (*entries)[new_position].tensor = tensor;
    (*entries)[new_position].gradient = gradient;
    (*count)++;
    return tensor_index_set(index, tensor, new_position);
}

static tensor_t *gradient_get(const gradient_entry_t *entries,
                              const tensor_index_t *index, tensor_t *tensor)
{
    size_t position = gradient_find(index, tensor);
    return position == SIZE_MAX ? NULL : entries[position].gradient;
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

static tensor_t *gradient_cast(tensor_t *tensor, dtype_t dtype)
{
    if (tensor->dtype == dtype)
    {
        tensor->ref_count++;
        return tensor;
    }
    if (fusion_active())
        return fusion_cast(tensor, dtype);

    tensor_t *result = cuda_tensor_create_empty_dtype(tensor->shape, tensor->ndims, dtype);
    if (!result)
        return NULL;
    tensor_t *packed = is_contiguous(tensor) ? NULL : cuda_tensor_clone(tensor);
    if (!is_contiguous(tensor) && !packed)
    {
        cuda_tensor_destroy(result);
        return NULL;
    }
    cudaError_t status = launch_tensor_cast(packed ? packed->data : tensor->data, result->data,
                                            tensor->dtype, dtype, tensor->total_size);
    if (packed) cuda_tensor_destroy(packed);
    if (status != cudaSuccess)
    {
        cuda_tensor_destroy(result);
        CUDA_THROW_RUNTIME("Gradient dtype conversion failed: %s", cudaGetErrorString(status));
        return NULL;
    }
    return result;
}

static tensor_t *gradient_unbroadcast(tensor_t *gradient, const tensor_t *target)
{
    if (gradient->ndims == target->ndims &&
        memcmp(gradient->shape, target->shape, target->ndims * sizeof(int)) == 0)
        return gradient;

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
    for (int axis = current->ndims - 1; axis >= 0; axis--)
    {
        int target_axis = axis - leading_axes;
        if (target_axis >= 0 && target->shape[target_axis] != 1)
        {
            if (current->shape[axis] != target->shape[target_axis])
            {
                CUDA_THROW_INVALID("Gradient shape cannot be reduced to its operand shape at axis %d (%d versus %d)",
                                   target_axis, current->shape[axis], target->shape[target_axis]);
                cuda_tensor_destroy(current);
                return NULL;
            }
            continue;
        }
        if (target_axis >= 0 && current->shape[axis] == 1)
            continue;

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

static tensor_t *autograd_where(tensor_t *condition, tensor_t *x, tensor_t *y)
{
    return fusion_active() ? fusion_where(condition, x, y)
                           : cuda_tensor_where(condition, x, y);
}

static tensor_t *gradient_for_product_reduction(tensor_t *input, tensor_t *upstream, int axis)
{
    tensor_t *temporaries[20] = {0};
    tensor_t *result = NULL;
    scalar_value_t zero = {0}, one = {0};
    zero.dtype = input->dtype;
    one.dtype = input->dtype;
    if (one.dtype == DTYPE_FLOAT64) one.v.f64 = 1.0;
    else one.v.f32 = 1.0f;

    temporaries[0] = gradient_expand_reduction(upstream, input, axis);
    temporaries[1] = cuda_tensor_reduce(input, axis, OP_REDUCE_PROD);
    if (temporaries[1])
    {
        temporaries[2] = gradient_expand_reduction(temporaries[1], input, axis);
        temporaries[1] = NULL;
    }
    temporaries[3] = cuda_scalar_op(input, zero, OP_EQ);
    if (temporaries[3])
        temporaries[4] = tensor_cast(temporaries[3], input->dtype);
    if (temporaries[4])
        temporaries[5] = cuda_tensor_reduce(temporaries[4], axis, OP_REDUCE_SUM);
    if (temporaries[5])
    {
        temporaries[6] = gradient_expand_reduction(temporaries[5], input, axis);
        temporaries[5] = NULL;
    }
    if (temporaries[6])
    {
        temporaries[7] = cuda_scalar_op(temporaries[6], zero, OP_EQ);
        temporaries[8] = cuda_scalar_op(temporaries[6], one, OP_EQ);
    }
    if (temporaries[0] && temporaries[2] && temporaries[7] && temporaries[8])
    {
        temporaries[9] = gradient_scale(input, 0.0);
        if (temporaries[9])
            temporaries[10] = cuda_scalar_op(temporaries[9], one, OP_ADD);
    }
    if (temporaries[3] && temporaries[10])
        temporaries[11] = autograd_where(temporaries[3], temporaries[10], input);
    if (temporaries[11])
        temporaries[12] = cuda_tensor_reduce(temporaries[11], axis, OP_REDUCE_PROD);
    if (temporaries[12])
    {
        temporaries[13] = gradient_expand_reduction(temporaries[12], input, axis);
        temporaries[12] = NULL;
    }
    if (temporaries[2])
        temporaries[14] = cuda_tensor_op(temporaries[2], input, OP_DIV);
    if (temporaries[7] && temporaries[14] && temporaries[9])
        temporaries[15] = autograd_where(temporaries[7], temporaries[14], temporaries[9]);
    if (temporaries[4] && temporaries[8])
        temporaries[17] = tensor_cast(temporaries[8], input->dtype);
    if (temporaries[4] && temporaries[17])
        temporaries[18] = cuda_tensor_op(temporaries[4], temporaries[17], OP_MUL);
    if (temporaries[18])
        temporaries[19] = cuda_scalar_op(temporaries[18], zero, OP_GT);
    if (temporaries[19] && temporaries[13] && temporaries[9])
        temporaries[16] = autograd_where(temporaries[19], temporaries[13], temporaries[9]);
    if (temporaries[15] && temporaries[16])
    {
        tensor_t *factor = cuda_tensor_op(temporaries[15], temporaries[16], OP_ADD);
        if (factor)
        {
            result = cuda_tensor_op(temporaries[0], factor, OP_MUL);
            cuda_tensor_destroy(factor);
        }
    }

    for (size_t i = 0; i < sizeof(temporaries) / sizeof(temporaries[0]); i++)
        if (temporaries[i]) cuda_tensor_destroy(temporaries[i]);
    if (!result && !EG(exception))
        CUDA_THROW_RUNTIME("Failed to construct product-reduction gradient");
    return result;
}

static tensor_t *gradient_for_reduction(tensor_t *output, tensor_t *upstream,
                                        autograd_node_t *node)
{
    tensor_t *input = node->a;
    int axis = node->parameter;
    if (node->op == OP_REDUCE_PROD)
        return gradient_for_product_reduction(input, upstream, axis);
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
                              gradient_entry_t **entries, size_t *count, size_t *capacity,
                              tensor_index_t *gradient_index)
{
    autograd_node_t *node = output->grad_fn;
    if (!node)
    {
        if (output->requires_grad)
            return gradient_add(entries, count, capacity, gradient_index, output, upstream);
        cuda_tensor_destroy(upstream);
        return 1;
    }

    if (node->kind == AUTOGRAD_CONCAT)
    {
        size_t axis_offset = 0;
        int success = 1;
        for (size_t i = 0; i < node->parent_count; i++)
        {
            tensor_t *parent = node->parents[i];
            if (parent->requires_grad)
            {
                tensor_t *part = NULL;
                if (fusion_active())
                {
                    int axes[MAX_DIMS], starts[MAX_DIMS], steps[MAX_DIMS];
                    size_t strides[MAX_DIMS];
                    for (int dimension = 0; dimension < parent->ndims; dimension++)
                    {
                        axes[dimension] = dimension;
                        starts[dimension] = 0;
                        steps[dimension] = 1;
                        strides[dimension] = upstream->strides[dimension];
                    }
                    starts[node->parameter] = (int)axis_offset;
                    part = fusion_slice(upstream, parent->shape, strides, parent->ndims,
                                        axes, starts, steps);
                }
                else
                {
                    size_t offset = axis_offset * upstream->strides[node->parameter];
                    part = cuda_tensor_create_view(
                        upstream, parent->shape, upstream->strides, parent->ndims,
                        offset, parent->total_size);
                    if (part) part->offset = 0;
                }
                if (!part || !gradient_add(entries, count, capacity, gradient_index, parent, part))
                {
                    success = 0;
                    break;
                }
            }
            axis_offset += (size_t)parent->shape[node->parameter];
        }
        cuda_tensor_destroy(upstream);
        return success;
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
                    {
                        dtype_t gradient_dtype = output->dtype == DTYPE_FLOAT64
                            ? DTYPE_FLOAT64 : DTYPE_FLOAT32;
                        tensor_t *base = gradient_cast(node->a, gradient_dtype);
                        tensor_t *exponent = gradient_cast(node->b, gradient_dtype);
                        temporary = base && exponent
                            ? cuda_tensor_op(base, exponent, OP_POW) : NULL;
                        if (temporary)
                        {
                            scalar_value_t one = {0};
                            one.dtype = gradient_dtype;
                            if (one.dtype == DTYPE_FLOAT64) one.v.f64 = 1.0;
                            else one.v.f32 = 1.0f;
                            tensor_t *exponent_minus_one = cuda_scalar_op(exponent, one, OP_SUB);
                            tensor_t *base_power = exponent_minus_one
                                ? cuda_tensor_op(base, exponent_minus_one, OP_POW) : NULL;
                            if (exponent_minus_one) cuda_tensor_destroy(exponent_minus_one);
                            if (base_power)
                            {
                                tensor_t *factor = cuda_tensor_op(exponent, base_power, OP_MUL);
                                if (factor) ga = cuda_tensor_op(upstream, factor, OP_MUL);
                                if (factor) cuda_tensor_destroy(factor);
                                cuda_tensor_destroy(base_power);
                            }
                            tensor_t *log_base = cuda_unary_op(base, OP_LOG);
                            tensor_t *factor_y = log_base ? cuda_tensor_op(temporary, log_base, OP_MUL) : NULL;
                            if (factor_y) gb = cuda_tensor_op(upstream, factor_y, OP_MUL);
                            if (factor_y) cuda_tensor_destroy(factor_y);
                            if (log_base) cuda_tensor_destroy(log_base);
                            cuda_tensor_destroy(temporary);
                        }
                        if (base) cuda_tensor_destroy(base);
                        if (exponent) cuda_tensor_destroy(exponent);
                    }
                    break;
                case OP_MAXIMUM:
                case OP_MINIMUM:
                    {
                        operation_type_t comparison = node->op == OP_MAXIMUM ? OP_GE : OP_LE;
                        tensor_t *mask = cuda_tensor_op(node->a, node->b, comparison);
                        tensor_t *zeros = mask ? gradient_scale(upstream, 0.0) : NULL;
                        if (mask && zeros)
                        {
                            ga = fusion_active() ? fusion_where(mask, upstream, zeros)
                                                 : cuda_tensor_where(mask, upstream, zeros);
                            gb = fusion_active() ? fusion_where(mask, zeros, upstream)
                                                 : cuda_tensor_where(mask, zeros, upstream);
                        }
                        if (zeros) cuda_tensor_destroy(zeros);
                        if (mask) cuda_tensor_destroy(mask);
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
                if (node->op == OP_ADD)
                    ga = upstream, upstream = NULL;
                else if (node->op == OP_MUL)
                    ga = cuda_scalar_op(upstream, node->scalar, OP_MUL);
                else if (node->op == OP_SUB)
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
                else if (node->op == OP_POW)
                {
                    double scalar_value = node->scalar.dtype == DTYPE_INT64
                        ? (double)node->scalar.v.i64
                        : node->scalar.dtype == DTYPE_FLOAT64
                            ? node->scalar.v.f64 : (double)node->scalar.v.f32;
                    scalar_value_t log_scalar = {0};
                    log_scalar.dtype = node->a->dtype;
                    if (log_scalar.dtype == DTYPE_FLOAT64) log_scalar.v.f64 = log(scalar_value);
                    else log_scalar.v.f32 = (float)log(scalar_value);
                    temporary = cuda_scalar_op(output, log_scalar, OP_MUL);
                    if (temporary) ga = cuda_tensor_op(upstream, temporary, OP_MUL);
                    if (temporary) cuda_tensor_destroy(temporary);
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
                lowered.dtype = node->a->dtype;
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
            else if (node->op == OP_MAXIMUM || node->op == OP_MINIMUM)
            {
                operation_type_t comparison;
                if (node->op == OP_MAXIMUM)
                    comparison = node->parameter ? OP_GT : OP_GE;
                else
                    comparison = node->parameter ? OP_LT : OP_LE;
                tensor_t *mask = node->parameter
                    ? cuda_scalar_op(node->a, node->scalar, comparison)
                    : cuda_scalar_op(node->a, node->scalar, comparison);
                tensor_t *zeros = mask ? gradient_scale(upstream, 0.0) : NULL;
                if (mask && zeros)
                    ga = fusion_active() ? fusion_where(mask, upstream, zeros)
                                         : cuda_tensor_where(mask, upstream, zeros);
                if (zeros) cuda_tensor_destroy(zeros);
                if (mask) cuda_tensor_destroy(mask);
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
                case OP_EXPM1:
                {
                    temporary = cuda_unary_op(node->a, OP_EXP);
                    if (temporary) ga = cuda_tensor_op(upstream, temporary, OP_MUL);
                    if (temporary) cuda_tensor_destroy(temporary);
                    break;
                }
                case OP_EXP:
                    ga = cuda_tensor_op(upstream, output, OP_MUL);
                    break;
                case OP_EXP2:
                {
                    scalar_value_t ln2 = {.v.f32 = 0.6931471805599453f,
                                          .dtype = DTYPE_FLOAT32};
                    temporary = cuda_scalar_op(output, ln2, OP_MUL);
                    if (temporary) ga = cuda_tensor_op(upstream, temporary, OP_MUL);
                    if (temporary) cuda_tensor_destroy(temporary);
                    break;
                }
                case OP_LOG:
                    ga = cuda_tensor_op(upstream, node->a, OP_DIV);
                    break;
                case OP_LOG1P:
                {
                    scalar_value_t one = {.v.f32 = 1.0f, .dtype = DTYPE_FLOAT32};
                    temporary = cuda_scalar_op(node->a, one, OP_ADD);
                    if (temporary) ga = cuda_tensor_op(upstream, temporary, OP_DIV);
                    if (temporary) cuda_tensor_destroy(temporary);
                    break;
                }
                case OP_LOG2:
                case OP_LOG10:
                {
                    scalar_value_t scale = {
                        .v.f32 = node->op == OP_LOG2
                            ? 0.6931471805599453f : 2.302585092994046f,
                        .dtype = DTYPE_FLOAT32};
                    temporary = cuda_scalar_op(node->a, scale, OP_MUL);
                    if (temporary) ga = cuda_tensor_op(upstream, temporary, OP_DIV);
                    if (temporary) cuda_tensor_destroy(temporary);
                    break;
                }
                case OP_ASIN:
                case OP_ACOS:
                case OP_ATAN:
                {
                    scalar_value_t one = {.v.f32 = 1.0f, .dtype = DTYPE_FLOAT32};
                    temporary = cuda_tensor_op(node->a, node->a, OP_MUL);
                    tensor_t *denominator_squared = temporary
                        ? (node->op == OP_ATAN
                            ? cuda_scalar_op(temporary, one, OP_ADD)
                            : cuda_inv_scalar_op(temporary, one, OP_SUB))
                        : NULL;
                    tensor_t *denominator = denominator_squared &&
                        node->op != OP_ATAN
                            ? cuda_unary_op(denominator_squared, OP_SQRT)
                            : denominator_squared;
                    if (denominator) ga = cuda_tensor_op(upstream, denominator, OP_DIV);
                    if (node->op == OP_ACOS && ga)
                    {
                        tensor_t *negative = gradient_negate(ga);
                        if (negative)
                        {
                            cuda_tensor_destroy(ga);
                            ga = negative;
                        }
                    }
                    if (denominator && denominator != denominator_squared)
                        cuda_tensor_destroy(denominator);
                    if (denominator_squared) cuda_tensor_destroy(denominator_squared);
                    if (temporary) cuda_tensor_destroy(temporary);
                    break;
                }
                case OP_SINH:
                case OP_COSH:
                {
                    temporary = cuda_unary_op(node->a,
                        node->op == OP_SINH ? OP_COSH : OP_SINH);
                    if (temporary) ga = cuda_tensor_op(upstream, temporary, OP_MUL);
                    if (temporary) cuda_tensor_destroy(temporary);
                    break;
                }
                case OP_TANH:
                {
                    scalar_value_t one = {.v.f32 = 1.0f, .dtype = DTYPE_FLOAT32};
                    temporary = cuda_tensor_op(output, output, OP_MUL);
                    tensor_t *derivative = temporary
                        ? cuda_inv_scalar_op(temporary, one, OP_SUB)
                        : NULL;
                    if (derivative) ga = cuda_tensor_op(upstream, derivative, OP_MUL);
                    if (derivative) cuda_tensor_destroy(derivative);
                    if (temporary) cuda_tensor_destroy(temporary);
                    break;
                }
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
                case OP_CEIL:
                case OP_FLOOR:
                case OP_ROUND:
                    ga = gradient_scale(upstream, 0.0);
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

        case AUTOGRAD_SCAN:
            {
                if (node->op == OP_CUMPROD)
                {
                    ga = cuda_tensor_cumulative_product_gradient(
                        node->a, upstream, node->parameter);
                    upstream = NULL;
                    break;
                }
                tensor_t *scan_gradient = upstream;
                upstream = NULL;
                int axis = node->parameter;
                int flip_axis = axis < 0 ? 0 : axis;
                if (axis == -1)
                {
                    int shape[MAX_DIMS];
                    memcpy(shape, node->a->shape, node->a->ndims * sizeof(int));
                    tensor_t *reshaped = cuda_tensor_reshape(
                        scan_gradient, shape, node->a->ndims);
                    cuda_tensor_destroy(scan_gradient);
                    scan_gradient = reshaped;
                }

                tensor_transform_spec_t reverse_spec = {0};
                reverse_spec.op = OP_FLIP;
                reverse_spec.output_ndims = scan_gradient
                    ? scan_gradient->ndims : 0;
                if (scan_gradient)
                {
                    memcpy(reverse_spec.output_shape, scan_gradient->shape,
                           scan_gradient->ndims * sizeof(int));
                    reverse_spec.axes[flip_axis] = 1;
                }

                tensor_t *reversed = scan_gradient
                    ? cuda_tensor_index_transform(scan_gradient, &reverse_spec)
                    : NULL;
                tensor_t *scanned = reversed
                    ? cuda_tensor_cumulative(reversed, axis, OP_CUMSUM) : NULL;
                if (scanned)
                    ga = cuda_tensor_index_transform(scanned, &reverse_spec);

                if (scanned) cuda_tensor_destroy(scanned);
                if (reversed) cuda_tensor_destroy(reversed);
                if (scan_gradient) cuda_tensor_destroy(scan_gradient);
                if (ga && axis == -1)
                {
                    int shape[MAX_DIMS];
                    memcpy(shape, node->a->shape, node->a->ndims * sizeof(int));
                    tensor_t *reshaped = cuda_tensor_reshape(
                        ga, shape, node->a->ndims);
                    cuda_tensor_destroy(ga);
                    ga = reshaped;
                }
            }
            break;

        case AUTOGRAD_REPEAT_COUNTS:
            ga = cuda_tensor_repeat_counts_backward(
                upstream, node->a, node->repeat_prefix_counts,
                node->repeat_count, node->parameter);
            upstream = NULL;
            break;

        case AUTOGRAD_MATMUL:
            if (node->a->ndims < 2 || node->b->ndims < 2)
            {
                CUDA_THROW_INVALID("Autograd matmul requires operands with rank at least 2");
                cuda_tensor_destroy(upstream);
                return 0;
            }
            {
                dtype_t gradient_dtype = autograd_is_low_precision_type(node->a->dtype)
                    ? DTYPE_FLOAT32 : node->a->dtype;
                tensor_t *a_values = gradient_cast(node->a, gradient_dtype);
                tensor_t *b_values = gradient_cast(node->b, gradient_dtype);
                tensor_t *upstream_values = gradient_cast(upstream, gradient_dtype);
                int a_axes[MAX_DIMS], b_axes[MAX_DIMS];
                for (int d = 0; d < node->a->ndims; d++) a_axes[d] = d;
                for (int d = 0; d < node->b->ndims; d++) b_axes[d] = d;
                int a_last = node->a->ndims - 1;
                int b_last = node->b->ndims - 1;
                int a_second_last = node->a->ndims - 2;
                int b_second_last = node->b->ndims - 2;
                a_axes[a_last] = a_second_last;
                a_axes[a_second_last] = a_last;
                b_axes[b_last] = b_second_last;
                b_axes[b_second_last] = b_last;
                tensor_t *b_transposed = b_values
                    ? cuda_tensor_transpose(b_values, b_axes, node->b->ndims) : NULL;
                if (upstream_values && b_transposed)
                {
                    ga = cuda_tensor_matmul(upstream_values, b_transposed);
                    cuda_tensor_destroy(b_transposed);
                }
                else if (b_transposed)
                    cuda_tensor_destroy(b_transposed);
                tensor_t *a_transposed = a_values
                    ? cuda_tensor_transpose(a_values, a_axes, node->a->ndims) : NULL;
                if (a_transposed && upstream_values)
                {
                    gb = cuda_tensor_matmul(a_transposed, upstream_values);
                    cuda_tensor_destroy(a_transposed);
                }
                else if (a_transposed)
                    cuda_tensor_destroy(a_transposed);
                if (a_values) cuda_tensor_destroy(a_values);
                if (b_values) cuda_tensor_destroy(b_values);
                if (upstream_values) cuda_tensor_destroy(upstream_values);
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
            else if (node->op == OP_BROADCAST)
            {
                ga = upstream;
                upstream = NULL;
            }
            else
            {
                CUDA_THROW_INVALID("Backward is not implemented for this tensor view");
                cuda_tensor_destroy(upstream);
                return 0;
            }
            break;

        case AUTOGRAD_SLICE:
            ga = cuda_tensor_slice_scatter(upstream, node->input_shape,
                node->input_ndims, node->axes, node->starts, node->steps);
            break;

        case AUTOGRAD_INDEX_TRANSFORM:
            ga = cuda_tensor_index_transform_backward(upstream,
                                                       &node->transform);
            break;

        case AUTOGRAD_UNSUPPORTED:
            CUDA_THROW_INVALID("Backward is not implemented for operation %d; gradient tracking cannot skip this path",
                               node->op);
            cuda_tensor_destroy(upstream);
            return 0;

        case AUTOGRAD_GATHER:
            {
                scalar_value_t zero = {0};
                zero.dtype = autograd_is_low_precision_type(node->a->dtype)
                    ? DTYPE_FLOAT32 : node->a->dtype;
                int shape[MAX_DIMS];
                memcpy(shape, node->a->shape, node->a->ndims * sizeof(int));
                tensor_t *empty = cuda_tensor_create_with_value(
                    shape, node->a->ndims, zero, zero.dtype);
                if (empty)
                {
                    ga = cuda_tensor_scatter_add(empty, node->b, upstream, node->parameter);
                    cuda_tensor_destroy(empty);
                }
            }
            break;

        case AUTOGRAD_SCATTER_ADD:
            ga = upstream;
            upstream = NULL;
            gc = cuda_tensor_gather(ga, node->b, node->parameter);
            break;

        case AUTOGRAD_CAST:
            ga = gradient_cast(upstream,
                autograd_is_low_precision_type(node->a->dtype)
                    ? DTYPE_FLOAT32 : node->a->dtype);
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
    return gradient_add(entries, count, capacity, gradient_index, node->a, ga) &&
           gradient_add(entries, count, capacity, gradient_index, node->b, gb) &&
           gradient_add(entries, count, capacity, gradient_index, node->c, gc);
}

int autograd_backward(tensor_t *tensor, tensor_t *gradient, int retain_graph)
{
    if (!tensor || !tensor->requires_grad)
    {
        CUDA_THROW_INVALID("backward() requires a tensor marked with requiresGrad()");
        return 0;
    }
    if (!autograd_is_matmul_differentiable_type(tensor->dtype))
    {
        CUDA_THROW_INVALID("backward() is only supported for float32, float64, float16 and bfloat16 tensors");
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
    tensor_index_t visited = {0}, gradient_index = {0};
    gradient_entry_t *entries = NULL;
    size_t count = 0, capacity = 0;
    int success = autograd_visit(tensor, &topology, &visited);
    int gradient_owned = 0;
    if (success && !gradient)
    {
        scalar_value_t one = {0};
        one.dtype = autograd_is_low_precision_type(tensor->dtype)
            ? DTYPE_FLOAT32 : tensor->dtype;
        if (one.dtype == DTYPE_FLOAT64) one.v.f64 = 1.0;
        else one.v.f32 = 1.0f;
        int shape[MAX_DIMS];
        memcpy(shape, tensor->shape, tensor->ndims * sizeof(int));
        gradient = cuda_tensor_create_with_value(shape, tensor->ndims, one, one.dtype);
        gradient_owned = gradient != NULL;
    }
    else if (success)
    {
        if (autograd_is_low_precision_type(tensor->dtype))
        {
            gradient = gradient_cast(gradient, DTYPE_FLOAT32);
            gradient_owned = gradient != NULL;
        }
        else
        {
            gradient->ref_count++;
            gradient_owned = 1;
        }
    }

    int suppression_active = 0;
    if (success && !gradient)
        success = 0;
    if (success)
    {
        autograd_suppressed++;
        suppression_active = 1;
        if (!gradient_add(&entries, &count, &capacity, &gradient_index, tensor, gradient))
            success = 0;
        gradient = NULL;
        gradient_owned = 0;
    }

    for (size_t i = topology.count; success && i > 0; i--)
    {
        tensor_t *current = topology.items[i - 1];
        if (!current->grad_fn)
            continue;
        tensor_t *upstream = gradient_get(entries, &gradient_index, current);
        if (!upstream)
            continue;
        size_t entry_index = gradient_find(&gradient_index, current);
        tensor_t *owned_upstream = entries[entry_index].gradient;
        entries[entry_index].gradient = NULL;
        if (!autograd_propagate(current, owned_upstream, &entries, &count, &capacity,
                                &gradient_index))
            success = 0;
    }
    if (suppression_active)
        autograd_suppressed--;

    for (size_t i = 0; success && i < count; i++)
    {
        tensor_t *leaf = entries[i].tensor;
        if (leaf->grad_fn || !leaf->requires_grad)
            continue;
        if (leaf->grad)
        {
            tensor_t *sum = cuda_tensor_op(leaf->grad, entries[i].gradient, OP_ADD);
            if (!sum)
            {
                success = 0;
                break;
            }
            autograd_clear_gradient(leaf);
            leaf->grad = sum;
        }
        else
        {
            leaf->grad = entries[i].gradient;
            entries[i].gradient = NULL;
        }
    }

    if (success && !retain_graph)
    {
        for (size_t i = 0; i < topology.count; i++)
            topology.items[i]->ref_count++;
        for (size_t i = 0; i < topology.count; i++)
        {
            tensor_t *current = topology.items[i];
            if (current->grad_fn)
            {
                current->autograd_consumed = 1;
                autograd_release_node(current);
            }
        }
        for (size_t i = 0; i < topology.count; i++)
            cuda_tensor_destroy(topology.items[i]);
    }

    if (gradient && gradient_owned) cuda_tensor_destroy(gradient);
    for (size_t i = 0; i < count; i++)
        if (entries[i].gradient) cuda_tensor_destroy(entries[i].gradient);
    if (entries) efree(entries);
    if (topology.items) efree(topology.items);
    tensor_index_free(&visited);
    tensor_index_free(&gradient_index);
    if (!success && !EG(exception))
        CUDA_THROW_RUNTIME("Autograd backward failed");
    return success;
}
