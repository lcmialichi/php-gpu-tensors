#include "fusion.h"
#include "cuda.h"
#include "cuda_array_ce.h"
#include "ca_private.h"
#include "compiler_ce.h"
#include "kernel_types.h"
#include "cuda_exceptions.h"
#include "zend_interfaces.h"
#include "zend_fibers.h"
#include "zend_smart_str.h"
#include <cuda.h>
#include <math.h>
#include <limits.h>
#include <string.h>

#define FUSION_MAX_NODES 512
#define FUSION_KERNEL_BUDGET 32

typedef enum
{
    FUSION_INPUT, FUSION_BINARY, FUSION_SCALAR, FUSION_UNARY,
    FUSION_REDUCE, FUSION_ARG_REDUCE, FUSION_MATMUL, FUSION_VIEW
} fusion_kind;

typedef struct fusion_node
{
    fusion_kind kind;
    tensor_t *a;
    tensor_t *b;
    operation_type_t op;
    scalar_value_t scalar;
    int parameter;
    int axes[MAX_DIMS];
} fusion_node;

typedef struct fusion_scope
{
    tensor_t **nodes;
    size_t count;
    size_t capacity;
    int compiling;
} fusion_scope;

typedef struct
{
    tensor_t *tensor;
    int cut;
    int output;
    int cost;
    int scheduled;
} fusion_item;

typedef struct
{
    size_t root;
    size_t *leaves;
    size_t leaf_count;
    char name[32];
    CUfunction function;
} fusion_step;

typedef struct
{
    fusion_item *items;
    size_t count;
    fusion_step *steps;
    size_t step_count;
    size_t kernel_count;
    size_t *inputs;
    size_t input_count;
    CUmodule module;
    CUcontext context;
    int device;
    zend_string *source;
    size_t executions;
} fusion_plan;

typedef struct
{
    fusion_scope *scope;
    fusion_plan *plan;
    zval outputs;
    zend_object std;
} fusion_graph;

static zend_class_entry *fusion_ce;
static zend_class_entry *fusion_graph_ce;
static zend_object_handlers fusion_graph_handlers;

#define Z_FUSION_GRAPH_P(zv) ((fusion_graph *)((char *)Z_OBJ_P(zv) - XtOffsetOf(fusion_graph, std)))

static int fusion_execute(fusion_plan *plan, tensor_t **values);
static fusion_plan *fusion_build_plan(tensor_t **roots, size_t root_count);

int fusion_active(void)
{
    return CUDA_G(fusion_scope) != NULL;
}

int fusion_check_mutation(void)
{
    if (fusion_active())
    {
        CUDA_THROW_RUNTIME("Mutation, custom kernel launches and device changes are not allowed during Fusion capture");
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

static tensor_t *fusion_record(fusion_kind kind, tensor_t *a, tensor_t *b,
                              operation_type_t op, const int *shape, int ndims, dtype_t dtype)
{
    fusion_scope *scope = CUDA_G(fusion_scope);
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
    efree(node);
}

static void fusion_scope_free(fusion_scope *scope, int invalidate)
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

static int fusion_inline(tensor_t *tensor)
{
    fusion_node *node = tensor->fusion;
    if (!node) return 0;
    return (node->kind == FUSION_BINARY || node->kind == FUSION_SCALAR) &&
        (node->op == OP_ADD || node->op == OP_SUB || node->op == OP_MUL ||
         node->op == OP_DIV || (node->op >= OP_GT && node->op <= OP_LE));
}

static size_t fusion_find(fusion_plan *plan, tensor_t *tensor)
{
    for (size_t i = 0; i < plan->count; i++)
        if (plan->items[i].tensor == tensor) return i;
    return SIZE_MAX;
}

static size_t fusion_collect(fusion_plan *plan, tensor_t *tensor)
{
    size_t id = fusion_find(plan, tensor);
    if (id != SIZE_MAX) return id;
    fusion_node *node = tensor->fusion;
    size_t a = SIZE_MAX, b = SIZE_MAX;
    if (node && node->a) a = fusion_collect(plan, node->a);
    if (node && node->b) b = fusion_collect(plan, node->b);
    id = plan->count++;
    plan->items = erealloc(plan->items, plan->count * sizeof(fusion_item));
    plan->items[id] = (fusion_item){ .tensor = tensor };
    if (node && node->kind != FUSION_INPUT)
    {
        if (fusion_inline(tensor))
        {
            int cost = 1;
            if (a != SIZE_MAX && !plan->items[a].cut) cost += plan->items[a].cost;
            if (b != SIZE_MAX && !plan->items[b].cut) cost += plan->items[b].cost;
            if (cost > FUSION_KERNEL_BUDGET)
            {
                if (a != SIZE_MAX && plan->items[a].cost) plan->items[a].cut = 1;
                if (b != SIZE_MAX && plan->items[b].cost) plan->items[b].cut = 1;
                cost = 1;
            }
            plan->items[id].cost = cost;
        }
        else
        {
            plan->items[id].cut = 1;
            if (a != SIZE_MAX && plan->items[a].cost) plan->items[a].cut = 1;
            if (b != SIZE_MAX && plan->items[b].cost) plan->items[b].cut = 1;
        }
    }
    return id;
}

static void fusion_schedule(fusion_plan *plan, size_t id);

static void fusion_dependencies(fusion_plan *plan, size_t id, size_t root)
{
    if (id != root && plan->items[id].cut)
    {
        fusion_schedule(plan, id);
        return;
    }
    fusion_node *node = plan->items[id].tensor->fusion;
    if (node && node->a) fusion_dependencies(plan, fusion_find(plan, node->a), root);
    if (node && node->b) fusion_dependencies(plan, fusion_find(plan, node->b), root);
}

static void fusion_schedule(fusion_plan *plan, size_t id)
{
    fusion_item *item = &plan->items[id];
    if (item->scheduled || !item->tensor->fusion || item->tensor->fusion->kind == FUSION_INPUT)
        return;
    fusion_dependencies(plan, id, id);
    item->scheduled = 1;
    plan->steps = erealloc(plan->steps, (plan->step_count + 1) * sizeof(fusion_step));
    fusion_step *step = &plan->steps[plan->step_count++];
    memset(step, 0, sizeof(*step));
    step->root = id;
    if (fusion_inline(item->tensor))
    {
        snprintf(step->name, sizeof(step->name), "fusion_%zu", plan->kernel_count++);
    }
}

static const char *fusion_ctype(dtype_t dtype)
{
    switch (dtype)
    {
        case DTYPE_FLOAT32: return "float";
        case DTYPE_FLOAT64: return "double";
        case DTYPE_INT8: return "signed char";
        case DTYPE_INT16: return "short";
        case DTYPE_INT32: return "int";
        case DTYPE_INT64: return "long long";
        case DTYPE_UINT8: return "unsigned char";
        case DTYPE_UINT16: return "unsigned short";
        case DTYPE_UINT32: return "unsigned int";
        case DTYPE_UINT64: return "unsigned long long";
        case DTYPE_BOOL: return "bool";
        default: return NULL;
    }
}

static const char *fusion_operator(operation_type_t op)
{
    switch (op)
    {
        case OP_ADD: return "+";
        case OP_SUB: return "-";
        case OP_MUL: return "*";
        case OP_DIV: return "/";
        case OP_GT: return ">";
        case OP_LT: return "<";
        case OP_EQ: return "==";
        case OP_NE: return "!=";
        case OP_GE: return ">=";
        case OP_LE: return "<=";
        default: return NULL;
    }
}

static void fusion_scalar_source(smart_str *source, scalar_value_t scalar)
{
    /* Bit patterns preserve NaNs, signed zero and locale-independent constants. */
    switch (scalar.dtype)
    {
        case DTYPE_FLOAT64:
        {
            uint64_t bits;
            memcpy(&bits, &scalar.v.f64, sizeof(bits));
            smart_str_append_printf(source, "__longlong_as_double(0x%llxULL)", (unsigned long long)bits);
            break;
        }
        case DTYPE_FLOAT32:
        {
            uint32_t bits;
            memcpy(&bits, &scalar.v.f32, sizeof(bits));
            smart_str_append_printf(source, "__int_as_float(0x%xU)", bits);
            break;
        }
        case DTYPE_INT64:
            smart_str_append_printf(source, "static_cast<long long>(0x%llxULL)", (unsigned long long)scalar.v.i64);
            break;
        case DTYPE_INT32: smart_str_append_printf(source, "%d", scalar.v.i32); break;
        case DTYPE_INT8: smart_str_append_printf(source, "%d", scalar.v.i8); break;
        case DTYPE_BOOL: smart_str_appends(source, scalar.v.b ? "true" : "false"); break;
        default: ZEND_ASSERT(0);
    }
}

static void fusion_emit(fusion_plan *plan, fusion_step *step, size_t id,
                        smart_str *body, unsigned char *emitted)
{
    if (emitted[id]) return;
    emitted[id] = 1;
    tensor_t *tensor = plan->items[id].tensor;
    tensor_t *root = plan->items[step->root].tensor;
    const char *type = fusion_ctype(tensor->dtype);
    fusion_node *node = tensor->fusion;
    if (id != step->root && (!fusion_inline(tensor) || plan->items[id].cut))
    {
        size_t leaf = step->leaf_count++;
        step->leaves = erealloc(step->leaves, step->leaf_count * sizeof(size_t));
        step->leaves[leaf] = id;
        smart_str_append_printf(body, "size_t o%zu=0;\n", id);
        size_t divisor = 1;
        for (int d = root->ndims - 1; d >= 0; d--)
        {
            int td = d - (root->ndims - tensor->ndims);
            if (td >= 0 && tensor->shape[td] != 1 && root->shape[d] != 0)
                smart_str_append_printf(body, "o%zu+=((i/%zuULL)%%%dULL)*%zuULL;\n",
                                        id, divisor, root->shape[d], tensor->strides[td]);
            divisor *= root->shape[d];
        }
        smart_str_append_printf(body, "%s v%zu=static_cast<const %s*>(p%zu)[o%zu];\n",
                                type, id, type, leaf, id);
        return;
    }
    size_t a = fusion_find(plan, node->a);
    fusion_emit(plan, step, a, body, emitted);
    size_t b = SIZE_MAX;
    if (node->b)
    {
        b = fusion_find(plan, node->b);
        fusion_emit(plan, step, b, body, emitted);
    }
    smart_str_append_printf(body, "%s v%zu=static_cast<%s>(", type, id, type);
    if (node->kind == FUSION_SCALAR)
    {
        smart_str_append_printf(body, "static_cast<%s>(", type);
        if (node->parameter) fusion_scalar_source(body, node->scalar);
        else smart_str_append_printf(body, "v%zu", a);
        smart_str_append_printf(body, ")%s static_cast<%s>(", fusion_operator(node->op), type);
        if (node->parameter) smart_str_append_printf(body, "v%zu", a);
        else fusion_scalar_source(body, node->scalar);
        smart_str_appends(body, ")");
    }
    else
    {
        smart_str_append_printf(body, "static_cast<%s>(v%zu)%s static_cast<%s>(v%zu)",
                                type, a, fusion_operator(node->op), type, b);
    }
    smart_str_appends(body, ");\n");
}

static void fusion_plan_free(fusion_plan *plan)
{
    if (!plan) return;
    if (plan->module)
    {
        CUresult error = cuModuleUnload(plan->module);
        if (error != CUDA_SUCCESS && !EG(exception))
            php_error_docref(NULL, E_WARNING, "Failed to unload fusion module (CUDA error %d)", error);
    }
    for (size_t i = 0; i < plan->step_count; i++)
        if (plan->steps[i].leaves) efree(plan->steps[i].leaves);
    if (plan->steps) efree(plan->steps);
    if (plan->items) efree(plan->items);
    if (plan->inputs) efree(plan->inputs);
    if (plan->source) zend_string_release(plan->source);
    efree(plan);
}

static fusion_plan *fusion_build_plan(tensor_t **roots, size_t root_count)
{
    fusion_plan *plan = ecalloc(1, sizeof(fusion_plan));
    for (size_t i = 0; i < root_count; i++)
    {
        size_t id = fusion_collect(plan, roots[i]);
        if (roots[i]->fusion && roots[i]->fusion->kind != FUSION_INPUT)
            plan->items[id].cut = 1;
        plan->items[id].output = 1;
    }
    for (size_t i = 0; i < root_count; i++)
        fusion_schedule(plan, fusion_find(plan, roots[i]));

    smart_str source = {0};
    const char **names = plan->kernel_count ? emalloc(plan->kernel_count * sizeof(char *)) : NULL;
    size_t name_count = 0;
    for (size_t i = 0; i < plan->step_count; i++)
    {
        fusion_step *step = &plan->steps[i];
        if (!step->name[0]) continue;
        names[name_count++] = step->name;
        smart_str body = {0};
        unsigned char *emitted = ecalloc(plan->count, 1);
        fusion_emit(plan, step, step->root, &body, emitted);
        efree(emitted);
        smart_str_append_printf(&source, "extern \"C\" __global__ void %s(", step->name);
        for (size_t j = 0; j < step->leaf_count; j++)
            smart_str_append_printf(&source, "const void* p%zu,", j);
        tensor_t *root = plan->items[step->root].tensor;
        smart_str_append_printf(&source,
            "void* out){for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;"
            "i<%zuULL;i+=(size_t)blockDim.x*gridDim.x){\n", root->total_size);
        if (body.s) smart_str_append(&source, body.s);
        smart_str_append_printf(&source, "static_cast<%s*>(out)[i]=v%zu;\n}}\n",
                                fusion_ctype(root->dtype), step->root);
        smart_str_free(&body);
    }
    smart_str_0(&source);
    plan->source = source.s ? source.s : ZSTR_EMPTY_ALLOC();
    cudaError_t runtime_error = cudaGetDevice(&plan->device);
    if (runtime_error == cudaSuccess) runtime_error = cudaFree(NULL);
    CUresult error = runtime_error == cudaSuccess ? cuCtxGetCurrent(&plan->context) : CUDA_ERROR_INVALID_CONTEXT;
    if (error != CUDA_SUCCESS || !plan->context)
    {
        CUDA_THROW_RUNTIME("Cannot access CUDA context for fusion (CUDA error %d)", error);
        if (names) efree(names);
        fusion_plan_free(plan);
        return NULL;
    }
    if (plan->kernel_count)
    {
        zval module;
        ZVAL_UNDEF(&module);
        int ok = cuda_compile_generated_source(plan->source, names, plan->kernel_count, &module);
        efree(names);
        if (ok)
            error = cuModuleLoadData(&plan->module, Z_CUDA_MODULE_P(&module)->ptx_code);
        if (!Z_ISUNDEF(module)) zval_ptr_dtor(&module);
        if (!ok || error != CUDA_SUCCESS)
        {
            if (!EG(exception)) CUDA_THROW_RUNTIME("Failed to load fusion PTX (CUDA error %d)", error);
            fusion_plan_free(plan);
            return NULL;
        }
        for (size_t i = 0; i < plan->step_count; i++)
        {
            fusion_step *step = &plan->steps[i];
            if (step->name[0] &&
                (error = cuModuleGetFunction(&step->function, plan->module, step->name)) != CUDA_SUCCESS)
            {
                CUDA_THROW_RUNTIME("Failed to resolve fused kernel %s (CUDA error %d)", step->name, error);
                fusion_plan_free(plan);
                return NULL;
            }
        }
    }
    return plan;
}

static int fusion_execute(fusion_plan *plan, tensor_t **values)
{
    int device;
    CUcontext context;
    cudaError_t runtime_error = cudaGetDevice(&device);
    CUresult error = cuCtxGetCurrent(&context);
    if (runtime_error != cudaSuccess || error != CUDA_SUCCESS ||
        device != plan->device || context != plan->context)
    {
        CUDA_THROW_RUNTIME("Fusion graph must execute on its compilation device and CUDA context");
        return 0;
    }
    for (size_t i = 0; i < plan->step_count; i++)
    {
        fusion_step *step = &plan->steps[i];
        tensor_t *tensor = plan->items[step->root].tensor;
        fusion_node *node = tensor->fusion;
        tensor_t *result = NULL;
        if (step->name[0])
        {
            result = cuda_tensor_create_empty_with_dtype(tensor->shape, tensor->ndims, tensor->dtype);
            if (!result) return 0;
            if (result->total_size)
            {
                void **args = emalloc((step->leaf_count + 1) * sizeof(void *));
                for (size_t j = 0; j < step->leaf_count; j++)
                    args[j] = &values[step->leaves[j]]->data;
                args[step->leaf_count] = &result->data;
                size_t blocks = (result->total_size - 1) / 256 + 1;
                if (blocks > 65535) blocks = 65535;
                error = cuLaunchKernel(step->function, (unsigned int)blocks, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
                efree(args);
                if (error != CUDA_SUCCESS)
                {
                    cuda_tensor_destroy(result);
                    CUDA_THROW_RUNTIME("Fused kernel launch failed (CUDA error %d)", error);
                    return 0;
                }
            }
        }
        else
        {
            tensor_t *a = values[fusion_find(plan, node->a)];
            tensor_t *b = node->b ? values[fusion_find(plan, node->b)] : NULL;
            tensor_t a_view, b_view;
            if (node->kind == FUSION_BINARY || node->kind == FUSION_REDUCE ||
                node->kind == FUSION_ARG_REDUCE)
            {
                /* View data already points at the slice; legacy launchers also add offset. */
                a_view = *a;
                a_view.offset = 0;
                a = &a_view;
                if (b)
                {
                    b_view = *b;
                    b_view.offset = 0;
                    b = &b_view;
                }
            }
            tensor_t *flat = NULL;
            int axis = node->parameter;
            if ((node->kind == FUSION_REDUCE || node->kind == FUSION_ARG_REDUCE) && axis == -1)
            {
                if (a->total_size > INT_MAX)
                {
                    CUDA_THROW_INVALID("Global fusion reduction exceeds the supported flatten size");
                    return 0;
                }
                int shape[] = {(int)a->total_size};
                flat = cuda_tensor_reshape(a, shape, 1);
                if (!flat) return 0;
                a = flat;
                axis = 0;
            }
            switch (node->kind)
            {
                case FUSION_BINARY: result = cuda_tensor_op(a, b, node->op); break;
                case FUSION_SCALAR:
                    result = node->parameter ? cuda_inv_scalar_op(a, node->scalar, node->op)
                                             : cuda_scalar_op(a, node->scalar, node->op);
                    break;
                case FUSION_UNARY: result = cuda_unary_op(a, node->op); break;
                case FUSION_REDUCE: result = cuda_tensor_reduce(a, axis, node->op); break;
                case FUSION_ARG_REDUCE: result = cuda_tensor_reduce_arg(a, axis, node->op); break;
                case FUSION_MATMUL: result = cuda_tensor_matmul(a, b); break;
                case FUSION_VIEW:
                    result = node->op == OP_RESHAPE
                        ? cuda_tensor_reshape(a, tensor->shape, tensor->ndims)
                        : cuda_tensor_transpose(a, node->axes, tensor->ndims);
                    break;
                default: ZEND_ASSERT(0);
            }
            if (flat) cuda_tensor_destroy(flat);
            if (!result)
            {
                if (!EG(exception)) CUDA_THROW_RUNTIME("Fusion execution boundary failed");
                return 0;
            }
        }
        values[step->root] = result;
    }
    runtime_error = cudaDeviceSynchronize();
    if (runtime_error != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Fusion synchronization failed: %s", cudaGetErrorString(runtime_error));
        return 0;
    }
    plan->executions++;
    return 1;
}

static tensor_t **fusion_values(fusion_plan *plan)
{
    tensor_t **values = ecalloc(plan->count, sizeof(tensor_t *));
    for (size_t i = 0; i < plan->count; i++)
        if (!plan->items[i].tensor->fusion) values[i] = plan->items[i].tensor;
    return values;
}

static void fusion_values_free(fusion_plan *plan, tensor_t **values)
{
    /* Even a failed launch can leave earlier kernels using these allocations. */
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess && !EG(exception))
        CUDA_THROW_RUNTIME("Fusion cleanup synchronization failed: %s", cudaGetErrorString(error));
    for (size_t i = 0; i < plan->count; i++)
        if (values[i] && values[i] != plan->items[i].tensor)
            cuda_tensor_destroy(values[i]);
    efree(values);
}

static int fusion_materialize_roots(tensor_t **roots, size_t count)
{
    fusion_scope *scope = CUDA_G(fusion_scope);
    CUDA_G(fusion_scope) = NULL;
    fusion_plan *plan = fusion_build_plan(roots, count);
    if (!plan)
    {
        CUDA_G(fusion_scope) = scope;
        return 0;
    }
    tensor_t **values = fusion_values(plan);
    int ok = 1;
    for (size_t i = 0; i < plan->count; i++)
    {
        if (plan->items[i].tensor->fusion && plan->items[i].tensor->fusion->kind == FUSION_INPUT)
        {
            CUDA_THROW_RUNTIME("Compiled graph placeholders cannot be materialized outside FusionGraph::run");
            ok = 0;
            break;
        }
    }
    if (ok) ok = fusion_execute(plan, values);
    if (ok)
    {
        for (size_t i = 0; i < plan->count; i++)
        {
            tensor_t *result = values[i];
            if (!result || result == plan->items[i].tensor || !result->base_tensor) continue;
            for (size_t j = 0; j < plan->count; j++)
            {
                if (result->base_tensor == values[j] && values[j] != plan->items[j].tensor)
                {
                    result->base_tensor = plan->items[j].tensor;
                    break;
                }
            }
        }
        for (size_t i = 0; i < plan->count; i++)
        {
            tensor_t *target = plan->items[i].tensor;
            if (!values[i] || values[i] == target) continue;
            tensor_t *result = values[i];
            int references = target->ref_count + result->ref_count - 1;
            fusion_release_node(target);
            if (target->shape) efree(target->shape);
            if (target->strides) efree(target->strides);
            *target = *result;
            target->ref_count = references;
            efree(result);
            values[i] = target;
        }
    }
    fusion_values_free(plan, values);
    fusion_plan_free(plan);
    CUDA_G(fusion_scope) = scope;
    return ok && !EG(exception);
}

int fusion_materialize(tensor_t *tensor)
{
    if (tensor->fusion_failed)
    {
        CUDA_THROW_RUNTIME("Tensor belongs to an aborted Fusion capture");
        return 0;
    }
    if (!tensor->fusion) return 1;
    if ((CUDA_G(fusion_scope) && CUDA_G(fusion_scope)->compiling) ||
        tensor->fusion->kind == FUSION_INPUT)
    {
        CUDA_THROW_RUNTIME("Fusion::compile cannot read tensor data during capture; use FusionGraph::run");
        return 0;
    }
    return fusion_materialize_roots(&tensor, 1);
}

static void fusion_wrap(zval *value, tensor_t *tensor)
{
    object_init_ex(value, cuda_array_ce);
    cuda_array_obj *object = Z_CUDA_ARRAY_P(value);
    object->tensor_handle = tensor;
    object->shape = zend_new_array(tensor->ndims);
    for (int i = 0; i < tensor->ndims; i++)
    {
        zval dim;
        ZVAL_LONG(&dim, tensor->shape[i]);
        zend_hash_next_index_insert(object->shape, &dim);
    }
}

static int fusion_output_roots(zval *output, tensor_t ***roots, size_t *count, HashTable *seen)
{
    ZVAL_DEREF(output);
    if (Z_TYPE_P(output) == IS_OBJECT && instanceof_function(Z_OBJCE_P(output), cuda_array_ce))
    {
        tensor_t *tensor = Z_CUDA_ARRAY_P(output)->tensor_handle;
        if (!tensor || tensor->fusion_failed)
        {
            CUDA_THROW_INVALID("Fusion callback returned an invalid tensor");
            return 0;
        }
        *roots = erealloc(*roots, (*count + 1) * sizeof(tensor_t *));
        (*roots)[(*count)++] = tensor;
        return 1;
    }
    if (Z_TYPE_P(output) == IS_ARRAY)
    {
        zend_ulong key = (zend_ulong)(uintptr_t)Z_ARRVAL_P(output);
        if (zend_hash_index_exists(seen, key))
        {
            CUDA_THROW_INVALID("Fusion outputs must not contain recursive arrays");
            return 0;
        }
        zend_hash_index_add_empty_element(seen, key);
        zval *value;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(output), value)
        {
            if (!fusion_output_roots(value, roots, count, seen)) return 0;
        }
        ZEND_HASH_FOREACH_END();
        zend_hash_index_del(seen, key);
        return 1;
    }
    CUDA_THROW_INVALID("Fusion callback must return a CudaArray or an array of CudaArray outputs");
    return 0;
}

static int fusion_begin(void)
{
    if (fusion_active())
    {
        CUDA_THROW_RUNTIME("Nested Fusion captures are not supported");
        return 0;
    }
    CUDA_G(fusion_scope) = ecalloc(1, sizeof(fusion_scope));
    zend_fiber_switch_block();
    return 1;
}

static int fusion_call(zend_fcall_info *fci, zend_fcall_info_cache *fcc, zval *output)
{
    fci->retval = output;
    ZVAL_UNDEF(output);
    if (zend_call_function(fci, fcc) == FAILURE && !EG(exception))
        CUDA_THROW_RUNTIME("Failed to invoke Fusion callback");
    return !EG(exception);
}

ZEND_METHOD(Fusion, __construct) {}

ZEND_METHOD(Fusion, run)
{
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    zend_bool enabled = 1;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_FUNC(fci, fcc)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(enabled)
    ZEND_PARSE_PARAMETERS_END();
    if (fusion_active())
    {
        CUDA_THROW_RUNTIME("Nested Fusion captures are not supported");
        RETURN_THROWS();
    }
    if (!enabled)
    {
        fusion_call(&fci, &fcc, return_value);
        return;
    }
    if (!fusion_begin()) RETURN_THROWS();
    fusion_scope *scope = CUDA_G(fusion_scope);
    int ok = fusion_call(&fci, &fcc, return_value);
    tensor_t **roots = NULL;
    size_t count = 0;
    HashTable seen;
    zend_hash_init(&seen, 8, NULL, NULL, 0);
    if (ok) ok = fusion_output_roots(return_value, &roots, &count, &seen);
    zend_hash_destroy(&seen);
    if (ok)
    {
        /* Materialize escaped tensors too, not just callback return values. */
        for (size_t i = 0; i < scope->count; i++)
        {
            tensor_t *tensor = scope->nodes[i];
            int internal = 1;
            for (size_t j = 0; j < scope->count; j++)
            {
                fusion_node *node = scope->nodes[j]->fusion;
                if (node && node->a == tensor) internal++;
                if (node && node->b == tensor) internal++;
            }
            if (tensor->fusion && tensor->ref_count > internal)
            {
                roots = erealloc(roots, (count + 1) * sizeof(tensor_t *));
                roots[count++] = tensor;
            }
        }
        if (count) ok = fusion_materialize_roots(roots, count);
    }
    if (roots) efree(roots);
    CUDA_G(fusion_scope) = NULL;
    fusion_scope_free(scope, !ok);
    zend_fiber_switch_unblock();
    if (!ok) RETURN_THROWS();
}

ZEND_METHOD(Fusion, compile)
{
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    HashTable *inputs;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_FUNC(fci, fcc)
        Z_PARAM_ARRAY_HT(inputs)
    ZEND_PARSE_PARAMETERS_END();
    if (!fusion_begin()) RETURN_THROWS();
    fusion_scope *scope = CUDA_G(fusion_scope);
    scope->compiling = 1;
    size_t count = zend_hash_num_elements(inputs);
    zval *arguments = count ? safe_emalloc(count, sizeof(zval), 0) : NULL;
    tensor_t **input_tensors = count ? safe_emalloc(count, sizeof(tensor_t *), 0) : NULL;
    size_t initialized = 0;
    int ok = 1;
    zval *input;
    ZEND_HASH_FOREACH_VAL(inputs, input)
    {
        ZVAL_DEREF(input);
        if (Z_TYPE_P(input) != IS_OBJECT || !instanceof_function(Z_OBJCE_P(input), cuda_array_ce) ||
            !Z_CUDA_ARRAY_P(input)->tensor_handle)
        {
            CUDA_THROW_INVALID("Fusion compilation inputs must be initialized CudaArray objects");
            ok = 0;
            break;
        }
        tensor_t *example = Z_CUDA_ARRAY_P(input)->tensor_handle;
        if (example->fusion || example->fusion_failed)
        {
            CUDA_THROW_INVALID("Fusion compilation inputs must be materialized tensors");
            ok = 0;
            break;
        }
        tensor_t *placeholder = fusion_record(FUSION_INPUT, NULL, NULL, OP_ADD,
                                              example->shape, example->ndims, example->dtype);
        if (!placeholder) { ok = 0; break; }
        if (example->ndims)
            memcpy(placeholder->strides, example->strides, example->ndims * sizeof(size_t));
        placeholder->is_contiguous_cached = -1;
        placeholder->fusion->parameter = (int)initialized;
        input_tensors[initialized] = placeholder;
        fusion_wrap(&arguments[initialized++], placeholder);
    }
    ZEND_HASH_FOREACH_END();
    zval outputs;
    ZVAL_UNDEF(&outputs);
    if (ok)
    {
        fci.params = arguments;
        fci.param_count = count;
        ok = fusion_call(&fci, &fcc, &outputs);
    }
    tensor_t **roots = NULL;
    size_t root_count = 0;
    HashTable seen;
    zend_hash_init(&seen, 8, NULL, NULL, 0);
    if (ok) ok = fusion_output_roots(&outputs, &roots, &root_count, &seen);
    zend_hash_destroy(&seen);
    fusion_plan *plan = ok ? fusion_build_plan(roots, root_count) : NULL;
    if (ok && !plan) ok = 0;
    if (ok)
    {
        plan->input_count = count;
        plan->inputs = count ? emalloc(count * sizeof(size_t)) : NULL;
        for (size_t i = 0; i < count; i++)
            plan->inputs[i] = fusion_find(plan, input_tensors[i]);
        object_init_ex(return_value, fusion_graph_ce);
        fusion_graph *graph = Z_FUSION_GRAPH_P(return_value);
        graph->scope = scope;
        graph->plan = plan;
        ZVAL_COPY_VALUE(&graph->outputs, &outputs);
    }
    else if (!Z_ISUNDEF(outputs)) zval_ptr_dtor(&outputs);
    if (roots) efree(roots);
    for (size_t i = 0; i < initialized; i++) zval_ptr_dtor(&arguments[i]);
    if (arguments) efree(arguments);
    if (input_tensors) efree(input_tensors);
    CUDA_G(fusion_scope) = NULL;
    if (!ok) fusion_scope_free(scope, 1);
    zend_fiber_switch_unblock();
    if (!ok) RETURN_THROWS();
}

static void fusion_copy_outputs(zval *target, zval *template, fusion_plan *plan, tensor_t **values)
{
    ZVAL_DEREF(template);
    if (Z_TYPE_P(template) == IS_OBJECT)
    {
        tensor_t *tensor = values[fusion_find(plan, Z_CUDA_ARRAY_P(template)->tensor_handle)];
        tensor->ref_count++;
        fusion_wrap(target, tensor);
        return;
    }
    array_init_size(target, zend_hash_num_elements(Z_ARRVAL_P(template)));
    zend_ulong index;
    zend_string *key;
    zval *value;
    ZEND_HASH_FOREACH_KEY_VAL(Z_ARRVAL_P(template), index, key, value)
    {
        zval copy;
        fusion_copy_outputs(&copy, value, plan, values);
        if (key) zend_hash_add_new(Z_ARRVAL_P(target), key, &copy);
        else zend_hash_index_add_new(Z_ARRVAL_P(target), index, &copy);
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(FusionGraph, __construct) {}

ZEND_METHOD(FusionGraph, run)
{
    zval *inputs;
    int count;
    ZEND_PARSE_PARAMETERS_START(0, -1)
        Z_PARAM_VARIADIC('*', inputs, count)
    ZEND_PARSE_PARAMETERS_END();
    fusion_graph *graph = Z_FUSION_GRAPH_P(ZEND_THIS);
    fusion_plan *plan = graph->plan;
    if (!plan || fusion_active())
    {
        CUDA_THROW_RUNTIME("FusionGraph cannot execute inside a capture or without a compiled plan");
        RETURN_THROWS();
    }
    if ((size_t)count != plan->input_count)
    {
        CUDA_THROW_INVALID("FusionGraph expects %zu inputs, got %d", plan->input_count, count);
        RETURN_THROWS();
    }
    tensor_t **values = fusion_values(plan);
    int ok = 1;
    for (int i = 0; i < count; i++)
    {
        zval *input = &inputs[i];
        ZVAL_DEREF(input);
        if (Z_TYPE_P(input) != IS_OBJECT || !instanceof_function(Z_OBJCE_P(input), cuda_array_ce) ||
            !Z_CUDA_ARRAY_P(input)->tensor_handle)
        {
            CUDA_THROW_INVALID("FusionGraph input %d must be an initialized CudaArray", i);
            ok = 0;
            break;
        }
        tensor_t *tensor = Z_CUDA_ARRAY_P(input)->tensor_handle;
        tensor_t *example = NULL;
        for (size_t j = 0; j < graph->scope->count; j++)
        {
            fusion_node *node = graph->scope->nodes[j]->fusion;
            if (node && node->kind == FUSION_INPUT && node->parameter == i)
            {
                example = graph->scope->nodes[j];
                break;
            }
        }
        if (!fusion_materialize(tensor)) { ok = 0; break; }
        if (tensor->dtype != example->dtype || tensor->ndims != example->ndims ||
            (tensor->ndims && (memcmp(tensor->shape, example->shape, tensor->ndims * sizeof(int)) ||
                              memcmp(tensor->strides, example->strides, tensor->ndims * sizeof(size_t)))))
        {
            CUDA_THROW_INVALID("FusionGraph input %d has incompatible shape, dtype or strides", i);
            ok = 0;
            break;
        }
        if (plan->inputs[i] != SIZE_MAX) values[plan->inputs[i]] = tensor;
    }
    if (ok) ok = fusion_execute(plan, values);
    if (ok) fusion_copy_outputs(return_value, &graph->outputs, plan, values);
    /* Replay inputs are borrowed, not allocations owned by this execution. */
    for (size_t i = 0; i < plan->input_count; i++)
        if (plan->inputs[i] != SIZE_MAX) values[plan->inputs[i]] = NULL;
    fusion_values_free(plan, values);
    if (!ok) RETURN_THROWS();
}

ZEND_METHOD(FusionGraph, getStats)
{
    ZEND_PARSE_PARAMETERS_NONE();
    fusion_plan *plan = Z_FUSION_GRAPH_P(ZEND_THIS)->plan;
    array_init(return_value);
    add_assoc_long(return_value, "nodes", plan ? plan->count : 0);
    add_assoc_long(return_value, "fusedKernels", plan ? plan->kernel_count : 0);
    add_assoc_long(return_value, "boundaries", plan ? plan->step_count - plan->kernel_count : 0);
    add_assoc_long(return_value, "executions", plan ? plan->executions : 0);
    add_assoc_long(return_value, "executionSteps", plan ? plan->step_count : 0);
    size_t intermediates = 0;
    if (plan)
        for (size_t i = 0; i < plan->step_count; i++)
            if (!plan->items[plan->steps[i].root].output &&
                plan->items[plan->steps[i].root].tensor->fusion->kind != FUSION_VIEW) intermediates++;
    add_assoc_long(return_value, "intermediateBuffers", intermediates);
}

ZEND_METHOD(FusionGraph, getSource)
{
    ZEND_PARSE_PARAMETERS_NONE();
    fusion_plan *plan = Z_FUSION_GRAPH_P(ZEND_THIS)->plan;
    if (!plan) RETURN_EMPTY_STRING();
    RETURN_STR_COPY(plan->source);
}

static zend_object *fusion_graph_create(zend_class_entry *ce)
{
    fusion_graph *graph = zend_object_alloc(sizeof(fusion_graph), ce);
    zend_object_std_init(&graph->std, ce);
    object_properties_init(&graph->std, ce);
    ZVAL_UNDEF(&graph->outputs);
    graph->std.handlers = &fusion_graph_handlers;
    return &graph->std;
}

static void fusion_graph_free(zend_object *object)
{
    fusion_graph *graph = (fusion_graph *)((char *)object - XtOffsetOf(fusion_graph, std));
    fusion_plan_free(graph->plan);
    if (!Z_ISUNDEF(graph->outputs)) zval_ptr_dtor(&graph->outputs);
    fusion_scope_free(graph->scope, 1);
    zend_object_std_dtor(object);
}

static HashTable *fusion_graph_gc(zend_object *object, zval **table, int *count)
{
    fusion_graph *graph = (fusion_graph *)((char *)object - XtOffsetOf(fusion_graph, std));
    *table = &graph->outputs;
    *count = Z_ISUNDEF(graph->outputs) ? 0 : 1;
    return zend_std_get_properties(object);
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_fusion_construct, 0, 0, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_fusion_run, 0, 1, IS_MIXED, 0)
    ZEND_ARG_TYPE_INFO(0, callback, IS_CALLABLE, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, enabled, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_fusion_compile, 0, 2, Cuda\\FusionGraph, 0)
    ZEND_ARG_TYPE_INFO(0, callback, IS_CALLABLE, 0)
    ZEND_ARG_TYPE_INFO(0, inputs, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_fusion_graph_run, 0, 0, IS_MIXED, 0)
    ZEND_ARG_VARIADIC_OBJ_INFO(0, inputs, Cuda\\CudaArray, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_fusion_stats, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_fusion_source, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

static const zend_function_entry fusion_methods[] = {
    ZEND_ME(Fusion, __construct, arginfo_fusion_construct, ZEND_ACC_PRIVATE)
    ZEND_ME(Fusion, run, arginfo_fusion_run, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(Fusion, compile, arginfo_fusion_compile, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_FE_END
};
static const zend_function_entry fusion_graph_methods[] = {
    ZEND_ME(FusionGraph, __construct, arginfo_fusion_construct, ZEND_ACC_PRIVATE)
    ZEND_ME(FusionGraph, run, arginfo_fusion_graph_run, ZEND_ACC_PUBLIC)
    ZEND_ME(FusionGraph, getStats, arginfo_fusion_stats, ZEND_ACC_PUBLIC)
    ZEND_ME(FusionGraph, getSource, arginfo_fusion_source, ZEND_ACC_PUBLIC)
    ZEND_FE_END
};

int fusion_init(void)
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, "Cuda\\Fusion", fusion_methods);
    fusion_ce = zend_register_internal_class(&ce);
    fusion_ce->ce_flags |= ZEND_ACC_FINAL | ZEND_ACC_NOT_SERIALIZABLE;
    INIT_CLASS_ENTRY(ce, "Cuda\\FusionGraph", fusion_graph_methods);
    fusion_graph_ce = zend_register_internal_class(&ce);
    fusion_graph_ce->ce_flags |= ZEND_ACC_FINAL | ZEND_ACC_NOT_SERIALIZABLE;
    fusion_graph_ce->create_object = fusion_graph_create;
    memcpy(&fusion_graph_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    fusion_graph_handlers.offset = XtOffsetOf(fusion_graph, std);
    fusion_graph_handlers.free_obj = fusion_graph_free;
    fusion_graph_handlers.get_gc = fusion_graph_gc;
    fusion_graph_handlers.clone_obj = NULL;
    return SUCCESS;
}
