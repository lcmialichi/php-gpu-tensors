#include "fusion_internal.h"

static const char *fusion_ctype(dtype_t dtype)
{
    static const char *types[] = {"float", "double", "signed char", "short", "int", "long long",
        "unsigned char", "unsigned short", "unsigned int", "unsigned long long", "bool"};
    ZEND_ASSERT(dtype <= DTYPE_BOOL);
    return types[dtype];
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
        default: ZEND_ASSERT(0); return NULL;
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

static int fusion_leaf(fusion_plan *plan, fusion_step *step, size_t id)
{
    for (size_t i = 0; i < step->root_count; i++)
        if (step->roots[i] == id) return 0;
    return !fusion_inline(plan->items[id].tensor) || plan->items[id].cut;
}

static void fusion_gather(fusion_plan *plan, fusion_step *step, size_t id, unsigned char *seen)
{
    if (seen[id]) return;
    seen[id] = 1;
    if (fusion_leaf(plan, step, id))
    {
        step->leaves = erealloc(step->leaves, (step->leaf_count + 1) * sizeof(size_t));
        step->leaves[step->leaf_count++] = id;
        return;
    }
    fusion_node *node = plan->items[id].tensor->fusion;
    tensor_t *children[] = {node->a, node->b, node->c};
    for (int i = 0; i < 3; i++)
        if (children[i]) fusion_gather(plan, step, fusion_find(plan, children[i]), seen);
}

static void fusion_params(smart_str *source, fusion_step *step, int declaration)
{
    for (size_t i = 0; i < step->leaf_count; i++)
    {
        if (declaration) smart_str_appends(source, "const void* ");
        smart_str_append_printf(source, "p%zu,", i);
    }
}

static void fusion_call_node(smart_str *source, fusion_step *step, size_t id, const char *index)
{
    smart_str_append_printf(source, "%s_n%zu(", step->name, id);
    fusion_params(source, step, 0);
    smart_str_appends(source, index);
    smart_str_appends(source, ")");
}

static void fusion_project(smart_str *source, tensor_t *parent, tensor_t *child, const char *name)
{
    smart_str_append_printf(source, "size_t %s=0;\n", name);
    size_t divisor = 1, stride = 1;
    for (int d = parent->ndims - 1; d >= 0; d--)
    {
        int axis = d - (parent->ndims - child->ndims);
        if (axis >= 0)
        {
            if (child->shape[axis] != 1 && parent->shape[d] != 0)
                smart_str_append_printf(source, "%s+=((i/%zuULL)%%%dULL)*%zuULL;\n",
                                        name, divisor, parent->shape[d], stride);
            stride *= child->shape[axis];
        }
        divisor *= parent->shape[d];
    }
}

static void fusion_unary_source(smart_str *source, fusion_node *node, dtype_t dtype)
{
    int floating = dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT64;
    const char *suffix = dtype == DTYPE_FLOAT64 ? "" : "f";
    switch (node->op)
    {
        case OP_NEG: smart_str_appends(source, "-a"); return;
        case OP_ABS:
            smart_str_appends(source, dtype_is_signed(dtype) ? "(a<0?-a:a)" : "a");
            return;
        case OP_FLOOR: case OP_CEIL: case OP_ROUND:
            if (!floating) { smart_str_appends(source, "a"); return; }
            break;
        default: break;
    }
    const char *name = NULL;
    switch (node->op)
    {
        case OP_EXP: name = "exp"; break;
        case OP_LOG: name = "log"; break;
        case OP_SQRT: name = "sqrt"; break;
        case OP_SIN: name = "sin"; break;
        case OP_COS: name = "cos"; break;
        case OP_TAN: name = "tan"; break;
        case OP_FLOOR: name = "floor"; break;
        case OP_CEIL: name = "ceil"; break;
        case OP_ROUND: name = "round"; break;
        default: ZEND_ASSERT(0);
    }
    smart_str_append_printf(source, "%s%s(%s)", name, suffix, floating ? "a" : "static_cast<float>(a)");
}

static void fusion_emit_node(fusion_plan *plan, fusion_step *step, size_t id,
                             smart_str *source, unsigned char *emitted)
{
    if (emitted[id]) return;
    emitted[id] = 1;
    tensor_t *tensor = plan->items[id].tensor;
    fusion_node *node = tensor->fusion;
    int leaf = fusion_leaf(plan, step, id);
    if (!leaf)
    {
        tensor_t *children[] = {node->a, node->b, node->c};
        for (int j = 0; j < 3; j++)
            if (children[j]) fusion_emit_node(plan, step, fusion_find(plan, children[j]), source, emitted);
    }
    const char *type = fusion_ctype(tensor->dtype);
    smart_str_append_printf(source, "__device__ __forceinline__ %s %s_n%zu(", type, step->name, id);
    fusion_params(source, step, 1);
    smart_str_appends(source, "size_t i){\n");
    if (leaf)
    {
        size_t parameter = 0;
        while (step->leaves[parameter] != id) parameter++;
        smart_str_appends(source, "size_t offset=0;\n");
        size_t divisor = 1;
        for (int d = tensor->ndims - 1; d >= 0; d--)
        {
            size_t stride = tensor->fusion && tensor->fusion->kind != FUSION_INPUT
                ? divisor : tensor->strides[d];
            if (tensor->shape[d])
                smart_str_append_printf(source, "offset+=((i/%zuULL)%%%dULL)*%zuULL;\n",
                                        divisor, tensor->shape[d], stride);
            divisor *= tensor->shape[d];
        }
        smart_str_append_printf(source, "return static_cast<const %s*>(p%zu)[offset];}\n", type, parameter);
        return;
    }
    size_t a = fusion_find(plan, node->a);
    if (node->kind == FUSION_VIEW)
    {
        if (node->op == OP_RESHAPE)
            smart_str_appends(source, "size_t ia=i;\n");
        else
        {
            smart_str_appends(source, "size_t ia=0;\n");
            size_t divisor = 1;
            for (int d = tensor->ndims - 1; d >= 0; d--)
            {
                size_t stride = 1;
                for (int j = node->axes[d] + 1; j < node->a->ndims; j++) stride *= node->a->shape[j];
                if (tensor->shape[d])
                    smart_str_append_printf(source, "ia+=((i/%zuULL)%%%dULL)*%zuULL;\n",
                                            divisor, tensor->shape[d], stride);
                divisor *= tensor->shape[d];
            }
        }
    }
    else fusion_project(source, tensor, node->a, "ia");
    smart_str_append_printf(source, "%s a=", fusion_ctype(node->a->dtype));
    fusion_call_node(source, step, a, "ia");
    smart_str_appends(source, ";\n");
    if (node->b)
    {
        fusion_project(source, tensor, node->b, "ib");
        smart_str_append_printf(source, "%s b=", fusion_ctype(node->b->dtype));
        fusion_call_node(source, step, fusion_find(plan, node->b), "ib");
        smart_str_appends(source, ";\n");
    }
    if (node->c)
    {
        fusion_project(source, tensor, node->c, "ic");
        smart_str_append_printf(source, "%s c=", fusion_ctype(node->c->dtype));
        fusion_call_node(source, step, fusion_find(plan, node->c), "ic");
        smart_str_appends(source, ";\n");
    }
    smart_str_append_printf(source, "return static_cast<%s>(", type);
    if (node->kind == FUSION_UNARY) fusion_unary_source(source, node, tensor->dtype);
    else if (node->kind == FUSION_CAST || node->kind == FUSION_VIEW) smart_str_appends(source, "a");
    else if (node->kind == FUSION_WHERE) smart_str_appends(source, "a!=0?b:c");
    else
    {
        smart_str_append_printf(source, "static_cast<%s>(", type);
        if (node->kind == FUSION_SCALAR && node->parameter) fusion_scalar_source(source, node->scalar);
        else smart_str_appends(source, "a");
        smart_str_append_printf(source, ")%s static_cast<%s>(", fusion_operator(node->op), type);
        if (node->kind == FUSION_SCALAR && !node->parameter) fusion_scalar_source(source, node->scalar);
        else smart_str_appends(source, node->kind == FUSION_SCALAR ? "a" : "b");
        smart_str_appends(source, ")");
    }
    smart_str_appends(source, ");}\n");
}

int fusion_generate_source(fusion_plan *plan)
{
    smart_str source = {0};
    for (size_t i = 0; i < plan->step_count; i++)
    {
        fusion_step *step = &plan->steps[i];
        if (!step->name[0]) continue;
        unsigned char *seen = ecalloc(plan->count, 1);
        for (size_t j = 0; j < step->root_count; j++)
            fusion_gather(plan, step, step->roots[j], seen);
        step->arguments = emalloc((step->leaf_count + step->root_count) * sizeof(void *));
        memset(seen, 0, plan->count);
        for (size_t j = 0; j < step->root_count; j++)
            fusion_emit_node(plan, step, step->roots[j], &source, seen);
        efree(seen);
        tensor_t *root = plan->items[step->root].tensor;
        smart_str_append_printf(&source, "extern \"C\" __global__ void %s(", step->name);
        fusion_params(&source, step, 1);
        for (size_t j = 0; j < step->root_count; j++)
            smart_str_append_printf(&source, "%svoid* out%zu", j ? "," : "", j);
        smart_str_append_printf(&source,
            "){for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;"
            "i<%zuULL;i+=(size_t)blockDim.x*gridDim.x){", root->total_size);
        for (size_t j = 0; j < step->root_count; j++)
        {
            tensor_t *output = plan->items[step->roots[j]].tensor;
            smart_str_append_printf(&source, "static_cast<%s*>(out%zu)[i]=", fusion_ctype(output->dtype), j);
            fusion_call_node(&source, step, step->roots[j], "i");
            smart_str_appends(&source, ";");
        }
        smart_str_appends(&source, "}}\n");
    }
    smart_str_0(&source);
    plan->source = source.s ? source.s : ZSTR_EMPTY_ALLOC();
    return 1;
}
