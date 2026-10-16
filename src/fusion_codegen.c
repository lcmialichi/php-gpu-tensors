#include "fusion_internal.h"

static const char *fusion_ctype(dtype_t dtype)
{
    static const char *types[] = {"float", "double", "signed char", "short", "int", "long long",
        "unsigned char", "unsigned short", "unsigned int", "unsigned long long", "bool"};
    if (dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16) return "float";
    ZEND_ASSERT(dtype <= DTYPE_BOOL);
    return types[dtype];
}

static const char *fusion_storage_type(dtype_t dtype)
{
    if (dtype == DTYPE_FLOAT16) return "__half";
    if (dtype == DTYPE_BFLOAT16) return "__nv_bfloat16";
    return fusion_ctype(dtype);
}

static const char *fusion_low_precision_to_float(dtype_t dtype)
{
    return dtype == DTYPE_FLOAT16 ? "__half2float" : "__bfloat162float";
}

static const char *fusion_float_to_low_precision(dtype_t dtype)
{
    return dtype == DTYPE_FLOAT16 ? "__float2half_rn" : "__float2bfloat16_rn";
}

static void fusion_scalar_source(smart_str *source, scalar_value_t scalar);

static const char *fusion_operator(operation_type_t op)
{
    switch (op)
    {
        case OP_ADD: return "+";
        case OP_SUB: return "-";
        case OP_MUL: return "*";
        case OP_DIV: return "/";
        case OP_MAXIMUM: return ">";
        case OP_MINIMUM: return "<";
        case OP_GT: return ">";
        case OP_LT: return "<";
        case OP_EQ: return "==";
        case OP_NE: return "!=";
        case OP_GE: return ">=";
        case OP_LE: return "<=";
        default: ZEND_ASSERT(0); return NULL;
    }
}

static void fusion_power_helpers(smart_str *source)
{
    smart_str_appends(source,
        "template<typename T> __device__ __forceinline__ T fusion_integer_power(T base,int exponent){"
        "T result=(T)1;while(exponent>0){if(exponent&1)result*=base;base*=base;exponent>>=1;}return result;}\n"
        "__device__ __forceinline__ float fusion_power(float a,float b){"
        "if(b==floorf(b)){int exponent=static_cast<int>(b);"
        "if(exponent==0)return 1.0f;if(exponent==1)return a;"
        "if(exponent==2)return a*a;if(exponent==3)return a*a*a;"
        "if(exponent<0)return 1.0f/fusion_integer_power(a,-exponent);"
        "return fusion_integer_power(a,exponent);}return powf(a,b);}\n"
        "__device__ __forceinline__ double fusion_power(double a,double b){"
        "if(b==floor(b)){int exponent=static_cast<int>(b);"
        "if(exponent==0)return 1.0;if(exponent==1)return a;"
        "if(exponent==2)return a*a;if(exponent==3)return a*a*a;"
        "if(exponent<0)return 1.0/fusion_integer_power(a,-exponent);"
        "return fusion_integer_power(a,exponent);}return pow(a,b);}\n");
}

static void fusion_power_operand(smart_str *source, fusion_node *node,
                                 const char *type, int left)
{
    smart_str_append_printf(source, "static_cast<%s>(", type);
    if (node->kind == FUSION_SCALAR && ((left && node->parameter) ||
                                         (!left && !node->parameter)))
        fusion_scalar_source(source, node->scalar);
    else
        smart_str_appends(source, node->kind == FUSION_SCALAR
            ? "a" : (left ? "a" : "b"));
    smart_str_appends(source, ")");
}

static int fusion_plan_uses_inline_power(fusion_plan *plan)
{
    for (size_t i = 0; i < plan->count; i++)
    {
        tensor_t *tensor = plan->items[i].tensor;
        fusion_node *node = tensor->fusion;
        if (node && node->op == OP_POW && fusion_inline(tensor))
            return 1;
    }
    return 0;
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
            if (child->shape[axis] != 1 && parent->shape[d] != 0 && divisor)
                smart_str_append_printf(source, "%s+=((i/%zuULL)%%%dULL)*%zuULL;\n",
                                        name, divisor, parent->shape[d], stride);
            stride *= child->shape[axis];
        }
        divisor *= parent->shape[d];
    }
}

static void fusion_unary_source(smart_str *source, fusion_node *node, dtype_t dtype)
{
    int floating = dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT64 ||
        dtype == DTYPE_FLOAT16 || dtype == DTYPE_BFLOAT16;
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
        case OP_EXPM1: name = "expm1"; break;
        case OP_EXP2: name = "exp2"; break;
        case OP_LOG: name = "log"; break;
        case OP_LOG1P: name = "log1p"; break;
        case OP_LOG2: name = "log2"; break;
        case OP_LOG10: name = "log10"; break;
        case OP_SQRT: name = "sqrt"; break;
        case OP_SIN: name = "sin"; break;
        case OP_COS: name = "cos"; break;
        case OP_TAN: name = "tan"; break;
        case OP_ASIN: name = "asin"; break;
        case OP_ACOS: name = "acos"; break;
        case OP_ATAN: name = "atan"; break;
        case OP_SINH: name = "sinh"; break;
        case OP_COSH: name = "cosh"; break;
        case OP_TANH: name = "tanh"; break;
        case OP_FLOOR: name = "floor"; break;
        case OP_CEIL: name = "ceil"; break;
        case OP_ROUND: name = "round"; break;
        default: ZEND_ASSERT(0);
    }
    smart_str_append_printf(source, "%s%s(%s)", name, suffix, floating ? "a" : "static_cast<float>(a)");
}

static void fusion_transform_source(smart_str *source, fusion_node *node)
{
    const tensor_transform_spec_t *spec = &node->transform;
    int flattened = (spec->op == OP_REPEAT || spec->op == OP_ROLL) &&
        spec->axis < 0;
    size_t divisor = 1;
    if (!flattened)
    {
        for (int axis = spec->output_ndims - 1; axis >= 0; axis--)
        {
            smart_str_append_printf(source, "size_t o%d=(i/%zuULL)%%%dULL;\n",
                                    axis, divisor, spec->output_shape[axis]);
            divisor *= (size_t)spec->output_shape[axis];
        }
    }

    if (spec->op == OP_REPEAT && spec->axis < 0)
    {
        smart_str_append_printf(source, "size_t flat=i/%dULL;\n", spec->repeats);
        divisor = 1;
        for (int axis = spec->input_ndims - 1; axis >= 0; axis--)
        {
            smart_str_append_printf(source, "size_t c%d=(flat/%zuULL)%%%dULL;\n",
                                    axis, divisor, spec->input_shape[axis]);
            divisor *= (size_t)spec->input_shape[axis];
        }
    }
    else if (spec->op == OP_ROLL && spec->axis < 0)
    {
        size_t total = 1;
        for (int axis = 0; axis < spec->input_ndims; axis++)
            total *= (size_t)spec->input_shape[axis];
        smart_str_append_printf(source,
            "size_t flat=i>=%lldULL?i-%lldULL:%zuULL-(%lldULL-i);\n",
            (long long)spec->shifts[0], (long long)spec->shifts[0], total,
            (long long)spec->shifts[0]);
        divisor = 1;
        for (int axis = spec->input_ndims - 1; axis >= 0; axis--)
        {
            smart_str_append_printf(source, "size_t c%d=(flat/%zuULL)%%%dULL;\n",
                                    axis, divisor, spec->input_shape[axis]);
            divisor *= (size_t)spec->input_shape[axis];
        }
    }
    else
    {
        int diagonal_axis = spec->output_ndims - 1;
        int retained_axis = 0;
        int leading_axes = spec->output_ndims - spec->input_ndims;
        for (int axis = 0; axis < spec->input_ndims; axis++)
        {
            if (spec->op == OP_DIAGONAL)
            {
                if (axis == spec->axis1)
                    smart_str_append_printf(source, "size_t c%d=o%d+%dULL;\n",
                        axis, diagonal_axis,
                        spec->offset < 0 ? -spec->offset : 0);
                else if (axis == spec->axis2)
                    smart_str_append_printf(source, "size_t c%d=o%d+%dULL;\n",
                        axis, diagonal_axis,
                        spec->offset > 0 ? spec->offset : 0);
                else
                    smart_str_append_printf(source, "size_t c%d=o%d;\n",
                        axis, retained_axis++);
            }
            else
            {
                int output_axis = axis + leading_axes;
                if (spec->op == OP_REPEAT && axis == spec->axis)
                    smart_str_append_printf(source, "size_t c%d=o%d/%dULL;\n",
                                            axis, output_axis, spec->repeats);
                else if (spec->op == OP_TILE)
                    smart_str_append_printf(source, "size_t c%d=o%d%%%dULL;\n",
                                            axis, output_axis, spec->input_shape[axis]);
                else if (spec->op == OP_FLIP && spec->axes[axis])
                    smart_str_append_printf(source, "size_t c%d=%dULL-1ULL-o%d;\n",
                                            axis, spec->input_shape[axis], output_axis);
                else if (spec->op == OP_ROLL && spec->shifts[axis])
                    smart_str_append_printf(source,
                        "size_t c%d=o%d>=%lldULL?o%d-%lldULL:%dULL-(%lldULL-o%d);\n",
                        axis, output_axis, (long long)spec->shifts[axis],
                        output_axis, (long long)spec->shifts[axis],
                        spec->input_shape[axis], (long long)spec->shifts[axis],
                        output_axis);
                else
                    smart_str_append_printf(source, "size_t c%d=o%d;\n",
                                            axis, output_axis);
            }
        }
    }

    smart_str_appends(source, "size_t ia=0;\n");
    divisor = 1;
    for (int axis = spec->input_ndims - 1; axis >= 0; axis--)
    {
        smart_str_append_printf(source, "ia+=c%d*%zuULL;\n", axis, divisor);
        divisor *= (size_t)spec->input_shape[axis];
    }
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
            if (tensor->shape[d] && divisor)
                smart_str_append_printf(source, "offset+=((i/%zuULL)%%%dULL)*%zuULL;\n",
                                        divisor, tensor->shape[d], stride);
            divisor *= tensor->shape[d];
        }
        if (tensor->dtype == DTYPE_FLOAT16 || tensor->dtype == DTYPE_BFLOAT16)
            smart_str_append_printf(source, "return %s(static_cast<const %s*>(p%zu)[offset]);}\n",
                                    fusion_low_precision_to_float(tensor->dtype),
                                    fusion_storage_type(tensor->dtype), parameter);
        else
            smart_str_append_printf(source, "return static_cast<const %s*>(p%zu)[offset];}\n",
                                    type, parameter);
        return;
    }
    if (node->kind == FUSION_SLICE_SCATTER)
    {
        size_t updates = fusion_find(plan, node->a);
        if (!node->a->total_size)
        {
            smart_str_append_printf(source, "return static_cast<%s>(0);}\n", type);
            return;
        }
        smart_str_appends(source, "size_t ia=0;bool selected=true;\n");
        size_t source_divisor = 1;
        for (int d = tensor->ndims - 1; d >= 0; d--)
        {
            int axis = -1;
            for (int j = 0; j < node->a->ndims; j++)
                if (node->axes[j] == d) axis = j;
            smart_str_append_printf(source,
                "size_t sc%d=(i/%zuULL)%%%dULL;\n",
                d, source_divisor, tensor->shape[d]);
            if (axis >= 0)
            {
                int start = node->slice_starts[d];
                int step_size = node->slice_steps[d];
                size_t end = (size_t)start +
                    (size_t)node->a->shape[axis] * (size_t)step_size;
                size_t updates_divisor = 1;
                for (int j = node->a->ndims - 1; j > axis; j--)
                    updates_divisor *= (size_t)node->a->shape[j];
                if (start > 0)
                    smart_str_append_printf(source, "if(sc%d<%dULL||", d, start);
                else
                    smart_str_appends(source, "if(");
                smart_str_append_printf(source,
                    "sc%d>=%zuULL||((sc%d-%dULL)%%%dULL)!=0)selected=false;\n"
                    "ia+=((sc%d-%dULL)/%dULL)*%zuULL;\n",
                    d, end, d, start, step_size,
                    d, start, step_size, updates_divisor);
            }
            else
                smart_str_append_printf(source, "if(sc%d!=%dULL)selected=false;\n",
                                        d, node->slice_starts[d]);
            source_divisor *= (size_t)tensor->shape[d];
        }
        smart_str_appends(source, "if(!selected)return static_cast<");
        smart_str_appends(source, type);
        smart_str_appends(source, ">(0);\nreturn static_cast<");
        smart_str_appends(source, type);
        smart_str_appends(source, ">(");
        fusion_call_node(source, step, updates, "ia");
        smart_str_appends(source, ");}\n");
        return;
    }
    if (node->kind == FUSION_TRANSFORM)
    {
        fusion_transform_source(source, node);
        smart_str_appends(source, "return ");
        fusion_call_node(source, step, fusion_find(plan, node->a), "ia");
        smart_str_appends(source, ";}\n");
        return;
    }
    size_t a = fusion_find(plan, node->a);
    if (node->kind == FUSION_VIEW)
    {
        if (node->op == OP_RESHAPE)
            smart_str_appends(source, "size_t ia=i;\n");
        else if (node->op == OP_SLICE)
        {
            smart_str_append_printf(source, "size_t ia=%zuULL;\n", node->slice_offset);
            size_t divisor = 1;
            for (int d = tensor->ndims - 1; d >= 0; d--)
            {
                if (tensor->shape[d] && divisor)
                    smart_str_append_printf(source, "ia+=((i/%zuULL)%%%dULL)*%zuULL;\n",
                                            divisor, tensor->shape[d], node->slice_strides[d]);
                divisor *= tensor->shape[d];
            }
        }
        else if (node->op == OP_BROADCAST)
            fusion_project(source, tensor, node->a, "ia");
        else
        {
            smart_str_appends(source, "size_t ia=0;\n");
            size_t divisor = 1;
            for (int d = tensor->ndims - 1; d >= 0; d--)
            {
                size_t stride = 1;
                for (int j = node->axes[d] + 1; j < node->a->ndims; j++) stride *= node->a->shape[j];
                if (tensor->shape[d] && divisor)
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
    int low_precision_output = tensor->dtype == DTYPE_FLOAT16 || tensor->dtype == DTYPE_BFLOAT16;
    if (low_precision_output)
        smart_str_append_printf(source, "return %s(%s(static_cast<float>(",
                                fusion_low_precision_to_float(tensor->dtype),
                                fusion_float_to_low_precision(tensor->dtype));
    else
        smart_str_append_printf(source, "return static_cast<%s>(", type);
    if (node->kind == FUSION_UNARY) fusion_unary_source(source, node, tensor->dtype);
    else if (node->kind == FUSION_CAST || node->kind == FUSION_VIEW) smart_str_appends(source, "a");
    else if (node->kind == FUSION_WHERE) smart_str_appends(source, "a!=0?b:c");
    else if (node->op == OP_MAXIMUM || node->op == OP_MINIMUM)
    {
        const char *comparison = node->op == OP_MAXIMUM ? ">" : "<";
        for (int part = 0; part < 4; part++)
        {
            int left_operand = part == 0 || part == 2;
            int operand_is_scalar = node->kind == FUSION_SCALAR;
            smart_str_append_printf(source, "static_cast<%s>(", type);
            if (operand_is_scalar && left_operand == node->parameter)
                fusion_scalar_source(source, node->scalar);
            else
                smart_str_appends(source, operand_is_scalar ? "a" : (left_operand ? "a" : "b"));
            smart_str_appends(source, ")");
            if (part == 0) smart_str_appends(source, comparison);
            else if (part == 1) smart_str_appends(source, "?");
            else if (part == 2) smart_str_appends(source, ":");
        }
    }
    else if (node->op == OP_POW)
    {
        smart_str_appends(source, "fusion_power(");
        fusion_power_operand(source, node, type, 1);
        smart_str_appends(source, ",");
        fusion_power_operand(source, node, type, 0);
        smart_str_appends(source, ")");
    }
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
    smart_str_appends(source, low_precision_output ? ")));}\n" : ");}\n");
}

int fusion_generate_source(fusion_plan *plan)
{
    smart_str source = {0};
    smart_str_appends(&source,
        "#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n");
    if (fusion_plan_uses_inline_power(plan))
        fusion_power_helpers(&source);
    for (size_t i = 0; i < plan->step_count; i++)
    {
        fusion_step *step = &plan->steps[i];
        if (!step->name[0]) continue;
        unsigned char *seen = ecalloc(plan->count, 1);
        for (size_t j = 0; j < step->root_count; j++)
            fusion_gather(plan, step, step->roots[j], seen);
        step->arguments = emalloc((step->leaf_count + step->root_count) * sizeof(void *));
        memset(seen, 0, plan->count);
        tensor_t *root = plan->items[step->root].tensor;
        if (root->total_size)
            for (size_t j = 0; j < step->root_count; j++)
                fusion_emit_node(plan, step, step->roots[j], &source, seen);
        efree(seen);
        smart_str_append_printf(&source, "extern \"C\" __global__ void %s(", step->name);
        fusion_params(&source, step, 1);
        for (size_t j = 0; j < step->root_count; j++)
            smart_str_append_printf(&source, "%svoid* out%zu", j ? "," : "", j);
        if (!root->total_size)
        {
            smart_str_appends(&source, "){}\n");
            continue;
        }
        smart_str_append_printf(&source,
            "){for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;"
            "i<%zuULL;i+=(size_t)blockDim.x*gridDim.x){", root->total_size);
        for (size_t j = 0; j < step->root_count; j++)
        {
            tensor_t *output = plan->items[step->roots[j]].tensor;
            if (output->dtype == DTYPE_FLOAT16 || output->dtype == DTYPE_BFLOAT16)
                smart_str_append_printf(&source, "static_cast<%s*>(out%zu)[i]=%s(",
                                        fusion_storage_type(output->dtype), j,
                                        fusion_float_to_low_precision(output->dtype));
            else
                smart_str_append_printf(&source, "static_cast<%s*>(out%zu)[i]=",
                                        fusion_storage_type(output->dtype), j);
            fusion_call_node(&source, step, step->roots[j], "i");
            smart_str_appends(&source,
                output->dtype == DTYPE_FLOAT16 || output->dtype == DTYPE_BFLOAT16 ? ");" : ";");
        }
        smart_str_appends(&source, "}}\n");
    }
    smart_str_0(&source);
    plan->source = source.s ? source.s : ZSTR_EMPTY_ALLOC();
    return 1;
}
