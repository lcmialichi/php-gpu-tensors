#include "fusion_internal.h"
#include "reduction_ops.h"
int fusion_inline(tensor_t *tensor)
{
    fusion_node *node = tensor->fusion;
    if (!node) return 0;
    if (node->kind == FUSION_UNARY || node->kind == FUSION_CAST ||
        node->kind == FUSION_WHERE || node->kind == FUSION_VIEW) return 1;
    return (node->kind == FUSION_BINARY || node->kind == FUSION_SCALAR) &&
        (node->op == OP_ADD || node->op == OP_SUB || node->op == OP_MUL ||
         node->op == OP_DIV || node->op == OP_MAXIMUM || node->op == OP_MINIMUM ||
         (node->op >= OP_GT && node->op <= OP_LE));
}

size_t fusion_find(fusion_plan *plan, tensor_t *tensor)
{
    for (size_t i = 0; i < plan->count; i++)
        if (plan->items[i].tensor == tensor) return i;
    return SIZE_MAX;
}

int fusion_preallocated(fusion_plan *plan, fusion_step *step)
{
    fusion_kind kind = plan->items[step->root].tensor->fusion->kind;
    return step->name[0] || kind == FUSION_MATMUL || kind == FUSION_REDUCE || kind == FUSION_ARG_REDUCE;
}

static size_t fusion_collect(fusion_plan *plan, tensor_t *tensor)
{
    size_t id = fusion_find(plan, tensor);
    if (id != SIZE_MAX) return id;
    fusion_node *node = tensor->fusion;
    size_t a = SIZE_MAX, b = SIZE_MAX, c = SIZE_MAX;
    if (node && node->a) a = fusion_collect(plan, node->a);
    if (node && node->b) b = fusion_collect(plan, node->b);
    if (node && node->c) c = fusion_collect(plan, node->c);
    id = plan->count++;
    plan->items = erealloc(plan->items, plan->count * sizeof(fusion_item));
    plan->items[id] = (fusion_item){ .tensor = tensor, .a = a, .b = b, .c = c, .slot = id };
    if (node && node->kind != FUSION_INPUT)
    {
        if (fusion_inline(tensor))
        {
            int cost = node->kind == FUSION_UNARY ? 4 :
                       node->kind == FUSION_WHERE ? 3 :
                       node->kind == FUSION_VIEW ? 2 : 1;
            if (tensor->dtype == DTYPE_FLOAT64) cost *= 2;
            if (a != SIZE_MAX && !plan->items[a].cut) cost += plan->items[a].cost;
            if (b != SIZE_MAX && !plan->items[b].cut) cost += plan->items[b].cost;
            if (c != SIZE_MAX && !plan->items[c].cut) cost += plan->items[c].cost;
            if (cost > FUSION_KERNEL_BUDGET)
            {
                size_t children[] = {a, b, c};
                for (int i = 0; i < 3; i++)
                    if (children[i] != SIZE_MAX && plan->items[children[i]].cost)
                    {
                        plan->items[children[i]].cut = 1;
                        plan->items[children[i]].reason = "expression-cost";
                    }
                cost = node->kind == FUSION_UNARY ? 4 : 1;
            }
            plan->items[id].cost = cost;
        }
        else
        {
            plan->items[id].cut = 1;
            plan->items[id].reason = node->kind == FUSION_MATMUL ? "native-matmul" :
                node->kind == FUSION_REDUCE || node->kind == FUSION_ARG_REDUCE ? "native-reduction" :
                "native-power";
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
    if (node && node->c) fusion_dependencies(plan, fusion_find(plan, node->c), root);
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
    step->roots = emalloc(sizeof(size_t));
    step->roots[0] = id;
    step->root_count = 1;
    if (fusion_inline(item->tensor) && !item->alias)
    {
        snprintf(step->name, sizeof(step->name), "fusion_%zu", plan->kernel_count++);
    }
}

void fusion_plan_free(fusion_plan *plan)
{
    if (!plan) return;
    CUcontext current = NULL, popped;
    CUresult context_error = cuCtxGetCurrent(&current);
    int pushed = context_error == CUDA_SUCCESS && plan->context && current != plan->context;
    if (pushed)
    {
        context_error = cuCtxPushCurrent(plan->context);
        if (context_error != CUDA_SUCCESS) pushed = 0;
    }
    fusion_cleanup_error("Selecting module context", context_error);
    if (plan->graph_exec) fusion_cleanup_error("Destroying graph executable", cuGraphExecDestroy(plan->graph_exec));
    if (plan->cuda_graph) fusion_cleanup_error("Destroying CUDA graph", cuGraphDestroy(plan->cuda_graph));
    if (plan->graph_nodes) efree(plan->graph_nodes);
    if (plan->graph_capture_stream)
    {
        cudaError_t error = cuda_reduction_release_stream((cudaStream_t)plan->graph_capture_stream);
        if (error != cudaSuccess)
            php_error_docref(NULL, E_WARNING, "Releasing graph reduction workspace failed: %s", cudaGetErrorString(error));
        fusion_cleanup_error("Destroying graph capture stream", cuStreamDestroy(plan->graph_capture_stream));
    }
    if (plan->sync_stream) fusion_cleanup_error("Destroying replay stream", cuStreamDestroy(plan->sync_stream));
    if (plan->sync_ready) fusion_cleanup_error("Destroying replay readiness event", cuEventDestroy(plan->sync_ready));
    if (plan->module)
    {
        CUresult error = cuModuleUnload(plan->module);
        if (error != CUDA_SUCCESS && !EG(exception))
            php_error_docref(NULL, E_WARNING, "Failed to unload fusion module (CUDA error %d)", error);
    }
    for (size_t i = 0; i < plan->step_count; i++)
    {
        if (plan->steps[i].leaves) efree(plan->steps[i].leaves);
        if (plan->steps[i].arguments) efree(plan->steps[i].arguments);
        efree(plan->steps[i].roots);
    }
    if (plan->workspace)
    {
        for (size_t i = 0; i < plan->count; i++)
            cuda_tensor_destroy(plan->workspace[i]);
        efree(plan->workspace);
    }
    if (plan->steps) efree(plan->steps);
    if (plan->items) efree(plan->items);
    if (plan->inputs) efree(plan->inputs);
    if (plan->input_examples) efree(plan->input_examples);
    if (plan->source) zend_string_release(plan->source);
    if (pushed) fusion_cleanup_error("Restoring module context", cuCtxPopCurrent(&popped));
    efree(plan);
}

static int fusion_depends(fusion_plan *plan, size_t id, size_t other, unsigned char *seen)
{
    if (id == other) return 1;
    if (seen[id]) return 0;
    seen[id] = 1;
    fusion_node *node = plan->items[id].tensor->fusion;
    if (!node) return 0;
    tensor_t *children[] = {node->a, node->b, node->c};
    for (int i = 0; i < 3; i++)
        if (children[i] && fusion_depends(plan, fusion_find(plan, children[i]), other, seen))
            return 1;
    return 0;
}

static void fusion_group_outputs(fusion_plan *plan)
{
    for (size_t i = 0; i + 1 < plan->step_count;)
    {
        fusion_step *first = &plan->steps[i], *next = &plan->steps[i + 1];
        int compatible = first->name[0] && next->name[0] && plan->items[next->root].output;
        tensor_t *shape = plan->items[next->root].tensor;
        unsigned char *seen = ecalloc(plan->count, 1);
        for (size_t j = 0; compatible && j < first->root_count; j++)
        {
            size_t root = first->roots[j];
            tensor_t *tensor = plan->items[root].tensor;
            compatible = plan->items[root].output && tensor->ndims == shape->ndims &&
                !memcmp(tensor->shape, shape->shape, shape->ndims * sizeof(int));
            if (compatible)
            {
                memset(seen, 0, plan->count);
                compatible = !fusion_depends(plan, root, next->root, seen);
                memset(seen, 0, plan->count);
                compatible = compatible && !fusion_depends(plan, next->root, root, seen);
            }
        }
        efree(seen);
        if (!compatible || first->root_count >= 4) { i++; continue; }
        first->roots = erealloc(first->roots, (first->root_count + 1) * sizeof(size_t));
        first->roots[first->root_count++] = next->root;
        efree(next->roots);
        memmove(next, next + 1, (plan->step_count - i - 2) * sizeof(fusion_step));
        plan->step_count--;
    }
    plan->kernel_count = 0;
    for (size_t i = 0; i < plan->step_count; i++)
        if (plan->steps[i].name[0])
            snprintf(plan->steps[i].name, sizeof(plan->steps[i].name), "fusion_%zu", plan->kernel_count++);
}

fusion_plan *fusion_build_plan(tensor_t **roots, size_t root_count)
{
    fusion_plan *plan = ecalloc(1, sizeof(fusion_plan));
    for (size_t i = 0; i < root_count; i++)
    {
        size_t id = fusion_collect(plan, roots[i]);
        if (roots[i]->fusion && roots[i]->fusion->kind != FUSION_INPUT)
            plan->items[id].cut = 1;
        plan->items[id].output = 1;
        if (!plan->items[id].reason) plan->items[id].reason = "output";
    }
    for (size_t i = 0; i < plan->count; i++)
    {
        fusion_node *node = plan->items[i].tensor->fusion;
        if (!node) continue;
        tensor_t *children[] = {node->a, node->b, node->c};
        for (int j = 0; j < 3; j++)
            if (children[j]) plan->items[fusion_find(plan, children[j])].consumers++;
    }
    for (size_t i = 0; i < plan->count; i++)
        if (plan->items[i].consumers > 1 && plan->items[i].cost >= 8 && !plan->items[i].cut)
        {
            plan->items[i].cut = 1;
            plan->items[i].reason = "shared-expensive-expression";
        }
    for (size_t i = 0; i < plan->count; i++)
    {
        fusion_item *item = &plan->items[i];
        fusion_node *node = item->tensor->fusion;
        if (!node || node->kind != FUSION_VIEW || !item->cut || item->output) continue;
        tensor_t *parent = plan->items[item->a].tensor;
        if (parent->fusion && parent->fusion->kind != FUSION_INPUT && !plan->items[item->a].cut) continue;
        int compatible = 1;
        for (size_t j = 0; j < plan->count; j++)
        {
            fusion_item *consumer = &plan->items[j];
            if (consumer->a != i && consumer->b != i && consumer->c != i) continue;
            fusion_node *operation = consumer->tensor->fusion;
            if (!operation || (operation->kind != FUSION_MATMUL &&
                operation->kind != FUSION_REDUCE && operation->kind != FUSION_ARG_REDUCE))
                compatible = 0;
            if (operation && (operation->kind == FUSION_REDUCE || operation->kind == FUSION_ARG_REDUCE) &&
                operation->parameter == -1 && !is_contiguous(item->tensor))
                compatible = 0;
        }
        if (compatible) { item->alias = 1; item->reason = "native-layout-alias"; }
    }
    for (size_t i = 0; i < root_count; i++)
        fusion_schedule(plan, fusion_find(plan, roots[i]));
    fusion_group_outputs(plan);

    plan->stream_compatible = 1;
    plan->graph_compatible = 1;
    for (size_t i = 0; i < plan->step_count; i++)
        if (!plan->steps[i].name[0] && !plan->items[plan->steps[i].root].alias)
        {
            plan->native_steps++;
            fusion_kind kind = plan->items[plan->steps[i].root].tensor->fusion->kind;
            if (kind != FUSION_MATMUL && kind != FUSION_REDUCE && kind != FUSION_ARG_REDUCE)
                plan->graph_compatible = 0;
            if (!fusion_preallocated(plan, &plan->steps[i]))
            {
                plan->stream_compatible = 0;
                plan->incompatibility = plan->items[plan->steps[i].root].reason;
            }
        }
    fusion_generate_source(plan);
    for (size_t i = 0; i < plan->step_count; i++)
    {
        fusion_step *step = &plan->steps[i];
        for (size_t j = 0; j < step->leaf_count; j++)
            plan->items[step->leaves[j]].last_use = i;
        if (!step->name[0])
        {
            fusion_item *item = &plan->items[step->root];
            if (item->a != SIZE_MAX) plan->items[item->a].last_use = i;
            if (item->b != SIZE_MAX) plan->items[item->b].last_use = i;
            if (item->c != SIZE_MAX) plan->items[item->c].last_use = i;
        }
    }
    for (size_t i = plan->count; i > 0; i--)
    {
        fusion_item *item = &plan->items[i - 1];
        if (item->alias && plan->items[item->a].last_use < item->last_use)
            plan->items[item->a].last_use = item->last_use;
    }
    for (size_t i = 0; i < plan->step_count; i++)
    {
        fusion_step *step = &plan->steps[i];
        fusion_item *item = &plan->items[step->root];
        if (item->output || step->root_count != 1 || !fusion_preallocated(plan, step)) continue;
        for (size_t j = 0; j < i; j++)
        {
            fusion_item *previous = &plan->items[plan->steps[j].root];
            if (previous->output || plan->steps[j].root_count != 1 || !fusion_preallocated(plan, &plan->steps[j]) ||
                previous->last_use >= i ||
                previous->tensor->dtype != item->tensor->dtype ||
                previous->tensor->ndims != item->tensor->ndims ||
                memcmp(previous->tensor->shape, item->tensor->shape, item->tensor->ndims * sizeof(int)))
                continue;
            int in_use = 0;
            for (size_t k = 0; k < i; k++)
            {
                fusion_item *used = &plan->items[plan->steps[k].root];
                if (used->slot == previous->slot && used->last_use >= i) in_use = 1;
            }
            if (!in_use) { item->slot = previous->slot; break; }
        }
    }
    const char **names = plan->kernel_count ? emalloc(plan->kernel_count * sizeof(char *)) : NULL;
    size_t name_count = 0;
    for (size_t i = 0; i < plan->step_count; i++)
        if (plan->steps[i].name[0]) names[name_count++] = plan->steps[i].name;
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
        int ok = fusion_load_module(plan, names);
        efree(names);
        if (!ok)
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
