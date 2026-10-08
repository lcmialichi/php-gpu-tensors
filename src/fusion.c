#include "fusion_internal.h"
#include "autograd.h"
#include "cuda_array_ce.h"
#include "reduction_ops.h"
#include "zend_interfaces.h"
#include "zend_fibers.h"
#include <time.h>

typedef struct
{
    fusion_scope *scope;
    fusion_plan *plan;
    zval outputs;
    zend_object std;
} fusion_graph;

static zend_class_entry *fusion_ce;
static zend_class_entry *fusion_graph_ce;
static zend_class_entry *fusion_execution_ce;
static zend_object_handlers fusion_graph_handlers;
static zend_object_handlers fusion_execution_handlers;

typedef struct
{
    zval graph;
    zval result;
    tensor_t **values;
    CUstream stream;
    CUevent event;
    int finished;
    CUresult error;
    tensor_t **retained;
    size_t retained_count;
    zend_object std;
} fusion_execution;

#define Z_FUSION_EXECUTION_P(zv) ((fusion_execution *)((char *)Z_OBJ_P(zv) - XtOffsetOf(fusion_execution, std)))

#define Z_FUSION_GRAPH_P(zv) ((fusion_graph *)((char *)Z_OBJ_P(zv) - XtOffsetOf(fusion_graph, std)))

static uint64_t fusion_time(void)
{
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0)
    {
        CUDA_THROW_RUNTIME("Cannot read the fusion profiling clock");
        return 0;
    }
    return (uint64_t)timestamp.tv_sec * UINT64_C(1000000000) + (uint64_t)timestamp.tv_nsec;
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
            struct autograd_node *grad_fn = target->grad_fn;
            tensor_t *gradient = target->grad;
            int requires_grad = target->requires_grad;
            fusion_release_node(target);
            if (target->shape) efree(target->shape);
            if (target->strides) efree(target->strides);
            *target = *result;
            target->ref_count = references;
            target->grad_fn = grad_fn;
            target->grad = gradient;
            target->requires_grad = requires_grad;
            efree(result);
            values[i] = target;
        }
    }
    fusion_values_free(plan, values);
    fusion_plan_free(plan);
    CUDA_G(fusion_scope) = scope;
    return ok && !EG(exception);
}

tensor_t *fusion_cast_eager(tensor_t *a, dtype_t dtype)
{
    if (!fusion_materialize(a)) return NULL;
    fusion_scope *scope = ecalloc(1, sizeof(fusion_scope));
    CUDA_G(fusion_scope) = scope;
    tensor_t *result = fusion_cast(a, dtype);
    int ok = result && fusion_materialize(result);
    CUDA_G(fusion_scope) = NULL;
    fusion_scope_free(scope, !ok);
    if (!ok && result) cuda_tensor_destroy(result);
    return ok ? result : NULL;
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
                if (node && node->c == tensor) internal++;
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
    zend_bool cuda_graph = 0;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_FUNC(fci, fcc)
        Z_PARAM_ARRAY_HT(inputs)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(cuda_graph)
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
        placeholder->requires_grad = example->requires_grad;
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
        plan->use_cuda_graph = cuda_graph && plan->graph_compatible;
        plan->cuda_graph_requested = cuda_graph;
        plan->compiled = 1;
        plan->input_count = count;
        plan->inputs = count ? emalloc(count * sizeof(size_t)) : NULL;
        plan->input_examples = count ? emalloc(count * sizeof(tensor_t *)) : NULL;
        for (size_t i = 0; i < count; i++)
        {
            plan->inputs[i] = fusion_find(plan, input_tensors[i]);
            plan->input_examples[i] = input_tensors[i];
        }
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

static tensor_t **fusion_bind(fusion_graph *graph, zval *inputs, int count)
{
    fusion_plan *plan = graph->plan;
    if (!plan || fusion_active())
    {
        CUDA_THROW_RUNTIME("FusionGraph cannot execute inside a capture or without a compiled plan");
        return NULL;
    }
    if ((size_t)count != plan->input_count)
    {
        CUDA_THROW_INVALID("FusionGraph expects %zu inputs, got %d", plan->input_count, count);
        return NULL;
    }
    if (plan->use_cuda_graph && plan->pending)
    {
        CUDA_THROW_RUNTIME("A CUDA Graph replay is already pending; wait before reusing this graph");
        return NULL;
    }
    if (!fusion_context_check(plan)) return NULL;
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
        tensor_t *example = plan->input_examples[i];
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
    if (!ok)
    {
        for (size_t i = 0; i < plan->input_count; i++)
            if (plan->inputs[i] != SIZE_MAX) values[plan->inputs[i]] = NULL;
        fusion_values_release(plan, values);
        return NULL;
    }
    return values;
}

static void fusion_unbind(fusion_plan *plan, tensor_t **values)
{
    for (size_t i = 0; i < plan->input_count; i++)
        if (plan->inputs[i] != SIZE_MAX) values[plan->inputs[i]] = NULL;
    fusion_values_release(plan, values);
}

static tensor_t *fusion_storage(tensor_t *tensor)
{
    while (tensor->base_tensor) tensor = tensor->base_tensor;
    return tensor;
}

static int fusion_execution_finish(fusion_execution *execution)
{
    fusion_graph *graph = Z_FUSION_GRAPH_P(&execution->graph);
    fusion_plan *plan = graph->plan;
    if (execution->finished)
    {
        if (execution->error != CUDA_SUCCESS)
            CUDA_THROW_RUNTIME("Asynchronous fusion failed (CUDA error %d)", execution->error);
        return execution->error == CUDA_SUCCESS;
    }
    if (!fusion_context_check(plan)) return 0;
    CUresult synchronization = execution->stream ? cuStreamSynchronize(execution->stream) : CUDA_SUCCESS;
    if (execution->stream) plan->synchronizations++;
    if (execution->error == CUDA_SUCCESS) execution->error = synchronization;
    execution->finished = 1;
    if (execution->error == CUDA_SUCCESS)
    {
        fusion_copy_outputs(&execution->result, &graph->outputs, plan, execution->values);
        plan->executions++;
    }
    else if (!EG(exception)) CUDA_THROW_RUNTIME("Asynchronous fusion failed (CUDA error %d)", execution->error);
    fusion_unbind(plan, execution->values);
    execution->values = NULL;
    for (size_t i = 0; i < execution->retained_count; i++)
    {
        fusion_storage(execution->retained[i])->fusion_readers--;
        cuda_tensor_destroy(execution->retained[i]);
    }
    execution->retained_count = 0;
    plan->pending--;
    CUDA_G(fusion_pending)--;
    return execution->error == CUDA_SUCCESS;
}

static int fusion_start(zval *graph_value, tensor_t **values, zval *result)
{
    fusion_graph *graph = Z_FUSION_GRAPH_P(graph_value);
    fusion_plan *plan = graph->plan;
    if (!plan->stream_compatible)
    {
        fusion_unbind(plan, values);
        CUDA_THROW_RUNTIME("runAsync requires a stream-compatible plan: %s", plan->incompatibility);
        return 0;
    }
    object_init_ex(result, fusion_execution_ce);
    fusion_execution *execution = Z_FUSION_EXECUTION_P(result);
    ZVAL_COPY(&execution->graph, graph_value);
    execution->values = values;
    execution->retained = ecalloc(plan->count, sizeof(tensor_t *));
    for (size_t i = 0; i < plan->count; i++)
    {
        if (!values[i]) continue;
        tensor_t *tensor = values[i];
        tensor->ref_count++;
        fusion_storage(tensor)->fusion_readers++;
        execution->retained[execution->retained_count++] = tensor;
    }
    CUresult error = cuStreamCreate(&execution->stream, CU_STREAM_NON_BLOCKING);
    if (error == CUDA_SUCCESS) error = cuEventCreate(&execution->event, CU_EVENT_DISABLE_TIMING);
    plan->pending++;
    CUDA_G(fusion_pending)++;
    int ok = error == CUDA_SUCCESS && fusion_enqueue(plan, values, execution->stream, FUSION_BUFFERS_LOCAL);
    if (ok) error = cuEventRecord(execution->event, execution->stream);
    if (!ok || error != CUDA_SUCCESS)
    {
        execution->error = error != CUDA_SUCCESS ? error : CUDA_ERROR_LAUNCH_FAILED;
        if (!EG(exception)) CUDA_THROW_RUNTIME("Cannot submit asynchronous fusion (CUDA error %d)", error);
        return 0;
    }
    return 1;
}

ZEND_METHOD(FusionGraph, run)
{
    zval *inputs;
    int count;
    ZEND_PARSE_PARAMETERS_START(0, -1)
        Z_PARAM_VARIADIC('*', inputs, count)
    ZEND_PARSE_PARAMETERS_END();
    fusion_graph *graph = Z_FUSION_GRAPH_P(ZEND_THIS);
    fusion_plan *plan = graph->plan;
    uint64_t start = plan && plan->profiling ? fusion_time() : 0;
    if (EG(exception)) RETURN_THROWS();
    tensor_t **values = fusion_bind(graph, inputs, count);
    if (!values) RETURN_THROWS();
    uint64_t bound = start ? fusion_time() : 0;
    if (EG(exception))
    {
        fusion_unbind(plan, values);
        RETURN_THROWS();
    }
    uint64_t executed = 0;
    int ok = fusion_execute(graph->plan, values);
    if (start) executed = fusion_time();
    if (ok) fusion_copy_outputs(return_value, &graph->outputs, graph->plan, values);
    if (!ok)
    {
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess && !EG(exception))
            CUDA_THROW_RUNTIME("Fusion cleanup failed: %s", cudaGetErrorString(error));
    }
    fusion_unbind(graph->plan, values);
    if (!ok || EG(exception)) RETURN_THROWS();
    if (start)
    {
        uint64_t collected = fusion_time();
        if (EG(exception)) RETURN_THROWS();
        plan->bind_ns += bound - start;
        plan->execute_ns += executed - bound;
        plan->collect_ns += collected - executed;
        plan->profiled_executions++;
    }
}

ZEND_METHOD(FusionGraph, runAsync)
{
    zval *inputs;
    int count;
    ZEND_PARSE_PARAMETERS_START(0, -1)
        Z_PARAM_VARIADIC('*', inputs, count)
    ZEND_PARSE_PARAMETERS_END();
    fusion_graph *graph = Z_FUSION_GRAPH_P(ZEND_THIS);
    tensor_t **values = fusion_bind(graph, inputs, count);
    if (!values) RETURN_THROWS();
    if (!fusion_start(ZEND_THIS, values, return_value)) RETURN_THROWS();
}

ZEND_METHOD(FusionExecution, __construct) {}

ZEND_METHOD(FusionExecution, wait)
{
    ZEND_PARSE_PARAMETERS_NONE();
    fusion_execution *execution = Z_FUSION_EXECUTION_P(ZEND_THIS);
    if (Z_ISUNDEF(execution->graph))
    {
        CUDA_THROW_RUNTIME("FusionExecution has no submitted execution");
        RETURN_THROWS();
    }
    if (!fusion_execution_finish(execution)) RETURN_THROWS();
    RETURN_COPY(&execution->result);
}

ZEND_METHOD(FusionExecution, isFinished)
{
    ZEND_PARSE_PARAMETERS_NONE();
    fusion_execution *execution = Z_FUSION_EXECUTION_P(ZEND_THIS);
    if (Z_ISUNDEF(execution->graph))
    {
        CUDA_THROW_RUNTIME("FusionExecution has no submitted execution");
        RETURN_THROWS();
    }
    if (execution->finished) RETURN_TRUE;
    if (!fusion_context_check(Z_FUSION_GRAPH_P(&execution->graph)->plan)) RETURN_THROWS();
    CUresult error = cuEventQuery(execution->event);
    if (error == CUDA_ERROR_NOT_READY) RETURN_FALSE;
    if (error != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME("Cannot query fusion completion (CUDA error %d)", error);
        RETURN_THROWS();
    }
    RETURN_TRUE;
}

ZEND_METHOD(Fusion, getCacheStats)
{
    ZEND_PARSE_PARAMETERS_NONE();
    fusion_cache_stats(return_value);
}

ZEND_METHOD(Fusion, clearCache)
{
    ZEND_PARSE_PARAMETERS_NONE();
    fusion_cache_shutdown();
}

ZEND_METHOD(FusionGraph, setProfiling)
{
    zend_bool enabled;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_BOOL(enabled)
    ZEND_PARSE_PARAMETERS_END();
    fusion_plan *plan = Z_FUSION_GRAPH_P(ZEND_THIS)->plan;
    if (!plan || plan->pending)
    {
        CUDA_THROW_RUNTIME("Profiling requires an initialized graph without pending executions");
        RETURN_THROWS();
    }
    plan->profiling = enabled;
    plan->profiled_executions = 0;
    plan->bind_ns = plan->execute_ns = plan->collect_ns = 0;
}

ZEND_METHOD(FusionGraph, getStats)
{
    ZEND_PARSE_PARAMETERS_NONE();
    fusion_plan *plan = Z_FUSION_GRAPH_P(ZEND_THIS)->plan;
    array_init(return_value);
    add_assoc_long(return_value, "nodes", plan ? plan->count : 0);
    add_assoc_long(return_value, "fusedKernels", plan ? plan->kernel_count : 0);
    size_t boundaries = 0, aliases = 0;
    if (plan)
        for (size_t i = 0; i < plan->step_count; i++)
        {
            if (plan->items[plan->steps[i].root].alias) aliases++;
            else if (!plan->steps[i].name[0]) boundaries++;
        }
    add_assoc_long(return_value, "boundaries", boundaries);
    add_assoc_long(return_value, "viewAliases", aliases);
    add_assoc_long(return_value, "executions", plan ? plan->executions : 0);
    add_assoc_long(return_value, "executionSteps", plan ? plan->step_count : 0);
    size_t intermediates = 0;
    if (plan)
        for (size_t i = 0; i < plan->step_count; i++)
            if (!plan->items[plan->steps[i].root].output && !plan->items[plan->steps[i].root].alias) intermediates++;
    add_assoc_long(return_value, "intermediateBuffers", intermediates);
    add_assoc_bool(return_value, "cacheHit", plan && plan->cache_hit);
    add_assoc_string(return_value, "backend", plan && plan->use_cuda_graph ? "cuda-graph" :
                     plan && plan->stream_compatible ? "stream" : "native");
    add_assoc_bool(return_value, "asyncCompatible", plan && plan->stream_compatible);
    add_assoc_bool(return_value, "cudaGraphCompatible", plan && plan->graph_compatible);
    if (plan && !plan->graph_compatible)
        add_assoc_string(return_value, "cudaGraphIncompatibility", plan->incompatibility ? plan->incompatibility : "native-graph-not-supported");
    else add_assoc_null(return_value, "cudaGraphIncompatibility");
    if (plan && plan->incompatibility) add_assoc_string(return_value, "incompatibility", plan->incompatibility);
    else add_assoc_null(return_value, "incompatibility");
    add_assoc_long(return_value, "graphLaunches", plan ? plan->graph_launches : 0);
    add_assoc_long(return_value, "bufferReuses", plan ? plan->buffer_reuses : 0);
    add_assoc_long(return_value, "pending", plan ? plan->pending : 0);
    add_assoc_bool(return_value, "cudaGraphRequested", plan && plan->cuda_graph_requested);
    add_assoc_long(return_value, "tensorAllocations", plan ? plan->tensor_allocations : 0);
    add_assoc_long(return_value, "synchronizations", plan ? plan->synchronizations : 0);
    size_t scratch = 0;
    if (plan && plan->workspace)
        for (size_t i = 0; i < plan->count; i++)
            if (plan->workspace[i]) scratch++;
    add_assoc_long(return_value, "workspaceBuffers", scratch);
    add_assoc_bool(return_value, "profiling", plan && plan->profiling);
    add_assoc_long(return_value, "profiledExecutions", plan ? plan->profiled_executions : 0);
    add_assoc_long(return_value, "bindTimeNs", plan ? plan->bind_ns : 0);
    add_assoc_long(return_value, "executeTimeNs", plan ? plan->execute_ns : 0);
    add_assoc_long(return_value, "collectTimeNs", plan ? plan->collect_ns : 0);
}

ZEND_METHOD(FusionGraph, getPlan)
{
    ZEND_PARSE_PARAMETERS_NONE();
    fusion_plan *plan = Z_FUSION_GRAPH_P(ZEND_THIS)->plan;
    array_init(return_value);
    if (!plan) return;
    for (size_t i = 0; i < plan->step_count; i++)
    {
        fusion_step *step = &plan->steps[i];
        zval entry;
        array_init(&entry);
        add_assoc_long(&entry, "step", i);
        add_assoc_long(&entry, "root", step->root);
        add_assoc_string(&entry, "kind", plan->items[step->root].alias ? "view" :
                         step->name[0] ? "fused" : "native");
        add_assoc_string(&entry, "reason", plan->items[step->root].reason ?
                         plan->items[step->root].reason : "dependency");
        add_assoc_long(&entry, "inputs", step->leaf_count);
        add_assoc_long(&entry, "outputs", step->root_count);
        add_next_index_zval(return_value, &entry);
    }
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

static zend_object *fusion_execution_create(zend_class_entry *ce)
{
    fusion_execution *execution = zend_object_alloc(sizeof(fusion_execution), ce);
    zend_object_std_init(&execution->std, ce);
    object_properties_init(&execution->std, ce);
    ZVAL_UNDEF(&execution->graph);
    ZVAL_UNDEF(&execution->result);
    execution->std.handlers = &fusion_execution_handlers;
    return &execution->std;
}

static void fusion_execution_free(zend_object *object)
{
    fusion_execution *execution = (fusion_execution *)((char *)object - XtOffsetOf(fusion_execution, std));
    if (!Z_ISUNDEF(execution->graph))
    {
        fusion_plan *plan = Z_FUSION_GRAPH_P(&execution->graph)->plan;
        CUcontext previous;
        CUresult error = cuCtxPushCurrent(plan->context);
        if (error == CUDA_SUCCESS)
        {
            if (!execution->finished) fusion_execution_finish(execution);
            if (execution->event) fusion_cleanup_error("Destroying completion event", cuEventDestroy(execution->event));
            if (execution->stream) {
                cudaError_t workspace_error = cuda_reduction_release_stream((cudaStream_t)execution->stream);
                if (workspace_error != cudaSuccess)
                    php_error_docref(NULL, E_WARNING, "Releasing reduction workspace failed: %s", cudaGetErrorString(workspace_error));
                fusion_cleanup_error("Destroying async stream", cuStreamDestroy(execution->stream));
            }
            fusion_cleanup_error("Restoring async context", cuCtxPopCurrent(&previous));
        }
        else php_error_docref(NULL, E_WARNING, "Cannot clean up fusion execution context (CUDA error %d)", error);
        zval_ptr_dtor(&execution->graph);
    }
    if (!Z_ISUNDEF(execution->result)) zval_ptr_dtor(&execution->result);
    if (execution->retained) efree(execution->retained);
    zend_object_std_dtor(object);
}

static HashTable *fusion_execution_gc(zend_object *object, zval **table, int *count)
{
    fusion_execution *execution = (fusion_execution *)((char *)object - XtOffsetOf(fusion_execution, std));
    *table = &execution->graph;
    *count = 2;
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
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, cudaGraph, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_fusion_graph_async, 0, 0, Cuda\\FusionExecution, 0)
    ZEND_ARG_VARIADIC_OBJ_INFO(0, inputs, Cuda\\CudaArray, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_fusion_wait, 0, 0, IS_MIXED, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_fusion_finished, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_fusion_clear, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_fusion_profiling, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, enabled, _IS_BOOL, 0)
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
    ZEND_ME(Fusion, getCacheStats, arginfo_fusion_stats, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(Fusion, clearCache, arginfo_fusion_clear, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_FE_END
};
static const zend_function_entry fusion_graph_methods[] = {
    ZEND_ME(FusionGraph, __construct, arginfo_fusion_construct, ZEND_ACC_PRIVATE)
    ZEND_ME(FusionGraph, run, arginfo_fusion_graph_run, ZEND_ACC_PUBLIC)
    ZEND_ME(FusionGraph, runAsync, arginfo_fusion_graph_async, ZEND_ACC_PUBLIC)
    ZEND_ME(FusionGraph, getStats, arginfo_fusion_stats, ZEND_ACC_PUBLIC)
    ZEND_ME(FusionGraph, setProfiling, arginfo_fusion_profiling, ZEND_ACC_PUBLIC)
    ZEND_ME(FusionGraph, getPlan, arginfo_fusion_stats, ZEND_ACC_PUBLIC)
    ZEND_ME(FusionGraph, getSource, arginfo_fusion_source, ZEND_ACC_PUBLIC)
    ZEND_FE_END
};
static const zend_function_entry fusion_execution_methods[] = {
    ZEND_ME(FusionExecution, __construct, arginfo_fusion_construct, ZEND_ACC_PRIVATE)
    ZEND_ME(FusionExecution, wait, arginfo_fusion_wait, ZEND_ACC_PUBLIC)
    ZEND_ME(FusionExecution, isFinished, arginfo_fusion_finished, ZEND_ACC_PUBLIC)
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
    INIT_CLASS_ENTRY(ce, "Cuda\\FusionExecution", fusion_execution_methods);
    fusion_execution_ce = zend_register_internal_class(&ce);
    fusion_execution_ce->ce_flags |= ZEND_ACC_FINAL | ZEND_ACC_NOT_SERIALIZABLE;
    fusion_execution_ce->create_object = fusion_execution_create;
    memcpy(&fusion_execution_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    fusion_execution_handlers.offset = XtOffsetOf(fusion_execution, std);
    fusion_execution_handlers.free_obj = fusion_execution_free;
    fusion_execution_handlers.get_gc = fusion_execution_gc;
    fusion_execution_handlers.clone_obj = NULL;
    return SUCCESS;
}
