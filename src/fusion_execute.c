#include "fusion_internal.h"
#include "matmul_kernels.h"
#include "backend_info.h"
#include "reduction_ops.h"
void fusion_cleanup_error(const char *operation, CUresult error)
{
    if (error != CUDA_SUCCESS)
        php_error_docref(NULL, E_WARNING, "%s failed during fusion cleanup (CUDA error %d)", operation, error);
}

static void fusion_graph_discard(fusion_plan *plan)
{
    if (plan->graph_exec) fusion_cleanup_error("Destroying failed graph executable", cuGraphExecDestroy(plan->graph_exec));
    if (plan->cuda_graph) fusion_cleanup_error("Destroying failed CUDA graph", cuGraphDestroy(plan->cuda_graph));
    if (plan->graph_nodes) efree(plan->graph_nodes);
    plan->graph_exec = NULL;
    plan->cuda_graph = NULL;
    plan->graph_nodes = NULL;
}
int fusion_context_check(fusion_plan *plan)
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
    return 1;
}

static int fusion_prepare_buffers(fusion_plan *plan, tensor_t **values, fusion_buffer_mode mode)
{
    int reusable = mode != FUSION_BUFFERS_FRESH;
    int persistent = mode == FUSION_BUFFERS_WORKSPACE;
    if (persistent && !plan->workspace) plan->workspace = ecalloc(plan->count, sizeof(tensor_t *));
    for (size_t i = 0; i < plan->step_count; i++)
    {
        fusion_step *step = &plan->steps[i];
        if (!fusion_preallocated(plan, step)) continue;
        for (size_t output = 0; output < step->root_count; output++)
        {
            size_t id = step->roots[output];
            tensor_t *tensor = plan->items[id].tensor;
            tensor_t *result = NULL;
            if (persistent && !plan->items[id].output)
            {
                size_t slot = plan->items[id].slot;
                result = plan->workspace[slot];
                if (!result)
                {
                    result = cuda_tensor_create_empty_with_dtype(tensor->shape, tensor->ndims, tensor->dtype);
                    if (!result) return 0;
                    plan->workspace[slot] = result;
                    plan->tensor_allocations++;
                }
                else plan->buffer_reuses++;
                result->ref_count++;
            }
            else if (reusable && !plan->items[id].output && plan->items[id].slot != id)
            {
                result = values[plan->items[id].slot];
                if (result) { result->ref_count++; plan->buffer_reuses++; }
            }
            if (!result)
            {
                result = cuda_tensor_create_empty_with_dtype(tensor->shape, tensor->ndims, tensor->dtype);
                if (!result) return 0;
                plan->tensor_allocations++;
            }
            values[id] = result;
        }
    }
    return 1;
}

static int fusion_native_submit(fusion_plan *plan, fusion_step *step, tensor_t **values, CUstream stream)
{
    fusion_item *item = &plan->items[step->root];
    fusion_node *node = item->tensor->fusion;
    tensor_t *a = values[item->a];
    tensor_t *b = item->b != SIZE_MAX ? values[item->b] : NULL;
    tensor_t *result = values[step->root];
    if (node->kind == FUSION_MATMUL)
    {
        int ok = a->ndims == 2 && b->ndims == 2
            ? cuda_matmul_launcher(a->data, b->data, result->data,
                                  a->shape[0], a->shape[1], b->shape[1],
                                  a->strides[0], a->strides[1], b->strides[0], b->strides[1],
                                  result->strides[0], result->strides[1], (cudaStream_t)stream)
            : cuda_batched_matmul_nd_launcher(a->data, b->data, result->data,
                                             a->shape, a->strides, a->ndims,
                                             b->shape, b->strides, b->ndims,
                                             result->shape, result->strides, result->ndims, (cudaStream_t)stream);
        if (!ok)
        {
            cuda_backend_info info = {0};
            cuda_blas_info(&info);
            CUDA_THROW_RUNTIME("Fusion matmul submission failed (cuBLAS status %d)", info.last_blas_status);
            return 0;
        }
        return 1;
    }
    int axis = node->parameter;
    if (axis == -1 && a->total_size > INT_MAX)
    {
        CUDA_THROW_INVALID("Global fusion reduction exceeds INT_MAX elements");
        return 0;
    }
    if (axis == -1 && !is_contiguous(a))
    {
        CUDA_THROW_RUNTIME("Global fusion reduction requires contiguous input storage");
        return 0;
    }
    int shape[] = {axis == -1 ? (int)a->total_size : 0};
    size_t stride[] = {1};
    if (node->kind == FUSION_ARG_REDUCE)
        launch_arg_reduction(a->data, result->data, a->dtype, node->op,
                             axis == -1 ? shape : a->shape, axis == -1 ? 1 : a->ndims,
                             result->shape, axis == -1 ? stride : a->strides,
                             result->ndims, axis == -1 ? 0 : axis, result->total_size, 0, (cudaStream_t)stream);
    else
        launch_reduction(a->data, result->data, a->dtype, node->op,
                         axis == -1 ? shape : a->shape, axis == -1 ? 1 : a->ndims,
                         result->shape, axis == -1 ? stride : a->strides,
                         result->ndims, axis == -1 ? 0 : axis, result->total_size, 0, (cudaStream_t)stream);
    cudaError_t error = cuda_reduction_status();
    if (error != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Fusion reduction submission failed: %s", cudaGetErrorString(error));
        return 0;
    }
    return 1;
}

static int fusion_alias_view(fusion_plan *plan, fusion_step *step, tensor_t **values)
{
    tensor_t *tensor = plan->items[step->root].tensor;
    fusion_node *node = tensor->fusion;
    tensor_t *source = values[plan->items[step->root].a];
    size_t strides[MAX_DIMS];
    size_t offset = 0;
    if (node->op == OP_TRANSPOSE)
        for (int d = 0; d < tensor->ndims; d++) strides[d] = source->strides[node->axes[d]];
    else if (node->op == OP_SLICE)
    {
        for (int d = 0; d < source->ndims; d++)
            offset += (size_t)node->slice_starts[d] * source->strides[d];
        for (int d = 0; d < tensor->ndims; d++)
            strides[d] = source->strides[node->axes[d]] * (size_t)node->slice_steps[node->axes[d]];
        if (!tensor->total_size) offset = 0;
    }
    else
    {
        if (!is_contiguous(source))
        {
            CUDA_THROW_RUNTIME("Fusion reshape alias requires contiguous source storage");
            return 0;
        }
        size_t stride = 1;
        for (int d = tensor->ndims - 1; d >= 0; d--)
        {
            strides[d] = stride;
            stride *= tensor->shape[d];
        }
    }
    values[step->root] = cuda_tensor_create_view(source, tensor->shape, strides, tensor->ndims, offset, tensor->total_size);
    if (values[step->root]) values[step->root]->offset = 0;
    return values[step->root] != NULL;
}

int fusion_enqueue(fusion_plan *plan, tensor_t **values, CUstream stream, fusion_buffer_mode mode)
{
    if (!fusion_context_check(plan)) return 0;
    if (!plan->stream_compatible)
    {
        CUDA_THROW_RUNTIME("Asynchronous fusion requires a stream-compatible plan: %s", plan->incompatibility);
        return 0;
    }
    if (!fusion_prepare_buffers(plan, values, mode)) return 0;
    int pooled = mode == FUSION_BUFFERS_WORKSPACE;
    CUevent ready = pooled ? plan->sync_ready : NULL;
    CUresult error = ready ? CUDA_SUCCESS : cuEventCreate(&ready, CU_EVENT_DISABLE_TIMING);
    if (pooled && error == CUDA_SUCCESS) plan->sync_ready = ready;
    if (error == CUDA_SUCCESS) error = cuEventRecord(ready, NULL);
    if (error == CUDA_SUCCESS) error = cuStreamWaitEvent(stream, ready, 0);
    if (ready && !pooled) fusion_cleanup_error("Destroying readiness event", cuEventDestroy(ready));
    if (error != CUDA_SUCCESS)
    {
        CUDA_THROW_RUNTIME("Cannot order fusion stream after input work (CUDA error %d)", error);
        return 0;
    }
    int constructing = plan->use_cuda_graph && !plan->graph_exec;
    if (constructing)
    {
        error = cuGraphCreate(&plan->cuda_graph, 0);
        plan->graph_nodes = ecalloc(plan->step_count, sizeof(CUgraphNode));
        if (error != CUDA_SUCCESS)
        {
            fusion_graph_discard(plan);
            CUDA_THROW_RUNTIME("Cannot create CUDA fusion graph (CUDA error %d)", error);
            return 0;
        }
    }
    CUgraphNode previous = NULL;
    for (size_t i = 0; i < plan->step_count; i++)
    {
        fusion_step *step = &plan->steps[i];
        if (plan->items[step->root].alias)
        {
            if (!fusion_alias_view(plan, step, values)) return 0;
            continue;
        }
        if (!step->name[0])
        {
            if (!fusion_native_submit(plan, step, values, stream)) return 0;
            continue;
        }
        tensor_t *result = values[step->root];
        if (!result->total_size) continue;
        void **args = step->arguments;
        for (size_t j = 0; j < step->leaf_count; j++)
            args[j] = &values[step->leaves[j]]->data;
        for (size_t j = 0; j < step->root_count; j++)
            args[step->leaf_count + j] = &values[step->roots[j]]->data;
        size_t blocks = (result->total_size - 1) / 256 + 1;
        if (blocks > 65535) blocks = 65535;
        if (plan->use_cuda_graph)
        {
            CUDA_KERNEL_NODE_PARAMS parameters = {0};
            parameters.func = step->function;
            parameters.gridDimX = (unsigned int)blocks;
            parameters.gridDimY = parameters.gridDimZ = 1;
            parameters.blockDimX = 256;
            parameters.blockDimY = parameters.blockDimZ = 1;
            parameters.kernelParams = args;
            if (constructing)
            {
                error = cuGraphAddKernelNode(&plan->graph_nodes[i], plan->cuda_graph,
                                            previous ? &previous : NULL, previous ? 1 : 0, &parameters);
                previous = plan->graph_nodes[i];
            }
            else error = cuGraphExecKernelNodeSetParams(plan->graph_exec, plan->graph_nodes[i], &parameters);
        }
        else error = cuLaunchKernel(step->function, (unsigned int)blocks, 1, 1, 256, 1, 1, 0, stream, args, NULL);
        if (error != CUDA_SUCCESS)
        {
            if (constructing) fusion_graph_discard(plan);
            CUDA_THROW_RUNTIME("Fusion kernel submission failed (CUDA error %d)", error);
            return 0;
        }
    }
    if (plan->use_cuda_graph)
    {
        if (constructing)
        {
#if CUDA_VERSION >= 11040
            error = cuGraphInstantiateWithFlags(&plan->graph_exec, plan->cuda_graph, 0);
#else
            error = cuGraphInstantiate(&plan->graph_exec, plan->cuda_graph, NULL, NULL, 0);
#endif
        }
        if (error == CUDA_SUCCESS) error = cuGraphLaunch(plan->graph_exec, stream);
        if (error != CUDA_SUCCESS)
        {
            if (constructing) fusion_graph_discard(plan);
            CUDA_THROW_RUNTIME("CUDA fusion graph launch failed (CUDA error %d)", error);
            return 0;
        }
        plan->graph_launches++;
    }
    return 1;
}

int fusion_execute(fusion_plan *plan, tensor_t **values)
{
    if (!fusion_context_check(plan)) return 0;
    CUresult error;
    cudaError_t runtime_error;
    if (plan->stream_compatible)
    {
        int persistent = plan->compiled;
        CUstream stream = persistent ? plan->sync_stream : NULL;
        error = stream ? CUDA_SUCCESS : cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
        if (error != CUDA_SUCCESS)
        {
            CUDA_THROW_RUNTIME("Cannot create fusion execution stream (CUDA error %d)", error);
            return 0;
        }
        if (persistent) plan->sync_stream = stream;
        int ok = fusion_enqueue(plan, values, stream, persistent ? FUSION_BUFFERS_WORKSPACE : FUSION_BUFFERS_FRESH);
        error = cuStreamSynchronize(stream);
        plan->synchronizations++;
        if (!persistent) fusion_cleanup_error("Destroying execution stream", cuStreamDestroy(stream));
        if (error != CUDA_SUCCESS && !EG(exception))
            CUDA_THROW_RUNTIME("Fusion stream failed (CUDA error %d)", error);
        if (ok && error == CUDA_SUCCESS) plan->executions++;
        return ok && error == CUDA_SUCCESS;
    }
    if (!fusion_prepare_buffers(plan, values, plan->compiled ? FUSION_BUFFERS_WORKSPACE : FUSION_BUFFERS_FRESH)) return 0;
    for (size_t i = 0; i < plan->step_count; i++)
    {
        fusion_step *step = &plan->steps[i];
        tensor_t *tensor = plan->items[step->root].tensor;
        fusion_node *node = tensor->fusion;
        tensor_t *result = values[step->root];
        if (step->name[0])
        {
            if (result->total_size)
            {
                void **args = step->arguments;
                for (size_t j = 0; j < step->leaf_count; j++)
                    args[j] = &values[step->leaves[j]]->data;
                for (size_t j = 0; j < step->root_count; j++)
                    args[step->leaf_count + j] = &values[step->roots[j]]->data;
                size_t blocks = (result->total_size - 1) / 256 + 1;
                if (blocks > 65535) blocks = 65535;
                error = cuLaunchKernel(step->function, (unsigned int)blocks, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
                if (error != CUDA_SUCCESS)
                {
                    CUDA_THROW_RUNTIME("Fused kernel launch failed (CUDA error %d)", error);
                    return 0;
                }
            }
        }
        else if (fusion_preallocated(plan, step))
        {
            if (!fusion_native_submit(plan, step, values, NULL)) return 0;
        }
        else if (plan->items[step->root].alias)
        {
            if (!fusion_alias_view(plan, step, values)) return 0;
            result = values[step->root];
        }
        else
        {
            tensor_t *source = values[plan->items[step->root].a];
            tensor_t *a = source;
            tensor_t *b = node->b ? values[plan->items[step->root].b] : NULL;
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
                flat = cuda_tensor_reshape(source, shape, 1);
                if (!flat) return 0;
                flat->offset = 0;
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
    plan->synchronizations++;
    if (runtime_error != cudaSuccess)
    {
        CUDA_THROW_RUNTIME("Fusion synchronization failed: %s", cudaGetErrorString(runtime_error));
        return 0;
    }
    plan->executions++;
    return 1;
}

tensor_t **fusion_values(fusion_plan *plan)
{
    tensor_t **values = ecalloc(plan->count, sizeof(tensor_t *));
    for (size_t i = 0; i < plan->count; i++)
        if (!plan->items[i].tensor->fusion) values[i] = plan->items[i].tensor;
    return values;
}

void fusion_values_free(fusion_plan *plan, tensor_t **values)
{
    /* Even a failed launch can leave earlier kernels using these allocations. */
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess && !EG(exception))
        CUDA_THROW_RUNTIME("Fusion cleanup synchronization failed: %s", cudaGetErrorString(error));
    fusion_values_release(plan, values);
}

void fusion_values_release(fusion_plan *plan, tensor_t **values)
{
    for (size_t i = 0; i < plan->count; i++)
        if (values[i] && values[i] != plan->items[i].tensor)
            cuda_tensor_destroy(values[i]);
    efree(values);
}
