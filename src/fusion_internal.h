#ifndef CUDA_FUSION_INTERNAL_H
#define CUDA_FUSION_INTERNAL_H

#include "fusion.h"
#include "cuda.h"
#include "cuda_exceptions.h"
#include "ca_private.h"
#include "compiler_ce.h"
#include "kernel_types.h"
#include "zend_smart_str.h"
#include <cuda.h>
#include <limits.h>
#include <string.h>

#define FUSION_MAX_NODES 512
#define FUSION_KERNEL_BUDGET 32

typedef enum
{
    FUSION_BUFFERS_FRESH, FUSION_BUFFERS_LOCAL, FUSION_BUFFERS_WORKSPACE
} fusion_buffer_mode;

typedef enum
{
    FUSION_INPUT, FUSION_BINARY, FUSION_SCALAR, FUSION_UNARY,
    FUSION_REDUCE, FUSION_ARG_REDUCE, FUSION_MATMUL, FUSION_VIEW,
    FUSION_CAST, FUSION_WHERE
} fusion_kind;

typedef struct fusion_node
{
    fusion_kind kind;
    tensor_t *a;
    tensor_t *b;
    tensor_t *c;
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
    size_t consumers;
    const char *reason;
    size_t a, b, c;
    size_t last_use;
    size_t slot;
    int alias;
} fusion_item;

typedef struct
{
    size_t root;
    size_t *roots;
    size_t root_count;
    size_t *leaves;
    size_t leaf_count;
    char name[32];
    CUfunction function;
    void **arguments;
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
    tensor_t **input_examples;
    tensor_t **workspace;
    int compiled;
    CUmodule module;
    CUcontext context;
    int device;
    zend_string *source;
    size_t executions;
    int cache_hit;
    int stream_compatible;
    int graph_compatible;
    int native_steps;
    CUstream sync_stream;
    CUevent sync_ready;
    const char *incompatibility;
    int use_cuda_graph;
    CUgraph cuda_graph;
    CUgraphExec graph_exec;
    CUgraphNode *graph_nodes;
    size_t graph_launches;
    size_t buffer_reuses;
    size_t pending;
    int cuda_graph_requested;
    size_t tensor_allocations;
    size_t synchronizations;
    int profiling;
    size_t profiled_executions;
    uint64_t bind_ns, execute_ns, collect_ns;
} fusion_plan;

tensor_t *fusion_record(fusion_kind kind, tensor_t *a, tensor_t *b,
                       operation_type_t op, const int *shape, int ndims, dtype_t dtype);
void fusion_scope_free(fusion_scope *scope, int invalidate);
int fusion_inline(tensor_t *tensor);
size_t fusion_find(fusion_plan *plan, tensor_t *tensor);
fusion_plan *fusion_build_plan(tensor_t **roots, size_t root_count);
void fusion_plan_free(fusion_plan *plan);
int fusion_generate_source(fusion_plan *plan);
int fusion_load_module(fusion_plan *plan, const char **names);
void fusion_cache_shutdown(void);
void fusion_cache_stats(zval *result);
int fusion_execute(fusion_plan *plan, tensor_t **values);
int fusion_enqueue(fusion_plan *plan, tensor_t **values, CUstream stream, fusion_buffer_mode mode);
int fusion_context_check(fusion_plan *plan);
tensor_t **fusion_values(fusion_plan *plan);
void fusion_values_free(fusion_plan *plan, tensor_t **values);
void fusion_values_release(fusion_plan *plan, tensor_t **values);
void fusion_cleanup_error(const char *operation, CUresult error);
int fusion_preallocated(fusion_plan *plan, fusion_step *step);

#endif
