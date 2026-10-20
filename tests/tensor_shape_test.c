#include "php.h"
#undef NDEBUG
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static int errors;
#define CUDA_EXCEPTIONS_H
#define CUDA_THROW_INVALID(...) do { errors++; } while (0)
#define CUDA_THROW_RUNTIME(...) do { errors++; } while (0)
#define CUDA_THROW_OOM(...) do { errors++; } while (0)
#undef emalloc
#undef efree
#define emalloc(size) malloc(size)
#define efree(pointer) free(pointer)
#include "../src/operations.c"
#include "../src/tensor.c"

static void release_view(tensor_t *view)
{
    assert(view && view->base_tensor);
    view->base_tensor->ref_count--;
    free(view->shape);
    free(view->strides);
    free(view);
}

static void check_view_storage(void)
{
    float storage[8] = {0};
    int root_shape[] = {8}, shape[] = {2, 2}, pair[] = {2}, empty[] = {0};
    size_t root_strides[] = {1}, strides[] = {4, 1}, unit[] = {1};
    tensor_t root = {.data = storage, .allocated_size = sizeof(storage),
        .element_size = sizeof(float), .dtype = DTYPE_FLOAT32, .ndims = 1,
        .shape = root_shape, .strides = root_strides, .total_size = 8, .ref_count = 1};
    tensor_t *view = cuda_tensor_create_view(&root, shape, strides, 2, 2, 4);
    assert(view && view->data == storage + 2 && root.ref_count == 2);
    tensor_t *nested = cuda_tensor_create_view(view, pair, unit, 1, 4, 2);
    assert(nested && nested->data == storage + 6 && view->ref_count == 2);
    release_view(nested);

    int previous_errors = errors;
    assert(!cuda_tensor_create_view(view, pair, unit, 1, 5, 2));
    assert(errors == previous_errors + 1 && view->ref_count == 1);
    release_view(view);
    assert(root.ref_count == 1);

    size_t transposed[] = {1, 4};
    view = cuda_tensor_create_view(&root, shape, transposed, 2, 2, 4);
    assert(view);
    release_view(view);
    previous_errors = errors;
    assert(!cuda_tensor_create_view(&root, shape, strides, 2, 3, 4));
    assert(errors == previous_errors + 1 && root.ref_count == 1);

    view = cuda_tensor_create_view(&root, NULL, NULL, 0, 7, 1);
    assert(view && view->data == storage + 7);
    release_view(view);
    previous_errors = errors;
    assert(!cuda_tensor_create_view(&root, NULL, NULL, 0, 8, 1));
    assert(errors == previous_errors + 1);

    view = cuda_tensor_create_view(&root, empty, unit, 1, 8, 0);
    assert(view && view->data == storage + 8);
    release_view(view);
    previous_errors = errors;
    assert(!cuda_tensor_create_view(&root, empty, unit, 1, 9, 0));
    assert(errors == previous_errors + 1);

    tensor_t empty_root = {.element_size = sizeof(float), .dtype = DTYPE_FLOAT32,
        .ndims = 1, .shape = empty, .strides = unit, .ref_count = 1};
    view = cuda_tensor_create_view(&empty_root, empty, unit, 1, 0, 0);
    assert(view && !view->data);
    release_view(view);
    previous_errors = errors;
    assert(!cuda_tensor_create_view(&empty_root, NULL, NULL, 0, 0, 1));
    assert(errors == previous_errors + 1 && empty_root.ref_count == 1);

#if PTRDIFF_MAX > INT_MAX
    int broadcast_shape[] = {INT_MAX, 2};
    size_t broadcast_strides[] = {0, 0};
    view = cuda_tensor_create_view(&root, broadcast_shape, broadcast_strides, 2,
                                   7, (size_t)INT_MAX * 2);
    assert(view && view->total_size > INT_MAX && view->data == storage + 7);
    release_view(view);
    previous_errors = errors;
    assert(!cuda_tensor_create_view(&root, broadcast_shape, broadcast_strides, 2,
                                    8, (size_t)INT_MAX * 2));
    assert(errors == previous_errors + 1);
#endif
    assert(root.ref_count == 1);
}

int main(void)
{
    size_t elements, bytes, strides[MAX_DIMS];
    assert(tensor_shape_metrics(NULL, 0, 8, NULL, &elements, &bytes));
    assert(elements == 1 && bytes == 8);
    assert(!tensor_shape_metrics(NULL, 1, 8, NULL, &elements, &bytes));
    int maximum_rank[MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    assert(tensor_shape_metrics(maximum_rank, MAX_DIMS, 4, strides, &elements, &bytes));
    assert(elements == 1 && bytes == 4 && strides[0] == 1);
    assert(!tensor_shape_metrics(maximum_rank, MAX_DIMS + 1, 4, NULL, &elements, &bytes));
    int negative[] = {-1};
    assert(!tensor_shape_metrics(negative, 1, 4, NULL, &elements, &bytes));
    assert(!tensor_shape_metrics(maximum_rank, 1, 0, NULL, &elements, &bytes));
    int byte_overflow[] = {INT_MAX, INT_MAX};
    assert(!tensor_shape_metrics(byte_overflow, 2, 8, NULL, &elements, &bytes));

#if PTRDIFF_MAX > INT_MAX
    int empty[] = {0, INT_MAX, INT_MAX};
    assert(tensor_shape_metrics(empty, 3, 4, strides, &elements, &bytes));
    assert(elements == 0 && bytes == 0);
    assert(strides[0] == (size_t)INT_MAX * INT_MAX);
    tensor_t a = {.ndims = 3, .shape = empty, .strides = strides};
    tensor_t b = {.ndims = 0};
    int shape[MAX_DIMS], ndims;
    size_t a_strides[MAX_DIMS] = {0}, b_strides[MAX_DIMS] = {0};
    assert(prepare_broadcast_operation(&a, &b, shape, &ndims, a_strides, b_strides, &elements));
    assert(elements == 0 && ndims == 3 && shape[0] == 0);
    assert(a_strides[0] == strides[0] && a_strides[0] > INT_MAX);
    assert(a_strides[1] == INT_MAX && a_strides[2] == 1);

    int stride_overflow[] = {0, INT_MAX, INT_MAX, 3};
    assert(!tensor_shape_metrics(stride_overflow, 4, 4, NULL, &elements, &bytes));
    a.shape = stride_overflow;
    a.ndims = 4;
    assert(!prepare_broadcast_operation(&a, &b, shape, &ndims, a_strides, b_strides, &elements));
    assert(errors == 1);
    int empty_suffix[] = {INT_MAX, INT_MAX, INT_MAX, 0};
    assert(tensor_shape_metrics(empty_suffix, 4, 8, strides, &elements, &bytes));
    assert(elements == 0 && bytes == 0 && strides[0] == 0);
    a.shape = empty_suffix;
    a.ndims = 4;
    elements = 123;
    assert(!calculate_reduction_shape(&a, 3, shape, &elements));
    assert(errors == 2 && elements == 123);
    assert(calculate_reduction_shape(&a, 0, shape, &elements) == 3);
    assert(elements == 0 && shape[2] == 0);
#endif
    int result_rank, result_shape[MAX_DIMS];
    assert(!prepare_matmul_result_shape(0, NULL, 2, maximum_rank, &result_rank, result_shape));
    check_view_storage();
    puts("tensor shape checks passed");
    return 0;
}
