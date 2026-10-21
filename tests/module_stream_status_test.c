#include <stdio.h>
#include <string.h>
#undef NDEBUG
#include <assert.h>

typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef enum
{
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_LAUNCH_FAILED = 719
} CUresult;

static int owner_token, caller_token, stream_token;
static CUcontext current;
static CUcontext previous;
static CUresult push_result, stream_result, pop_result;
static char calls[5];
static int call_count;
static int expected_synchronize;

static CUresult cuCtxPushCurrent(CUcontext context)
{
    calls[call_count++] = 'P';
    assert(context == (CUcontext)&owner_token);
    if (push_result == CUDA_SUCCESS)
    {
        previous = current;
        current = context;
    }
    return push_result;
}

static CUresult cuStreamQuery(CUstream stream)
{
    calls[call_count++] = 'Q';
    assert(!expected_synchronize);
    assert(current == (CUcontext)&owner_token && stream == (CUstream)&stream_token);
    return stream_result;
}

static CUresult cuStreamSynchronize(CUstream stream)
{
    calls[call_count++] = 'S';
    assert(expected_synchronize);
    assert(current == (CUcontext)&owner_token && stream == (CUstream)&stream_token);
    return stream_result;
}

static CUresult cuCtxPopCurrent(CUcontext *popped)
{
    calls[call_count++] = 'O';
    assert(current == (CUcontext)&owner_token);
    if (pop_result == CUDA_SUCCESS)
    {
        *popped = current;
        current = previous;
    }
    return pop_result;
}

#include "module_stream_status.h"

int main(void)
{
    const CUresult results[] = {CUDA_SUCCESS, CUDA_ERROR_NOT_READY, CUDA_ERROR_LAUNCH_FAILED};
    int cases = 0;
    for (int synchronize = 0; synchronize <= 1; synchronize++)
    {
        expected_synchronize = synchronize;
        for (int push_failure = 0; push_failure <= 1; push_failure++)
            for (int pop_failure = 0; pop_failure <= 1; pop_failure++)
                for (int outcome = 0; outcome < 3; outcome++)
                {
                    current = (CUcontext)&caller_token;
                    push_result = push_failure ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS;
                    pop_result = pop_failure ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS;
                    stream_result = results[outcome];
                    call_count = 0;
                    memset(calls, 0, sizeof(calls));

                    module_stream_status status = module_stream_check(
                        (CUcontext)&owner_token, (CUstream)&stream_token, synchronize);

                    CUresult expected = push_failure ? push_result : stream_result;
                    if (!push_failure && pop_failure && stream_result != CUDA_ERROR_LAUNCH_FAILED)
                        expected = pop_result;
                    assert(status.result == expected);
                    assert(status.restore_result == (push_failure ? CUDA_SUCCESS : pop_result));
                    assert(strcmp(calls, push_failure ? "P" : (synchronize ? "PSO" : "PQO")) == 0);
                    assert(current == (CUcontext)((!push_failure && pop_failure) ? &owner_token : &caller_token));
                    cases++;
                }
    }
    assert(cases == 24);
    puts("24 simulated driver cases passed: owner context, pending work, sync/query failures and restoration failures");
    return 0;
}
