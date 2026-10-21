#ifndef MODULE_STREAM_STATUS_H
#define MODULE_STREAM_STATUS_H

#include <stddef.h>

/* CUDA driver types and entry points must be declared by the includer. */
typedef struct
{
    CUresult result;
    CUresult restore_result;
} module_stream_status;

static inline module_stream_status module_stream_check(CUcontext context, CUstream stream, int synchronize)
{
    module_stream_status status = {cuCtxPushCurrent(context), CUDA_SUCCESS};
    if (status.result != CUDA_SUCCESS)
        return status;

    status.result = synchronize ? cuStreamSynchronize(stream) : cuStreamQuery(stream);
    CUcontext popped = NULL;
    status.restore_result = cuCtxPopCurrent(&popped);
    if (status.restore_result != CUDA_SUCCESS &&
        (status.result == CUDA_SUCCESS || status.result == CUDA_ERROR_NOT_READY))
        status.result = status.restore_result;

    return status;
}

#endif
