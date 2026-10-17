<?php

/** Compiled backend capabilities and actual per-thread/request dispatch counters; no GPU work is submitted. */
function cuda_get_backend_info(): array {}

/**
 * Select strict FP32 or opt-in TF32 math for cuBLAS GEMMs.
 *
 * @throws \PhpTensors\InvalidArgumentException|\PhpTensors\RuntimeException
 */
function cuda_set_matmul_precision(string $precision): void {}

/**
 * Explicitly benchmark and cache a cuBLASLt algorithm for a compatible 2D FP32 GEMM.
 * This blocks while testing zero-workspace candidates and must be called outside Fusion capture.
 * @throws \PhpTensors\InvalidArgumentException|\PhpTensors\RuntimeException
 * @return array{backend:string,precision:string,selectedCandidate:int,candidateCount:int,candidates:list<array{algorithmId:int,workspaceBytes:int,medianMs:float,selected:bool}>}
 */
function cuda_tune_matmul(PhpTensors\CudaArray $left, PhpTensors\CudaArray $right): array {}

/** Return the number of CUDA devices visible to this process. @throws \PhpTensors\RuntimeException */
function cuda_get_device_count(): int
{
}

/** Read device properties. @throws \PhpTensors\RuntimeException */
function cuda_get_device_info(int $deviceId = 0): array
{
}

/** Select the CUDA device for the current process. @throws \PhpTensors\RuntimeException */
function cuda_set_device(int $deviceId): bool
{
}

/** Return the current CUDA device ID. @throws \PhpTensors\RuntimeException */
function cuda_get_current_device(): int
{
}

/** Return free, total, and used device memory in bytes. @throws \PhpTensors\RuntimeException */
function cuda_get_memory_info(): array
{
}

/** Reset the active CUDA device and release its context. @throws \PhpTensors\RuntimeException */
function cuda_device_reset(): bool
{
}

/** Read the installed NVIDIA driver version. @throws \PhpTensors\RuntimeException */
function cuda_get_driver_version(): array
{
}

/** Read the CUDA runtime version. @throws \PhpTensors\RuntimeException */
function cuda_get_runtime_version(): array
{
}

/** Block until queued device work finishes. @throws \PhpTensors\RuntimeException */
function cuda_synchronize(): bool
{
}

/** Return and clear the last CUDA error, or null when no error is pending. */
function cuda_get_last_error(): ?array
{
}

/** Clear the last CUDA error. */
function cuda_clear_error(): bool
{
}

/** Check peer access between two CUDA devices. @throws \PhpTensors\RuntimeException */
function cuda_get_peer_access(int $device1, int $device2): bool
{
}