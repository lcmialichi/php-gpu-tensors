<?php

/** Compiled backend capabilities and actual per-thread/request dispatch counters; no GPU work is submitted. */
function cuda_get_backend_info(): array {}

/** Return the number of CUDA devices visible to this process. @throws \Cuda\RuntimeException */
function cuda_get_device_count(): int
{
}

/** Read device properties. @throws \Cuda\RuntimeException */
function cuda_get_device_info(int $deviceId = 0): array
{
}

/** Select the CUDA device for the current process. @throws \Cuda\RuntimeException */
function cuda_set_device(int $deviceId): bool
{
}

/** Return the current CUDA device ID. @throws \Cuda\RuntimeException */
function cuda_get_current_device(): int
{
}

/** Return free, total, and used device memory in bytes. @throws \Cuda\RuntimeException */
function cuda_get_memory_info(): array
{
}

/** Reset the active CUDA device and release its context. @throws \Cuda\RuntimeException */
function cuda_device_reset(): bool
{
}

/** Read the installed NVIDIA driver version. @throws \Cuda\RuntimeException */
function cuda_get_driver_version(): array
{
}

/** Read the CUDA runtime version. @throws \Cuda\RuntimeException */
function cuda_get_runtime_version(): array
{
}

/** Block until queued device work finishes. @throws \Cuda\RuntimeException */
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

/** Check peer access between two CUDA devices. @throws \Cuda\RuntimeException */
function cuda_get_peer_access(int $device1, int $device2): bool
{
}