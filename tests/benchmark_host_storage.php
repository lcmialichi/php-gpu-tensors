<?php
declare(strict_types=1);

use PhpTensors\HostArray;

if (!extension_loaded('phptensors')) {
    fwrite(STDERR, "Load the phptensors extension before running this benchmark.\n");
    exit(1);
}
if (getenv('USE_ZEND_ALLOC') === '0') {
    fwrite(STDERR, "Enable Zend allocation accounting to measure native storage.\n");
    exit(1);
}
$gpuTransfers = ($argv[1] ?? '') === '--gpu';
if (isset($argv[1]) && !$gpuTransfers) {
    fwrite(STDERR, "Usage: php tests/benchmark_host_storage.php [--gpu]\n");
    exit(1);
}
if ($gpuTransfers && (!class_exists('PhpTensors\\NN', false) || cuda_get_device_count() < 1)) {
    fwrite(STDERR, "GPU transfer measurement requires a CUDA build and visible GPU.\n");
    exit(1);
}

function medianMilliseconds(callable $operation): float
{
    $operation();
    $times = [];
    for ($i = 0; $i < 7; $i++) {
        $start = hrtime(true);
        $result = $operation();
        $times[] = (hrtime(true) - $start) / 1e6;
        unset($result);
    }
    sort($times);
    return $times[3];
}

$count = 1024 * 1024;
$results = [];
foreach (['float32' => 4, 'float64' => 8] as $dtype => $width) {
    $before = memory_get_usage(false);
    $host = HostArray::ones([$count], $dtype);
    $allocated = memory_get_usage(false) - $before;
    $buffer = $host->toBuffer();
    $importMs = medianMilliseconds(fn() => HostArray::fromBuffer($buffer, [$count], $dtype));
    $exportMs = medianMilliseconds(fn() => $host->toBuffer());
    $arithmeticMs = medianMilliseconds(fn() => $host + $host);
    $results[$dtype] = [
        'elements' => $count,
        'payload_bytes' => $count * $width,
        'allocated_bytes_including_metadata' => $allocated,
        'buffer_import_median_ms' => $importMs,
        'buffer_export_median_ms' => $exportMs,
        'add_median_ms' => $arithmeticMs,
        'buffer_import_payload_gib_per_second' => ($count * $width / (1024 ** 3)) / ($importMs / 1000),
        'buffer_export_payload_gib_per_second' => ($count * $width / (1024 ** 3)) / ($exportMs / 1000),
        'add_million_elements_per_second' => $count / ($arithmeticMs * 1000),
    ];
    if ($gpuTransfers) {
        $gpu = $host->toGpu();
        cuda_synchronize();
        if ($gpu->toHost()->toBuffer() !== $buffer) {
            throw new RuntimeException("$dtype GPU round trip changed the bytes");
        }
        $uploadMs = medianMilliseconds(function () use ($host) {
            $result = $host->toGpu();
            cuda_synchronize();
            return $result;
        });
        $downloadMs = medianMilliseconds(fn() => $gpu->toHost());
        $results[$dtype]['gpu_upload_median_ms'] = $uploadMs;
        $results[$dtype]['gpu_download_median_ms'] = $downloadMs;
        $results[$dtype]['gpu_upload_payload_gib_per_second'] = ($count * $width / (1024 ** 3)) / ($uploadMs / 1000);
        $results[$dtype]['gpu_download_payload_gib_per_second'] = ($count * $width / (1024 ** 3)) / ($downloadMs / 1000);
        unset($gpu);
    }
    unset($host, $buffer);
}
echo json_encode(['php' => PHP_VERSION, 'zts' => (bool)PHP_ZTS,
    'cuda_build' => class_exists('PhpTensors\\NN', false),
    'gpu_transfers' => $gpuTransfers,
    'iterations' => 7, 'results' => $results], JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR), "\n";
