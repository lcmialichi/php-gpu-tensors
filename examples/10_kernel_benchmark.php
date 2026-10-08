<?php
declare(strict_types=1);

use Cuda\CudaArray;

if (!extension_loaded('cuda') || cuda_get_device_count() < 1) {
    throw new RuntimeException('CUDA extension and GPU required.');
}
$options = getopt('', ['output:', 'binary:', 'precision:']);
$precision = $options['precision'] ?? 'fp32-strict';
cuda_set_matmul_precision($precision);
$cases = [];
foreach ([1048576, 16777216] as $count) {
    $pattern = pack('g*', ...array_map(static fn($i) => $i / 16, range(1, 16)));
    $x = CudaArray::fromBuffer(str_repeat($pattern, intdiv($count, 16)), [$count]);
    $cases["sum-$count"] = [static fn() => $x->sum(), $count * 17 / 32, 1];
    $cases["mean-$count"] = [static fn() => $x->mean(), 17 / 32, 1];
}
foreach ([1024, 4096] as $size) {
    $x = CudaArray::ones([$size, $size]);
    $cases["axis0-$size"] = [static fn() => $x->sum(0), $size, $size];
    $cases["axis1-$size"] = [static fn() => $x->sum(1), $size, $size];
}
foreach ([
    ['small-32x64x32', 32, 64, 32],
    ['square-64', 64, 64, 64],
    ['square-128', 128, 128, 128],
    ['square-256', 256, 256, 256],
    ['square-512', 512, 512, 512],
    ['square-1024', 1024, 1024, 1024],
    ['square-2048', 2048, 2048, 2048],
    ['square-4096', 4096, 4096, 4096],
    ['skinny-16x1024x16', 16, 1024, 16],
    ['mlp-32x512x256', 32, 512, 256],
    ['mlp-128x512x256', 128, 512, 256],
    ['classifier-512x256x10', 512, 256, 10],
] as [$name, $rows, $inner, $cols]) {
    $a = CudaArray::ones([$rows, $inner]);
    $b = CudaArray::ones([$inner, $cols]);
    $cases["matmul-$name"] = [static fn() => $a->matmul($b), $inner, $rows * $cols, 'matmul'];
}
$transposed = CudaArray::ones([128, 128])->transpose();
$transposedRight = CudaArray::ones([128, 128]);
$cases['matmul-transposed-128'] = [static fn() => $transposed->matmul($transposedRight), 128, 128 * 128, 'matmul'];
$strided = CudaArray::ones([64, 128])->slice(':, ::2');
$stridedRight = CudaArray::ones([64, 64]);
$cases['matmul-strided-64'] = [static fn() => $strided->matmul($stridedRight), 64, 64 * 64, 'matmul'];
$batched = CudaArray::ones([4, 64, 64]);
$cases['matmul-batched-4x64'] = [static fn() => $batched->matmul($batched), 64, 4 * 64 * 64, 'matmul'];
$broadcast = CudaArray::ones([1, 64, 64]);
$cases['matmul-broadcast-4x64'] = [static fn() => $broadcast->matmul($batched), 64, 4 * 64 * 64, 'matmul'];
$a = CudaArray::ones([1, 100000]);
$b = CudaArray::ones([100000, 1]);
$cases['matmul-dot-100000'] = [static fn() => $a->matmul($b), 100000, 1, 'matmul'];
$report = [
    'php' => PHP_VERSION, 'gpu' => cuda_get_device_info(),
    'runtime' => cuda_get_runtime_version(), 'driver' => cuda_get_driver_version(),
    'precision' => $precision,
    'generatedAt' => gmdate(DATE_ATOM),
    'binarySha256' => isset($options['binary']) ? hash_file('sha256', $options['binary']) : null,
    'benchmarkSha256' => hash_file('sha256', __FILE__),
    'timing' => 'CPU wall clock, warm resident inputs, allocations and final synchronization included; destruction excluded; 5 warmups/25 samples.',
    'cases' => [],
];
foreach ($cases as $name => $case) {
    [$operation, $expected, $count] = $case;
    $isMatmul = ($case[3] ?? null) === 'matmul';
    cuda_synchronize();
    $result = $operation();
    cuda_synchronize();
    $bytes = $result->toBuffer();
    if (strlen($bytes) !== $count * 4) throw new RuntimeException("$name output shape mismatch");
    for ($offset = 0; $offset < $count; $offset += 65536) {
        $chunkCount = min(65536, $count - $offset);
        foreach (unpack('g*', substr($bytes, $offset * 4, $chunkCount * 4)) as $value) {
            if (!is_finite($value) || abs($value - $expected) > 1e-5 + 1e-4 * abs($expected)) {
                throw new RuntimeException("$name numerical mismatch");
            }
        }
    }
    unset($result, $bytes);
    for ($i = 0; $i < 5; $i++) { $result = $operation(); cuda_synchronize(); unset($result); }
    $samples = [];
    for ($i = 0; $i < 25; $i++) {
        cuda_synchronize();
        $start = hrtime(true);
        $result = $operation();
        cuda_synchronize();
        $samples[] = (hrtime(true) - $start) / 1e6;
        unset($result);
    }
    $sorted = $samples;
    sort($sorted);
    $row = ['name' => $name, 'medianMs' => $sorted[12], 'samplesMs' => $samples];
    if ($isMatmul) {
        $row['backend'] = cuda_get_backend_info()['lastMatmul'];
    }
    $report['cases'][] = $row;
    printf("%-28s %.4f ms%s\n", $name, $row['medianMs'],
        $isMatmul ? ' (' . $row['backend'] . ')' : '');
}
if (function_exists('cuda_get_backend_info')) $report['backends'] = cuda_get_backend_info();
if (isset($options['output']) && file_put_contents($options['output'],
    json_encode($report, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL) === false) {
    throw new RuntimeException('Cannot save benchmark report.');
}
