<?php
declare(strict_types=1);

use Cuda\CudaArray;

if (!extension_loaded('cuda') || cuda_get_device_count() < 1) {
    throw new RuntimeException('CUDA extension and GPU required.');
}
$options = getopt('', ['output:', 'binary:']);
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
foreach ([64, 128, 512] as $size) {
    $a = CudaArray::ones([$size, $size]);
    $cases["matmul-$size"] = [static fn() => $a->matmul($a), $size, $size * $size];
}
$a = CudaArray::ones([1, 100000]);
$b = CudaArray::ones([100000, 1]);
$cases['dot-100000'] = [static fn() => $a->matmul($b), 100000, 1];
$report = [
    'php' => PHP_VERSION, 'gpu' => cuda_get_device_info(),
    'runtime' => cuda_get_runtime_version(), 'driver' => cuda_get_driver_version(),
    'generatedAt' => gmdate(DATE_ATOM),
    'binarySha256' => isset($options['binary']) ? hash_file('sha256', $options['binary']) : null,
    'benchmarkSha256' => hash_file('sha256', __FILE__),
    'timing' => 'CPU wall clock, warm resident inputs, allocations and final synchronization included; destruction excluded; 5 warmups/25 samples.',
    'cases' => [],
];
foreach ($cases as $name => [$operation, $expected, $count]) {
    cuda_synchronize();
    $result = $operation();
    cuda_synchronize();
    $bytes = $result->toBuffer();
    if (strlen($bytes) !== $count * 4) throw new RuntimeException("$name output shape mismatch");
    foreach (unpack('g*', $bytes) as $value)
        if (!is_finite($value) || abs($value - $expected) > 1e-5 + 1e-4 * abs($expected))
            throw new RuntimeException("$name numerical mismatch");
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
    $report['cases'][] = $row;
    printf("%-18s %.4f ms\n", $name, $row['medianMs']);
}
if (function_exists('cuda_get_backend_info')) $report['backends'] = cuda_get_backend_info();
if (isset($options['output']) && file_put_contents($options['output'],
    json_encode($report, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL) === false) {
    throw new RuntimeException('Cannot save benchmark report.');
}
