<?php
declare(strict_types=1);

use PhpTensors\CudaArray;
use PhpTensors\HostArray;

ini_set('display_errors', 'stderr');

function transferProcess(array $command): string
{
    $process = proc_open($command, [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
    if (!is_resource($process)) throw new RuntimeException('Cannot start transfer benchmark worker');
    $output = stream_get_contents($pipes[1]);
    $errors = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    $status = proc_close($process);
    if ($status !== 0 || $errors !== '') {
        throw new RuntimeException("Transfer benchmark worker failed (exit $status): $errors");
    }
    return $output;
}

function transferInteger(string $value, string $name, int $min, int $max): int
{
    $number = filter_var($value, FILTER_VALIDATE_INT);
    if (!is_int($number) || $number < $min || $number > $max) {
        throw new InvalidArgumentException("$name must be an integer between $min and $max");
    }
    return $number;
}

function transferFingerprint(string $path): string
{
    $hash = hash_file('sha256', $path);
    if ($hash === false) throw new RuntimeException("Cannot fingerprint $path");
    return $hash;
}

function transferMeasure(callable $operation, string $expected, array $shape, string $dtype,
    int $warmups, int $samples, bool $gpu): array
{
    $validate = static function (mixed $result) use ($expected, $shape, $dtype): void {
        $bytes = is_string($result) ? $result : $result->toBuffer();
        if ($bytes !== $expected ||
            (!is_string($result) && ($result->getShape() !== $shape || $result->dtype() !== $dtype))) {
            throw new RuntimeException('Transfer benchmark correctness check failed');
        }
    };
    if ($gpu) cuda_synchronize();
    $before = memory_get_usage(false);
    $start = hrtime(true);
    $result = $operation();
    if ($gpu) cuda_synchronize();
    $first = (hrtime(true) - $start) / 1e6;
    $liveBytes = memory_get_usage(false) - $before;
    if ($first < 0) throw new RuntimeException('Transfer clock moved backwards');
    $validate($result);
    unset($result);
    $batch = static function (int $iterations) use ($operation, $gpu): float {
        if ($gpu) cuda_synchronize();
        $start = hrtime(true);
        for ($i = 0; $i < $iterations; $i++) {
            $result = $operation();
            if ($gpu) cuda_synchronize();
            unset($result);
        }
        return (hrtime(true) - $start) / 1e6;
    };
    $iterations = 1;
    while (($calibrationMs = $batch($iterations)) < 1.0) {
        if ($calibrationMs < 0 || $iterations >= 1048576) {
            throw new RuntimeException('Cannot resolve a one-millisecond transfer timing window');
        }
        $iterations *= 2;
    }
    for ($i = 0; $i < $warmups; $i++) $batch($iterations);
    $times = [];
    for ($i = 0; $i < $samples; $i++) {
        $elapsed = $batch($iterations);
        if ($elapsed <= 0) throw new RuntimeException('Nonpositive transfer timing window');
        $times[] = $elapsed / $iterations;
    }
    $result = $operation();
    if ($gpu) cuda_synchronize();
    $validate($result);
    unset($result);
    $sorted = $times;
    sort($sorted);
    $middle = intdiv($samples, 2);
    $median = $samples % 2 ? $sorted[$middle] : ($sorted[$middle - 1] + $sorted[$middle]) / 2;
    return [
        'validated' => true,
        'first_sample_ms' => $first > 0 ? $first : null,
        'first_sample_below_clock_resolution' => $first === 0.0,
        'iterations_per_warm_sample' => $iterations,
        'calibration_window_ms' => $calibrationMs,
        'warm_samples_ms' => $times,
        'warm_median_ms' => $median,
        'warm_p95_ms' => $sorted[(int)ceil($samples * 0.95) - 1],
        'warm_payload_bytes_per_second' => strlen($expected) / ($median / 1000),
        'first_output_live_zend_allocation_bytes' => $liveBytes,
    ];
}

function transferCase(array $case): array
{
    if (!extension_loaded('phptensors')) throw new RuntimeException('Load the phptensors extension');
    if (getenv('USE_ZEND_ALLOC') === '0') {
        throw new RuntimeException('Enable Zend allocation accounting for transfer measurements');
    }
    if (PHP_OS_FAMILY !== 'Linux') throw new RuntimeException('Transfer RSS accounting requires Linux');
    $gpuMode = $case['mode'] === 'gpu';
    if ($gpuMode && (!class_exists(CudaArray::class, false) || !class_exists('PhpTensors\\NN', false))) {
        throw new RuntimeException('GPU transfer measurement requires a CUDA-enabled build');
    }
    if ($gpuMode && cuda_get_device_count() < 1) {
        throw new RuntimeException('GPU transfer measurement requires a visible CUDA device');
    }
    $fixtures = require __DIR__ . '/fixtures/transfer_contract.php';
    $pattern = $fixtures['patterns'][$case['dtype']];
    $width = intdiv(strlen($pattern), 8);
    $rows = intdiv($case['elements'], 8);
    $buffer = str_repeat($pattern, $rows);
    $shape = [$rows, 8];
    $expected = $buffer;
    $makeView = static fn($x) => $x;
    switch ($case['layout']) {
        case 'offset':
            $buffer = str_repeat("\0", strlen($pattern)) . $buffer;
            $shape = [$rows + 1, 8];
            $makeView = static fn($x) => $x->slice([1, null]);
            break;
        case 'transpose':
            $makeView = static fn($x) => $x->transpose();
            $expected = '';
            for ($i = 0; $i < 8; $i++) $expected .= str_repeat(substr($pattern, $i * $width, $width), $rows);
            break;
        case 'strided':
            $makeView = static fn($x) => $x->slice(':, 1:8:2');
            $row = '';
            foreach ([1, 3, 5, 7] as $i) $row .= substr($pattern, $i * $width, $width);
            $expected = str_repeat($row, $rows);
            break;
        case 'broadcast':
            $buffer = $pattern;
            $shape = [1, 8];
            $makeView = static fn($x) => $x->broadcastTo([$rows, 8]);
            break;
    }
    $host = $makeView(HostArray::fromBuffer($buffer, $shape, $case['dtype'], $case['pinned']));
    $logicalShape = $host->getShape();
    $operations = [
        'host_buffer_export' => static fn() => $host->toBuffer(),
    ];
    if ($case['layout'] === 'contiguous') {
        $operations['host_buffer_import'] = static fn() =>
            HostArray::fromBuffer($buffer, $shape, $case['dtype'], $case['pinned']);
    }
    $environment = [
        'php' => PHP_VERSION, 'zts' => (bool)PHP_ZTS,
        'extension_version' => phpversion('phptensors'),
        'cuda_build' => class_exists('PhpTensors\\NN', false),
        'os' => php_uname('s'), 'kernel' => php_uname('r'), 'architecture' => php_uname('m'),
    ];
    if ($gpuMode) {
        $device = $makeView(CudaArray::fromBuffer($buffer, $shape, $case['dtype']));
        $operations['upload'] = static fn() => $host->toGpu();
        $operations['download'] = static fn() => $device->toHost();
        $operations['device_buffer_export'] = static fn() => $device->toBuffer();
        $environment['gpu'] = cuda_get_device_info();
        $environment['runtime'] = cuda_get_runtime_version();
        $environment['driver'] = cuda_get_driver_version();
        $environment['memory_pool_size'] = ini_get('phptensors.cuda.memory_pool_size');
        $case['device_memory_before'] = cuda_get_memory_info();
    }
    $case['shape'] = $logicalShape;
    $case['logical_elements'] = $host->getSize();
    $case['payload_bytes'] = strlen($expected);
    $case['source_buffer_bytes'] = strlen($buffer);
    $case['process_id'] = getmypid();
    $case['resident_zend_bytes_before_measurements'] = memory_get_usage(false);
    foreach ($operations as $name => $operation) {
        $case['operations'][$name] = transferMeasure($operation, $expected, $logicalShape,
            $case['dtype'], $case['warmups'], $case['samples'], $gpuMode);
    }
    if ($gpuMode) $case['device_memory_after'] = cuda_get_memory_info();
    $usage = getrusage();
    if (!is_array($usage) || !isset($usage['ru_maxrss'])) {
        throw new RuntimeException('Cannot read process peak RSS');
    }
    $case['process_peak_rss_bytes'] = $usage['ru_maxrss'] * 1024;
    $case['process_peak_zend_bytes'] = memory_get_peak_usage(false);
    return ['environment' => $environment, 'case' => $case];
}

if (count($argv) === 2 && str_starts_with($argv[1], '--worker=')) {
    $case = json_decode(substr($argv[1], 9), true, 512, JSON_THROW_ON_ERROR);
    echo json_encode(transferCase($case), JSON_THROW_ON_ERROR | JSON_PRESERVE_ZERO_FRACTION), "\n";
    exit(0);
}

if ($argc === 2 && $argv[1] === '--help') {
    echo "Usage: php -n tests/benchmark_transfers.php --binary=MODULE --revision=COMMIT_OR_TAG\n",
        "  [--mode=cpu|gpu] [--dtypes=float32,bfloat16,...] [--sizes=1024,65536,1048576]\n",
        "  [--layouts=contiguous,offset,transpose,strided,broadcast] [--warmups=2] [--samples=7]\n",
        "Sizes are source element counts, multiples of eight. JSON is written to stdout.\n";
    exit(0);
}

$options = [
    'mode' => 'cpu', 'sizes' => '1024,65536,1048576',
    'layouts' => 'contiguous,offset,transpose,strided,broadcast', 'warmups' => '2', 'samples' => '7',
];
foreach (array_slice($argv, 1) as $argument) {
    if (!preg_match('/^--(binary|revision|mode|sizes|dtypes|layouts|warmups|samples)=(.+)$/D', $argument, $match)) {
        throw new InvalidArgumentException("Invalid option: $argument; use --help");
    }
    $options[$match[1]] = $match[2];
}
$binary = isset($options['binary']) ? realpath($options['binary']) : false;
if ($binary === false || !is_file($binary)) throw new InvalidArgumentException('--binary must name an extension file');
if (!isset($options['revision']) || trim($options['revision']) === '') {
    throw new InvalidArgumentException('--revision must identify the source commit or tag');
}
if (!in_array($options['mode'], ['cpu', 'gpu'], true)) throw new InvalidArgumentException('--mode must be cpu or gpu');
$warmups = transferInteger($options['warmups'], '--warmups', 0, 500);
$samples = transferInteger($options['samples'], '--samples', 1, 500);
$fixtures = require __DIR__ . '/fixtures/transfer_contract.php';
$dtypes = isset($options['dtypes']) ? explode(',', $options['dtypes']) : array_keys($fixtures['patterns']);
foreach ($dtypes as $dtype) {
    if (!isset($fixtures['patterns'][$dtype])) throw new InvalidArgumentException("Unknown dtype: $dtype");
}
$layouts = explode(',', $options['layouts']);
foreach ($layouts as $layout) {
    if (!in_array($layout, ['contiguous', 'offset', 'transpose', 'strided', 'broadcast'], true)) {
        throw new InvalidArgumentException("Unknown layout: $layout");
    }
}
$sizes = [];
foreach (explode(',', $options['sizes']) as $size) {
    $elements = transferInteger($size, '--sizes', 8, 16777216);
    if ($elements % 8 !== 0) throw new InvalidArgumentException('--sizes must be multiples of eight');
    $sizes[] = $elements;
}
$report = [
    'schema_version' => 1, 'generated_at' => gmdate(DATE_ATOM),
    'source_revision' => $options['revision'], 'revision_source' => 'user-supplied',
    'binary_sha256' => transferFingerprint($binary),
    'benchmark_sha256' => transferFingerprint(__FILE__),
    'fixture_sha256' => transferFingerprint(__DIR__ . '/fixtures/transfer_contract.php'),
    'command' => array_merge([PHP_BINARY], $argv),
    'mode' => $options['mode'],
    'methodology' => [
        'isolation' => 'Fresh PHP process per dtype/size/layout/pinning case; inputs prepared before timing.',
        'timing' => 'Wall clock includes output allocation and GPU completion. Warm batches also include output destruction and PHP loop overhead. Correctness checks are outside timing.',
        'first_sample' => 'First measured call after source/runtime setup; not cold CUDA initialization or a pristine allocator.',
        'warm_reuse' => 'Same resident inputs reused; each call allocates and releases a result. Batches calibrated to at least 1 ms; raw samples and p95 are per-call batch means, not individual-call tail latency.',
        'destination' => 'No preallocated-destination API is measured.',
        'memory' => 'Peak RSS/Zend cover the whole isolated case, including inputs, validation and timing; live allocation is output-resident Zend delta, not peak scratch.',
        'device_memory' => 'Before/after device snapshots only, not device high-water memory or pool allocation peaks.',
    ],
    'cases' => [],
];
foreach ($dtypes as $dtype) foreach ($sizes as $elements) foreach ($layouts as $layout) {
    foreach ($options['mode'] === 'gpu' ? [false, true] : [false] as $pinned) {
        $case = compact('dtype', 'elements', 'layout', 'pinned', 'warmups', 'samples') + ['mode' => $options['mode']];
        $worker = json_decode(transferProcess([PHP_BINARY, '-n', '-d', 'memory_limit=-1',
            '-d', 'extension=' . $binary, __FILE__,
            '--worker=' . json_encode($case, JSON_THROW_ON_ERROR)]), true, 512, JSON_THROW_ON_ERROR);
        $report['environment'] = $worker['environment'];
        $report['cases'][] = $worker['case'];
    }
}
echo json_encode($report, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR | JSON_PRESERVE_ZERO_FRACTION), "\n";
