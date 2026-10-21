--TEST--
Transfer benchmark validates raw bytes and reports isolated allocation-inclusive latency, throughput and memory
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (PHP_OS_FAMILY !== 'Linux') die('skip Linux RSS accounting required');
if (!function_exists('proc_open')) die('skip proc_open required');
if (getenv('USE_ZEND_ALLOC') === '0') die('skip Zend allocation accounting required');
?>
--FILE--
<?php
function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}
function benchmark(array $changes = []): array
{
    $options = array_replace([
        'binary' => realpath(__DIR__ . '/../modules/phptensors.so'),
        'revision' => 'benchmark-contract-fixture',
        'dtypes' => 'float16,uint64', 'sizes' => '16', 'warmups' => '0', 'samples' => '2',
    ], $changes);
    $command = [PHP_BINARY, '-n', __DIR__ . '/benchmark_transfers.php'];
    foreach ($options as $name => $value) $command[] = "--$name=$value";
    $process = proc_open($command, [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
    check(is_resource($process), 'start benchmark subprocess');
    $output = stream_get_contents($pipes[1]);
    $errors = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    return [proc_close($process), $output, $errors];
}
[$status, $output, $errors] = benchmark();
check($status === 0 && $errors === '', "benchmark failed: $errors");
$report = json_decode($output, true, 512, JSON_THROW_ON_ERROR);
check($report['schema_version'] === 1 && $report['mode'] === 'cpu', 'schema and mode');
check($report['source_revision'] === 'benchmark-contract-fixture', 'explicit source revision');
foreach (['binary_sha256', 'benchmark_sha256', 'fixture_sha256'] as $key) {
    check((bool)preg_match('/^[0-9a-f]{64}$/D', $report[$key]), "$key fingerprint");
}
check(count($report['cases']) === 10, 'two dtypes across five layouts');
check(count(array_unique(array_column($report['cases'], 'process_id'))) === 10, 'each case uses a fresh worker');
foreach ($report['cases'] as $case) {
    $width = $case['dtype'] === 'float16' ? 2 : 8;
    $elements = $case['layout'] === 'strided' ? 8 : 16;
    check($case['logical_elements'] === $elements && $case['payload_bytes'] === $elements * $width, 'logical payload');
    check($case['process_peak_rss_bytes'] > 0 && $case['process_peak_zend_bytes'] > 0, 'case memory peaks');
    check(!$case['pinned'] && !isset($case['device_memory_before']), 'CPU report never claims GPU evidence');
    check(count($case['operations']) === ($case['layout'] === 'contiguous' ? 2 : 1), 'expected operations');
    foreach ($case['operations'] as $measurement) {
        $sorted = $measurement['warm_samples_ms'];
        sort($sorted);
        check(count($sorted) === 2 && $measurement['validated'] === true, 'checked raw samples');
        check(($measurement['first_sample_ms'] > 0 && !$measurement['first_sample_below_clock_resolution']) ||
            ($measurement['first_sample_ms'] === null && $measurement['first_sample_below_clock_resolution']),
            'first sample is resolved or explicitly unmeasurable');
        check($measurement['iterations_per_warm_sample'] >= 1 && $measurement['calibration_window_ms'] >= 1.0,
            'batch duration resolves clock precision');
        $median = ($sorted[0] + $sorted[1]) / 2;
        check($measurement['warm_median_ms'] === $median, 'even sample median');
        check($measurement['warm_p95_ms'] === $sorted[1], 'nearest-rank p95');
        $rate = $case['payload_bytes'] / ($median / 1000);
        check(abs($measurement['warm_payload_bytes_per_second'] - $rate) <= $rate * 1e-12, 'throughput uses logical payload');
        check($measurement['first_output_live_zend_allocation_bytes'] >= $case['payload_bytes'], 'output allocation counted');
    }
}
[$status, $output, $errors] = benchmark([
    'dtypes' => 'float32', 'sizes' => '8', 'layouts' => 'contiguous', 'samples' => '3', 'warmups' => '1',
]);
check($status === 0 && $errors === '', "odd-sample benchmark failed: $errors");
$odd = json_decode($output, true, 512, JSON_THROW_ON_ERROR);
foreach ($odd['cases'][0]['operations'] as $measurement) {
    $sorted = $measurement['warm_samples_ms'];
    sort($sorted);
    check(count($sorted) === 3 && $measurement['warm_median_ms'] === $sorted[1], 'odd sample median');
    check($measurement['warm_p95_ms'] === $sorted[2], 'odd sample p95');
}
[$status, $output, $errors] = benchmark([
    'dtypes' => 'float32', 'sizes' => '1048576', 'layouts' => 'transpose', 'samples' => '1',
]);
check($status === 0 && $errors === '', "large packed benchmark failed: $errors");
$large = json_decode($output, true, 512, JSON_THROW_ON_ERROR)['cases'][0];
check($large['payload_bytes'] === 4194304, 'large case has a four MiB logical payload');
check($large['process_peak_zend_bytes'] - $large['resident_zend_bytes_before_measurements'] <=
    3 * $large['payload_bytes'] + 1048576, 'packed export stays below PHP scalar-array expansion budget');
echo "Byte checks, payload sizes, raw samples, median/p95, throughput and memory shape verified\n";
foreach ([
    ['samples' => '0'], ['warmups' => '-1'], ['sizes' => '9'], ['sizes' => '0'],
    ['dtypes' => 'invalid'], ['layouts' => 'invalid'], ['mode' => 'invalid'],
    ['revision' => ''], ['unknown' => '1'],
] as $invalid) {
    [$status, $output, $errors] = benchmark($invalid);
    check($status !== 0 && $output === '' && $errors !== '', 'invalid CLI must fail without success-shaped JSON');
}
$originalAllocator = getenv('USE_ZEND_ALLOC');
putenv('USE_ZEND_ALLOC=0');
[$status, $output, $errors] = benchmark(['layouts' => 'contiguous', 'dtypes' => 'float32']);
putenv($originalAllocator === false ? 'USE_ZEND_ALLOC' : 'USE_ZEND_ALLOC=' . $originalAllocator);
check($status !== 0 && $output === '' && str_contains($errors, 'Enable Zend allocation accounting'),
    'disabled accounting must fail without misleading memory metrics');
echo "Invalid benchmark inputs fail explicitly before emitting a report\n";
?>
--EXPECT--
Byte checks, payload sizes, raw samples, median/p95, throughput and memory shape verified
Invalid benchmark inputs fail explicitly before emitting a report
