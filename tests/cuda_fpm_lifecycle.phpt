--TEST--
PHP-FPM CUDA workers recycle and reset request state with pending Fusion work
--SKIPIF--
<?php
if (!extension_loaded('phptensors') || !class_exists('PhpTensors\\NN', false)) die('skip CUDA backend unavailable');
if (PHP_OS_FAMILY !== 'Linux' || !function_exists('proc_open')) die('skip Linux and proc_open required');
$fpm = getenv('TEST_PHP_FPM_EXECUTABLE');
if (!$fpm || !is_executable($fpm)) die('skip set TEST_PHP_FPM_EXECUTABLE to a matching PHP-FPM binary');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}
function readExact($stream, int $bytes): string
{
    $result = '';
    while (strlen($result) < $bytes) {
        $chunk = fread($stream, $bytes - strlen($result));
        check($chunk !== false && $chunk !== '', 'truncated or timed out FastCGI response');
        $result .= $chunk;
    }
    return $result;
}
function record($stream, int $type, string $data): void
{
    $record = pack('CCnnCC', 1, $type, 1, strlen($data), 0, 0) . $data;
    while ($record !== '') {
        $written = fwrite($stream, $record);
        check($written !== false && $written > 0, 'FastCGI write');
        $record = substr($record, $written);
    }
}
function request(string $socket, string $script, bool $cpuOnly): array
{
    $stream = stream_socket_client('unix://' . $socket, $errno, $error, 10);
    check(is_resource($stream), "FastCGI connect: $error");
    stream_set_timeout($stream, 20);
    try {
        record($stream, 1, pack('nCxxxxx', 1, 0));
        $params = '';
        foreach (['SCRIPT_FILENAME' => $script, 'REQUEST_METHOD' => 'GET',
                  'QUERY_STRING' => $cpuOnly ? 'cpu=1' : '',
                  'SERVER_PROTOCOL' => 'HTTP/1.1', 'CONTENT_LENGTH' => '0'] as $key => $value) {
            check(strlen($key) < 128 && strlen($value) < 128, 'FastCGI parameter length');
            $params .= chr(strlen($key)) . chr(strlen($value)) . $key . $value;
        }
        record($stream, 4, $params);
        record($stream, 4, '');
        record($stream, 5, '');
        $output = '';
        $errors = '';
        do {
            $header = unpack('Cversion/Ctype/nid/nlength/Cpadding/Creserved', readExact($stream, 8));
            $data = readExact($stream, $header['length']);
            readExact($stream, $header['padding']);
            if ($header['type'] === 6) $output .= $data;
            if ($header['type'] === 7) $errors .= $data;
        } while ($header['type'] !== 3);
        check($errors === '', 'FastCGI errors: ' . $errors);
        $parts = explode("\r\n\r\n", $output, 2);
        check(count($parts) === 2, 'FastCGI HTTP headers: ' . $output);
        $result = json_decode($parts[1], true, 512, JSON_THROW_ON_ERROR);
        check(!isset($result['error']), 'worker error: ' . ($result['error'] ?? ''));
        return $result;
    } finally {
        fclose($stream);
    }
}

$directory = sys_get_temp_dir() . '/phptensors-fpm-' . getmypid();
check(mkdir($directory, 0700), 'temporary FPM directory');
$socket = $directory . '/fpm.sock';
$script = $directory . '/request.php';
$config = $directory . '/fpm.conf';
$extension = realpath(__DIR__ . '/../modules/phptensors.so');
$user = fileowner($directory);
$group = filegroup($directory);
$workerCode = <<<'PHP'
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;
function verify(bool $ok, string $message): void {
    if (!$ok) throw new RuntimeException($message);
}
set_error_handler(function ($level, $message) { throw new ErrorException($message, 0, $level); });
try {
    verify(Fusion::getCacheStats()['maxEntries'] === 16, 'request-local Fusion metadata');
    if (isset($_GET['cpu'])) {
        verify((new PhpTensors\HostArray([1, 2]))->toArray() === [1.0, 2.0], 'CPU-only request');
        echo json_encode(['cpu' => true], JSON_THROW_ON_ERROR);
        return;
    }
    $before = cuda_get_backend_info();
    verify($before['precision'] === 'fp32-strict', 'precision leaked between requests');
    foreach (['cublasCalls', 'cublasLtCalls', 'builtinMatmulCalls',
              'cubReductionCalls', 'coalescedReductionCalls', 'genericReductionCalls'] as $counter) {
        verify($before[$counter] === 0, 'backend counter leaked: ' . $counter);
    }
    verify(Fusion::getCacheStats()['entries'] === 0, 'Fusion cache leaked between requests');
    verify(cuda_set_device(0), 'pending execution leaked between requests');
    $tensor = CudaArray::ones([512, 512]);
    verify($tensor->matmul($tensor)->toArray()[0][0] === 512.0, 'worker matmul');
    $graph = Fusion::compile(fn($x) => ($x + 1)->sum(), [$tensor], cudaGraph: true);
    verify($graph->run($tensor)->toArray() === [524288.0], 'worker graph reduction');
    $info = cuda_get_device_info();
    $tf32 = $before['cublas'] && $info['compute_capability_major'] >= 8;
    if ($tf32) cuda_set_matmul_precision('tf32');
    $pending = $graph->runAsync($tensor);
    verify($graph->getStats()['pending'] === 1, 'work must remain pending until request shutdown');
    $memory = cuda_get_memory_info();
    $status = file_get_contents('/proc/self/status');
    verify(preg_match('/^VmRSS:\s+(\d+)\s+kB/m', $status, $matches) === 1, 'worker RSS');
    echo json_encode(['pid' => getmypid(), 'rss' => (int)$matches[1] * 1024,
        'free' => $memory['free_memory'], 'tf32' => $tf32], JSON_THROW_ON_ERROR);
} catch (Throwable $error) {
    http_response_code(500);
    echo json_encode(['error' => (string)$error], JSON_THROW_ON_ERROR);
}
PHP;
$configuration = <<<FPM
[global]
pid = $directory/fpm.pid
error_log = $directory/fpm.log
daemonize = no
[cuda]
listen = $socket
user = $user
group = $group
pm = static
pm.max_children = 1
pm.max_requests = 8
clear_env = no
catch_workers_output = yes
request_terminate_timeout = 20s
php_admin_flag[log_errors] = on
php_admin_value[error_log] = $directory/php.log
FPM;
$process = null;
try {
    check(file_put_contents($script, $workerCode) !== false, 'request fixture');
    check(file_put_contents($config, $configuration) !== false, 'FPM configuration');
    $process = proc_open([getenv('TEST_PHP_FPM_EXECUTABLE'), '-n', '-d', 'extension=' . $extension,
        '-d', 'phptensors.cuda.memory_pool_size=64M', '-F', '-R', '-y', $config],
        [0 => ['file', '/dev/null', 'r'], 1 => ['file', $directory . '/stdout.log', 'w'],
         2 => ['file', $directory . '/stderr.log', 'w']], $pipes);
    check(is_resource($process), 'starting PHP-FPM');
    for ($attempt = 0; $attempt < 100 && !file_exists($socket); $attempt++) {
        check(proc_get_status($process)['running'], 'PHP-FPM exited during startup');
        usleep(50000);
    }
    check(file_exists($socket), 'PHP-FPM did not become responsive');
    $workers = [];
    for ($i = 0; $i < 32; $i++) {
        $cpuOnly = $i % 8 < 2;
        $result = request($socket, $script, $cpuOnly);
        if ($cpuOnly) {
            check(($result['cpu'] ?? false) === true, 'CPU-only response');
            continue;
        }
        $workers[$result['pid']][] = $result;
    }
    check(count($workers) === 4, 'expected four recycled workers');
    foreach ($workers as $requests) {
        check(count($requests) === 6, 'expected six requests per worker');
        $warm = $requests[1];
        foreach (array_slice($requests, 2) as $result) {
            check($result['rss'] <= $warm['rss'] + 32 * 1024 * 1024, 'worker host memory keeps growing');
            check($result['free'] >= $warm['free'] - 64 * 1024 * 1024, 'worker GPU memory keeps growing');
        }
    }
    echo "24 GPU and eight CPU-only requests completed across four recycled FPM workers\n";
    echo "precision, counters, cache and pending work isolated; warm memory bounded\n";
} finally {
    if (is_resource($process)) {
        if (proc_get_status($process)['running']) proc_terminate($process, 3 /* SIGQUIT */);
        proc_close($process);
    }
    foreach (['php.log', 'fpm.log', 'stderr.log'] as $name) {
        if (file_exists($directory . '/' . $name)) {
            $log = file_get_contents($directory . '/' . $name);
            if (preg_match('/(?:PHP (?:Warning|Fatal)|segfault|SIGSEGV|exited on signal)/i', $log))
                echo "FPM failure log: $log\n";
        }
    }
    foreach (['request.php', 'fpm.conf', 'fpm.sock', 'fpm.pid', 'fpm.log', 'php.log', 'stdout.log', 'stderr.log'] as $name) {
        if (file_exists($directory . '/' . $name)) unlink($directory . '/' . $name);
    }
    rmdir($directory);
}
?>
--EXPECT--
24 GPU and eight CPU-only requests completed across four recycled FPM workers
precision, counters, cache and pending work isolated; warm memory bounded
