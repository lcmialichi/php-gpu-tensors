--TEST--
Lazy CUDA pool validates sizes without affecting CPU startup or allowing runtime INI changes
--SKIPIF--
<?php
if (!extension_loaded('phptensors') || !class_exists('PhpTensors\\NN', false)) die('skip CUDA backend unavailable');
if (!function_exists('proc_open')) die('skip proc_open required');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}
$original = ini_get('phptensors.cuda.memory_pool_size');
$warnings = [];
set_error_handler(function ($level, $message) use (&$warnings) {
    $warnings[] = $message;
    return true;
});
$changed = ini_set('phptensors.cuda.memory_pool_size', '16M');
restore_error_handler();
check($changed === false && ini_get('phptensors.cuda.memory_pool_size') === $original, 'runtime INI must remain unchanged');
check(count($warnings) === 1 && str_contains($warnings[0], 'has no effect'), 'runtime INI warning');
check(PhpTensors\CudaArray::ones([1])->toArray() === [1.0], 'first allocation after rejected INI change');
echo "runtime configuration is immutable\n";

$extension = realpath(__DIR__ . '/../modules/phptensors.so');
$script = <<<'PHP'
$host = new PhpTensors\HostArray([1, 2]);
if ($host->toArray() !== [1.0, 2.0]) throw new RuntimeException('CPU startup');
try {
    if (PhpTensors\CudaArray::ones([1])->toArray() !== [1.0]) throw new RuntimeException('GPU result');
    echo 'valid';
} catch (PhpTensors\InvalidArgumentException $error) {
    if (!str_contains($error->getMessage(), 'CUDA memory pool size')) throw $error;
    echo 'invalid';
}
PHP;
foreach (['', '0', '-1', '0G', '1T', '1Gjunk', '18446744073709551616', '18446744073709551615G',
          '18446744073709551615',
          '16M', '16384K', '16777216', '16m', '1G'] as $index => $size) {
    $process = proc_open([PHP_BINARY, '-n', '-d', 'extension=' . $extension,
        '-d', 'phptensors.cuda.memory_pool_size=' . $size, '-r', $script],
        [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
    check(is_resource($process), 'starting PHP subprocess');
    $output = stream_get_contents($pipes[1]);
    $errors = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    check(proc_close($process) === 0 && $errors === '', "pool configuration subprocess: $size: $errors");
    check($output === ($index < 9 ? 'invalid' : 'valid'), "pool configuration result: $size: $output");
}
echo "empty, negative, overflow and malformed sizes rejected; byte and K/M/G syntax validated\n";
?>
--EXPECT--
runtime configuration is immutable
empty, negative, overflow and malformed sizes rejected; byte and K/M/G syntax validated
