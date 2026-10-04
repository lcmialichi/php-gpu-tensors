<?php
declare(strict_types=1);

use Cuda\CudaArray;
use Cuda\Fusion;

if (!extension_loaded('cuda') || cuda_get_device_count() < 1) {
    throw new RuntimeException('This example requires the CUDA extension and a visible GPU.');
}

$options = getopt('', ['benchmark', 'elements:', 'iterations:']);
$elements = filter_var($options['elements'] ?? 65536, FILTER_VALIDATE_INT);
$iterations = filter_var($options['iterations'] ?? 100, FILTER_VALIDATE_INT);
if ($elements === false || $elements < 1 || $iterations === false || $iterations < 1) {
    throw new InvalidArgumentException('elements and iterations must be positive integers.');
}

$a = CudaArray::full([$elements], 1.0);
$b = CudaArray::full([$elements], 2.0);
$c = CudaArray::full([$elements], 3.0);
$expression = static fn($a, $b, $c) => ($a + $b * $c)->sqrt()->neg();

Fusion::clearCache();
$start = hrtime(true);
$stream = Fusion::compile($expression, [$a, $b, $c]);
$coldCompileMs = (hrtime(true) - $start) / 1e6;
$start = hrtime(true);
$cudaGraph = Fusion::compile($expression, [$a, $b, $c], cudaGraph: true);
$cachedCompileMs = (hrtime(true) - $start) / 1e6;
$execution = $cudaGraph->runAsync($a, $b, $c);
$result = $execution->wait();

if (!array_key_exists('benchmark', $options)) {
    echo 'First result: ', $result[0], PHP_EOL;
    echo json_encode($cudaGraph->getStats(), JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR), PHP_EOL;
    exit(0);
}

$eager = $expression($a, $b, $c);
if ($eager->toHost()->toBuffer() !== $result->toHost()->toBuffer()) {
    throw new RuntimeException('Fused output differs from the eager reference.');
}
unset($eager, $execution, $result);

function elapsedPerIteration(callable $operation, int $iterations): float
{
    for ($i = 0; $i < 3; $i++) {
        $operation();
    }
    $start = hrtime(true);
    for ($i = 0; $i < $iterations; $i++) {
        $operation();
    }
    return (hrtime(true) - $start) / $iterations / 1e3;
}

$modes = [
    'eager' => static fn() => $expression($a, $b, $c),
    'scoped_cached' => static fn() => Fusion::run(static fn() => $expression($a, $b, $c)),
    'compiled_stream' => static fn() => $stream->run($a, $b, $c),
    'cuda_graph' => static fn() => $cudaGraph->run($a, $b, $c),
];
$timings = [];
foreach ($modes as $name => $operation) {
    $timings[$name] = elapsedPerIteration($operation, $iterations);
}
echo json_encode([
    'php' => PHP_VERSION,
    'zts' => PHP_ZTS,
    'device' => cuda_get_device_info(),
    'elements' => $elements,
    'iterations' => $iterations,
    'coldCompileMs' => $coldCompileMs,
    'cachedCompileMs' => $cachedCompileMs,
    'microsecondsPerIteration' => $timings,
    'streamPlan' => $stream->getStats(),
    'cudaGraphPlan' => $cudaGraph->getStats(),
    'cache' => Fusion::getCacheStats(),
], JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR), PHP_EOL;
