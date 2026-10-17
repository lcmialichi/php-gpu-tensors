--TEST--
Fusion inlines floating-point powers with eager-compatible results
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion;

function check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}

foreach (['float32', 'float64'] as $dtype) {
    $base = A::fromFlatArray([2.0, 4.0, 9.0], [3], $dtype);
    $exponent = A::fromFlatArray([3.0, -2.0, 0.5], [3], $dtype);
    $graph = Fusion::compile(fn($x, $y) => ($x ** $y) + $x, [$base, $exponent]);
    $stats = $graph->getStats();
    check($stats['fusedKernels'] === 1 && $stats['executionSteps'] === 1 &&
        $stats['intermediateBuffers'] === 0, "$dtype power was not fused into one kernel");
    $fused = $graph->run($base, $exponent);
    $eager = ($base ** $exponent) + $base;
    check($fused->dtype() === $dtype, "$dtype power result dtype mismatch");
    check($fused->toBuffer() === $eager->toBuffer(), "$dtype tensor power mismatch");

    $scalarGraph = Fusion::compile(fn($x) => ($x ** -2.0) + ($x ** 0.5), [$base]);
    $scalarStats = $scalarGraph->getStats();
    check($scalarStats['fusedKernels'] === 1 && $scalarStats['executionSteps'] === 1 &&
        $scalarStats['intermediateBuffers'] === 0,
        "$dtype scalar powers were not fused into one kernel");
    $scalarFused = $scalarGraph->run($base);
    $scalarEager = ($base ** -2.0) + ($base ** 0.5);
    check($scalarFused->toBuffer() === $scalarEager->toBuffer(),
        "$dtype scalar power mismatch");

    $inverseGraph = Fusion::compile(fn($x) => (2.0 ** $x) + $x, [$base]);
    $inverseFused = $inverseGraph->run($base);
    $inverseEager = (2.0 ** $base) + $base;
    check($inverseFused->toBuffer() === $inverseEager->toBuffer(),
        "$dtype inverse scalar power mismatch");
}

$integers = new A([2, 3], 'int32');
$integerGraph = Fusion::compile(fn($x) => $x ** 3, [$integers]);
check($integerGraph->run($integers)->toBuffer() === ($integers ** 3)->toBuffer(),
    'integer-output power fallback mismatch');

foreach (['float16', 'bfloat16'] as $dtype) {
    $low = A::fromFlatArray([1.5, 2.0, 3.0], [3], $dtype);
    $graph = Fusion::compile(fn($x) => ($x ** 2) + $x, [$low]);
    check($graph->run($low)->toBuffer() === (($low ** 2) + $low)->toBuffer(),
        "$dtype power rounding mismatch");
}

echo "Fusion power checks passed\n";
?>
--EXPECT--
Fusion power checks passed
