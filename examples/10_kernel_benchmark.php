<?php

declare(strict_types=1);

use Cuda\CudaArray;
use Cuda\Fusion;

/**
 * Kernel Benchmark Suite
 *
 * Measures the wall-clock cost of the extension's core operations
 * (reductions, matmul, fusion plans, memory-pool reuse) and writes a
 * JSON report that can be compared across runs, drivers or GPUs.
 */

if (!extension_loaded('cuda') || cuda_get_device_count() < 1) {
    throw new RuntimeException('CUDA extension and GPU required.');
}

// --- 1. CLI options ---

function benchmarkOptionInt(array $options, string $name, int $default, int $min, int $max): int
{
    if (!isset($options[$name])) {
        return $default;
    }

    $value = filter_var($options[$name], FILTER_VALIDATE_INT);
    if (!is_int($value) || $value < $min || $value > $max) {
        throw new InvalidArgumentException("--$name must be an integer between $min and $max.");
    }

    return $value;
}

$options = getopt('', ['output:', 'binary:', 'precision:', 'warmups:', 'samples:', 'pool-iterations:']);

$warmups = benchmarkOptionInt($options, 'warmups', 5, 0, 500);
$sampleCount = benchmarkOptionInt($options, 'samples', 25, 1, 500);
$poolIterations = benchmarkOptionInt($options, 'pool-iterations', 16, 1, 500);
$precision = $options['precision'] ?? 'fp32-strict';

cuda_set_matmul_precision($precision);

// --- 2. Benchmark case builders ---
//
// Every case is an array shaped as:
//   [callable $operation, float $expectedValue, int $elementCount, ?string $kind, array $details]
// `$kind === 'matmul'` makes the runner record the backend used for the call.
// `$details` lets a case attach extra fields (e.g. a fusion plan name) to its report row.

function buildReductionCases(): array
{
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

    return $cases;
}

function buildMatmulCases(): array
{
    $cases = [];

    $shapes = [
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
    ];

    foreach ($shapes as [$name, $rows, $inner, $cols]) {
        $a = CudaArray::ones([$rows, $inner]);
        $b = CudaArray::ones([$inner, $cols]);

        $cases["matmul-$name"] = [static fn() => $a->matmul($b), $inner, $rows * $cols, 'matmul'];
    }

    $transposed = CudaArray::ones([128, 128])->transpose();
    $transposedRight = CudaArray::ones([128, 128]);
    $cases['matmul-transposed-128'] = [
        static fn() => $transposed->matmul($transposedRight), 128, 128 * 128, 'matmul',
    ];

    $strided = CudaArray::ones([64, 128])->slice(':, ::2');
    $stridedRight = CudaArray::ones([64, 64]);
    $cases['matmul-strided-64'] = [
        static fn() => $strided->matmul($stridedRight), 64, 64 * 64, 'matmul',
    ];

    $batched = CudaArray::ones([4, 64, 64]);
    $cases['matmul-batched-4x64'] = [
        static fn() => $batched->matmul($batched), 64, 4 * 64 * 64, 'matmul',
    ];

    $broadcast = CudaArray::ones([1, 64, 64]);
    $cases['matmul-broadcast-4x64'] = [
        static fn() => $broadcast->matmul($batched), 64, 4 * 64 * 64, 'matmul',
    ];

    $dotLeft = CudaArray::ones([1, 100000]);
    $dotRight = CudaArray::ones([100000, 1]);
    $cases['matmul-dot-100000'] = [
        static fn() => $dotLeft->matmul($dotRight), 100000, 1, 'matmul',
    ];

    return $cases;
}

/**
 * Builds the elementwise/reduction fusion cases plus the compiled plans
 * they rely on. The plans are returned separately so the caller can add
 * their `getStats()`/`getPlan()` output to the final report.
 */
function buildFusionCases(): array
{
    $cases = [];

    $elementwiseInput = CudaArray::ones([262144]);
    $elementwise = static fn($x) => ($x * 1.25 + 0.5)->sqrt();
    $elementwisePlan = Fusion::compile($elementwise, [$elementwiseInput]);
    $elementwiseGraph = Fusion::compile($elementwise, [$elementwiseInput], cudaGraph: true);
    $elementwiseExpected = sqrt(1.75);

    $cases['elementwise-eager-262144'] = [
        static fn() => $elementwise($elementwiseInput), $elementwiseExpected, 262144,
    ];
    $cases['elementwise-fusion-262144'] = [
        static fn() => $elementwisePlan->run($elementwiseInput), $elementwiseExpected, 262144, null,
        ['fusionPlan' => 'elementwise-stream'],
    ];
    $cases['elementwise-cuda-graph-262144'] = [
        static fn() => $elementwiseGraph->run($elementwiseInput), $elementwiseExpected, 262144, null,
        ['fusionPlan' => 'elementwise-cuda-graph'],
    ];

    $reductionInput = CudaArray::ones([128, 1024]);
    $reduction = static fn($x) => ($x * 1.25)->sum(1);
    $reductionPlan = Fusion::compile($reduction, [$reductionInput]);
    $reductionGraph = Fusion::compile($reduction, [$reductionInput], cudaGraph: true);

    $cases['reduce-eager-128x1024'] = [
        static fn() => $reduction($reductionInput), 1280, 128,
    ];
    $cases['reduce-fusion-128x1024'] = [
        static fn() => $reductionPlan->run($reductionInput), 1280, 128, null,
        ['fusionPlan' => 'reduction-stream'],
    ];
    $cases['reduce-cuda-graph-128x1024'] = [
        static fn() => $reductionGraph->run($reductionInput), 1280, 128, null,
        ['fusionPlan' => 'reduction-cuda-graph'],
    ];

    $plans = [
        'elementwise-stream' => $elementwisePlan,
        'elementwise-cuda-graph' => $elementwiseGraph,
        'reduction-stream' => $reductionPlan,
        'reduction-cuda-graph' => $reductionGraph,
    ];

    return [$cases, $plans];
}

/**
 * Builds cases that repeatedly allocate and free same-shape outputs to
 * exercise the memory pool's reuse path between default-stream launches.
 */
function buildPoolReuseCases(int $poolIterations): array
{
    $cases = [];

    $sizes = [
        ['small-256k', 65536],
        ['medium-2m', 524288],
        ['large-8m', 2097152],
    ];

    foreach ($sizes as [$name, $count]) {
        $input = CudaArray::ones([$count]);

        $churn = static function () use ($input, $poolIterations) {
            for ($iteration = 0; $iteration < $poolIterations; $iteration++) {
                $result = $input->add(1);
                if ($iteration + 1 < $poolIterations) {
                    unset($result);
                }
            }

            return $result;
        };

        $cases["pool-reuse-$name"] = [$churn, 2, $count, null, [
            'iterationsPerSample' => $poolIterations,
            'outputBytes' => $count * 4,
        ]];
    }

    return $cases;
}

// --- 3. Run a single case: validate correctness, warm up, then sample timings ---

function assertBenchmarkResult(string $name, CudaArray $result, float $expected, int $count): void
{
    $bytes = $result->toBuffer();
    if (strlen($bytes) !== $count * 4) {
        throw new RuntimeException("$name output shape mismatch");
    }

    for ($offset = 0; $offset < $count; $offset += 65536) {
        $chunkCount = min(65536, $count - $offset);
        $values = unpack('g*', substr($bytes, $offset * 4, $chunkCount * 4));

        foreach ($values as $value) {
            if (!is_finite($value) || abs($value - $expected) > 1e-5 + 1e-4 * abs($expected)) {
                throw new RuntimeException("$name numerical mismatch");
            }
        }
    }
}

function runBenchmarkCase(string $name, array $case, int $warmups, int $sampleCount): array
{
    [$operation, $expected, $count] = $case;
    $isMatmul = ($case[3] ?? null) === 'matmul';
    $details = $case[4] ?? [];

    // Correctness check against the expected value, run once outside the timed samples.
    cuda_synchronize();
    $result = $operation();
    cuda_synchronize();
    assertBenchmarkResult($name, $result, $expected, $count);
    unset($result);

    for ($i = 0; $i < $warmups; $i++) {
        $result = $operation();
        cuda_synchronize();
        unset($result);
    }

    $samples = [];
    for ($i = 0; $i < $sampleCount; $i++) {
        cuda_synchronize();
        $start = hrtime(true);
        $result = $operation();
        cuda_synchronize();
        $samples[] = (hrtime(true) - $start) / 1e6;
        unset($result);
    }

    $sorted = $samples;
    sort($sorted);

    $middle = intdiv($sampleCount, 2);
    $median = $sampleCount % 2
        ? $sorted[$middle]
        : ($sorted[$middle - 1] + $sorted[$middle]) / 2;

    $mean = array_sum($samples) / $sampleCount;
    $variance = 0.0;
    foreach ($samples as $sample) {
        $variance += ($sample - $mean) ** 2;
    }

    $row = [
        'name' => $name,
        'medianMs' => $median,
        'minMs' => $sorted[0],
        'p95Ms' => $sorted[max(0, (int) ceil($sampleCount * 0.95) - 1)],
        'maxMs' => $sorted[$sampleCount - 1],
        'meanMs' => $mean,
        'stddevMs' => sqrt($variance / $sampleCount),
        'samplesMs' => $samples,
    ];

    foreach ($details as $key => $value) {
        $row[$key] = $value;
    }

    if ($isMatmul) {
        $row['backend'] = cuda_get_backend_info()['lastMatmul'];
    }

    return $row;
}

// --- 4. Assemble every case and the base report ---

[$fusionCases, $fusionPlans] = buildFusionCases();

$cases = array_merge(
    buildReductionCases(),
    buildMatmulCases(),
    $fusionCases,
    buildPoolReuseCases($poolIterations)
);

$report = [
    'php' => PHP_VERSION,
    'gpu' => cuda_get_device_info(),
    'runtime' => cuda_get_runtime_version(),
    'driver' => cuda_get_driver_version(),
    'precision' => $precision,
    'memoryPoolLimit' => ini_get('cuda.memory_size'),
    'configuration' => [
        'warmups' => $warmups,
        'samples' => $sampleCount,
        'poolIterations' => $poolIterations,
    ],
    'generatedAt' => gmdate(DATE_ATOM),
    'binarySha256' => isset($options['binary']) ? hash_file('sha256', $options['binary']) : null,
    'benchmarkSha256' => hash_file('sha256', __FILE__),
    'timing' => 'CPU wall clock with warm resident inputs; operation allocations and final '
        . 'synchronization included; returned-output destruction excluded; intermediate '
        . 'temporaries are destroyed inside the timed operation.',
    'poolChurnMethodology' => 'Each pool-reuse case repeatedly creates and destroys same-shape '
        . 'outputs between default-stream launches, then synchronizes once. This measures '
        . 'end-to-end reuse cost, not allocator-only time or internal pool counters.',
    'cases' => [],
];

// --- 5. Run every case, printing progress and collecting report rows ---

foreach ($cases as $name => $case) {
    $row = runBenchmarkCase($name, $case, $warmups, $sampleCount);
    $report['cases'][] = $row;

    $backendSuffix = isset($row['backend']) ? " ({$row['backend']})" : '';
    printf(
        "%-34s median %8.4f ms  p95 %8.4f ms%s\n",
        $name,
        $row['medianMs'],
        $row['p95Ms'],
        $backendSuffix
    );
}

// --- 6. Attach fusion plan details and backend info, then write the report ---

foreach ($fusionPlans as $name => $plan) {
    $report['fusionPlans'][$name] = [
        'stats' => $plan->getStats(),
        'steps' => $plan->getPlan(),
    ];
}

if (function_exists('cuda_get_backend_info')) {
    $report['backends'] = cuda_get_backend_info();
}

if (isset($options['output'])) {
    $json = json_encode($report, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR) . PHP_EOL;
    if (file_put_contents($options['output'], $json) === false) {
        throw new RuntimeException('Cannot save benchmark report.');
    }
}
