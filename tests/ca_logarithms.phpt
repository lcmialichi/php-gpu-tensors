--TEST--
CudaArray extended unary math eager, Fusion and autograd
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion;

function check_close(array $actual, array $expected, float $tolerance, string $message): void {
    if (count($actual) !== count($expected)) {
        throw new RuntimeException($message);
    }
    foreach ($actual as $index => $value) {
        if (abs($value - $expected[$index]) > $tolerance) {
            throw new RuntimeException($message);
        }
    }
}

$input = A::fromFlatArray([1, 2, 4, 8], [4]);
check_close($input->exp2()->toArray(), [2, 4, 16, 256], 1e-5,
    'exp2 result mismatch');
check_close($input->expm1()->toArray(),
    [expm1(1), expm1(2), expm1(4), expm1(8)], 1e-4,
    'expm1 result mismatch');
check_close($input->log2()->toArray(), [0, 1, 2, 3], 1e-5,
    'log2 result mismatch');
check_close($input->log10()->toArray(),
    [0, log10(2), log10(4), log10(8)], 1e-5, 'log10 result mismatch');
check_close(A::fromFlatArray([1e-7, 1e-5], [2])->log1p()->toArray(),
    [log1p(1e-7), log1p(1e-5)], 1e-11, 'log1p precision mismatch');

$graph = Fusion::compile(
    fn($x) => $x->exp2() + $x->expm1() + $x->log2() +
        $x->log10() + $x->log1p(),
    [$input]);
$expected = $input->exp2()->add($input->expm1())->add($input->log2())
    ->add($input->log10())->add($input->log1p());
check_close($graph->run($input)->toArray(), $expected->toArray(), 1e-5,
    'fused logarithm results mismatch');
if ($graph->getStats()['fusedKernels'] !== 1) {
    throw new RuntimeException('logarithm operations were not fused');
}

$tracked = A::fromFlatArray([1, 2, 4], [3])->requiresGrad();
$loss = $tracked->exp2()->sum()->add($tracked->expm1()->sum())
    ->add($tracked->log2()->sum())->add($tracked->log10()->sum())
    ->add($tracked->log1p()->sum());
$loss->backward();
$ln2 = log(2);
$ln10 = log(10);
check_close($tracked->grad()->toArray(), [
    $ln2 * 2 + exp(1) + 1 / $ln2 + 1 / $ln10 + 1 / 2,
    $ln2 * 4 + exp(2) + 1 / (2 * $ln2) + 1 / (2 * $ln10) + 1 / 3,
    $ln2 * 16 + exp(4) + 1 / (4 * $ln2) + 1 / (4 * $ln10) + 1 / 5,
], 5e-5, 'logarithm gradient mismatch');

$fusedTracked = A::fromFlatArray([1, 2, 4], [3])->requiresGrad();
$seed = A::ones([1]);
[$fusedLoss, $fusedGradient] = Fusion::run(function () use ($fusedTracked, $seed) {
    $loss = $fusedTracked->exp2()->sum()->add($fusedTracked->expm1()->sum())
        ->add($fusedTracked->log2()->sum())
        ->add($fusedTracked->log10()->sum())
        ->add($fusedTracked->log1p()->sum());
    $loss->backward($seed);
    return [$loss, $fusedTracked->grad()];
});
check_close($fusedGradient->toArray(), $tracked->grad()->toArray(), 1e-5,
    'Fusion logarithm gradient mismatch');

$trigValues = [-0.5, 0.2, 0.7];
$trigInput = A::fromFlatArray($trigValues, [3]);
$trigMethods = ['asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh'];
$trigExpected = array_map(static fn($method) =>
    array_map($method, $trigValues), $trigMethods);
foreach ($trigMethods as $index => $method) {
    check_close($trigInput->$method()->toArray(), $trigExpected[$index], 1e-5,
        "$method eager result mismatch");
}
$trigGraph = Fusion::compile(function ($x) {
    return $x->asin() + $x->acos() + $x->atan() +
        $x->sinh() + $x->cosh() + $x->tanh();
}, [$trigInput]);
$trigFusedExpected = $trigInput->asin()->add($trigInput->acos())
    ->add($trigInput->atan())->add($trigInput->sinh())
    ->add($trigInput->cosh())->add($trigInput->tanh());
check_close($trigGraph->run($trigInput)->toArray(),
    $trigFusedExpected->toArray(), 1e-5, 'fused trig results mismatch');

$trigTracked = A::fromFlatArray($trigValues, [3])->requiresGrad();
$trigLoss = $trigTracked->asin()->sum()
    ->add($trigTracked->acos()->sum()->multiply(2))
    ->add($trigTracked->atan()->sum())
    ->add($trigTracked->sinh()->sum())
    ->add($trigTracked->cosh()->sum())
    ->add($trigTracked->tanh()->sum());
$trigLoss->backward();
$trigExpectedGradient = array_map(static function (float $value): float {
    $inverseSqrt = 1 / sqrt(1 - $value * $value);
    return -$inverseSqrt + 1 / (1 + $value * $value) +
        cosh($value) + sinh($value) + 1 - tanh($value) ** 2;
}, $trigValues);
check_close($trigTracked->grad()->toArray(), $trigExpectedGradient, 1e-5,
    'trig autograd mismatch');

foreach (['float16', 'bfloat16'] as $dtype) {
    $lowPrecision = A::fromFlatArray([0.25, 0.5], [2], $dtype);
    $eagerLowPrecision = $lowPrecision->exp2() + $lowPrecision->log2() +
        $lowPrecision->expm1() + $lowPrecision->log1p() +
        $lowPrecision->asin() + $lowPrecision->acos() +
        $lowPrecision->atan() + $lowPrecision->sinh() +
        $lowPrecision->cosh() + $lowPrecision->tanh();
    $lowPrecisionGraph = Fusion::compile(
        fn($x) => $x->exp2() + $x->log2() + $x->expm1() + $x->log1p() +
            $x->asin() + $x->acos() + $x->atan() + $x->sinh() +
            $x->cosh() + $x->tanh(),
        [$lowPrecision]);
    if ($lowPrecisionGraph->run($lowPrecision)->toBuffer() !==
        $eagerLowPrecision->toBuffer()) {
        throw new RuntimeException("$dtype extended unary Fusion rounding mismatch");
    }
}

echo "CudaArray extended unary math checks passed\n";
?>
--EXPECT--
CudaArray extended unary math checks passed
