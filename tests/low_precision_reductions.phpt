--TEST--
Low-precision reductions accumulate in FP32 in eager, Fusion and autograd
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion;

function closeEnough(float $actual, float $expected): bool {
    return abs($actual - $expected) < 0.0001;
}

foreach (['float16', 'bfloat16'] as $dtype) {
    $input = A::fromFlatArray([1.0, 2.0, 3.0, 4.0], [2, 2], $dtype);
    $sum = $input->sum();
    $mean = $input->mean();
    $product = $input->prod();
    $minimum = $input->min();
    $maximum = $input->max();
    $argMaximum = $input->argMax();
    $variance = $input->var();
    $deviation = $input->std();
    if ($sum->dtype() !== 'float32' || $mean->dtype() !== 'float32' ||
        $product->dtype() !== 'float32' || $minimum->dtype() !== 'float32' ||
        $maximum->dtype() !== 'float32' || $argMaximum->dtype() !== 'int32' ||
        $variance->dtype() !== 'float32' || $deviation->dtype() !== 'float32' ||
        !closeEnough($sum->item(), 10.0) || !closeEnough($mean->item(), 2.5) ||
        !closeEnough($product->item(), 24.0) || !closeEnough($minimum->item(), 1.0) ||
        !closeEnough($maximum->item(), 4.0) || $argMaximum->item() !== 3 ||
        $input->all()->item() !== true || $input->any()->item() !== true ||
        !closeEnough($variance->item(), 1.25) ||
        !closeEnough($deviation->item(), sqrt(1.25))) {
        throw new RuntimeException("$dtype eager reduction mismatch");
    }

    $axisSum = $input->sum(0)->toArray();
    if ($axisSum !== [4.0, 6.0]) {
        throw new RuntimeException("$dtype axis sum mismatch");
    }

    $graph = Fusion::compile(
        fn($x) => [
            $x->sum(), $x->mean(), $x->prod(), $x->min(), $x->max(),
            $x->all(), $x->any(), $x->argMax(), $x->var(), $x->std()
        ],
        [$input]
    );
    [$fusedSum, $fusedMean, $fusedProduct, $fusedMinimum, $fusedMaximum,
        $fusedAll, $fusedAny, $fusedArgMaximum, $fusedVariance, $fusedDeviation] =
        $graph->run($input);
    if ($fusedSum->dtype() !== 'float32' ||
        $fusedProduct->dtype() !== 'float32' ||
        $fusedMinimum->dtype() !== 'float32' ||
        $fusedMaximum->dtype() !== 'float32' ||
        $fusedArgMaximum->dtype() !== 'int32' ||
        !closeEnough($fusedSum->item(), 10.0) ||
        !closeEnough($fusedMean->item(), 2.5) ||
        !closeEnough($fusedProduct->item(), 24.0) ||
        !closeEnough($fusedMinimum->item(), 1.0) ||
        !closeEnough($fusedMaximum->item(), 4.0) ||
        $fusedAll->item() !== true || $fusedAny->item() !== true ||
        $fusedArgMaximum->item() !== 3 ||
        !closeEnough($fusedVariance->item(), 1.25) ||
        !closeEnough($fusedDeviation->item(), sqrt(1.25))) {
        throw new RuntimeException("$dtype Fusion reduction mismatch");
    }
    $asyncGraph = Fusion::compile(fn($x) => $x->sum(), [$input], true);
    if ($asyncGraph->getStats()['cudaGraphCompatible'] !== true ||
        !closeEnough($asyncGraph->runAsync($input)->wait()->item(), 10.0)) {
        throw new RuntimeException("$dtype async/CUDA Graph reduction mismatch");
    }

    $tracked = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)->requiresGrad();
    $tracked->sum()->backward();
    if ($tracked->grad()->dtype() !== 'float32' ||
        $tracked->grad()->toArray() !== [1.0, 1.0, 1.0]) {
        throw new RuntimeException("$dtype eager sum gradient mismatch");
    }

    $meanTracked = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)->requiresGrad();
    $meanTracked->mean()->backward();
    foreach ($meanTracked->grad()->toArray() as $value) {
        if (!closeEnough($value, 1.0 / 3.0)) {
            throw new RuntimeException("$dtype eager mean gradient mismatch");
        }
    }

    $fusedTracked = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)->requiresGrad();
    $gradientGraph = Fusion::compile(function($x) {
        $sum = $x->sum();
        $sum->backward();
        return [$sum, $x->grad()];
    }, [$fusedTracked]);
    [, $gradient] = $gradientGraph->run($fusedTracked);
    if ($gradient->dtype() !== 'float32' ||
        $gradient->toArray() !== [1.0, 1.0, 1.0]) {
        throw new RuntimeException("$dtype Fusion sum gradient mismatch");
    }
}

echo "low-precision reductions passed\n";
?>
--EXPECT--
low-precision reductions passed
