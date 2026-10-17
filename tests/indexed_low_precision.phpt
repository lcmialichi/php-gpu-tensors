--TEST--
Low-precision gather supports eager and Fusion with FP32 gradients
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion;

foreach (['float16', 'bfloat16'] as $dtype) {
    $input = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype);
    $indices = A::fromFlatArray([2, 0], [2], 'int32');
    $eager = $input->gather($indices);
    if ($eager->dtype() !== $dtype || $eager->toArray() !== [3.0, 1.0]) {
        throw new RuntimeException("$dtype eager gather mismatch");
    }
    $graph = Fusion::compile(fn($x, $i) => $x->gather($i), [$input, $indices]);
    $fused = $graph->run($input, $indices);
    if ($fused->dtype() !== $dtype || $fused->toBuffer() !== $eager->toBuffer()) {
        throw new RuntimeException("$dtype Fusion gather mismatch");
    }

    $tracked = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)->requiresGrad();
    $singleIndex = A::fromFlatArray([2], [1], 'int32');
    $tracked->gather($singleIndex)->backward();
    if ($tracked->grad()->dtype() !== 'float32' ||
        $tracked->grad()->toArray() !== [0.0, 0.0, 1.0]) {
        throw new RuntimeException("$dtype eager gather gradient mismatch");
    }

    $fusedTracked = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)->requiresGrad();
    $graph = Fusion::compile(function($x, $i) {
        $output = $x->gather($i);
        $output->backward();
        return [$output, $x->grad()];
    }, [$fusedTracked, $singleIndex]);
    [, $gradient] = $graph->run($fusedTracked, $singleIndex);
    if ($gradient->dtype() !== 'float32' ||
        $gradient->toArray() !== [0.0, 0.0, 1.0]) {
        throw new RuntimeException("$dtype Fusion gather gradient mismatch");
    }

    $scatterIndices = A::fromFlatArray([2, 0, 2], [3], 'int32');
    $base = A::zeros([3], $dtype);
    $updates = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype);
    $scattered = $base->scatterAdd($scatterIndices, $updates);
    if ($scattered->dtype() !== 'float32' ||
        $scattered->toArray() !== [2.0, 0.0, 4.0]) {
        throw new RuntimeException("$dtype eager scatterAdd mismatch");
    }

    $trackedBase = A::zeros([3], $dtype)->requiresGrad();
    $trackedUpdates = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)
        ->requiresGrad();
    $trackedBase->scatterAdd($scatterIndices, $trackedUpdates)->sum()->backward();
    if ($trackedBase->grad()->dtype() !== 'float32' ||
        $trackedBase->grad()->toArray() !== [1.0, 1.0, 1.0] ||
        $trackedUpdates->grad()->dtype() !== 'float32' ||
        $trackedUpdates->grad()->toArray() !== [1.0, 1.0, 1.0]) {
        throw new RuntimeException("$dtype eager scatterAdd gradient mismatch");
    }

    $fusedBase = A::zeros([3], $dtype)->requiresGrad();
    $fusedUpdates = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)
        ->requiresGrad();
    $scatterGraph = Fusion::compile(function($x, $i, $u) {
        $output = $x->scatterAdd($i, $u);
        $output->sum()->backward();
        return [$output, $x->grad(), $u->grad()];
    }, [$fusedBase, $scatterIndices, $fusedUpdates]);
    [$fusedScattered, $baseGradient, $updatesGradient] =
        $scatterGraph->run($fusedBase, $scatterIndices, $fusedUpdates);
    if ($fusedScattered->dtype() !== 'float32' ||
        $fusedScattered->toArray() !== [2.0, 0.0, 4.0] ||
        $baseGradient->toArray() !== [1.0, 1.0, 1.0] ||
        $updatesGradient->toArray() !== [1.0, 1.0, 1.0]) {
        throw new RuntimeException("$dtype Fusion scatterAdd mismatch");
    }
}

echo "low-precision gather passed\n";
?>
--EXPECT--
low-precision gather passed
