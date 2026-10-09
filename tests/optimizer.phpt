--TEST--
Cuda\Optimizer performs functional SGD and AdamW updates in eager and Fusion
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
use Cuda\Optimizer;

function checkClose(float $actual, float $expected): void
{
    if (abs($actual - $expected) > 1e-5) {
        throw new RuntimeException("Expected $expected, got $actual");
    }
}

$parameter = CudaArray::fromFlatArray([2.0], [1])->requiresGrad();
$sgd = Optimizer::sgd(0.1, 0.9);
$sgdState = $sgd->initState([$parameter]);
$sgd->zeroGrad([$parameter]);
($parameter * $parameter)->sum()->backward();
$sgdStep = $sgd->step([$parameter], [$parameter->grad()], $sgdState);
checkClose($parameter->toArray()[0], 2.0);
checkClose($sgdStep['parameters'][0]->toArray()[0], 1.6);
checkClose($sgdStep['state']['velocity'][0]->toArray()[0], 4.0);
echo "eager SGD and functional parameter update\n";

$doubleParameter = CudaArray::fromFlatArray([2.0], [1], 'float64')->requiresGrad();
$doubleState = $sgd->initState([$doubleParameter]);
if ($doubleState['velocity'][0]->dtype() !== 'float64') {
    throw new RuntimeException('Optimizer state did not preserve float64 dtype');
}
$sgd->zeroGrad([$doubleParameter]);
($doubleParameter * $doubleParameter)->sum()->backward();
$doubleStep = $sgd->step([$doubleParameter], [$doubleParameter->grad()], $doubleState);
if ($doubleStep['parameters'][0]->dtype() !== 'float64') {
    throw new RuntimeException('Optimizer update did not preserve float64 dtype');
}
echo "float64 state and update\n";

$adamParameter = CudaArray::fromFlatArray([2.0], [1])->requiresGrad();
$adamW = Optimizer::adamW(0.1, 0.9, 0.999, 1e-8, 0.1);
$adamState = $adamW->initState([$adamParameter]);
$adamW->zeroGrad([$adamParameter]);
($adamParameter * $adamParameter)->sum()->backward();
$adamStep = $adamW->step([$adamParameter], [$adamParameter->grad()], $adamState, null, [false]);
checkClose($adamStep['parameters'][0]->toArray()[0], 1.9);
checkClose($adamStep['state']['firstMoment'][0]->toArray()[0], 0.4);
checkClose($adamStep['state']['secondMoment'][0]->toArray()[0], 0.016);
checkClose($adamStep['state']['beta1Power']->toArray()[0], 0.9);
checkClose($adamStep['state']['beta2Power']->toArray()[0], 0.999);
$decayedAdamStep = $adamW->step(
    [$adamParameter],
    [$adamParameter->grad()],
    $adamState,
    null,
    [true]
);
checkClose($decayedAdamStep['parameters'][0]->toArray()[0], 1.88);
echo "AdamW moments, bias correction, and decay mask\n";

try {
    $sgd->step(
        [$parameter],
        [CudaArray::fromFlatArray([1.0, 2.0], [2])],
        $sgdState
    );
    throw new RuntimeException('Mismatched gradient shape was accepted');
} catch (Cuda\InvalidArgumentException) {
}

$fusionParameter = CudaArray::fromFlatArray([2.0], [1])->requiresGrad();
$fusionState = $sgd->initState([$fusionParameter]);
$learningRate = CudaArray::fromFlatArray([0.1], [1]);
$graph = Fusion::compile(function ($parameter, $velocity, $rate) use ($sgd) {
    $sgd->zeroGrad([$parameter]);
    ($parameter * $parameter)->sum()->backward();
    return $sgd->step(
        [$parameter],
        [$parameter->grad()],
        ['velocity' => [$velocity]],
        $rate
    );
}, [$fusionParameter, $fusionState['velocity'][0], $learningRate]);
$fusionStep = $graph->run($fusionParameter, $fusionState['velocity'][0], $learningRate);
checkClose($fusionStep['parameters'][0]->toArray()[0], 1.6);
checkClose($fusionStep['state']['velocity'][0]->toArray()[0], 4.0);
echo "Fusion-compatible functional step\n";

try {
    Optimizer::adamW(0.1, 1.0);
    throw new RuntimeException('Invalid beta1 was accepted');
} catch (Cuda\InvalidArgumentException) {
}
echo "optimizer option validation\n";
?>
--EXPECT--
eager SGD and functional parameter update
float64 state and update
AdamW moments, bias correction, and decay mask
Fusion-compatible functional step
optimizer option validation
