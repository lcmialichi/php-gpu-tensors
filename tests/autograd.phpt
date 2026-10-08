--TEST--
Autograd propagates eager and Fusion gradients through MLP operations
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;

function check(bool $ok): void
{
    if (!$ok) throw new RuntimeException('autograd check failed');
}
function checkClose(float $actual, float $expected): void
{
    check(abs($actual - $expected) < 1e-5);
}

$left = CudaArray::fromFlatArray([1, 2, 3, 4], [2, 2])->requiresGrad();
$right = CudaArray::fromFlatArray([10, 20], [1, 2])->requiresGrad();
$loss = ($left * $right)->sum();
$loss->backward();
check($left->grad()->toArray() === [[10.0, 20.0], [10.0, 20.0]]);
check($right->grad()->toArray() === [[4.0, 6.0]]);
echo "eager broadcast gradients\n";

$ties = CudaArray::fromFlatArray([3, 3, 1], [3])->requiresGrad();
$ties->max()->backward();
check($ties->grad()->toArray() === [0.5, 0.5, 0.0]);
echo "max tie gradients\n";

$branch = CudaArray::fromFlatArray([-1, 2], [2])->requiresGrad();
$condition = $branch->gt(0);
$branchLoss = CudaArray::where($condition, $branch * $branch, $branch->neg())->sum();
$branchLoss->backward();
check($branch->grad()->toArray() === [-1.0, 4.0]);
echo "where gradients\n";

$seeded = CudaArray::fromFlatArray([1, 2], [2])->requiresGrad();
$nonScalar = $seeded * $seeded;
try {
    $nonScalar->backward();
    throw new RuntimeException('non-scalar backward accepted no seed');
} catch (Cuda\InvalidArgumentException) {
}
$seeded->zeroGrad();
$nonScalar->backward(CudaArray::fromFlatArray([1, 1], [2]));
check($seeded->grad()->toArray() === [2.0, 4.0]);
$nonScalar->backward(CudaArray::fromFlatArray([1, 1], [2]));
check($seeded->grad()->toArray() === [4.0, 8.0]);
$seeded->zeroGrad();
check($seeded->grad() === null);
$detached = ($seeded * 2)->detach()->requiresGrad();
$detached->sum()->backward();
check($detached->grad()->toArray() === [1.0, 1.0]);
check($seeded->grad() === null);
echo "explicit seeds, accumulation and detach\n";

$input = CudaArray::fromFlatArray([1, 2, 3, 4], [2, 2])->requiresGrad();
$weight = CudaArray::fromFlatArray([1, 0, 0, 1], [2, 2])->requiresGrad();
$graph = Fusion::compile(function ($x, $w) {
    $loss = $x->matmul($w)->mean();
    $loss->backward();
    return [
        'loss' => $loss,
        'inputGradient' => $x->grad(),
        'weightGradient' => $w->grad(),
    ];
}, [$input, $weight]);
$outputs = $graph->run($input, $weight);
check($outputs['loss']->toArray() === [2.5]);
check($outputs['inputGradient']->toArray() === [[0.25, 0.25], [0.25, 0.25]]);
check($outputs['weightGradient']->toArray() === [[1.0, 1.0], [1.5, 1.5]]);
echo "Fusion backward graph\n";

$logits = CudaArray::fromFlatArray([1, 2, 3, 0.5, 0.5, -1], [2, 3])->requiresGrad();
$labels = CudaArray::fromFlatArray([0, 0, 1, 1, 0, 0], [2, 3]);
$crossEntropy = Fusion::compile(function ($scores, $targets) {
    $shifted = $scores - $scores->max(1)->reshape([2, 1]);
    $exponentials = $shifted->exp();
    $denominator = $exponentials->sum(1)->reshape([2, 1]);
    $targetLogit = ($shifted * $targets)->sum(1)->reshape([2, 1]);
    $loss = ($denominator->log() - $targetLogit)->sum(0);
    $loss->backward();
    return [$loss, $scores->grad()];
}, [$logits, $labels]);
[$crossEntropyLoss, $logitsGradient] = $crossEntropy->run($logits, $labels);
checkClose($crossEntropyLoss->toArray()[0], 1.206522);
$gradientValues = $logitsGradient->toArray();
checkClose($gradientValues[0][0], 0.090031);
checkClose($gradientValues[0][1], 0.244728);
checkClose($gradientValues[0][2], -0.334759);
checkClose($gradientValues[1][0], -0.550184);
checkClose($gradientValues[1][1], 0.449816);
checkClose($gradientValues[1][2], 0.100368);
echo "fused softmax cross-entropy gradients\n";

$fusedInput = CudaArray::fromFlatArray([2, 3], [2])->requiresGrad();
$fusedOutput = Fusion::run(fn() => $fusedInput * $fusedInput);
$fusedOutput->sum()->backward();
check($fusedInput->grad()->toArray() === [4.0, 6.0]);
echo "Fusion materialization retains gradient history\n";
?>
--EXPECT--
eager broadcast gradients
max tie gradients
where gradients
explicit seeds, accumulation and detach
Fusion backward graph
fused softmax cross-entropy gradients
Fusion materialization retains gradient history
