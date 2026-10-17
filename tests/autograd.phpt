--TEST--
Autograd propagates eager and Fusion gradients through MLP operations
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;

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
} catch (PhpTensors\InvalidArgumentException) {
}
$seeded->zeroGrad();
$nonScalar->backward(CudaArray::fromFlatArray([1, 1], [2]), true);
check($seeded->grad()->toArray() === [2.0, 4.0]);
$nonScalar->backward(CudaArray::fromFlatArray([1, 1], [2]), true);
check($seeded->grad()->toArray() === [4.0, 8.0]);
$seeded->zeroGrad();
check($seeded->grad() === null);
$detached = ($seeded * 2)->detach()->requiresGrad();
$detached->sum()->backward();
check($detached->grad()->toArray() === [1.0, 1.0]);
check($seeded->grad() === null);
echo "explicit seeds, accumulation and detach\n";

$powerBase = CudaArray::fromFlatArray([2.0], [1])->requiresGrad();
$powerExponent = CudaArray::fromFlatArray([3.0], [1])->requiresGrad();
($powerBase ** $powerExponent)->sum()->backward();
check($powerBase->grad()->toArray() === [12.0]);
checkClose($powerExponent->grad()->toArray()[0], 8.0 * log(2.0));
$scalarBaseExponent = CudaArray::fromFlatArray([2.0], [1])->requiresGrad();
(3.0 ** $scalarBaseExponent)->sum()->backward();
checkClose($scalarBaseExponent->grad()->toArray()[0], 9.0 * log(3.0));
echo "binary power gradients\n";

$product = CudaArray::fromFlatArray([2.0, 3.0, 4.0], [3])->requiresGrad();
$product->prod()->backward();
check($product->grad()->toArray() === [12.0, 8.0, 6.0]);
$productWithZero = CudaArray::fromFlatArray([0.0, 2.0, 3.0], [3])->requiresGrad();
$productWithZero->prod()->backward();
check($productWithZero->grad()->toArray() === [6.0, 0.0, 0.0]);
$productWithMultipleZeros = CudaArray::fromFlatArray([0.0, 0.0, 3.0], [3])->requiresGrad();
$productWithMultipleZeros->prod()->backward();
check($productWithMultipleZeros->grad()->toArray() === [0.0, 0.0, 0.0]);
$axisProduct = CudaArray::fromFlatArray([2.0, 3.0, 4.0, 5.0], [2, 2])->requiresGrad();
$axisProduct->prod(1)->sum()->backward();
check($axisProduct->grad()->toArray() === [[3.0, 2.0], [5.0, 4.0]]);
$fusedProductInput = CudaArray::fromFlatArray([2.0, 3.0, 4.0], [3])->requiresGrad();
$fusedProduct = Fusion::compile(function ($x) {
    $loss = $x->prod();
    $loss->backward();
    return [$loss, $x->grad()];
}, [$fusedProductInput]);
[$fusedProductLoss, $fusedProductGradient] = $fusedProduct->run($fusedProductInput);
check($fusedProductLoss->toArray() === [24.0]);
check($fusedProductGradient->toArray() === [12.0, 8.0, 6.0]);
echo "product reduction gradients\n";

$sliceInput = CudaArray::fromFlatArray([1.0, 2.0, 3.0], [3])->requiresGrad();
($sliceInput->slice('0:2')->sum() + $sliceInput->sum())->backward();
check($sliceInput->grad()->toArray() === [2.0, 2.0, 1.0]);
$legacySliceInput = CudaArray::fromFlatArray([1.0, 2.0, 3.0], [3])->requiresGrad();
$legacySliceInput(1)->sum()->backward();
check($legacySliceInput->grad()->toArray() === [0.0, 1.0, 0.0]);
echo "slice gradient scatter\n";

$fusedSliceInput = CudaArray::fromFlatArray([1, 2, 3, 4, 5, 6], [2, 3])->requiresGrad();
$fusedSliceGraph = Fusion::compile(function ($x) {
    $loss = $x->slice(':, ::2')->sum();
    $loss->backward();
    return ['loss' => $loss, 'gradient' => $x->grad()];
}, [$fusedSliceInput]);
$fusedSliceResult = $fusedSliceGraph->run($fusedSliceInput);
check($fusedSliceResult['loss']->toArray() === [14.0]);
check($fusedSliceResult['gradient']->toArray() === [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]);

$fusedIndexedSliceInput = CudaArray::fromFlatArray([1, 2, 3, 4, 5, 6], [3, 2])->requiresGrad();
$fusedIndexedSliceGraph = Fusion::compile(function ($x) {
    $loss = $x->slice('1')->sum();
    $loss->backward();
    return $x->grad();
}, [$fusedIndexedSliceInput]);
$fusedIndexedSliceGradient = $fusedIndexedSliceGraph->run($fusedIndexedSliceInput);
check($fusedIndexedSliceGradient->toArray() === [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]);

$fusedEmptySliceInput = CudaArray::fromFlatArray([1, 2, 3], [3])->requiresGrad();
$fusedEmptySliceGraph = Fusion::compile(function ($x) {
    $loss = $x->slice('3:3')->sum();
    $loss->backward();
    return ['loss' => $loss, 'gradient' => $x->grad()];
}, [$fusedEmptySliceInput]);
$fusedEmptySliceResult = $fusedEmptySliceGraph->run($fusedEmptySliceInput);
check($fusedEmptySliceResult['loss']->toArray() === [0.0]);
check($fusedEmptySliceResult['gradient']->toArray() === [0.0, 0.0, 0.0]);
echo "Fusion slice scatter gradients\n";

$concatLeft = CudaArray::fromFlatArray([1.0, 2.0], [2])->requiresGrad();
$concatRight = CudaArray::fromFlatArray([3.0], [1])->requiresGrad();
$concatLeft->concat([$concatRight], 0)->sum()->backward();
check($concatLeft->grad()->toArray() === [1.0, 1.0]);
check($concatRight->grad()->toArray() === [1.0]);
$concatRowsLeft = CudaArray::fromFlatArray([1.0, 2.0, 3.0, 4.0], [2, 2])->requiresGrad();
$concatRowsRight = CudaArray::fromFlatArray([5.0, 6.0], [2, 1])->requiresGrad();
$concatRowsLeft->concat([$concatRowsRight], 1)->sum()->backward();
check($concatRowsLeft->grad()->toArray() === [[1.0, 1.0], [1.0, 1.0]]);
check($concatRowsRight->grad()->toArray() === [[1.0], [1.0]]);
echo "concat gradients\n";

$batchedLeft = CudaArray::fromFlatArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])->requiresGrad();
$batchedRight = CudaArray::fromFlatArray([1, 0, 0, 1], [2, 2])->requiresGrad();
$batchedLeft->matmul($batchedRight)->sum()->backward();
check($batchedLeft->grad()->toArray() === [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]);
check($batchedRight->grad()->toArray() === [[16.0, 16.0], [20.0, 20.0]]);
echo "batched matmul gradients\n";

$mutated = CudaArray::fromFlatArray([2.0], [1])->requiresGrad();
$mutatedLoss = ($mutated * $mutated)->sum();
$mutated[0] = 10;
try {
    $mutatedLoss->backward();
    throw new RuntimeException('backward accepted modified saved tensor');
} catch (PhpTensors\InvalidArgumentException) {
}
$mutatedThroughView = CudaArray::fromFlatArray([2.0, 3.0], [2])->requiresGrad();
$viewMutationLoss = ($mutatedThroughView * $mutatedThroughView)->sum();
$mutationView = $mutatedThroughView->slice('0:1');
$mutationView[0] = 11;
try {
    $viewMutationLoss->backward();
    throw new RuntimeException('backward accepted a mutation through a view');
} catch (PhpTensors\InvalidArgumentException) {
}
echo "saved tensor version checks\n";

$retained = CudaArray::fromFlatArray([2.0], [1])->requiresGrad();
$retainedLoss = ($retained * $retained)->sum();
$retainedLoss->backward(null, true);
$retainedLoss->backward(retainGraph: true);
check($retained->grad()->toArray() === [8.0]);
$retainedLoss->backward();
check($retained->grad()->toArray() === [12.0]);
try {
    $retainedLoss->backward();
    throw new RuntimeException('backward accepted consumed graph');
} catch (PhpTensors\InvalidArgumentException) {
}
echo "retainGraph and graph release\n";

$roundingInput = CudaArray::fromFlatArray([0.1, 1.6], [2])->requiresGrad();
($roundingInput->ceil() + $roundingInput->floor() + $roundingInput->round())->sum()->backward();
check($roundingInput->grad()->toArray() === [0.0, 0.0]);
$castInput = CudaArray::fromFlatArray([1.5], [1])->requiresGrad();
try {
    $castInput->astype('int32');
    throw new RuntimeException('gradient-tracked cast to integer was accepted');
} catch (PhpTensors\InvalidArgumentException) {
}
echo "rounding derivatives and cast validation\n";

$castGradient = CudaArray::fromFlatArray([1.5, -2.0], [2])->requiresGrad();
($castGradient->astype('float64') * 2.0)->sum()->backward();
check($castGradient->grad()->dtype() === 'float32');
check($castGradient->grad()->toArray() === [2.0, 2.0]);
$fusedCastGradient = CudaArray::fromFlatArray([1.5, -2.0], [2])->requiresGrad();
$fusedCastGraph = Fusion::compile(function ($x) {
    $loss = ($x->astype('float64') * 2.0)->sum();
    $loss->backward();
    return $x->grad();
}, [$fusedCastGradient]);
check($fusedCastGraph->run($fusedCastGradient)->toArray() === [2.0, 2.0]);
echo "widened cast backward gradients\n";

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
binary power gradients
product reduction gradients
slice gradient scatter
Fusion slice scatter gradients
concat gradients
batched matmul gradients
saved tensor version checks
retainGraph and graph release
rounding derivatives and cast validation
widened cast backward gradients
Fusion backward graph
fused softmax cross-entropy gradients
Fusion materialization retains gradient history
