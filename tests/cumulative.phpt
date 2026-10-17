--TEST--
CudaArray cumulative sums and products across axes, layouts and Fusion
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

$matrix = A::fromFlatArray([1, 2, 3, 4, 5, 6], [2, 3]);
check($matrix->cumsum()->getShape() === [6] &&
    $matrix->cumsum()->toArray() === [1.0, 3.0, 6.0, 10.0, 15.0, 21.0],
    'flattened cumulative sum mismatch');
check($matrix->cumsum(0)->toArray() === [[1.0, 2.0, 3.0], [5.0, 7.0, 9.0]],
    'axis-zero cumulative sum mismatch');
check($matrix->cumsum(-1)->toArray() === [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]],
    'negative-axis cumulative sum mismatch');
check($matrix->cumprod()->toArray() === [1.0, 2.0, 6.0, 24.0, 120.0, 720.0],
    'flattened cumulative product mismatch');
check($matrix->cumprod(1)->toArray() === [[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]],
    'axis cumulative product mismatch');

$transposed = $matrix->transpose();
check($transposed->cumsum(0)->toArray() === [[1.0, 4.0], [3.0, 9.0], [6.0, 15.0]],
    'transposed cumulative sum mismatch');
$strided = A::fromFlatArray(range(1, 24), [4, 6])->slice('::2, 1::2');
check($strided->cumsum(1)->toArray() === [[2.0, 6.0, 12.0], [14.0, 30.0, 48.0]],
    'strided cumulative sum mismatch');
check($strided->cumsum()->toArray() === [2.0, 6.0, 12.0, 26.0, 42.0, 60.0],
    'flattened strided cumulative sum mismatch');

foreach (['float32', 'float64', 'int8', 'int16', 'int32', 'int64',
          'uint8', 'uint16', 'uint32', 'uint64'] as $dtype) {
    $typed = A::fromFlatArray([1, 2, 3], [3], $dtype);
    $sum = $typed->cumsum();
    $product = $typed->cumprod();
    $expectedDtype = in_array($dtype, ['float32', 'float64'], true)
        ? $dtype : ($dtype === 'uint64' ? 'uint64' : 'int64');
    check($sum->dtype() === $expectedDtype &&
        array_map('floatval', $sum->toArray()) === [1.0, 3.0, 6.0],
        "$dtype cumulative sum mismatch");
    check(array_map('floatval', $product->toArray()) === [1.0, 2.0, 6.0],
        "$dtype cumulative product mismatch");
}
foreach (['float16', 'bfloat16'] as $dtype) {
    $typed = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype);
    $sum = $typed->cumsum();
    $product = $typed->cumprod();
    check($sum->dtype() === 'float32' &&
        $sum->toArray() === [1.0, 3.0, 6.0] &&
        $product->dtype() === 'float32' &&
        $product->toArray() === [1.0, 2.0, 6.0],
        "$dtype cumulative scan mismatch");

    $fused = Fusion::compile(
        fn($x) => [$x->cumsum(), $x->cumprod()],
        [$typed],
        true
    );
    [$fusedSum, $fusedProduct] = $fused->run($typed);
    check($fusedSum->dtype() === 'float32' &&
        $fusedSum->toArray() === [1.0, 3.0, 6.0] &&
        $fusedProduct->dtype() === 'float32' &&
        $fusedProduct->toArray() === [1.0, 2.0, 6.0],
        "$dtype Fusion cumulative scan mismatch");
    check($fused->getStats()['asyncCompatible'] === true &&
        $fused->getStats()['cudaGraphCompatible'] === true,
        "$dtype cumulative scans should support async and CUDA Graph replay");
    [$asyncSum, $asyncProduct] = $fused->runAsync($typed)->wait();
    check($asyncSum->toArray() === [1.0, 3.0, 6.0] &&
        $asyncProduct->toArray() === [1.0, 2.0, 6.0],
        "$dtype asynchronous Fusion cumulative scan mismatch");

    $tracked = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)
        ->requiresGrad();
    $tracked->cumprod()->sum()->backward();
    if ($tracked->grad()->dtype() !== 'float32' ||
        $tracked->grad()->toArray() !== [9.0, 4.0, 2.0]) {
        throw new RuntimeException("$dtype cumulative product gradient mismatch");
    }
    $fusedTracked = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)
        ->requiresGrad();
    $gradientGraph = Fusion::compile(function($x) {
        $loss = $x->cumprod()->sum();
        $loss->backward();
        return [$loss, $x->grad()];
    }, [$fusedTracked]);
    [, $gradient] = $gradientGraph->run($fusedTracked);
    if ($gradient->dtype() !== 'float32' ||
        $gradient->toArray() !== [9.0, 4.0, 2.0]) {
        throw new RuntimeException("$dtype Fusion cumulative product gradient mismatch");
    }
}
check(A::fromFlatArray([true, false, true], [3], 'bool')
    ->cumprod()->toArray() === [1, 0, 0],
    'boolean cumulative product mismatch');
check(A::fromFlatArray([true, false, true], [3], 'bool')
    ->cumsum()->toArray() === [1, 1, 2],
    'boolean cumulative sum mismatch');

$longRows = A::ones([2, 1025], 'float64')->cumsum(1)->toArray();
check($longRows[0][512] === 513.0 && $longRows[0][1024] === 1025.0 &&
    $longRows[1][0] === 1.0 && $longRows[1][1024] === 1025.0,
    'multi-tile segmented scan mismatch');

$empty = A::zeros([2, 0, 3]);
check($empty->cumsum(1)->getShape() === [2, 0, 3] &&
    $empty->cumprod()->getShape() === [0],
    'empty cumulative shape mismatch');

$fused = Fusion::run(fn() => $matrix->cumsum(1)->add(1));
check($fused->toArray() === [[2.0, 4.0, 7.0], [5.0, 10.0, 16.0]],
    'synchronous Fusion cumulative boundary mismatch');
$graph = Fusion::compile(fn($x) => $x->cumprod(1), [$matrix]);
check($graph->run($matrix)->toArray() === [[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]],
    'compiled Fusion cumulative boundary mismatch');
$asyncScan = $graph->runAsync($matrix)->wait();
check($asyncScan->toArray() === [[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]],
    'asynchronous Fusion cumulative scan mismatch');
$capturedScan = Fusion::compile(fn($x) => $x->cumsum(1), [$matrix], true);
check($capturedScan->getStats()['cudaGraphCompatible'] === true &&
    $capturedScan->run($matrix)->toArray() ===
        [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]],
    'CUDA Graph cumulative scan mismatch');
$transposedScan = Fusion::compile(fn($x) => $x->cumsum(0), [$transposed]);
check($transposedScan->getStats()['asyncCompatible'] === false &&
    $transposedScan->getStats()['cudaGraphCompatible'] === false,
    'strided non-last-axis scans must retain synchronous execution');
try {
    $transposedScan->runAsync($transposed);
    throw new RuntimeException('asynchronous strided cumulative scan was accepted');
} catch (PhpTensors\RuntimeException $error) {
}

$tracked = A::fromFlatArray([1.0, 2.0, 3.0], [3])->requiresGrad();
$tracked->cumsum()->sum()->backward();
check($tracked->grad()->toArray() === [3.0, 2.0, 1.0],
    'flattened cumulative sum gradient mismatch');
$trackedRows = A::fromFlatArray([1.0, 2.0, 3.0, 4.0], [2, 2])->requiresGrad();
$trackedRows->cumsum(1)->sum()->backward();
check($trackedRows->grad()->toArray() === [[2.0, 1.0], [2.0, 1.0]],
    'axis cumulative sum gradient mismatch');
$trackedStrided = A::fromFlatArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    ->transpose()->requiresGrad();
$trackedStrided->cumsum(0)->sum()->backward();
check($trackedStrided->grad()->toArray() === [[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]],
    'strided cumulative sum gradient mismatch');
$trackedFused = A::fromFlatArray([1.0, 2.0, 3.0], [3])->requiresGrad();
[, $fusedGradient] = Fusion::run(function () use ($trackedFused) {
    $loss = $trackedFused->cumsum()->sum();
    $loss->backward();
    return [$loss, $trackedFused->grad()];
});
check($fusedGradient->toArray() === [3.0, 2.0, 1.0],
    'Fusion cumulative sum gradient mismatch');
$trackedProduct = A::fromFlatArray([2.0, 3.0, 4.0], [3])->requiresGrad();
$trackedProduct->cumprod()->sum()->backward();
check($trackedProduct->grad()->toArray() === [16.0, 10.0, 6.0],
    'cumulative product gradient mismatch');
$trackedProductWithZero = A::fromFlatArray([0.0, 2.0, 3.0], [3])->requiresGrad();
$trackedProductWithZero->cumprod()->sum()->backward();
check($trackedProductWithZero->grad()->toArray() === [9.0, 0.0, 0.0],
    'cumulative product gradient with a zero mismatch');
$trackedProductWithMultipleZeros = A::fromFlatArray([0.0, 2.0, 0.0, 4.0], [4])
    ->requiresGrad();
$trackedProductWithMultipleZeros->cumprod()->sum()->backward();
check($trackedProductWithMultipleZeros->grad()->toArray() === [3.0, 0.0, 0.0, 0.0],
    'cumulative product gradient with multiple zeros mismatch');
$trackedProductRows = A::fromFlatArray([2.0, 3.0, 4.0, 5.0], [2, 2])
    ->requiresGrad();
$trackedProductRows->cumprod(1)->sum()->backward();
check($trackedProductRows->grad()->toArray() === [[4.0, 2.0], [6.0, 4.0]],
    'axis cumulative product gradient mismatch');
$trackedStridedProduct = A::fromFlatArray([2.0, 3.0, 4.0, 5.0], [2, 2])
    ->transpose()->requiresGrad();
$trackedStridedProduct->cumprod(1)->sum()->backward();
check($trackedStridedProduct->grad()->toArray() === [[5.0, 2.0], [6.0, 3.0]],
    'strided cumulative product gradient mismatch');
$trackedFusedProduct = A::fromFlatArray([2.0, 3.0, 4.0], [3])->requiresGrad();
[, $fusedProductGradient] = Fusion::run(function () use ($trackedFusedProduct) {
    $loss = $trackedFusedProduct->cumprod()->sum();
    $loss->backward();
    return [$loss, $trackedFusedProduct->grad()];
});
check($fusedProductGradient->toArray() === [16.0, 10.0, 6.0],
    'Fusion cumulative product gradient mismatch');

echo "cumulative operations passed\n";
?>
--EXPECT--
cumulative operations passed
