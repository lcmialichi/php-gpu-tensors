--TEST--
FP16 and BF16 conversion, storage, matmul and Fusion matmul
--SKIPIF--
<?php
if (!extension_loaded('phptensors') || cuda_get_device_count() < 1) die('skip CUDA GPU required');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion;

function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}

$half = A::fromFlatArray([1, 2, 3, 4], [2, 2], 'float16');
check($half->dtype() === 'float16', 'FP16 dtype was not preserved');
check($half->toArray() === [[1.0, 2.0], [3.0, 4.0]], 'FP16 export failed');
check(A::fromBuffer($half->toBuffer(), [2, 2], 'float16')->toArray() === $half->toArray(),
    'FP16 packed-buffer roundtrip failed');
check($half->toHost()->toArray() === $half->toArray(), 'FP16 host-array export failed');
check((clone $half->transpose())->toArray() === [[1.0, 3.0], [2.0, 4.0]],
    'FP16 non-contiguous clone failed');
check($half->astype('float32')->toArray() === [[1.0, 2.0], [3.0, 4.0]],
    'FP16 widening conversion failed');

$halfLimits = A::fromFlatArray([65504.0, 65520.0, 2 ** -24, 2 ** -25], [4], 'float16')->toArray();
check($halfLimits[0] === 65504.0, 'FP16 maximum finite value changed');
check(is_infinite($halfLimits[1]), 'FP16 overflow did not produce infinity');
check($halfLimits[2] === 5.960464477539063e-8, 'FP16 minimum subnormal was not preserved');
check($halfLimits[3] === 0.0, 'FP16 underflow did not round to zero');

$bfloat = A::fromFlatArray([1.0, 1.00390625, 1.0078125, 3.14159265], [4], 'bfloat16');
$bfloatValues = $bfloat->toArray();
check($bfloat->dtype() === 'bfloat16', 'BF16 dtype was not preserved');
check($bfloatValues[0] === 1.0 && $bfloatValues[1] === 1.0,
    'BF16 round-to-nearest-even conversion failed');
check($bfloatValues[2] === 1.0078125, 'BF16 representable value changed');
check(abs($bfloatValues[3] - 3.140625) < 0.000001, 'BF16 conversion tolerance exceeded');
check($bfloat->astype('float64')->toArray() === $bfloatValues,
    'BF16 widening conversion failed');
$bfloatLimits = A::fromFlatArray([3.4028234e38, 2 ** -133, 2 ** -134], [3], 'bfloat16')->toArray();
check(is_infinite($bfloatLimits[0]), 'BF16 overflow did not produce infinity');
check($bfloatLimits[1] > 0.0 && $bfloatLimits[2] === 0.0,
    'BF16 subnormal or underflow conversion failed');

foreach (['float16', 'bfloat16'] as $dtype) {
    check(A::full([2], 1.5, $dtype)->toArray() === [1.5, 1.5],
        "$dtype full allocation failed");
    $random = A::rand([32], -1.0, 1.0, $dtype)->toArray();
    foreach ($random as $value)
        check($value >= -1.0 && $value <= 1.0, "$dtype random value is out of range");

    $left = A::fromFlatArray([1, 2, 3, 4, 2, 0, 0, 2], [2, 2, 2], $dtype);
    $right = A::fromFlatArray([5, 6, 1, 2], [2, 2, 1], $dtype);
    $product = $left->matmul($right);
    check($product->dtype() === $dtype, "$dtype batched matmul output dtype changed");
    check($product->toArray() === [[[17.0], [39.0]], [[2.0], [4.0]]],
        "$dtype batched matmul result mismatch");

    $broadcastLeft = A::fromFlatArray(
        [1, 2, 3, 4, 2, 0, 0, 2], [2, 1, 2, 2], $dtype);
    $broadcastRight = A::fromFlatArray(
        [5, 6, 1, 2, 2, 1], [1, 3, 2, 1], $dtype);
    check($broadcastLeft->matmul($broadcastRight)->toArray() === [
        [[[17.0], [39.0]], [[5.0], [11.0]], [[4.0], [10.0]]],
        [[[10.0], [12.0]], [[2.0], [4.0]], [[4.0], [2.0]]],
    ], "$dtype broadcast matmul result mismatch");

    $graph = Fusion::compile(fn($a, $b) => $a->matmul($b), [$left, $right]);
    $fused = $graph->run($left, $right);
    check($fused->dtype() === $dtype && $fused->toArray() === $product->toArray(),
        "$dtype Fusion matmul differed from eager result");

    $identity = array_fill(0, 64 * 64, 0.0);
    for ($i = 0; $i < 64; $i++) $identity[$i * 64 + $i] = 1.0;
    $largeProduct = A::ones([64, 64], $dtype)
        ->matmul(A::fromFlatArray($identity, [64, 64], $dtype));
    check($largeProduct->toArray() === array_fill(0, 64, array_fill(0, 64, 1.0)),
        "$dtype 2D matmul result mismatch");
    check(in_array(cuda_get_backend_info()['lastMatmul'],
        ['cublas-low-precision', 'builtin-low-precision'], true),
        "$dtype 2D matmul did not use a low-precision backend");

    $approximate = A::fromFlatArray([0.1, 0.2, 0.3, 0.4], [2, 2], $dtype)
        ->matmul(A::fromFlatArray([0.5, 0.6, 0.7, 0.8], [2, 2], $dtype))
        ->toArray();
    $expected = [[0.19, 0.22], [0.43, 0.50]];
    foreach ($expected as $row => $values)
        foreach ($values as $column => $value)
            check(abs($approximate[$row][$column] - $value) < 0.005,
                "$dtype matmul exceeded the numerical tolerance");

    $scalarLeft = A::fromFlatArray([1, 2], [1, 2], $dtype)->requiresGrad();
    $scalarRight = A::fromFlatArray([3, 4], [2, 1], $dtype)->requiresGrad();
    $scalarProduct = $scalarLeft->matmul($scalarRight);
    $scalarProduct->backward();
    check($scalarProduct->toArray() === [[11.0]], "$dtype tracked matmul result mismatch");
    check($scalarLeft->grad()->dtype() === 'float32' &&
        $scalarLeft->grad()->toArray() === [[3.0, 4.0]],
        "$dtype left gradient must accumulate in float32");
    check($scalarRight->grad()->dtype() === 'float32' &&
        $scalarRight->grad()->toArray() === [[1.0], [2.0]],
        "$dtype right gradient must accumulate in float32");

    $seededLeft = A::fromFlatArray([1, 2, 3, 4], [2, 2], $dtype)->requiresGrad();
    $seededRight = A::fromFlatArray([5, 6, 7, 8], [2, 2], $dtype)->requiresGrad();
    $seededProduct = $seededLeft->matmul($seededRight);
    $seededProduct->backward(A::fromFlatArray([1, 0, 0, 1], [2, 2], $dtype));
    check($seededLeft->grad()->dtype() === 'float32' &&
        $seededLeft->grad()->toArray() === [[5.0, 7.0], [6.0, 8.0]],
        "$dtype explicit seed gradient mismatch");
    check($seededRight->grad()->toArray() === [[1.0, 3.0], [2.0, 4.0]],
        "$dtype right explicit seed gradient mismatch");

    $broadcastTrackedLeft = A::fromFlatArray(
        [1, 2, 3, 4, 2, 0, 0, 2], [2, 1, 2, 2], $dtype)->requiresGrad();
    $broadcastTrackedRight = A::fromFlatArray(
        [5, 6, 1, 2, 2, 1], [1, 3, 2, 1], $dtype)->requiresGrad();
    $broadcastProduct = $broadcastTrackedLeft->matmul($broadcastTrackedRight);
    $broadcastProduct->backward(A::ones([2, 3, 2, 1], $dtype));
    check($broadcastTrackedLeft->grad()->dtype() === 'float32' &&
        $broadcastTrackedLeft->grad()->toArray() === [
            [[[8.0, 9.0], [8.0, 9.0]]],
            [[[8.0, 9.0], [8.0, 9.0]]],
        ], "$dtype broadcasted left gradient mismatch");
    check($broadcastTrackedRight->grad()->toArray() === [
        [[[6.0], [8.0]], [[6.0], [8.0]], [[6.0], [8.0]]],
    ], "$dtype broadcasted right gradient mismatch");

    $transposedLeaf = A::fromFlatArray([1, 2, 3, 4], [2, 2], $dtype)->requiresGrad();
    $transposedLeaf->transpose()->matmul(A::ones([2, 1], $dtype))
        ->backward(A::ones([2, 1], $dtype));
    check($transposedLeaf->grad()->dtype() === 'float32' &&
        $transposedLeaf->grad()->toArray() === [[1.0, 1.0], [1.0, 1.0]],
        "$dtype transposed-view gradient mismatch");

    $fusedLeft = A::fromFlatArray([1, 2], [1, 2], $dtype)->requiresGrad();
    $fusedRight = A::fromFlatArray([3, 4], [2, 1], $dtype)->requiresGrad();
    $gradientGraph = Fusion::compile(function ($a, $b) {
        $product = $a->matmul($b);
        $product->backward();
        return [$product, $a->grad(), $b->grad()];
    }, [$fusedLeft, $fusedRight]);
    [$fusedProduct, $fusedLeftGradient, $fusedRightGradient] =
        $gradientGraph->run($fusedLeft, $fusedRight);
    check($fusedProduct->toArray() === [[11.0]] &&
        $fusedLeftGradient->dtype() === 'float32' &&
        $fusedLeftGradient->toArray() === [[3.0, 4.0]] &&
        $fusedRightGradient->toArray() === [[1.0], [2.0]],
        "$dtype Fusion matmul backward mismatch");

    $castTracked = A::ones([2, 2], $dtype)->requiresGrad();
    $castTracked->astype('float32')->sum()->backward();
    check($castTracked->grad()->dtype() === 'float32' &&
        $castTracked->grad()->toArray() === [[1.0, 1.0], [1.0, 1.0]],
        "$dtype widening cast gradient mismatch");
}

echo "FP16/BF16 conversion and matmul checks passed\n";
?>
--EXPECT--
FP16/BF16 conversion and matmul checks passed
