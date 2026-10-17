--TEST--
FP16/BF16 eager and Fusion elementwise parity with FP32 gradients
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

foreach (['float16', 'bfloat16'] as $dtype) {
    $input = A::fromFlatArray([0.5, 1.25, 2.0], [3], $dtype);
    $other = A::fromFlatArray([1.0, 0.25, 2.0], [3], $dtype);
    $eager = $input->sin() * 2 + $other;
    $graph = Fusion::compile(fn($x, $y) => $x->sin() * 2 + $y, [$input, $other]);
    $fused = $graph->run($input, $other);
    check($eager->dtype() === $dtype, "$dtype eager output dtype mismatch");
    check($fused->dtype() === $dtype, "$dtype Fusion output dtype mismatch");
    check($fused->toBuffer() === $eager->toBuffer(), "$dtype Fusion rounding mismatch");

    $inverseEager = 2 - $input;
    $inverseFused = Fusion::run(fn() => 2 - $input);
    check($inverseFused->toBuffer() === $inverseEager->toBuffer(),
        "$dtype inverse scalar mismatch");

    $column = A::fromFlatArray([1.0, 2.0], [2, 1], $dtype);
    $row = A::fromFlatArray([0.25, 0.5, 0.75], [1, 3], $dtype);
    $broadcastEager = $column + $row;
    $broadcastGraph = Fusion::compile(fn($x, $y) => $x + $y, [$column, $row]);
    check($broadcastGraph->run($column, $row)->toBuffer() === $broadcastEager->toBuffer(),
        "$dtype broadcast result mismatch");

    $eagerTracked = A::fromFlatArray([1.0, 2.0], [2], $dtype)->requiresGrad();
    $eagerSeed = A::fromFlatArray([1.0, 1.0], [2], $dtype);
    ($eagerTracked * $eagerTracked)->backward($eagerSeed);
    check($eagerTracked->grad()->dtype() === 'float32' &&
        $eagerTracked->grad()->toArray() === [2.0, 4.0],
        "$dtype eager gradient mismatch");

    $tracked = A::fromFlatArray([1.0, 2.0], [2], $dtype)->requiresGrad();
    $seed = A::fromFlatArray([1.0, 1.0], [2], $dtype);
    [$product, $gradient] = Fusion::run(function () use ($tracked, $seed) {
        $product = $tracked * $tracked;
        $product->backward($seed);
        return [$product, $tracked->grad()];
    });
    check($product->dtype() === $dtype, "$dtype gradient output dtype mismatch");
    check($gradient->dtype() === 'float32', "$dtype gradient accumulation dtype mismatch");
    check($gradient->toArray() === [2.0, 4.0], "$dtype gradient values mismatch");

    $condition = A::fromFlatArray([true, false, true], [3], 'bool');
    $selectedTrue = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype);
    $selectedFalse = A::fromFlatArray([4.0, 5.0, 6.0], [3], $dtype);
    $eagerWhere = A::where($condition, $selectedTrue, $selectedFalse);
    $fusedWhere = Fusion::compile(
        fn($mask, $x, $y) => A::where($mask, $x, $y),
        [$condition, $selectedTrue, $selectedFalse]
    )->run($condition, $selectedTrue, $selectedFalse);
    check($fusedWhere->dtype() === $dtype &&
        $fusedWhere->toBuffer() === $eagerWhere->toBuffer(),
        "$dtype where result mismatch");

    $trackedTrue = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)->requiresGrad();
    $trackedFalse = A::fromFlatArray([4.0, 5.0, 6.0], [3], $dtype)->requiresGrad();
    $seed = A::ones([3], $dtype);
    A::where($condition, $trackedTrue, $trackedFalse)->backward($seed);
    $expectedTrueGradient = $trackedTrue->grad()->toArray();
    $expectedFalseGradient = $trackedFalse->grad()->toArray();

    $fusedTrackedTrue = A::fromFlatArray([1.0, 2.0, 3.0], [3], $dtype)->requiresGrad();
    $fusedTrackedFalse = A::fromFlatArray([4.0, 5.0, 6.0], [3], $dtype)->requiresGrad();
    [, $fusedTrueGradient, $fusedFalseGradient] = Fusion::run(
        function () use ($condition, $fusedTrackedTrue, $fusedTrackedFalse, $seed) {
            $output = A::where($condition, $fusedTrackedTrue, $fusedTrackedFalse);
            $output->backward($seed);
            return [$output, $fusedTrackedTrue->grad(), $fusedTrackedFalse->grad()];
        }
    );
    check($fusedTrueGradient->toArray() === $expectedTrueGradient &&
        $fusedFalseGradient->toArray() === $expectedFalseGradient,
        "$dtype where gradients mismatch");
}

$mask = A::fromFlatArray([0, 1], [2], 'float16');
$true = new A([10.0, 20.0]);
$false = new A([30.0, 40.0]);
$where = A::where($mask, $true, $false);
$fusedWhere = Fusion::run(fn() => A::where($mask, $true, $false));
check($where->toArray() === [30.0, 20.0], 'low-precision where condition mismatch');
check($fusedWhere->toBuffer() === $where->toBuffer(), 'Fusion low-precision where condition mismatch');
echo "FP16/BF16 eager and Fusion elementwise checks passed\n";
?>
--EXPECT--
FP16/BF16 eager and Fusion elementwise checks passed
