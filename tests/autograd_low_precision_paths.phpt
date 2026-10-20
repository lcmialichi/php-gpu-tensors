--TEST--
Low-precision concat and slices retain FP32 gradient paths in eager and Fusion
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion as F;
function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}
foreach (['float16', 'bfloat16'] as $dtype) {
    foreach ([false, true] as $fused) {
        $a = (new A([[1, 2], [3, 4]], $dtype))->requiresGrad();
        $b = (new A([[5], [6]], $dtype))->requiresGrad();
        $compute = function () use ($a, $b) {
            $joined = $a->concat([$b], 1);
            $selected = $joined->slice(':, ::2')->copy();
            $loss = ($selected * $selected)->sum();
            $loss->backward();
            return [$loss, $a->grad(), $b->grad()];
        };
        [$loss, $ga, $gb] = $fused ? F::run($compute) : $compute();
        check($loss->item() === 71.0, "$dtype concat/slice forward");
        check($ga !== null && $gb !== null, "$dtype missing gradient path");
        check($ga->dtype() === 'float32' && $gb->dtype() === 'float32', "$dtype gradient dtype");
        check($ga->toArray() === [[2.0, 0.0], [6.0, 0.0]], "$dtype slice scatter gradient");
        check($gb->toArray() === [[10.0], [12.0]], "$dtype concat gradient");

        $x = (new A([1, 2], $dtype))->requiresGrad();
        $repeat = function () use ($x) {
            $x->concat([$x])->sum()->backward();
            return $x->grad();
        };
        $gradient = $fused ? F::run($repeat) : $repeat();
        check($gradient !== null && $gradient->toArray() === [2.0, 2.0], "$dtype repeated parent accumulation");

        $branch = (new A([-1, 2], $dtype))->requiresGrad();
        $piecewise = function () use ($branch) {
            A::where($branch->gt(0), $branch * $branch, $branch->neg())->sum()->backward();
            return $branch->grad();
        };
        $gradient = $fused ? F::run($piecewise) : $piecewise();
        check($gradient !== null && $gradient->toArray() === [-1.0, 4.0], "$dtype conditional gradient");

        $x = (new A([1, 2], $dtype))->requiresGrad();
        try {
            $unsupported = function () use ($x) { $x->eq($x)->sum()->backward(); };
            if ($fused) F::run($unsupported); else $unsupported();
            throw new RuntimeException("$dtype unsupported comparison derivative was accepted");
        } catch (PhpTensors\InvalidArgumentException $error) {
            check(str_contains($error->getMessage(), 'gradient tracking cannot skip this path'),
                  "$dtype comparison silently dropped its gradient edge");
        }
    }
}
foreach (['float32', 'float64', 'float16', 'bfloat16'] as $dtype) {
    foreach ([false, true] as $fused) {
        $empty = A::fromBuffer('', [0, 2], $dtype)->requiresGrad();
        $compute = function () use ($empty) {
            $empty->concat([$empty], 1)->sum()->backward();
            return $empty->grad();
        };
        $gradient = $fused ? F::run($compute) : $compute();
        check($gradient !== null && $gradient->getShape() === [0, 2] &&
              $gradient->toArray() === [], "$dtype empty concat gradient footprint");
    }
}
echo "Low-precision concat, slice and repeated-parent gradients passed\n";
echo "Unsupported derivatives and empty gradient views checked\n";
?>
--EXPECT--
Low-precision concat, slice and repeated-parent gradients passed
Unsupported derivatives and empty gradient views checked
