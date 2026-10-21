--TEST--
Extrema split tied gradients equally and preserve precision, zero branches and clamp composition
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
function check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}
foreach (['float32', 'float64', 'float16', 'bfloat16'] as $dtype) {
    $gradientDtype = $dtype === 'float64' ? 'float64' : 'float32';
    foreach ([false, true] as $fused) {
        $run = fn(callable $operation) => $fused ? F::run($operation) : $operation();
        foreach (['maximum', 'minimum'] as $operation) {
            $a = (new A([1, 3, 2], $dtype))->requiresGrad();
            $b = (new A([1, 2, 3], $dtype))->requiresGrad();
            $run(function () use ($a, $b, $operation, $dtype) {
                $a->$operation($b)->backward(new A([2, 4, 6], $dtype));
                return [$a->grad(), $b->grad()];
            });
            $expectedA = $operation === 'maximum' ? [1.0, 4.0, 0.0] : [1.0, 0.0, 6.0];
            $expectedB = $operation === 'maximum' ? [1.0, 0.0, 6.0] : [1.0, 4.0, 0.0];
            check($a->grad()->toArray() === $expectedA && $b->grad()->toArray() === $expectedB,
                  "$dtype $operation binary tie sharing fused=$fused");
            check($a->grad()->dtype() === $gradientDtype && $b->grad()->dtype() === $gradientDtype,
                  "$dtype $operation gradient dtype");

            $x = (new A([0, 1, 2], $dtype))->requiresGrad();
            $run(function () use ($x, $operation) {
                $x->$operation(1)->sum()->backward();
                return $x->grad();
            });
            check($x->grad()->toArray() === ($operation === 'maximum'
                ? [0.0, 0.5, 1.0] : [1.0, 0.5, 0.0]), "$dtype $operation scalar tie");

            $a = (new A([[1], [2]], $dtype))->requiresGrad();
            $b = (new A([[1, 2]], $dtype))->requiresGrad();
            $run(function () use ($a, $b, $operation) {
                $a->$operation($b)->sum()->backward();
                return [$a->grad(), $b->grad()];
            });
            $expectedA = $operation === 'maximum' ? [[0.5], [1.5]] : [[1.5], [0.5]];
            $expectedB = $operation === 'maximum' ? [[0.5, 1.5]] : [[1.5, 0.5]];
            check($a->grad()->toArray() === $expectedA && $b->grad()->toArray() === $expectedB,
                  "$dtype $operation broadcast tie accumulation");

            $x = (new A([1, 2], $dtype))->requiresGrad();
            $run(function () use ($x, $operation) {
                $x->$operation($x)->sum()->backward();
                return $x->grad();
            });
            check($x->grad()->toArray() === [1.0, 1.0], "$dtype repeated-parent tie");
            $x = (new A([1], $dtype))->requiresGrad();
            $run(function () use ($x, $dtype, $operation) {
                $x->$operation(new A([1], $dtype))->sum()->backward();
                return $x->grad();
            });
            check($x->grad()->toArray() === [0.5], "$dtype constant operand tie");

            $x = (new A([1], $dtype))->requiresGrad();
            $bound = $operation === 'maximum' ? 1.00000001 : 0.99999999;
            $run(function () use ($x, $operation, $bound) {
                $x->$operation($bound)->sum()->backward();
                return $x->grad();
            });
            check($x->grad()->toArray() === [0.0],
                  "$dtype $operation predicate rounded a promoted scalar");
            check($x->grad()->dtype() === $gradientDtype,
                  "$dtype $operation promoted scalar gradient dtype");

            $a = (new A([1], $dtype))->requiresGrad();
            $b = (new A([1], 'float64'))->requiresGrad();
            $run(function () use ($a, $b, $operation) {
                ($a->$operation($b)->sum() + ($a * 3)->sum())->backward();
                return [$a->grad(), $b->grad()];
            });
            check($a->grad()->dtype() === $gradientDtype && $a->grad()->toArray() === [3.5] &&
                  $b->grad()->dtype() === 'float64' && $b->grad()->toArray() === [0.5],
                  "$dtype $operation mixed-precision accumulation");

            $delta = ['float32' => 2 ** -23, 'float64' => 2 ** -52,
                      'float16' => 2 ** -10, 'bfloat16' => 2 ** -7][$dtype];
            $a = (new A([1], $dtype))->requiresGrad();
            $b = (new A([1 + $delta], $dtype))->requiresGrad();
            $run(function () use ($a, $b, $operation) {
                $a->$operation($b)->sum()->backward();
                return [$a->grad(), $b->grad()];
            });
            check($a->grad()->toArray() === [$operation === 'maximum' ? 0.0 : 1.0] &&
                  $b->grad()->toArray() === [$operation === 'maximum' ? 1.0 : 0.0],
                  "$dtype $operation confused adjacent values with ties");

            $base = (new A([[1, 2], [1, 3]], $dtype))->requiresGrad();
            $other = (new A([[1, 1], [2, 3]], $dtype))->requiresGrad();
            $run(function () use ($base, $other, $operation) {
                $base->transpose()->$operation($other)->sum()->backward();
                return [$base->grad(), $other->grad()];
            });
            check($base->grad()->toArray() === [[0.5, 0.5], [0.5, 0.5]] &&
                  $other->grad()->toArray() === [[0.5, 0.5], [0.5, 0.5]], "$dtype strided tie masks");

            $a = (new A([INF, -INF, -0.0, 0.0], $dtype))->requiresGrad();
            $b = (new A([INF, -INF, 0.0, -0.0], $dtype))->requiresGrad();
            $run(function () use ($a, $b, $operation, $dtype) {
                $a->$operation($b)->backward(A::ones([4], $dtype));
                return [$a->grad(), $b->grad()];
            });
            check($a->grad()->toArray() === [0.5, 0.5, 0.5, 0.5] &&
                  $b->grad()->toArray() === [0.5, 0.5, 0.5, 0.5], "$dtype IEEE equal-value ties");

            $a = (new A([1, 3, 1], $dtype))->requiresGrad();
            $b = (new A([1, 2, 2], $dtype))->requiresGrad();
            $run(function () use ($a, $b, $dtype, $operation) {
                $a->$operation($b)->backward(new A([INF, NAN, INF], $dtype));
                return [$a->grad(), $b->grad()];
            });
            $ga = $a->grad()->toArray();
            $gb = $b->grad()->toArray();
            check($ga[0] === INF && $gb[0] === INF, "$dtype nonfinite tied seed");
            check($operation === 'maximum'
                ? (is_nan($ga[1]) && $ga[2] === 0.0 && $gb[1] === 0.0 && $gb[2] === INF)
                : ($ga[1] === 0.0 && $ga[2] === INF && is_nan($gb[1]) && $gb[2] === 0.0),
                "$dtype nonfinite inactive extrema branches");
            $x = (new A([0, 1, 2], $dtype))->requiresGrad();
            $run(function () use ($x, $operation, $dtype) {
                $x->$operation(1)->backward(new A([INF, INF, NAN], $dtype));
                return $x->grad();
            });
            $gradient = $x->grad()->toArray();
            check($operation === 'maximum'
                ? ($gradient[0] === 0.0 && $gradient[1] === INF && is_nan($gradient[2]))
                : ($gradient === [INF, INF, 0.0]), "$dtype nonfinite scalar tie seed");
        }
        foreach (['max', 'min'] as $operation) {
            $x = (new A($operation === 'max' ? [3, 3, 1] : [1, 1, 3], $dtype))->requiresGrad();
            $run(function () use ($x, $operation) {
                $output = $x->$operation();
                $output->backward(A::full([1], INF, $output->dtype()));
                return $x->grad();
            });
            check($x->grad()->toArray() === [INF, INF, 0.0], "$dtype reduction inactive zero");
        }
        $x = (new A([-1, 0, 1, 2, 3], $dtype))->requiresGrad();
        $run(function () use ($x) {
            $x->clamp(0, 2)->sum()->backward();
            return $x->grad();
        });
        check($x->grad()->toArray() === [0.0, 0.5, 1.0, 0.5, 0.0], "$dtype clamp boundaries");
        $x = (new A([0, 1, 2], $dtype))->requiresGrad();
        $run(function () use ($x) {
            $x->clamp(1, 1)->sum()->backward();
            return $x->grad();
        });
        check($x->grad()->toArray() === [0.0, 0.25, 0.0], "$dtype zero-width clamp composition");
        $empty = A::fromBuffer('', [0, 2], $dtype)->requiresGrad();
        $run(function () use ($empty) {
            $empty->maximum($empty)->sum()->backward();
            return $empty->grad();
        });
        check($empty->grad()->getShape() === [0, 2] && $empty->grad()->toArray() === [],
              "$dtype empty extrema gradient");
    }
}
$a = (new A([1, 1], 'float64'))->requiresGrad();
$b = (new A([1, 1], 'float64'))->requiresGrad();
$graph = F::compile(function ($a, $b) {
    $a->maximum($b)->sum()->backward();
    return [$a->grad(), $b->grad()];
}, [$a, $b]);
[$ga, $gb] = $graph->run(new A([2, 1], 'float64'), new A([1, 1], 'float64'));
check($ga->toArray() === [1.0, 0.5] && $gb->toArray() === [0.0, 0.5], 'Tie replay retained capture masks');
echo "Extrema ties, precision and composed boundary gradients passed\n";
?>
--EXPECT--
Extrema ties, precision and composed boundary gradients passed
