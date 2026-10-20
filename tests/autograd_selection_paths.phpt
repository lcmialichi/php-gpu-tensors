--TEST--
Conditional gradients preserve predicate histories and inactive branches receive genuine zeros
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
    foreach ([false, true] as $fused) {
        $run = fn(callable $operation) => $fused ? F::run($operation) : $operation();
        $source = (new A([-1, 2], $dtype))->requiresGrad();
        $branch = (new A([3, 4], $dtype))->requiresGrad();
        $run(function () use ($source, $branch) {
            $history = $source * 2;
            A::where($history->gt(0), $branch, $branch->neg())->sum()->backward();
            check($source->grad() === null, 'Predicate received a gradient');
            $history->sum()->backward();
            return [$source->grad(), $branch->grad()];
        });
        check($source->grad()->toArray() === [2.0, 2.0], "$dtype predicate history was consumed");
        check($branch->grad()->toArray() === [-1.0, 1.0], "$dtype branch derivatives");

        $left = (new A([1, 2], $dtype))->requiresGrad();
        $right = (new A([3, 4], $dtype))->requiresGrad();
        $run(function () use ($left, $right, $dtype) {
            $selected = A::where(new A([1, 0], 'int32'), $left, $right);
            $selected->backward(new A([INF, NAN], $dtype));
            return [$left->grad(), $right->grad()];
        });
        $leftGradient = $left->grad()->toArray();
        $rightGradient = $right->grad()->toArray();
        check($leftGradient[0] === INF && $leftGradient[1] === 0.0 &&
              $rightGradient[0] === 0.0 && is_nan($rightGradient[1]),
              "$dtype inactive where branch was contaminated by nonfinite seed");
        foreach (['ceil', 'floor', 'round'] as $operation) {
            $input = (new A([1.25, -2.5], $dtype))->requiresGrad();
            $run(function () use ($input, $operation, $dtype) {
                $input->$operation()->backward(new A([INF, NAN], $dtype));
                return $input->grad();
            });
            check($input->grad()->toArray() === [0.0, 0.0], "$dtype $operation zero derivative");
        }
        foreach (['maximum', 'minimum'] as $operation) {
            $left = (new A([2, 1], $dtype))->requiresGrad();
            $right = (new A([1, 2], $dtype))->requiresGrad();
            $run(function () use ($left, $right, $dtype, $operation) {
                $left->$operation($right)->backward(new A([INF, NAN], $dtype));
                return [$left->grad(), $right->grad()];
            });
            $active = $operation === 'maximum' ? $left->grad()->toArray() : $right->grad()->toArray();
            $inactive = $operation === 'maximum' ? $right->grad()->toArray() : $left->grad()->toArray();
            check($active[0] === INF && $active[1] === 0.0 &&
                  $inactive[0] === 0.0 && is_nan($inactive[1]), "$dtype $operation masked zeros");
        }
        $left = (new A([[1, 2]], $dtype))->requiresGrad();
        $right = (new A([[3], [4]], $dtype))->requiresGrad();
        $run(function () use ($left, $right) {
            A::where(new A([[1], [0]], 'int32'), $left, $right)->sum()->backward();
            return [$left->grad(), $right->grad()];
        });
        check($left->grad()->toArray() === [[1.0, 1.0]] &&
              $right->grad()->toArray() === [[0.0], [2.0]], "$dtype broadcast branch gradients");
    }
}
$source = (new A([-1, 2]))->requiresGrad();
$condition = $source->gt(0);
$branch = (new A([3, 4]))->requiresGrad();
$loss = A::where($condition, $branch, $branch->neg())->sum();
$source[0] = 7;
$loss->backward();
check($branch->grad()->toArray() === [-1.0, 1.0], 'Predicate-source mutation invalidated saved mask');

$condition = (new A([0, 1]))->requiresGrad();
$branch = (new A([3, 4]))->requiresGrad();
$loss = A::where($condition, $branch, $branch->neg())->sum();
$condition[0] = 1;
try {
    $loss->backward();
    throw new RuntimeException('Saved mask mutation was accepted');
} catch (PhpTensors\InvalidArgumentException $error) {
    check(str_contains($error->getMessage(), 'modified in place'), 'Saved mask mutation error');
}
check($branch->grad() === null, 'Failed mask validation committed a partial gradient');
A::where($condition, $branch, $branch->neg())->sum()->backward();
check($branch->grad()->toArray() === [1.0, 1.0], 'Fresh graph did not recover after mask mutation');

foreach ([false, true] as $fused) {
    $run = fn(callable $operation) => $fused ? F::run($operation) : $operation();
    $source = (new A([-1, 2]))->requiresGrad();
    $output = $run(fn() => A::where($source->gt(0), new A([3, 4]), new A([5, 6])));
    check($output->toArray() === [5.0, 4.0] && $source->grad() === null,
          'Condition-only tracking created a branch gradient');
    try {
        $output->backward(new A([1, 1]));
        throw new RuntimeException('Condition-only output unexpectedly tracked gradients');
    } catch (PhpTensors\InvalidArgumentException $error) {
        check(str_contains($error->getMessage(), 'requiresGrad'), 'Condition-only backward error');
    }
    try {
        $run(function () use ($source) {
            $mask = $source->gt(0);
            $loss = A::where($mask, $mask, new A([0, 0]))->sum();
            $loss->backward();
            return $loss;
        });
        throw new RuntimeException('An unsupported mask derivative was silently skipped as a branch');
    } catch (PhpTensors\InvalidArgumentException $error) {
        check(str_contains($error->getMessage(), 'not implemented'), 'Numeric mask branch derivative error');
    }
}
$left = (new A([1, 2]))->requiresGrad();
$right = (new A([3, 4]))->requiresGrad();
$graph = F::compile(function ($mask, $a, $b) {
    A::where($mask, $a, $b)->backward(new A([INF, NAN]));
    return [$a->grad(), $b->grad()];
}, [new A([1, 0], 'int32'), $left, $right]);
[$ga, $gb] = $graph->run(new A([0, 1], 'int32'), $left, $right);
$a = $ga->toArray();
$b = $gb->toArray();
check($a[0] === 0.0 && is_nan($a[1]) && $b[0] === INF && $b[1] === 0.0,
      'Conditional gradient replay froze masks or contaminated zero branches');
echo "Conditional histories and zero-gradient branches passed\n";
?>
--EXPECT--
Conditional histories and zero-gradient branches passed
