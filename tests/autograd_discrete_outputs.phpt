--TEST--
Discrete reductions require explicit detach for tracked tensors in eager and Fusion
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
function needsDetach(callable $operation): void {
    try {
        $operation();
        throw new RuntimeException('Discrete reduction silently removed the gradient path');
    } catch (PhpTensors\InvalidArgumentException $error) {
        check(str_contains($error->getMessage(), 'detach'), 'Missing explicit detach guidance');
    }
}
$expected = [
    'argMax' => [[2], [1, 1], [0, 0]],
    'argMin' => [[1], [0, 0], [1, 1]],
    'all' => [[false], [true, false], [true, false]],
    'any' => [[true], [true, true], [true, true]],
];
foreach (['float32', 'float64', 'float16', 'bfloat16'] as $dtype) {
    foreach ([false, true] as $fused) {
        $run = fn(callable $operation) => $fused ? F::run($operation) : $operation();
        $input = (new A([[1, -2], [3, 0]], $dtype))->requiresGrad();
        foreach ($expected as $operation => $values) {
            needsDetach(fn() => $run(fn() => $input->$operation()));
            foreach ([null, 0, 1, -1] as $axis)
                needsDetach(fn() => $run(fn() => $input->$operation($axis)));
            needsDetach(fn() => $run(fn() => $input->transpose()->$operation()));
            foreach ([null, 0, 1, -1] as $axis) {
                $output = $run(fn() => $input->detach()->$operation($axis));
                $expectedValues = $axis === null ? $values[0] : ($axis === 0 ? $values[1] : $values[2]);
                check($output->toArray() === $expectedValues, "$dtype $operation detached axis");
                check($output->dtype() === (str_starts_with($operation, 'arg') ? 'int32' : 'bool'),
                      "$dtype $operation discrete dtype");
            }
        }
        $empty = A::fromBuffer('', [0, 2], $dtype)->requiresGrad();
        foreach (array_keys($expected) as $operation)
            needsDetach(fn() => $run(fn() => $empty->$operation()));
        check($run(fn() => $empty->detach()->all())->toArray() === [true], 'Empty detached all');
        check($run(fn() => $empty->detach()->any())->toArray() === [false], 'Empty detached any');
        needsDetach(fn() => $run(fn() => $input->gt(0)->all()));
        $run(function () use ($input) {
            ($input * $input)->sum()->backward();
            return $input->grad();
        });
        check($input->grad()->toArray() === [[2.0, -4.0], [6.0, 0.0]],
              "$dtype rejection damaged subsequent backward");
        foreach (['sum' => 2.0, 'mean' => 0.5, 'max' => 3.0, 'min' => -2.0,
                  'prod' => 0.0, 'argMax' => 2, 'argMin' => 1, 'all' => false, 'any' => true] as $op => $value) {
            check($run(fn() => $input->detach()->$op(null))->item() === $value,
                  "$dtype $op null axis must reduce globally");
        }
    }
}
$input = (new A([1, 2]))->requiresGrad();
foreach (array_keys($expected) as $operation)
    needsDetach(fn() => F::compile(fn($x) => $x->$operation(), [$input]));
$graph = F::compile(fn($x) => [$x->detach()->argMax(), $x->detach()->all()], [$input]);
[$index, $all] = $graph->run((new A([5, 0]))->requiresGrad());
check($index->toArray() === [0] && $all->toArray() === [false], 'Detached reduction replay');
$plain = new A([[1, 2], [3, 4]]);
foreach (['sum', 'mean', 'max', 'min', 'prod', 'argMax', 'argMin', 'all', 'any'] as $operation) {
    foreach ([0.5, '1', true, [], new stdClass(), PHP_INT_MAX] as $axis) {
        try {
            $plain->$operation($axis);
            throw new RuntimeException("$operation accepted an invalid axis");
        } catch (PhpTensors\InvalidArgumentException $expected) {
        }
    }
}
echo "Explicit discrete autograd boundaries passed\n";
?>
--EXPECT--
Explicit discrete autograd boundaries passed
