--TEST--
Autograd defines singularities, constant standard deviation and explicit compiled-graph boundaries
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion as F;

function values(A $gradient, array $expected, string $dtype, string $label): void
{
    if ($gradient->dtype() !== ($dtype === 'float64' ? 'float64' : 'float32')) {
        throw new RuntimeException("$label: gradient dtype changed");
    }
    $flat = [];
    $array = $gradient->toArray();
    array_walk_recursive($array, function (float $value) use (&$flat): void { $flat[] = $value; });
    if (count($flat) !== count($expected)) throw new RuntimeException("$label: gradient size changed");
    foreach ($flat as $index => $actual) {
        $reference = $expected[$index];
        $valid = is_nan($reference) ? is_nan($actual)
            : (is_infinite($reference) ? $actual === $reference
               : is_finite($actual) && abs($actual - $reference) <= 3e-6 * (1 + abs($reference)));
        if (!$valid) throw new RuntimeException("$label $dtype index $index: expected $reference, got $actual");
    }
}

foreach (['float32', 'float64', 'float16', 'bfloat16'] as $dtype) {
    foreach ([false, true] as $fused) {
        $run = fn(callable $callback) => $fused ? F::run($callback) : $callback();
        foreach (['log' => 1, 'log2' => 1 / log(2), 'log10' => 1 / log(10)] as $operation => $unit) {
            $x = (new A([-1, -0.0, 0.0, 1, INF, NAN], $dtype))->requiresGrad();
            $gradient = $run(function () use ($x, $operation): A {
                $x->$operation()->sum()->backward();
                return $x->grad();
            });
            values($gradient, [NAN, -INF, INF, $unit, 0, NAN], $dtype, $operation);
        }
        $x = (new A([-2, -1, 0, 1, INF, NAN], $dtype))->requiresGrad();
        $gradient = $run(function () use ($x): A {
            $x->log1p()->sum()->backward();
            return $x->grad();
        });
        values($gradient, [NAN, INF, 1, 0.5, 0, NAN], $dtype, 'log1p');

        foreach (['sqrt', 'asin', 'acos'] as $operation) {
            $input = $operation === 'sqrt' ? [-1, -0.0, 0.0, 1, INF, NAN] : [-2, -1, 0, 1, 2, NAN];
            $x = (new A($input, $dtype))->requiresGrad();
            $gradient = $run(function () use ($x, $operation): A {
                $x->$operation()->sum()->backward();
                return $x->grad();
            });
            $expected = $operation === 'sqrt' ? [NAN, -INF, INF, 0.5, 0, NAN]
                : ($operation === 'asin' ? [NAN, INF, 1, INF, NAN, NAN] : [NAN, -INF, -1, -INF, NAN, NAN]);
            values($gradient, $expected, $dtype, $operation);
        }
        foreach (['abs', 'ceil', 'floor', 'round'] as $operation) {
            $x = (new A([-2, -0.0, 0.0, 2, NAN, INF], $dtype))->requiresGrad();
            $gradient = $run(function () use ($x, $operation): A {
                $x->$operation()->sum()->backward();
                return $x->grad();
            });
            values($gradient, $operation === 'abs' ? [-1, 0, 0, 1, 0, 1] : [0, 0, 0, 0, 0, 0],
                   $dtype, $operation);
        }
        $base = (new A([0, 0, 0, -2, 1], $dtype))->requiresGrad();
        $exponent = (new A([0, 1, 2, 3, NAN], $dtype))->requiresGrad();
        [$ga, $gb] = $run(function () use ($base, $exponent): array {
            ($base ** $exponent)->sum()->backward();
            return [$base->grad(), $exponent->grad()];
        });
        values($ga, [0, 1, 0, 12, NAN], $dtype, 'power base partial');
        values($gb, [-INF, 0, 0, NAN, 0], $dtype, 'power exponent partial');

        $base = (new A([-2, 0, INF, NAN], $dtype))->requiresGrad();
        $exponent = (new A([0, 0, 0, 0], $dtype))->requiresGrad();
        $gradient = $run(function () use ($base, $exponent): A {
            $output = $base ** $exponent;
            $output->backward(new A([NAN, INF, NAN, INF], $output->dtype()));
            return $base->grad();
        });
        values($gradient, [0, 0, 0, 0], $dtype, 'constant tensor exponent');

        $x = (new A([-1, 0, 1, 2], $dtype))->requiresGrad();
        $gradient = $run(function () use ($x): A {
            (0 ** $x)->sum()->backward();
            return $x->grad();
        });
        values($gradient, [-INF, -INF, 0, 0], $dtype, 'zero scalar base');

        foreach ([null, 1] as $axis) {
            foreach ([0, 1] as $correction) {
                $x = (new A([[2, 2], [2, 2]], $dtype))->requiresGrad();
                $gradient = $run(function () use ($x, $axis, $correction): A {
                    $output = $x->std($axis, $correction);
                    $output->backward(A::full($output->getShape(), NAN, $output->dtype()));
                    return $x->grad();
                });
                values($gradient, [0, 0, 0, 0], $dtype, 'constant std');
            }
        }
        $x = (new A([[1, 1], [1, 3]], $dtype))->requiresGrad();
        $gradient = $run(function () use ($x): A {
            $output = $x->std(1);
            $output->backward(new A([INF, 2], $output->dtype()));
            return $x->grad();
        });
        values($gradient, [0, 0, -1, 1], $dtype, 'mixed constant std');
    }
}

foreach (['float32', 'float64', 'float16', 'bfloat16'] as $dtype) {
    $example = (new A([1, 1], $dtype))->requiresGrad();
    foreach ([
        fn($x) => $x,
        fn($x) => ['nested' => [$x * $x]],
        function ($x): array {
            $other = $x + 1;
            ($x * 2)->sum()->backward();
            return [$other, $x->grad()];
        },
        function ($x): A {
            ($x * $x)->sum()->backward();
            return $x * $x;
        },
    ] as $callback) {
        try {
            F::compile($callback, [$example]);
            throw new RuntimeException('Compiled forward silently detached a tracked output');
        } catch (PhpTensors\InvalidArgumentException $error) {
            if (!str_contains($error->getMessage(), 'backward')) throw $error;
        }
    }
    $detached = F::compile(fn($x) => ($x * $x)->detach(), [$example]);
    if ($detached->run(new A([2, 3], $dtype))->toArray() !== [4.0, 9.0]) {
        throw new RuntimeException('Explicit compiled detach changed values');
    }
    $graph = F::compile(function ($x): array {
        $std = $x->std();
        $std->backward(retainGraph: true);
        return [$std, $x->grad()];
    }, [$example]);
    foreach ([[2, 2], [1, 3]] as $input) {
        [$std, $gradient] = $graph->run(new A($input, $dtype));
        values($gradient, $input[0] === $input[1] ? [0, 0] : [-0.5, 0.5], $dtype, 'std replay');
    }
    $exampleExponent = (new A([0, 0], $dtype))->requiresGrad();
    $power = F::compile(function ($a, $b): array {
        $loss = ($a ** $b)->sum();
        $loss->backward();
        return [$loss, $a->grad(), $b->grad()];
    }, [$example, $exampleExponent]);
    [, $ga, $gb] = $power->run(new A([0, 2], $dtype), new A([2, 3], $dtype));
    values($ga, [0, 12], $dtype, 'power replay base');
    values($gb, [0, 8 * log(2)], $dtype, 'power replay exponent');
    [, $ga, $gb] = $power->run(new A([1, -2], $dtype), new A([NAN, 0], $dtype));
    values($ga, [NAN, 0], $dtype, 'power replay singular base');
    values($gb, [0, NAN], $dtype, 'power replay singular exponent');
    $x = (new A([2], $dtype))->requiresGrad();
    F::run(fn() => $x * $x)->sum()->backward();
    values($x->grad(), [4], $dtype, 'backward after Fusion run');
    $x->zeroGrad();
    $gradient = F::run(function () use ($x): A {
        ($x * $x)->sum()->backward();
        ($x * $x)->sum()->backward();
        return $x->grad();
    });
    values($gradient, [8], $dtype, 'recomputed consumed expression');
    foreach ([false, true] as $cudaGraph) {
        $inference = F::compile(fn($input) => $input * $input,
                               [new A([1, 2], $dtype)], $cudaGraph);
        foreach (['run', 'runAsync'] as $method) {
            try {
                $inference->$method((new A([1, 2], $dtype))->requiresGrad());
                throw new RuntimeException('Inference replay silently detached a tracked input');
            } catch (PhpTensors\InvalidArgumentException $error) {
                if (!str_contains($error->getMessage(), 'before compile')) throw $error;
            }
        }
        if ($inference->run(new A([2, 3], $dtype))->toArray() !== [4.0, 9.0]) {
            throw new RuntimeException('Rejected tracked replay left the graph unusable');
        }
    }
}
echo "Singularities, constant std and compiled autograd boundaries passed\n";
?>
--EXPECT--
Singularities, constant std and compiled autograd boundaries passed
