--TEST--
Autograd agrees with FP64 finite differences and preserves scalar power precision and constant derivatives
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion as F;

$failures = [];

function closeGradient(float $actual, float $expected, string $label,
                       float $absolute = 2e-10, float $relative = 1e-9): void
{
    global $failures;
    if (!is_finite($actual) || !is_finite($expected) ||
        abs($actual - $expected) > $absolute + $relative * abs($expected)) {
        $failures[] = "$label: expected $expected, got $actual";
    }
}

function tensorInputs(array $specs, bool $tracked): array
{
    return array_map(function (array $spec) use ($tracked): A {
        $tensor = A::fromFlatArray($spec[0], $spec[1], 'float64');
        return $tracked ? $tensor->requiresGrad() : $tensor;
    }, $specs);
}

function numericalGradient(string $name, callable $objective, array $specs): void
{
    $expected = [];
    foreach ($specs as $argument => $spec) {
        foreach ($spec[0] as $index => $value) {
            $step = 1e-4 * max(1.0, abs($value));
            $samples = [];
            foreach ([-2, -1, 1, 2] as $offset) {
                $sample = $specs;
                $sample[$argument][0][$index] += $offset * $step;
                $samples[$offset] = $objective(...tensorInputs($sample, false))->item();
            }
            $expected[$argument][$index] =
                ($samples[-2] - 8 * $samples[-1] + 8 * $samples[1] - $samples[2]) / (12 * $step);
        }
    }
    foreach ([false, true] as $fused) {
        $inputs = tensorInputs($specs, true);
        $backward = function () use ($objective, $inputs): array {
            $objective(...$inputs)->backward();
            return array_map(function (A $input): A {
                $gradient = $input->grad();
                if ($gradient === null) throw new RuntimeException('Missing numerical-check gradient');
                return $gradient;
            }, $inputs);
        };
        $gradients = $fused ? F::run($backward) : $backward();
        foreach ($gradients as $argument => $gradient) {
            if ($gradient->dtype() !== 'float64' || $gradient->getShape() !== $specs[$argument][1]) {
                throw new RuntimeException("$name: gradient dtype or shape changed");
            }
            $values = $gradient->toArray();
            $flat = [];
            array_walk_recursive($values, function (float $value) use (&$flat): void {
                $flat[] = $value;
            });
            if (count($flat) !== count($expected[$argument])) {
                throw new RuntimeException("$name: gradient element count changed");
            }
            foreach ($flat as $index => $actual) {
                closeGradient($actual, $expected[$argument][$index],
                              "$name " . ($fused ? 'Fusion' : 'eager') . " argument $argument index $index");
            }
        }
    }
}

$weights = A::fromFlatArray([0.3, -0.7, 1.1, 2.0], [2, 2], 'float64');
$positive = [[[0.4, 0.7, 1.2, 1.6], [2, 2]]];
$signed = [[[-0.6, -0.2, 0.3, 0.7], [2, 2]]];
foreach (['exp', 'expm1', 'exp2', 'log', 'log1p', 'log2', 'log10', 'sqrt',
          'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
          'abs', 'neg', 'ceil', 'floor', 'round'] as $operation) {
    numericalGradient($operation, fn(A $x): A => ($x->$operation() * $weights)->sum(),
                      in_array($operation, ['log', 'log2', 'log10', 'sqrt'], true) ? $positive : $signed);
}

$binaryInputs = [$positive[0], [[0.9, 1.3], [1, 2]]];
foreach (['add', 'subtract', 'multiply', 'divide', 'power', 'maximum', 'minimum'] as $operation) {
    numericalGradient("broadcast $operation",
        fn(A $a, A $b): A => ($a->$operation($b) * $weights)->sum(), $binaryInputs);
}
numericalGradient('scalar power', fn(A $x): A => (($x ** 2.3) * $weights)->sum(), $positive);
numericalGradient('inverse scalar power', fn(A $x): A => ((1.7 ** $x) * $weights)->sum(), $signed);
numericalGradient('inverse scalar division', fn(A $x): A => ((1.7 / $x) * $weights)->sum(), $positive);
numericalGradient('repeated parent', fn(A $x): A => (($x * $x + $x) * $weights)->sum(), $signed);
numericalGradient('transpose and slice',
    fn(A $x): A => ($x->transpose()->slice(':, 0:1')->flip(0) * $weights)->sum(), $signed);
numericalGradient('repeat', fn(A $x): A => ($x->repeat([1, 2], 1) ** 2)->sum(), $signed);
numericalGradient('product with zero', fn(A $x): A => $x->prod(1)->sum(),
                  [[[0.0, 2.0, 3.0, 4.0], [2, 2]]]);
numericalGradient('cumprod with zeros', fn(A $x): A => ($x->cumprod(1) * $weights)->sum(),
                  [[[0.0, 2.0, 0.0, 0.0], [2, 2]]]);
numericalGradient('cumsum', fn(A $x): A => ($x->cumsum(0) * $weights)->sum(), $signed);
numericalGradient('variance', fn(A $x): A => $x->var(1, 1)->sum(), $signed);
numericalGradient('standard deviation', fn(A $x): A => $x->std(1)->sum(), $signed);
numericalGradient('softmax', fn(A $x): A => ($x->softmax(1) * $weights)->sum(), $signed);
numericalGradient('layerNorm',
    fn(A $x, A $scale, A $bias): A => ($x->layerNorm([1], 0.05, $scale, $bias) * $weights)->sum(),
    [$signed[0], [[0.8, 1.3], [2]], [[0.2, -0.1], [2]]]);

foreach (['float32' => 2e-6, 'float64' => 2e-12,
          'float16' => 4e-3, 'bfloat16' => 2e-2] as $dtype => $tolerance) {
    foreach ([false, true] as $fused) {
            foreach ([3, 2.5, -2, true] as $exponent) {
                $x = (new A([0.5, 2.0], $dtype))->requiresGrad();
            $backward = function () use ($x, $exponent): A {
                ($x ** $exponent)->sum()->backward();
                return $x->grad();
            };
            $gradient = $fused ? F::run($backward) : $backward();
            foreach ($gradient->toArray() as $index => $actual) {
                closeGradient($actual, $exponent * [0.5, 2.0][$index] ** ($exponent - 1),
                              "$dtype scalar power $exponent", $tolerance, $tolerance);
            }
        }
        $x = (new A([0.5, 2.0], $dtype))->requiresGrad();
        $backward = function () use ($x): A {
            (1.7 ** $x)->sum()->backward();
            return $x->grad();
        };
        $gradient = $fused ? F::run($backward) : $backward();
        foreach ($gradient->toArray() as $index => $actual) {
            closeGradient($actual, 1.7 ** [0.5, 2.0][$index] * log(1.7),
                          "$dtype inverse power", $tolerance, $tolerance);
        }
        foreach ([0, 0.0, false] as $exponent) {
            $x = (new A([-2.0, 0.0, 2.0, INF, NAN], $dtype))->requiresGrad();
            $backward = function () use ($x, $exponent): A {
                $output = $x ** $exponent;
                $output->backward(new A([INF, NAN, 1, INF, NAN], $output->dtype()));
                return $x->grad();
            };
            $gradient = $fused ? F::run($backward) : $backward();
            if ($gradient->toArray() !== [0.0, 0.0, 0.0, 0.0, 0.0]) {
                $failures[] = "$dtype constant zero power produced a nonzero or non-finite derivative";
            }
        }
        foreach ([1, 1.0, true] as $base) {
            $x = (new A([-INF, 0.0, INF, NAN], $dtype))->requiresGrad();
            $backward = function () use ($x, $base): A {
                $output = $base ** $x;
                $output->backward(new A([INF, NAN, INF, NAN], $output->dtype()));
                return $x->grad();
            };
            $gradient = $fused ? F::run($backward) : $backward();
            if ($gradient->toArray() !== [0.0, 0.0, 0.0, 0.0]) {
                $failures[] = "$dtype constant unit base produced a nonzero or non-finite derivative";
            }
        }
    }
    foreach ([3, 2.5, -2, 0] as $exponent) {
        $example = (new A([0.5, 2.0], $dtype))->requiresGrad();
        $graph = F::compile(function (A $x) use ($exponent): array {
            $loss = ($x ** $exponent)->sum();
            $loss->backward();
            return [$loss, $x->grad()];
        }, [$example]);
        foreach ([[0.25, 3.0], [0.5, 2.0]] as $values) {
            [, $gradient] = $graph->run(new A($values, $dtype));
            $expectedDtype = $dtype === 'float64' ? 'float64' : 'float32';
            if ($gradient->dtype() !== $expectedDtype || $gradient->getShape() !== [2]) {
                throw new RuntimeException("$dtype replay power gradient dtype or shape changed");
            }
            foreach ($gradient->toArray() as $index => $actual) {
                $expected = $exponent === 0 ? 0.0 : $exponent * $values[$index] ** ($exponent - 1);
                closeGradient($actual, $expected, "$dtype replay power $exponent", $tolerance, $tolerance);
            }
        }
    }
}

$example = (new A([0.4, 1.6], 'float64'))->requiresGrad();
$graph = F::compile(function (A $x): A {
    ($x->exp2() + $x->log2() + $x->log10())->sum()->backward();
    return $x->grad();
}, [$example]);
foreach ([[0.7, 1.2], [0.4, 1.6]] as $values) {
    $gradient = $graph->run(new A($values, 'float64'));
    foreach ($gradient->toArray() as $index => $actual) {
        $x = $values[$index];
        closeGradient($actual, log(2.0) * 2 ** $x + 1 / ($x * log(2.0)) + 1 / ($x * log(10.0)),
                      'FP64 logarithmic replay', 2e-12, 2e-12);
    }
}

$x = (new A([2.0], 'float64'))->requiresGrad();
($x ** 3)->sum()->backward();
try {
    $x->grad()->backward();
    throw new RuntimeException('Backward unexpectedly built a higher-order graph');
} catch (PhpTensors\InvalidArgumentException $error) {
    if (!str_contains($error->getMessage(), 'requiresGrad')) throw $error;
}

if ($failures) throw new RuntimeException(implode("\n", $failures));
echo "FP64 finite differences agree in eager and Fusion\n";
echo "Scalar powers preserve precision and constant derivatives across floating dtypes\n";
?>
--EXPECT--
FP64 finite differences agree in eager and Fusion
Scalar powers preserve precision and constant derivatives across floating dtypes
