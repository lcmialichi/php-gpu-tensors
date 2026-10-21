--TEST--
Every public floating tensor operation preserves gradients or explicitly rejects its unsupported path
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion as F;

function weighted(A $output): A
{
    $count = $output->getSize();
    return $output * A::fromFlatArray($count ? range(1, $count) : [],
                                     $output->getShape(), $output->dtype());
}

function checkPath(string $name, string $dtype, bool $fused, callable $forward,
                   array $specs, array $expected): void
{
    $inputs = array_map(fn(array $spec): A =>
        A::fromFlatArray($spec[0], $spec[1], $spec[2] ?? $dtype)->requiresGrad(), $specs);
    $backward = function () use ($inputs, $forward): array {
        $forward(...$inputs)->sum()->backward();
        return array_map(function (A $input): A {
            if ($input->grad() === null) throw new RuntimeException('Missing gradient path');
            return $input->grad();
        }, $inputs);
    };
    $gradients = $fused ? F::run($backward) : $backward();
    foreach ($gradients as $argument => $gradient) {
        $sourceDtype = $specs[$argument][2] ?? $dtype;
        $gradientDtype = $sourceDtype === 'float64' ? 'float64' : 'float32';
        if ($gradient->dtype() !== $gradientDtype || $gradient->getShape() !== $specs[$argument][1]) {
            throw new RuntimeException("$name $dtype: gradient dtype/shape mismatch");
        }
        $flat = [];
        $values = $gradient->toArray();
        array_walk_recursive($values, function (float $value) use (&$flat): void { $flat[] = $value; });
        if (count($flat) !== count($expected[$argument])) {
            throw new RuntimeException("$name $dtype: gradient element count mismatch");
        }
        $tolerance = ['float64' => 2e-12, 'float32' => 3e-6,
                      'float16' => 6e-3, 'bfloat16' => 5e-2][$sourceDtype];
        foreach ($flat as $index => $actual) {
            $reference = $expected[$argument][$index];
            if (!is_finite($actual) || abs($actual - $reference) > $tolerance * (1 + abs($reference))) {
                throw new RuntimeException("$name $dtype " . ($fused ? 'Fusion' : 'eager') .
                    " argument $argument index $index: expected $reference, got $actual");
            }
        }
    }
}

function rejects(callable $callback, string $message): void
{
    try {
        $callback();
        throw new RuntimeException("Unsupported path was accepted: $message");
    } catch (PhpTensors\InvalidArgumentException $error) {
        if (!str_contains($error->getMessage(), $message)) throw $error;
    }
}

$matrix = [[[1, 2, 3, 4], [2, 2]]];
$pair = [$matrix[0], [[2, 4], [1, 2]]];
$unary = [
    'exp' => exp(0.5), 'expm1' => exp(0.5), 'exp2' => log(2) * 2 ** 0.5,
    'sqrt' => 1 / (2 * sqrt(0.5)), 'log' => 2, 'log1p' => 1 / 1.5,
    'log2' => 2 / log(2), 'log10' => 2 / log(10),
    'sin' => cos(0.5), 'cos' => -sin(0.5), 'tan' => 1 / cos(0.5) ** 2,
    'asin' => 1 / sqrt(0.75), 'acos' => -1 / sqrt(0.75), 'atan' => 1 / 1.25,
    'sinh' => cosh(0.5), 'cosh' => sinh(0.5), 'tanh' => 1 - tanh(0.5) ** 2,
    'abs' => 1, 'neg' => -1, 'ceil' => 0, 'floor' => 0, 'round' => 0,
];
$cases = [
    'add' => [fn($a, $b) => $a + $b, $pair, [[1, 1, 1, 1], [2, 2]]],
    'subtract' => [fn($a, $b) => $a - $b, $pair, [[1, 1, 1, 1], [-2, -2]]],
    'multiply' => [fn($a, $b) => $a * $b, $pair, [[2, 4, 2, 4], [4, 6]]],
    'divide' => [fn($a, $b) => $a / $b, $pair, [[0.5, 0.25, 0.5, 0.25], [-1, -0.375]]],
    'power' => [fn($a, $b) => $a ** $b, $pair,
                [[2, 32, 6, 256], [9 * log(3), 16 * log(2) + 256 * log(4)]]],
    'maximum' => [fn($a, $b) => $a->maximum($b), $pair, [[0, 0, 1, 0.5], [1, 1.5]]],
    'minimum' => [fn($a, $b) => $a->minimum($b), $pair, [[1, 1, 0, 0.5], [1, 0.5]]],
    'reshape' => [fn($x) => weighted($x->reshape([4, 1])), $matrix, [[1, 2, 3, 4]]],
    'flatten' => [fn($x) => weighted($x->flatten()), $matrix, [[1, 2, 3, 4]]],
    'squeeze' => [fn($x) => weighted($x->unsqueeze(0)->squeeze(0)), $matrix, [[1, 2, 3, 4]]],
    'copy' => [fn($x) => weighted($x->copy()), $matrix, [[1, 2, 3, 4]]],
    'transpose' => [fn($x) => weighted($x->transpose()), $matrix, [[1, 3, 2, 4]]],
    'swapaxes' => [fn($x) => weighted($x->swapaxes(0, 1)), $matrix, [[1, 3, 2, 4]]],
    'moveaxis' => [fn($x) => weighted($x->moveaxis(0, 1)), $matrix, [[1, 3, 2, 4]]],
    'broadcast' => [fn($x) => weighted($x->broadcastTo([2, 2])),
                    [[[1, 2], [1, 2]]], [[4, 6]]],
    'diagonal' => [fn($x) => weighted($x->diagonal()), $matrix, [[1, 0, 0, 2]]],
    'offset diagonal' => [fn($x) => weighted($x->diagonal(1)), $matrix, [[0, 1, 0, 0]]],
    'repeat' => [fn($x) => weighted($x->repeat(2, 1)), $matrix, [[3, 7, 11, 15]]],
    'repeat counts' => [fn($x) => weighted($x->repeat([0, 2], 1)), $matrix, [[0, 3, 0, 7]]],
    'tile' => [fn($x) => weighted($x->tile([2, 1])), $matrix, [[6, 8, 10, 12]]],
    'flip' => [fn($x) => weighted($x->flip(0)), $matrix, [[3, 4, 1, 2]]],
    'roll' => [fn($x) => weighted($x->roll(1, 1)), $matrix, [[2, 1, 4, 3]]],
    'slice' => [fn($x) => weighted($x->slice(':, ::2')), $matrix, [[1, 0, 2, 0]]],
    'invoke' => [fn($x) => weighted($x(1)), $matrix, [[0, 0, 1, 2]]],
    'dimension read' => [fn($x) => weighted($x[1]), $matrix, [[0, 0, 1, 2]]],
    'concat' => [fn($x) => weighted($x->concat([$x], 1)), $matrix, [[4, 6, 12, 14]]],
    'gather' => [fn($x) => weighted($x->gather(new A([[1, 1], [0, 0]], 'int32'), 1)),
                  $matrix, [[0, 3, 7, 0]]],
    'scatterAdd' => [fn($x, $u) => weighted($x->scatterAdd(new A([[1, 1], [0, 0]], 'int32'), $u, 1)),
                      [$matrix[0], $matrix[0]], [[1, 2, 3, 4], [2, 2, 3, 3]]],
    'sum' => [fn($x) => $x->sum(), $matrix, [[1, 1, 1, 1]]],
    'mean' => [fn($x) => weighted($x->mean(1)), $matrix, [[0.5, 0.5, 1, 1]]],
    'max' => [fn($x) => weighted($x->max(1)), $matrix, [[0, 1, 0, 2]]],
    'min' => [fn($x) => weighted($x->min(1)), $matrix, [[1, 0, 2, 0]]],
    'prod' => [fn($x) => weighted($x->prod(1)), $matrix, [[2, 1, 8, 6]]],
    'cumsum axis' => [fn($x) => weighted($x->cumsum(1)), $matrix, [[3, 2, 7, 4]]],
    'cumsum global' => [fn($x) => weighted($x->cumsum()), $matrix, [[10, 9, 7, 4]]],
    'cumprod axis' => [fn($x) => weighted($x->cumprod(1)), $matrix, [[5, 2, 19, 12]]],
    'cumprod global' => [fn($x) => weighted($x->cumprod()), $matrix, [[119, 59, 38, 24]]],
    'var' => [fn($x) => weighted($x->var(1)), $matrix, [[-0.5, 0.5, -1, 1]]],
    'std' => [fn($x) => weighted($x->std(1)), $matrix, [[-0.5, 0.5, -1, 1]]],
    'softmax' => [fn($x) => weighted($x->softmax(1)), [[[0, 0, 0, 0], [2, 2]]],
                   [[-0.25, 0.25, -0.25, 0.25]]],
    'layerNorm' => [fn($x, $s, $b) => weighted($x->layerNorm([1], 1.0, $s, $b)),
        [[[-1, 1, -1, 1], [2, 2]], [[1, 1], [2]], [[0, 0], [2]]],
        [[-0.25 / sqrt(2), 0.25 / sqrt(2), -0.25 / sqrt(2), 0.25 / sqrt(2)],
         [-4 / sqrt(2), 6 / sqrt(2)], [4, 6]]],
    'where' => [fn($a, $b) => weighted(A::where(new A([[1, 0], [0, 1]], 'bool'), $a, $b)),
                 $pair, [[1, 0, 0, 4], [3, 2]]],
    'clamp' => [fn($x) => $x->clamp(0, 2), [[[-1, 0, 1, 2, 3], [5]]], [[0, 0.5, 1, 0.5, 0]]],
    'strided unary' => [fn($x) => weighted($x->transpose()->exp()), $matrix,
                        [[exp(1), 3 * exp(2), 2 * exp(3), 4 * exp(4)]]],
    'strided scalar' => [fn($x) => weighted($x->transpose() * 2), $matrix, [[2, 6, 4, 8]]],
    'strided binary' => [fn($x, $b) => weighted($x->transpose() * $b), $pair,
                         [[2, 6, 8, 16], [7, 22]]],
    'strided reduction' => [fn($x) => weighted($x->transpose()->prod(1)), $matrix, [[3, 8, 1, 4]]],
    'strided scan axis' => [fn($x) => weighted($x->transpose()->cumsum(1)), $matrix, [[3, 7, 2, 4]]],
    'strided scan global' => [fn($x) => weighted($x->transpose()->cumsum()), $matrix, [[10, 7, 9, 4]]],
    'strided gather' => [fn($x) => weighted($x->transpose()->gather(new A([[1, 1], [0, 0]], 'int32'), 1)),
                          $matrix, [[0, 7, 3, 0]]],
    'strided cast' => [fn($x) => weighted($x->transpose()->astype('float64')), $matrix, [[1, 3, 2, 4]]],
    'strided copy' => [fn($x) => weighted($x->transpose()->copy()), $matrix, [[1, 3, 2, 4]]],
    'empty transforms' => [fn($x) => $x->transpose()->flip()->tile([1, 2])->repeat(2, 0),
                            [[[], [0, 2]]], [[]]],
    'empty gather' => [fn($x) => $x->gather(A::zeros([2, 0], 'int32'), 1), $matrix, [[0, 0, 0, 0]]],
    'empty scatter' => [fn($x, $u) => weighted($x->scatterAdd(A::zeros([2, 0], 'int32'), $u, 1)),
                         [$matrix[0], [[], [2, 0]]], [[1, 2, 3, 4], []]],
];

foreach (['float32', 'float64', 'float16', 'bfloat16'] as $dtype) {
    foreach ([false, true] as $fused) {
        foreach ($unary as $operation => $derivative) {
            checkPath($operation, $dtype, $fused, fn($x) => $x->$operation(),
                      [[[0.5], [1]]], [[$derivative]]);
            checkPath("$operation empty", $dtype, $fused, fn($x) => $x->$operation(),
                      [[[], [0]]], [[]]);
        }
        foreach ($cases as $name => [$forward, $specs, $expected]) {
            checkPath($name, $dtype, $fused, $forward, $specs, $expected);
        }
        foreach ([
            'scalar add' => [fn($x) => $x + 2, 1], 'inverse add' => [fn($x) => 2 + $x, 1],
            'scalar sub' => [fn($x) => $x - 2, 1], 'inverse sub' => [fn($x) => 2 - $x, -1],
            'scalar mul' => [fn($x) => $x * 2, 2], 'inverse mul' => [fn($x) => 2 * $x, 2],
            'scalar div' => [fn($x) => $x / 2, 0.5], 'inverse div' => [fn($x) => 2 / $x, -8],
            'scalar power' => [fn($x) => $x ** 3, 0.75],
            'inverse power' => [fn($x) => 2 ** $x, sqrt(2) * log(2)],
        ] as $name => [$forward, $derivative]) {
            checkPath($name, $dtype, $fused, $forward, [[[0.5], [1]]], [[$derivative]]);
        }
        foreach (['float32', 'float64', 'float16', 'bfloat16'] as $otherDtype) {
            foreach (['add', 'subtract', 'multiply', 'divide', 'power', 'maximum', 'minimum'] as $operation) {
                [$forward, $specs, $expected] = $cases[$operation];
                $specs[1][2] = $otherDtype;
                checkPath("mixed $operation $otherDtype", $dtype, $fused, $forward, $specs, $expected);
            }
        }
        $casts = ['float32' => ['float32', 'float64'], 'float64' => ['float64'],
                  'float16' => ['float16', 'float32', 'float64'],
                  'bfloat16' => ['bfloat16', 'float32', 'float64']][$dtype];
        foreach (['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16', 'int32',
                  'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'] as $target) {
            if (in_array($target, $casts, true)) {
                checkPath("cast $target", $dtype, $fused, fn($x) => weighted($x->astype($target)),
                          $matrix, [[1, 2, 3, 4]]);
                checkPath("empty cast $target", $dtype, $fused, fn($x) => $x->astype($target),
                          [[[], [0, 2]]], [[]]);
            } else {
                $invalid = fn() => (new A([1], $dtype))->requiresGrad()->astype($target);
                rejects(fn() => $fused ? F::run($invalid) : $invalid(), 'Cannot');
            }
        }
        foreach (['gt', 'lt', 'eq', 'ne', 'ge', 'le'] as $operation) {
            foreach ([false, true] as $tensorOperand) {
                $invalid = function () use ($dtype, $operation, $tensorOperand): A {
                    $x = (new A([1], $dtype))->requiresGrad();
                    $result = $x->$operation($tensorOperand ? new A([1], $dtype) : 1);
                    ($result + $x)->sum()->backward();
                    return $result;
                };
                rejects(fn() => $fused ? F::run($invalid) : $invalid(), 'Backward is not implemented');
            }
        }
        if ($dtype === 'float64') {
            $invalid = fn() => (new A([[1]], $dtype))->requiresGrad()->matmul(new A([[2]], $dtype));
            rejects(fn() => $fused ? F::run($invalid) : $invalid(), 'matmul');
        } else {
            checkPath('matmul', $dtype, $fused, fn($a, $b) => $a->matmul($b),
                      [$matrix[0], $matrix[0]], [[3, 7, 3, 7], [4, 4, 6, 6]]);
        }
    }
}

foreach (['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'] as $dtype) {
    foreach ([false, true] as $fused) {
        $invalid = fn() => (new A([1], $dtype))->requiresGrad();
        rejects(fn() => $fused ? F::run($invalid) : $invalid(), 'Gradient tracking supports');
    }
}
echo "Complete floating operation/dtype path matrix passed in eager and Fusion\n";
?>
--EXPECT--
Complete floating operation/dtype path matrix passed in eager and Fusion
