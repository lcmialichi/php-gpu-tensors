--TEST--
AdamW fuses dependent outputs without changing functional state or retained snapshots
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;
use PhpTensors\Optimizer;

function inputs(array $parameters, array $gradients, array $state, CudaArray $rate): array
{
    return [$rate, $state['beta1Power'], $state['beta2Power'],
        ...$parameters, ...$gradients, ...$state['firstMoment'], ...$state['secondMoment']];
}
function tensors(array $result): array
{
    return [...$result['parameters'], ...$result['state']['firstMoment'],
        ...$result['state']['secondMoment'], $result['state']['beta1Power'], $result['state']['beta2Power']];
}
function checkStep(array $actual, array $expected, float $tolerance): void
{
    foreach (tensors($actual) as $i => $tensor) {
        $delta = ($tensor - tensors($expected)[$i])->abs()->max()->item();
        if (!is_finite($delta) || $delta > $tolerance) {
            throw new RuntimeException("AdamW output $i differs: $delta.");
        }
    }
}

foreach (['float32' => 1e-6, 'float64' => 1e-12] as $dtype => $tolerance) {
    foreach ([false, true] as $cudaGraph) {
        $optimizer = Optimizer::adamW(0.001, 0.8, 0.95, 1e-6, 0.02);
        $parameters = [CudaArray::full([4, 8], 0.25, $dtype), CudaArray::full([1, 8], -0.5, $dtype)];
        $initialParameters = $parameters;
        $initialBytes = array_map(fn($p) => $p->toBuffer(), $parameters);
        $state = $optimizer->initState($parameters);
        $referenceParameters = $parameters;
        $referenceState = $state;
        $values = [-8.0, -5.0, -4.99, -0.0, 1e-7, 4.99, 5.0, 8.0];
        $gradients = [CudaArray::fromFlatArray([...$values, ...$values, ...$values, ...$values],
            [8, 4], $dtype)->transpose(), CudaArray::fromFlatArray($values, [1, 8], $dtype)];
        $rate = CudaArray::full([1], 0.001, $dtype);
        $body = function ($rate, $beta1Power, $beta2Power, ...$tensors) use ($optimizer) {
            [$parameters, $gradients, $firstMoment, $secondMoment] = array_chunk($tensors, 2);
            foreach ($gradients as $i => $gradient) {
                $gradient = CudaArray::where($gradient->gt(5.0), CudaArray::full([1], 5.0, $gradient->dtype()), $gradient);
                $gradients[$i] = CudaArray::where($gradient->lt(-5.0), CudaArray::full([1], -5.0, $gradient->dtype()), $gradient);
            }
            return $optimizer->step($parameters, $gradients, [
                'firstMoment' => $firstMoment, 'secondMoment' => $secondMoment,
                'beta1Power' => $beta1Power, 'beta2Power' => $beta2Power,
            ], $rate, [true, false]);
        };
        $graph = Fusion::compile($body, inputs($parameters, $gradients, $state, $rate), cudaGraph: $cudaGraph);
        $grouped = array_filter($graph->getPlan(), fn($step) => $step['kind'] === 'fused' && $step['outputs'] >= 3);
        if (count($grouped) < 2 || $graph->getStats()['fusedKernels'] > 12) {
            throw new RuntimeException('AdamW did not fuse dependent weight and moment outputs.');
        }
        $saved = [];
        for ($step = 0; $step < 12; $step++) {
            $rate = CudaArray::full([1], $step === 3 ? 0.0 : 0.002 / ($step + 1), $dtype);
            $sign = $step % 2 ? -1.0 : 1.0;
            $signed = [($gradients[0]->transpose() * $sign)->transpose(), $gradients[1] * $sign];
            $expected = $body(...inputs($referenceParameters, $signed, $referenceState, $rate));
            $actual = $graph->run(...inputs($parameters, $signed, $state, $rate));
            checkStep($actual, $expected, $tolerance);
            if ($step === 0) {
                $saved = $actual;
                $savedBytes = array_map(fn($tensor) => $tensor->toBuffer(), tensors($actual));
            }
            $parameters = $actual['parameters'];
            $state = $actual['state'];
            $referenceParameters = $expected['parameters'];
            $referenceState = $expected['state'];
        }
        if (array_map(fn($tensor) => $tensor->toBuffer(), tensors($saved)) !== $savedBytes) {
            throw new RuntimeException('A later replay overwrote saved parameters or moments.');
        }
        if (array_map(fn($tensor) => $tensor->toBuffer(), $initialParameters) !== $initialBytes) {
            throw new RuntimeException('An optimizer replay mutated its initial parameters.');
        }
        if (array_map(fn($tensor) => $tensor->toBuffer(), $referenceParameters) === $initialBytes) {
            throw new RuntimeException('Training did not update parameters.');
        }
        $pending = $graph->runAsync(...inputs($parameters, $gradients, $state, $rate));
        $expected = $body(...inputs($parameters, $gradients, $state, $rate));
        checkStep($pending->wait(), $expected, $tolerance);
        try {
            $bad = [CudaArray::zeros([1], $dtype), $gradients[1]];
            $graph->run(...inputs($parameters, $bad, $state, $rate));
            throw new RuntimeException('Changed gradient shape was accepted.');
        } catch (PhpTensors\InvalidArgumentException) {
        }
        checkStep($graph->run(...inputs($parameters, $gradients, $state, $rate)), $expected, $tolerance);
        $special = [NAN, INF, -INF, -5.0, 5.0, 0.0, 1.0, -1.0];
        $nonfinite = [CudaArray::fromFlatArray([...$special, ...$special, ...$special, ...$special],
            [8, 4], $dtype)->transpose(), CudaArray::fromFlatArray($special, [1, 8], $dtype)];
        $expected = $body(...inputs($parameters, $nonfinite, $state, $rate));
        $actual = $graph->run(...inputs($parameters, $nonfinite, $state, $rate));
        foreach (tensors($expected) as $i => $tensor) {
            $format = $dtype === 'float32' ? 'g*' : 'e*';
            $got = array_values(unpack($format, tensors($actual)[$i]->toBuffer()));
            foreach (array_values(unpack($format, $tensor->toBuffer())) as $j => $value) {
                if (is_nan($value) ? !is_nan($got[$j]) :
                    (is_infinite($value) ? $got[$j] !== $value :
                        (!is_finite($got[$j]) || abs($value - $got[$j]) > $tolerance))) {
                    throw new RuntimeException('Non-finite gradient behavior changed.');
                }
            }
        }
    }
}
echo "float32/float64 AdamW groups weights and both moments\n";
echo "variable rates, zero rate, signed clipping boundaries and strided gradients match eager\n";
echo "retained state, async replay and binding failure recovery checked\n";
echo "NaN propagation and infinite-gradient clipping preserved\n";
?>
--EXPECT--
float32/float64 AdamW groups weights and both moments
variable rates, zero rate, signed clipping boundaries and strided gradients match eager
retained state, async replay and binding failure recovery checked
NaN propagation and infinite-gradient clipping preserved
