--TEST--
Dependent pointwise outputs share a bounded region while preserving boundaries and side exits
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;

function checkOutputs(array $actual, array $expected): void
{
    foreach ($expected as $i => $tensor) {
        if ($actual[$i]->toBuffer() !== $tensor->toBuffer()) {
            throw new RuntimeException("Dependent output $i differs.");
        }
    }
}
$x = CudaArray::fromFlatArray([1, 2, 3, 4], [2, 2]);
$body = function ($x) {
    $moment = $x * 0.9 + 1;
    $variance = ($x * $x + 1)->sqrt();
    return [$moment / $variance, $moment, $variance];
};
foreach ([false, true] as $cudaGraph) {
    $graph = Fusion::compile($body, [$x], cudaGraph: $cudaGraph);
    if ($graph->getStats()['fusedKernels'] !== 1 || $graph->getPlan()[0]['outputs'] !== 3) {
        throw new RuntimeException('Dependent outputs were not grouped.');
    }
    $first = $graph->run($x);
    checkOutputs($first, $body($x));
    checkOutputs($graph->run($x * 2), $body($x * 2));
    checkOutputs($first, $body($x));
    checkOutputs($graph->runAsync($x)->wait(), $body($x));
    if (!$cudaGraph) {
        $one = $graph->runAsync($x);
        $two = $graph->runAsync($x * 2);
        checkOutputs($graph->run($x * 3), $body($x * 3));
        checkOutputs($two->wait(), $body($x * 2));
        checkOutputs($one->wait(), $body($x));
    }
}
checkOutputs(Fusion::run(fn() => $body($x)), $body($x));
echo "dependent outputs grouped with independent retained storage\n";

$cases = [
    function ($x) {
        $scratch = $x + 1;
        $moment = $scratch * 2;
        return [$moment / 3, $moment, $scratch->sum(0)];
    },
    function ($x) {
        $y = $x + 1;
        return [$y * 2, $y, $y->sum(0)];
    },
    function ($x) {
        $y = $x + 1;
        return [$y + $y->sum(0), $y];
    },
    function ($x) {
        $y = $x + 1;
        return [$y + $y->transpose(), $y];
    },
    function ($x) {
        $y = $x + 1;
        return [$y->astype('float64') * 2, $y];
    },
    function ($x) {
        $y = $x + 1;
        $z = $y;
        for ($i = 0; $i < 160; $i++) $z = $z + 1;
        return [$z, $y];
    },
    function ($x) {
        $result = [];
        for ($i = 0; $i < 6; $i++) {
            $x = $x + 1;
            $result[] = $x;
        }
        return $result;
    },
];
foreach ($cases as $body) {
    foreach ([false, true] as $cudaGraph) {
        $graph = Fusion::compile($body, [$x], cudaGraph: $cudaGraph);
        $first = $graph->run($x);
        checkOutputs($first, $body($x));
        checkOutputs($graph->run($x * 2), $body($x * 2));
        checkOutputs($first, $body($x));
        foreach ($graph->getPlan() as $step) {
            if ($step['outputs'] > 4) {
                throw new RuntimeException('Region output budget exceeded.');
            }
        }
    }
}
echo "native side exits, boundary reentry, mixed dtypes and region budgets checked\n";
$empty = CudaArray::zeros([0, 2]);
$body = function ($x) {
    $y = $x + 1;
    return [$y + 1, $y];
};
checkOutputs(Fusion::compile($body, [$empty])->run($empty), $body($empty));
$special = CudaArray::fromFlatArray([NAN, INF, -INF, -0.0], [4]);
$body = function ($x) {
    $y = $x * 2;
    return [$y + 1, $y];
};
$actual = Fusion::compile($body, [$special])->run($special);
foreach ($body($special) as $i => $tensor) {
    foreach ($tensor->toArray() as $j => $value) {
        $got = $actual[$i]->toArray()[$j];
        if (is_nan($value) ? !is_nan($got) : ($value !== $got)) {
            throw new RuntimeException('Non-finite output changed.');
        }
    }
}
echo "empty and non-finite dependent outputs preserved\n";
?>
--EXPECT--
dependent outputs grouped with independent retained storage
native side exits, boundary reentry, mixed dtypes and region budgets checked
empty and non-finite dependent outputs preserved
