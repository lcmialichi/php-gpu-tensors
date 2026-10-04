--TEST--
Fusion unaries, where, casts, layout transforms, CSE and grouped outputs
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
function check(bool $ok): void { if (!$ok) throw new RuntimeException('check failed'); }
$a = new CudaArray([[1, 2, 3], [4, 5, 6]]);
$b = new CudaArray([2, 3, 4]);
foreach (['neg', 'abs', 'exp', 'sqrt', 'log', 'sin', 'cos', 'tan', 'floor', 'ceil', 'round'] as $method) {
    $graph = Fusion::compile(fn($x) => ($x + 0.25)->$method() + 1, [$a]);
    check($graph->getStats()['fusedKernels'] === 1);
    check($graph->getStats()['boundaries'] === 0);
    check($graph->run($a)->toArray() === ($a + 0.25)->$method()->add(1)->toArray());
}
echo "unaries fused\n";
$mask = new CudaArray([[1], [0]], 'bool');
$graph = Fusion::compile(fn($mask, $a, $b) => CudaArray::where($mask, $a + 1, ($a * $b)->neg()), [$mask, $a, $b]);
check($graph->getStats()['fusedKernels'] === 1);
check($graph->run($mask, $a, $b)->toArray() === CudaArray::where($mask, $a + 1, ($a * $b)->neg())->toArray());
echo "where fused\n";
$graph = Fusion::compile(fn($x) => ($x * 2)->transpose()->astype('float64') + 1, [$a]);
check($graph->getStats()['fusedKernels'] === 1);
$expected = [[3.0, 9.0], [5.0, 11.0], [7.0, 13.0]];
check($graph->run($a)->toArray() === $expected);
check(Fusion::run(fn() => ($a * 2)->transpose()->astype('float64') + 1)->toArray() === $expected);
check($a->astype('float64')->dtype() === 'float64');
check($a->transpose()->astype('float64')->toArray() === [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
echo "layouts and casts fused\n";
$graph = Fusion::compile(fn($x, $y) => ['a' => ($x * $y) + 1, 'b' => ($x * $y) - 1], [$a, $b]);
check($graph->getStats()['fusedKernels'] === 1);
check($graph->getPlan()[0]['outputs'] === 2);
$outputs = $graph->run($a, $b);
check($outputs['a']->toArray() === ($a * $b + 1)->toArray());
check($outputs['b']->toArray() === ($a * $b - 1)->toArray());
echo "grouped outputs\n";
$graph = Fusion::compile(function ($x) { $unused = ($x + 1)->exp(); return $x * 2; }, [$a]);
check($graph->getStats()['nodes'] === 2);
echo "dead nodes eliminated\n";
$integer = new CudaArray([1, 2, 3], 'int16');
check(Fusion::run(fn() => ($integer + 1)->astype('float32'))->toArray() === [2.0, 3.0, 4.0]);
echo "integer cast\n";
?>
--EXPECT--
unaries fused
where fused
layouts and casts fused
grouped outputs
dead nodes eliminated
integer cast
