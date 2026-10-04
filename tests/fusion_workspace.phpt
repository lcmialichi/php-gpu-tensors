--TEST--
Mixed Fusion replay reuses scratch, preserves outputs, profiles phases and aliases native layouts
--INI--
cuda.memory_size=16M
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
function check(bool $ok): void { if (!$ok) throw new RuntimeException('workspace check failed'); }
$a = new CudaArray([[1, 2], [3, 4]]);
$unused = new CudaArray([9]);
$graph = Fusion::compile(function ($x, $unused) {
    $y = $x + 1;
    return $y->transpose()->matmul($y)->sum(0) * 2;
}, [$a, $unused]);
check($graph->getStats()['viewAliases'] === 1);
check(in_array('view', array_column($graph->getPlan(), 'kind'), true));
$first = $graph->run($a, $unused);
check($first->toArray() === [92.0, 120.0]);
$warm = $graph->getStats();
check($warm['workspaceBuffers'] > 0);
check($warm['synchronizations'] === 1);
check($warm['profiledExecutions'] === 0);
$graph->setProfiling(true);
for ($i = 0; $i < 100; $i++) check($graph->run($a, $unused)->toArray() === [92.0, 120.0]);
$stats = $graph->getStats();
check($stats['tensorAllocations'] - $warm['tensorAllocations'] === 100);
check($stats['workspaceBuffers'] === $warm['workspaceBuffers']);
check($stats['bufferReuses'] > $warm['bufferReuses']);
check($stats['synchronizations'] === 101);
check($stats['profiledExecutions'] === 100);
check($stats['bindTimeNs'] > 0 && $stats['executeTimeNs'] > 0 && $stats['collectTimeNs'] > 0);
check($first->toArray() === [92.0, 120.0]);
echo "persistent scratch and profiling checked\n";
$graph->setProfiling(false);
check($graph->getStats()['profiledExecutions'] === 0);
check($graph->getStats()['bindTimeNs'] === 0);
try { $graph->run(new CudaArray([1]), $unused); echo "MISSED\n"; }
catch (Cuda\InvalidArgumentException $error) { echo "binding failure rejected\n"; }
check($graph->run($a, $unused)->toArray() === [92.0, 120.0]);
$sliced = (new CudaArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))[1];
$layout = Fusion::compile(fn($x) => $x->transpose()->matmul($x), [$sliced]);
check($layout->getStats()['viewAliases'] === 1);
check($layout->run($sliced)->toArray() === [[74.0, 86.0], [86.0, 100.0]]);
echo "offset native alias checked\n";
$global = Fusion::compile(fn($x) => $x->transpose()->sum(), [$a]);
check($global->getStats()['viewAliases'] === 0);
check($global->run($a)->toArray() === [10.0]);
echo "non-contiguous global reduction materialized\n";
$batched = CudaArray::ones([2, 4, 4]);
$batch = Fusion::compile(fn($x) => ($x + 1)->matmul($x) + 1, [$batched]);
for ($i = 0; $i < 20; $i++) check($batch->run($batched)->toArray()[1][3][3] === 9.0);
echo "batched host descriptors checked\n";
$integer = new CudaArray([[1, 2], [3, 4]], 'int16');
$mean = Fusion::compile(fn($x) => ($x + 1)->mean(0) * 2, [$integer]);
check($mean->run($integer)->dtype() === 'float64');
check($mean->run($integer)->toArray() === [6.0, 8.0]);
$arg = Fusion::compile(fn($x) => ($x + 1)->argMax(1), [$a]);
check($arg->run($a)->toArray() === [1, 1]);
echo "reduction dtype and arg output checked\n";
$async = Fusion::compile(fn($x) => $x + 1, [$a]);
$execution = $async->runAsync($a);
try { $async->setProfiling(true); echo "MISSED\n"; }
catch (Cuda\RuntimeException $error) { echo "pending profiling change rejected\n"; }
$execution->wait();
$large = CudaArray::ones([64, 128]);
$twice = CudaArray::full([64, 128], 2);
$thrice = CudaArray::full([64, 128], 3);
$parallel = Fusion::compile(fn($x) => $x->transpose()->matmul($x)->sum(1) + 1, [$large]);
$one = $parallel->runAsync($large);
$two = $parallel->runAsync($twice);
check($parallel->run($thrice)->toArray() === array_fill(0, 128, 73729.0));
check($large->matmul($large->transpose())->toArray()[0][0] === 128.0);
check($two->wait()->toArray() === array_fill(0, 128, 32769.0));
check($one->wait()->toArray() === array_fill(0, 128, 8193.0));
check($parallel->getStats()['pending'] === 0);
echo "concurrent native streams and eager cuBLAS checked\n";
$invalid = Fusion::compile(fn($x) => [($x + 1)->sum(0), $x->sum()], [$a->transpose()]);
try { $invalid->runAsync($a->transpose()); echo "MISSED\n"; }
catch (Cuda\RuntimeException $error) {
    check(str_contains($error->getMessage(), 'contiguous'));
    echo "partial native submission cleaned up\n";
}
check($invalid->getStats()['pending'] === 0);
$a[0][0] = 1;
check($graph->run($a, $unused)->toArray() === [92.0, 120.0]);
?>
--EXPECT--
persistent scratch and profiling checked
binding failure rejected
offset native alias checked
non-contiguous global reduction materialized
batched host descriptors checked
reduction dtype and arg output checked
pending profiling change rejected
concurrent native streams and eager cuBLAS checked
partial native submission cleaned up
