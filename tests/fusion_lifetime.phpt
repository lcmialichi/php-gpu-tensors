--TEST--
Repeated async and CUDA Graph replay stays within a small memory budget
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
function check(bool $ok): void { if (!$ok) throw new RuntimeException('check failed'); }
$a = CudaArray::ones([4096]);
$expression = function ($x) {
    for ($i = 0; $i < 130; $i++) $x = $x + 1;
    return $x;
};
$stream = Fusion::compile($expression, [$a]);
$graph = Fusion::compile($expression, [$a], cudaGraph: true);
$stream->run($a);
$graph->run($a);
$before = memory_get_usage();
for ($i = 0; $i < 100; $i++) {
    $one = $stream->runAsync($a);
    $two = $stream->runAsync($a);
    check($one->wait()[0] === 131.0);
    check($two->wait()[0] === 131.0);
    check($graph->run($a)[0] === 131.0);
    unset($one, $two);
}
check(memory_get_usage() - $before < 1024 * 1024);
check($stream->getStats()['pending'] === 0);
check($graph->getStats()['pending'] === 0);
echo "bounded replay lifetime\n";
for ($i = 0; $i < 25; $i++) {
    $view = Fusion::run(fn() => (new CudaArray([[1, 2], [3, 4]]) + 1)->transpose());
    check($view->toArray() === [[2.0, 4.0], [3.0, 5.0]]);
    unset($view);
}
echo "views released\n";
$matrix = new CudaArray([[1, 2], [3, 4]]);
$reduce = Fusion::compile(fn($x) => $x->sum() * 2, [$matrix[1]]);
for ($i = 0; $i < 25; $i++) check($reduce->run($matrix[1])->toArray() === [14.0]);
check($matrix[1]->reshape([1, 2])->toArray() === [[3.0, 4.0]]);
check($matrix->toArray() === [[1.0, 2.0], [3.0, 4.0]]);
echo "native view reduction released\n";
try { $graph->run(CudaArray::ones([8])); }
catch (Cuda\InvalidArgumentException $error) { echo "bad replay rejected\n"; }
check($graph->run($a)[0] === 131.0);
echo "replay recovered\n";
$large = CudaArray::ones([262144]);
$many = Fusion::compile(function ($x) {
    $outputs = [];
    for ($i = 0; $i < 20; $i++) $outputs[] = $x + $i;
    return $outputs;
}, [$large], cudaGraph: true);
try { $many->runAsync($large); echo "MISSED\n"; }
catch (Cuda\OutOfMemoryException $error) { echo "allocation failure preserved\n"; }
check($many->getStats()['pending'] === 0);
$large[0] = 2;
check($graph->run($a)[0] === 131.0);
echo "failed submission released\n";
?>
--EXPECT--
bounded replay lifetime
views released
native view reduction released
bad replay rejected
replay recovered
allocation failure preserved
failed submission released
