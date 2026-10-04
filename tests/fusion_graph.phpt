--TEST--
Fusion compilation emits one kernel without intermediate buffers and replays new inputs
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
$a = new CudaArray([1, 2]);
$b = new CudaArray([3, 4]);
$c = new CudaArray([5, 6]);
$calls = 0;
$graph = Fusion::compile(function ($a, $b, $c) use (&$calls) {
    $calls++;
    return $a + $b * $c;
}, inputs: [$a, $b, $c]);
$stats = $graph->getStats();
var_dump($stats['fusedKernels'], $stats['executionSteps'], $stats['intermediateBuffers']);
var_dump(substr_count($graph->getSource(), '__global__ void'));
$first = $graph->run($a, $b, $c);
$second = $graph->run($c, $a, $b);
unset($a, $b, $c);
var_dump($first->toArray(), $second->toArray(), $calls, $graph->getStats()['executions']);
unset($graph);
var_dump($first->toArray());
?>
--EXPECT--
int(1)
int(1)
int(0)
int(1)
array(2) {
  [0]=>
  float(16)
  [1]=>
  float(26)
}
array(2) {
  [0]=>
  float(8)
  [1]=>
  float(14)
}
int(1)
int(2)
array(2) {
  [0]=>
  float(16)
  [1]=>
  float(26)
}
