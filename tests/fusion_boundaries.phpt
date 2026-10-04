--TEST--
Fusion plans compile multiple segments together around reductions and matmul
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
$a = new CudaArray([[1, 2], [3, 4]]);
$b = new CudaArray([[5, 6], [7, 8]]);
$graph = Fusion::compile(fn($a, $b) => ($a * 2 + 1)->matmul($b) * 3 + 2, [$a, $b]);
$stats = $graph->getStats();
var_dump($stats['fusedKernels'], $stats['boundaries'], substr_count($graph->getSource(), '__global__ void'));
var_dump($graph->run($a, $b)->toArray() === (($a * 2 + 1)->matmul($b) * 3 + 2)->toArray());
$graph = Fusion::compile(fn($a) => ($a * 2 + 1)->sum(1) * 3, [$a]);
var_dump($graph->getStats()['fusedKernels'], $graph->getStats()['boundaries'], $graph->run($a)->toArray());
$graph = Fusion::compile(fn($a) => ($a + 1)->sum() * 2, [$a]);
var_dump($graph->run($a)->toArray());
$graph = Fusion::compile(function ($a) {
    $x = $a * 2;
    return ['x' => $x, 'y' => [$x + 1]];
}, [$a]);
var_dump($graph->run($a)['y'][0]->toArray() === ($a * 2 + 1)->toArray());
$graph = Fusion::compile(function ($a) {
    for ($i = 0; $i < 70; $i++) $a = $a + 1;
    return $a;
}, [$a]);
var_dump($graph->getStats()['fusedKernels'] > 1, $graph->run($a)->toArray());
$graph = Fusion::compile(fn($a) => ($a * 2)->sqrt() + 1, [$a]);
var_dump($graph->getStats()['boundaries'], $graph->run($a)->toArray() === ($a * 2)->sqrt()->add(1)->toArray());
$graph = Fusion::compile(fn($a) => ($a + 1)->transpose() * 2, [$a]);
var_dump($graph->run($a)->toArray());
$view = Fusion::run(fn() => ($a + 1)->transpose());
var_dump($view->getShape(), ($view * 2)->getShape());
$graph = Fusion::compile(fn($a) => ($a * 2)->reshape([4]) + 1, [$a]);
var_dump($graph->run($a)->toArray());
$graph = Fusion::compile(fn($a) => $a, [$a]);
var_dump($graph->getStats()['fusedKernels'], $graph->run($a)->toArray() === $a->toArray());
?>
--EXPECT--
int(2)
int(1)
int(2)
bool(true)
int(2)
int(1)
array(2) {
  [0]=>
  float(24)
  [1]=>
  float(48)
}
array(1) {
  [0]=>
  float(28)
}
bool(true)
bool(true)
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(71)
    [1]=>
    float(72)
  }
  [1]=>
  array(2) {
    [0]=>
    float(73)
    [1]=>
    float(74)
  }
}
int(1)
bool(true)
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(4)
    [1]=>
    float(8)
  }
  [1]=>
  array(2) {
    [0]=>
    float(6)
    [1]=>
    float(10)
  }
}
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(2)
}
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(2)
}
array(4) {
  [0]=>
  float(3)
  [1]=>
  float(5)
  [2]=>
  float(7)
  [3]=>
  float(9)
}
int(0)
bool(true)
