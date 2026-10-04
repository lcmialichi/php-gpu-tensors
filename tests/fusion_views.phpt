--TEST--
Fusion boundaries consume offset and non-contiguous views without double offsets
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
$base = new CudaArray([[10, 20, 30], [40, 50, 60]]);
$slice = $base[1];
$graph = Fusion::compile(fn($x) => $x->sum() + 1, [$slice]);
var_dump($graph->run($slice)->toArray());
var_dump(Fusion::run(fn() => $slice->sum())->toArray());
$transpose = $base->transpose();
$graph = Fusion::compile(fn($x) => $x->sum(1) * 2, [$transpose]);
var_dump($graph->run($transpose)->toArray());
$graph = Fusion::compile(fn($x) => $x->neg() + 1, [$transpose]);
var_dump($graph->run($transpose)->toArray());
$graph = Fusion::compile(fn($x) => $x->power($x) * 0, [new CudaArray([2, 3])]);
var_dump($graph->run(new CudaArray([2, 3]))->toArray());
$result = Fusion::run(fn() => (new CudaArray([[1, 2], [3, 4]]) + 1)->transpose());
unset($base, $slice, $transpose, $graph);
var_dump(($result * 2)->toArray());
?>
--EXPECT--
array(1) {
  [0]=>
  float(151)
}
array(1) {
  [0]=>
  float(150)
}
array(3) {
  [0]=>
  float(100)
  [1]=>
  float(140)
  [2]=>
  float(180)
}
array(3) {
  [0]=>
  array(2) {
    [0]=>
    float(-9)
    [1]=>
    float(-39)
  }
  [1]=>
  array(2) {
    [0]=>
    float(-19)
    [1]=>
    float(-49)
  }
  [2]=>
  array(2) {
    [0]=>
    float(-29)
    [1]=>
    float(-59)
  }
}
array(2) {
  [0]=>
  float(0)
  [1]=>
  float(0)
}
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
