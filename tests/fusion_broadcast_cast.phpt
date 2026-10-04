--TEST--
Fusion broadcasting, view strides, scalar order and per-node dtype casts
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
$a = new CudaArray([[1], [2]], 'int32');
$b = new CudaArray([[3, 4, 5]], 'float32');
$c = new CudaArray([2, 3, 4], 'float64');
$graph = Fusion::compile(fn($a, $b, $c) => $a + $b * $c, [$a, $b, $c]);
$result = $graph->run($a, $b, $c);
var_dump($result->dtype(), $result->getShape(), $result->toArray() === ($a + $b * $c)->toArray());
var_dump($result->toArray());
$row = new CudaArray([[10, 20, 30], [40, 50, 60]]);
$view = $row->transpose();
$graph = Fusion::compile(fn($v) => 100 - $v * 2, [$view]);
var_dump($graph->run($view)->toArray());
$slice = $row[1];
var_dump(Fusion::run(fn() => $slice * 2 + 1)->toArray());
$narrow = new CudaArray([120], 'int8');
$wide = new CudaArray([1], 'int32');
$graph = Fusion::compile(fn($x, $y) => ($x * 2) + $y, [$narrow, $wide]);
var_dump($graph->run($narrow, $wide)->toArray() === (($narrow * 2) + $wide)->toArray());
$u16 = CudaArray::fromBuffer(pack('v*', 1000, 2000), [2], 'uint16');
$float = new CudaArray([0.5, 1.5]);
var_dump(Fusion::run(fn() => $u16 + $float)->toArray() === ($u16 + $float)->toArray());
?>
--EXPECT--
string(7) "float64"
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(3)
}
bool(true)
array(2) {
  [0]=>
  array(3) {
    [0]=>
    float(7)
    [1]=>
    float(13)
    [2]=>
    float(21)
  }
  [1]=>
  array(3) {
    [0]=>
    float(8)
    [1]=>
    float(14)
    [2]=>
    float(22)
  }
}
array(3) {
  [0]=>
  array(2) {
    [0]=>
    float(80)
    [1]=>
    float(20)
  }
  [1]=>
  array(2) {
    [0]=>
    float(60)
    [1]=>
    float(0)
  }
  [2]=>
  array(2) {
    [0]=>
    float(40)
    [1]=>
    float(-20)
  }
}
array(3) {
  [0]=>
  float(81)
  [1]=>
  float(101)
  [2]=>
  float(121)
}
bool(true)
bool(true)
