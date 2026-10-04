--TEST--
Fusion preserves per-node rounding, scalar bit patterns, comparisons and integer promotion
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
$a = new CudaArray([-1.0]);
$b = new CudaArray([1.0 + 2 ** -23]);
$c = new CudaArray([1.0 - 2 ** -23]);
$graph = Fusion::compile(fn($a, $b, $c) => $a + $b * $c, [$a, $b, $c]);
var_dump($graph->run($a, $b, $c)->toHost()->toBuffer() === ($a + $b * $c)->toHost()->toBuffer());
$x = new CudaArray([1.0, 2.0]);
var_dump(Fusion::run(fn() => $x * -0.0)->toHost()->toBuffer() === ($x * -0.0)->toHost()->toBuffer());
$result = Fusion::run(fn() => ($x * 2)->gt($x + 1));
var_dump($result->toArray() === ($x * 2)->gt($x + 1)->toArray());
var_dump(Fusion::run(fn() => $x + NAN)->toArray());
$a = new CudaArray([10, 20], 'int32');
$b = new CudaArray([3, 4], 'int32');
var_dump(Fusion::run(fn() => $a / $b + 1)->toArray() === ($a / $b + 1)->toArray());
$graph = Fusion::compile(fn($x) => ($x + 1)->astype('float32') * 2, [$x]);
var_dump($graph->run($x)->toArray());
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
array(2) {
  [0]=>
  float(NAN)
  [1]=>
  float(NAN)
}
bool(true)
array(2) {
  [0]=>
  float(4)
  [1]=>
  float(6)
}
