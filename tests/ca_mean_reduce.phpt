--TEST--
Cuda Array Mean reduce
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$ca = new PhpTensors\CudaArray([[1, 2, 3], [4, 5, 6]], 'float32');
var_dump($ca->mean()->toArray());
var_dump($ca->mean(0)->toArray());
var_dump($ca->mean(1)->toArray());
$integers = new PhpTensors\CudaArray([1, 2, 4], 'int32');
var_dump($integers->mean()->dtype());
var_dump(abs($integers->mean()->toArray()[0] - (7 / 3)) < 1e-12);
?>
--EXPECT--
array(1) {
  [0]=>
  float(3.5)
}
array(3) {
  [0]=>
  float(2.5)
  [1]=>
  float(3.5)
  [2]=>
  float(4.5)
}
array(2) {
  [0]=>
  float(2)
  [1]=>
  float(5)
}
string(7) "float64"
bool(true)