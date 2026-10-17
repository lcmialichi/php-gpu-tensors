--TEST--
Cuda Array Max reduce
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$ca = new PhpTensors\CudaArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
var_dump($ca->max(1)->toArray());
var_dump($ca->max(0)->toArray());
var_dump($ca->max(2)->toArray());
var_dump($ca->max()->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(4)
  }
  [1]=>
  array(2) {
    [0]=>
    float(7)
    [1]=>
    float(8)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(5)
    [1]=>
    float(6)
  }
  [1]=>
  array(2) {
    [0]=>
    float(7)
    [1]=>
    float(8)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(2)
    [1]=>
    float(4)
  }
  [1]=>
  array(2) {
    [0]=>
    float(6)
    [1]=>
    float(8)
  }
}
array(1) {
  [0]=>
  float(8)
}
