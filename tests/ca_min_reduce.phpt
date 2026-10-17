--TEST--
Cuda Array Min reduce
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$ca = new PhpTensors\CudaArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
var_dump($ca->min(1)->toArray());
var_dump($ca->min(0)->toArray());
var_dump($ca->min(2)->toArray());
var_dump($ca->min()->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1)
    [1]=>
    float(2)
  }
  [1]=>
  array(2) {
    [0]=>
    float(5)
    [1]=>
    float(6)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1)
    [1]=>
    float(2)
  }
  [1]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(4)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1)
    [1]=>
    float(3)
  }
  [1]=>
  array(2) {
    [0]=>
    float(5)
    [1]=>
    float(7)
  }
}
array(1) {
  [0]=>
  float(1)
}

