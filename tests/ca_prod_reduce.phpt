--TEST--
Cuda Array Prod reduce
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$ca = new PhpTensors\CudaArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
var_dump($ca->prod(1)->toArray());
var_dump($ca->prod(0)->toArray());
var_dump($ca->prod(2)->toArray());
var_dump($ca->prod()->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(8)
  }
  [1]=>
  array(2) {
    [0]=>
    float(35)
    [1]=>
    float(48)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(5)
    [1]=>
    float(12)
  }
  [1]=>
  array(2) {
    [0]=>
    float(21)
    [1]=>
    float(32)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(2)
    [1]=>
    float(12)
  }
  [1]=>
  array(2) {
    [0]=>
    float(30)
    [1]=>
    float(56)
  }
}
array(1) {
  [0]=>
  float(40320)
}

