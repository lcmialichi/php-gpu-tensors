--TEST--
Cuda Array Subtraction
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$a = new PhpTensors\CudaArray([4, 1, 2, 3]);
$b = new PhpTensors\CudaArray([[4, 1, 2, 3], [6, 7, 8, 9]]);
$result = $a->subtract($b);
var_dump($result->toArray());

$a = new PhpTensors\CudaArray([[22, 32], [47, 61]]);
$b = new PhpTensors\CudaArray([[[4, 1],[ 2, 3]], [[6, 7], [8, 9]]]);
$result = $a->subtract($b);
var_dump($result->toArray());

$result = $a->subtract(11);
var_dump($result->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(4) {
    [0]=>
    float(0)
    [1]=>
    float(0)
    [2]=>
    float(0)
    [3]=>
    float(0)
  }
  [1]=>
  array(4) {
    [0]=>
    float(-2)
    [1]=>
    float(-6)
    [2]=>
    float(-6)
    [3]=>
    float(-6)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(18)
      [1]=>
      float(31)
    }
    [1]=>
    array(2) {
      [0]=>
      float(45)
      [1]=>
      float(58)
    }
  }
  [1]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(16)
      [1]=>
      float(25)
    }
    [1]=>
    array(2) {
      [0]=>
      float(39)
      [1]=>
      float(52)
    }
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(11)
    [1]=>
    float(21)
  }
  [1]=>
  array(2) {
    [0]=>
    float(36)
    [1]=>
    float(50)
  }
}