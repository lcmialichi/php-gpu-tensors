--TEST--
Cuda Array Multiplication overload
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$a = new PhpTensors\CudaArray([4, 1, 2, 3]);
$b = new PhpTensors\CudaArray([[4, 1, 2, 3], [6, 7, 8, 9]]);
$result = $a * $b;
var_dump($result->toArray());

$a = new PhpTensors\CudaArray([[22, 32], [47, 61]]);
$b = new PhpTensors\CudaArray([[[4, 1],[ 2, 3]], [[6, 7], [8, 9]]]);
$result = $a * $b;
var_dump($result->toArray());

$result = $a * 11;
var_dump($result->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(4) {
    [0]=>
    float(16)
    [1]=>
    float(1)
    [2]=>
    float(4)
    [3]=>
    float(9)
  }
  [1]=>
  array(4) {
    [0]=>
    float(24)
    [1]=>
    float(7)
    [2]=>
    float(16)
    [3]=>
    float(27)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(88)
      [1]=>
      float(32)
    }
    [1]=>
    array(2) {
      [0]=>
      float(94)
      [1]=>
      float(183)
    }
  }
  [1]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(132)
      [1]=>
      float(224)
    }
    [1]=>
    array(2) {
      [0]=>
      float(376)
      [1]=>
      float(549)
    }
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(242)
    [1]=>
    float(352)
  }
  [1]=>
  array(2) {
    [0]=>
    float(517)
    [1]=>
    float(671)
  }
}
