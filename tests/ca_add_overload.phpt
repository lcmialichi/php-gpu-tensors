--TEST--
Cuda Array Addition overload
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$a = new PhpTensors\CudaArray([4, 1, 2, 3]);
$b = new PhpTensors\CudaArray([[4, 1, 2, 3], [6, 7, 8, 9]]);
$result = $a + $b;
var_dump($result->toArray());

$a = new PhpTensors\CudaArray([[22, 32], [47, 61]]);
$b = new PhpTensors\CudaArray([[[4, 1],[ 2, 3]], [[6, 7], [8, 9]]]);
$result = $a + $b;
var_dump($result->toArray());

$result = $a + 11;
var_dump($result->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(4) {
    [0]=>
    float(8)
    [1]=>
    float(2)
    [2]=>
    float(4)
    [3]=>
    float(6)
  }
  [1]=>
  array(4) {
    [0]=>
    float(10)
    [1]=>
    float(8)
    [2]=>
    float(10)
    [3]=>
    float(12)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(26)
      [1]=>
      float(33)
    }
    [1]=>
    array(2) {
      [0]=>
      float(49)
      [1]=>
      float(64)
    }
  }
  [1]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(28)
      [1]=>
      float(39)
    }
    [1]=>
    array(2) {
      [0]=>
      float(55)
      [1]=>
      float(70)
    }
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(33)
    [1]=>
    float(43)
  }
  [1]=>
  array(2) {
    [0]=>
    float(58)
    [1]=>
    float(72)
  }
}