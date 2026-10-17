--TEST--
Cuda Array Division overload
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$a = new PhpTensors\CudaArray([4, 1, 2, 3]);
$b = new PhpTensors\CudaArray([[4, 1, 2, 3], [6, 7, 8, 9]]);
$result = $a / $b;
var_dump($result->toArray());

$a = new PhpTensors\CudaArray([[22, 32], [47, 61]]);
$b = new PhpTensors\CudaArray([[[4, 1],[ 2, 3]], [[6, 7], [8, 9]]]);
$result = $a / $b;
var_dump($result->toArray());

$result = $a / 11;
var_dump($result->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(4) {
    [0]=>
    float(1)
    [1]=>
    float(1)
    [2]=>
    float(1)
    [3]=>
    float(1)
  }
  [1]=>
  array(4) {
    [0]=>
    float(0.6666666865348816)
    [1]=>
    float(0.1428571492433548)
    [2]=>
    float(0.25)
    [3]=>
    float(0.3333333432674408)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(5.5)
      [1]=>
      float(32)
    }
    [1]=>
    array(2) {
      [0]=>
      float(23.5)
      [1]=>
      float(20.33333396911621)
    }
  }
  [1]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(3.6666667461395264)
      [1]=>
      float(4.5714287757873535)
    }
    [1]=>
    array(2) {
      [0]=>
      float(5.875)
      [1]=>
      float(6.777777671813965)
    }
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(2)
    [1]=>
    float(2.909090995788574)
  }
  [1]=>
  array(2) {
    [0]=>
    float(4.2727274894714355)
    [1]=>
    float(5.545454502105713)
  }
}
