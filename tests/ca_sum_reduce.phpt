--TEST--
Cuda Array Sum reduce
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$ca = new PhpTensors\CudaArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
var_dump($ca->sum(1)->toArray());
var_dump($ca->sum(0)->toArray());
var_dump($ca->sum(2)->toArray());
var_dump($ca->sum()->toArray());
$rows = array_fill(0, 4096, [1, 2]);
$large = new PhpTensors\CudaArray($rows);
var_dump($large->sum(1)->toArray() === array_fill(0, 4096, 3.0));
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(4)
    [1]=>
    float(6)
  }
  [1]=>
  array(2) {
    [0]=>
    float(12)
    [1]=>
    float(14)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(6)
    [1]=>
    float(8)
  }
  [1]=>
  array(2) {
    [0]=>
    float(10)
    [1]=>
    float(12)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(7)
  }
  [1]=>
  array(2) {
    [0]=>
    float(11)
    [1]=>
    float(15)
  }
}
array(1) {
  [0]=>
  float(36)
}
bool(true)
