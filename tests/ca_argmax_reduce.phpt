--TEST--
Cuda Array ArgMax reduce
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$ca = new PhpTensors\CudaArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
var_dump($ca->argMax(1)->toArray());
var_dump($ca->argMax(0)->toArray());
var_dump($ca->argMax(2)->toArray());
var_dump($ca->argMax()->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(1)
  }
  [1]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(1)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(1)
  }
  [1]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(1)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(1)
  }
  [1]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(1)
  }
}
array(1) {
  [0]=>
  int(7)
}
