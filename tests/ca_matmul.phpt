--TEST--
CudaArray 2D and batched matrix multiplication
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
$a = new PhpTensors\CudaArray([[1, 2], [3, 4]]);
$b = new PhpTensors\CudaArray([[5, 6], [7, 8]]);
var_dump($a->matmul($b)->toArray());

$a = new PhpTensors\CudaArray([[[1, 2], [3, 4]], [[2, 0], [1, 2]]]);
$b = new PhpTensors\CudaArray([[[5, 6], [7, 8]], [[1, 2], [3, 4]]]);
var_dump($a->matmul($b)->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(19)
    [1]=>
    float(22)
  }
  [1]=>
  array(2) {
    [0]=>
    float(43)
    [1]=>
    float(50)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(19)
      [1]=>
      float(22)
    }
    [1]=>
    array(2) {
      [0]=>
      float(43)
      [1]=>
      float(50)
    }
  }
  [1]=>
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
      float(7)
      [1]=>
      float(10)
    }
  }
}