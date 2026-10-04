--TEST--
CudaArray astype retains same-dtype ownership and supports safe conversions
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
$original = new Cuda\CudaArray([1, 2]);
$alias = $original->astype('float32');
unset($original);
var_dump($alias->toArray());
unset($alias);
echo "released\n";
var_dump((new Cuda\CudaArray([1]))->astype('float64')->toArray());
try {
  (new Cuda\CudaArray([1]))->astype('int32');
} catch (Cuda\InvalidArgumentException $error) {
  echo "unsafe cast\n";
}
?>
--EXPECT--
array(2) {
  [0]=>
  float(1)
  [1]=>
  float(2)
}
released
array(1) {
  [0]=>
  float(1)
}
unsafe cast