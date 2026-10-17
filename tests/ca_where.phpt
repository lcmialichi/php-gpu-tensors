--TEST--
CudaArray where with contiguous and broadcast operands
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
$condition = PhpTensors\CudaArray::fromBuffer(pack('C*', 1, 0, 1), [3], 'bool');
$x = PhpTensors\CudaArray::fromBuffer(pack('g*', 10, 20, 30), [3]);
$y = PhpTensors\CudaArray::fromBuffer(pack('g*', 1, 2, 3), [3]);
var_dump(PhpTensors\CudaArray::where($condition, $x, $y)->toArray());

$condition = PhpTensors\CudaArray::fromBuffer(pack('C*', 1, 0), [2, 1], 'bool');
$x = PhpTensors\CudaArray::fromBuffer(pack('V*', 10, 20, 30), [1, 3], 'int32');
$y = PhpTensors\CudaArray::fromBuffer(pack('V*', 1, 2, 3, 4, 5, 6), [2, 3], 'int32');
var_dump(PhpTensors\CudaArray::where($condition, $x, $y)->toArray());

$condition = PhpTensors\CudaArray::fromBuffer(pack('C*', 1, 0, 1, 0), [2, 2], 'bool')->transpose();
$x = PhpTensors\CudaArray::fromBuffer(pack('V*', 10, 20, 30, 40), [2, 2], 'int32');
$y = PhpTensors\CudaArray::fromBuffer(pack('V*', 1, 2, 3, 4), [2, 2], 'int32');
var_dump(PhpTensors\CudaArray::where($condition, $x, $y)->toArray());

try {
  PhpTensors\CudaArray::where($condition, $x, new PhpTensors\CudaArray([[1.0]]));
} catch (PhpTensors\InvalidArgumentException $error) {
    echo "dtype mismatch\n";
}
try {
  $incompatible = PhpTensors\CudaArray::fromBuffer(pack('V*', 1, 2, 3, 4, 5, 6, 7, 8, 9), [3, 3], 'int32');
  PhpTensors\CudaArray::where($condition, $x, $incompatible);
} catch (PhpTensors\InvalidArgumentException $error) {
  echo "shape mismatch\n";
}
?>
--EXPECT--
array(3) {
  [0]=>
  float(10)
  [1]=>
  float(2)
  [2]=>
  float(30)
}
array(2) {
  [0]=>
  array(3) {
    [0]=>
    int(10)
    [1]=>
    int(20)
    [2]=>
    int(30)
  }
  [1]=>
  array(3) {
    [0]=>
    int(4)
    [1]=>
    int(5)
    [2]=>
    int(6)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(10)
    [1]=>
    int(20)
  }
  [1]=>
  array(2) {
    [0]=>
    int(3)
    [1]=>
    int(4)
  }
}
dtype mismatch
shape mismatch