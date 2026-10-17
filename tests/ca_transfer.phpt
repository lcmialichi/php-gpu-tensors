--TEST--
CudaArray transfer to PHP array and host tensor
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
$tensor = new PhpTensors\CudaArray([[1, 2], [3, 4]], 'int32');
var_dump($tensor->toArray());
$constructed = new PhpTensors\HostArray([[1, 2], [3, 4]], 'int32');
var_dump($constructed->toArray() === [[1, 2], [3, 4]]);
var_dump($constructed->toGpu()->toArray() === [[1, 2], [3, 4]]);
try {
  new PhpTensors\HostArray([[1, 2], [3]], 'int32');
} catch (PhpTensors\InvalidArgumentException $error) {
  echo "invalid host shape\n";
}
$pinned = PhpTensors\HostArray::fromBuffer(pack('V*', 1, 2, 3, 4), [2, 2], 'int32', true);
var_dump($pinned->isPinned());
var_dump($pinned->toGpu()->toArray() === [[1, 2], [3, 4]]);
var_dump($constructed->toBuffer() === pack('V*', 1, 2, 3, 4));
var_dump($constructed[1]->toGpu()->toArray() === [3, 4]);
$host = $tensor->toHost();
var_dump($host instanceof PhpTensors\HostArray);
var_dump($host->toArray());
var_dump($host->toGpu()->toArray());
$packed = PhpTensors\HostArray::fromBuffer(pack('V*', 1, 2, 3, 4), [2, 2], 'int32');
var_dump($packed->toGpu()->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(2)
  }
  [1]=>
  array(2) {
    [0]=>
    int(3)
    [1]=>
    int(4)
  }
}
bool(true)
bool(true)
invalid host shape
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(2)
  }
  [1]=>
  array(2) {
    [0]=>
    int(3)
    [1]=>
    int(4)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(2)
  }
  [1]=>
  array(2) {
    [0]=>
    int(3)
    [1]=>
    int(4)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(2)
  }
  [1]=>
  array(2) {
    [0]=>
    int(3)
    [1]=>
    int(4)
  }
}