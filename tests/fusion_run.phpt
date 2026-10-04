--TEST--
Fusion scoped execution, eager opt-out, temporary and escaped tensor lifetimes
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
$a = new CudaArray([1, 2, 3]);
$b = new CudaArray([4, 5, 6]);
$c = new CudaArray([7, 8, 9]);
$expected = ($a + $b * $c)->toArray();
$fused = Fusion::run(fn() => $a + $b * $c);
var_dump($fused->toArray() === $expected);
var_dump(Fusion::run(fn() => $a + $b * $c, enabled: false)->toArray() === $expected);
var_dump(Fusion::run(fn() => (new CudaArray([2, 3])) * 2 + 1)->toArray());
$escaped = null;
$outputs = Fusion::run(function () use ($a, $b, &$escaped) {
    $escaped = $a * 2;
    $product = $b * 3;
    var_dump($product->getShape(), $product->dtype());
    return ['first' => $product + $a, 'nested' => [$product - 1]];
});
var_dump($escaped->toArray(), $outputs['first']->toArray(), $outputs['nested'][0]->toArray());
$result = Fusion::run(function () use ($a) {
    $x = $a * 2;
    var_dump($x->toArray());
    return $x + 1;
});
var_dump($result->toArray());
?>
--EXPECT--
bool(true)
bool(true)
array(2) {
  [0]=>
  float(5)
  [1]=>
  float(7)
}
array(1) {
  [0]=>
  int(3)
}
string(7) "float32"
array(3) {
  [0]=>
  float(2)
  [1]=>
  float(4)
  [2]=>
  float(6)
}
array(3) {
  [0]=>
  float(13)
  [1]=>
  float(17)
  [2]=>
  float(21)
}
array(3) {
  [0]=>
  float(11)
  [1]=>
  float(14)
  [2]=>
  float(17)
}
array(3) {
  [0]=>
  float(2)
  [1]=>
  float(4)
  [2]=>
  float(6)
}
array(3) {
  [0]=>
  float(3)
  [1]=>
  float(5)
  [2]=>
  float(7)
}
