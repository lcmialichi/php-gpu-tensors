--TEST--
Cuda Array Concat
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip');
?>
--FILE--
<?php
$c1 = PhpTensors\CudaArray::full([2, 2, 2], 10);
$c2 = PhpTensors\CudaArray::full([2, 2, 2], -10);
var_dump($c1->concat([$c2])->toArray());
var_dump($c2->concat([$c1], axis: 1)->toArray());
var_dump($c2->concat([$c1], axis: 2)->toArray());
?>
--EXPECT--
array(4) {
  [0]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(10)
      [1]=>
      float(10)
    }
    [1]=>
    array(2) {
      [0]=>
      float(10)
      [1]=>
      float(10)
    }
  }
  [1]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(10)
      [1]=>
      float(10)
    }
    [1]=>
    array(2) {
      [0]=>
      float(10)
      [1]=>
      float(10)
    }
  }
  [2]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
    }
    [1]=>
    array(2) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
    }
  }
  [3]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
    }
    [1]=>
    array(2) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
    }
  }
}
array(2) {
  [0]=>
  array(4) {
    [0]=>
    array(2) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
    }
    [1]=>
    array(2) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
    }
    [2]=>
    array(2) {
      [0]=>
      float(10)
      [1]=>
      float(10)
    }
    [3]=>
    array(2) {
      [0]=>
      float(10)
      [1]=>
      float(10)
    }
  }
  [1]=>
  array(4) {
    [0]=>
    array(2) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
    }
    [1]=>
    array(2) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
    }
    [2]=>
    array(2) {
      [0]=>
      float(10)
      [1]=>
      float(10)
    }
    [3]=>
    array(2) {
      [0]=>
      float(10)
      [1]=>
      float(10)
    }
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    array(4) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
      [2]=>
      float(10)
      [3]=>
      float(10)
    }
    [1]=>
    array(4) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
      [2]=>
      float(10)
      [3]=>
      float(10)
    }
  }
  [1]=>
  array(2) {
    [0]=>
    array(4) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
      [2]=>
      float(10)
      [3]=>
      float(10)
    }
    [1]=>
    array(4) {
      [0]=>
      float(-10)
      [1]=>
      float(-10)
      [2]=>
      float(10)
      [3]=>
      float(10)
    }
  }
}
