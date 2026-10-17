--TEST--
CudaArray import from packed bytes, raw files and NumPy arrays
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
$bytes = pack('g*', 1, 2, 3, 4);
$raw = tempnam(sys_get_temp_dir(), 'cuda-raw-');
$npy = tempnam(sys_get_temp_dir(), 'cuda-npy-');
try {
    var_dump(PhpTensors\CudaArray::fromBuffer($bytes, [2, 2])->toArray());
    file_put_contents($raw, $bytes);
    var_dump(PhpTensors\CudaArray::fromFile($raw, [2, 2])->toArray());

    $header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }";
    $padding = (16 - ((10 + strlen($header) + 1) % 16)) % 16;
    $header .= str_repeat(' ', $padding) . "\n";
    file_put_contents($npy, "\x93NUMPY" . chr(1) . chr(0) . pack('v', strlen($header)) . $header . $bytes);
    var_dump(PhpTensors\CudaArray::fromNpy($npy)->toArray());

    $halfBytes = pack('v*', 0x3c00, 0x4000, 0x4200, 0x4400);
    $halfHeader = "{'descr': '<f2', 'fortran_order': False, 'shape': (2, 2), }";
    $padding = (16 - ((10 + strlen($halfHeader) + 1) % 16)) % 16;
    $halfHeader .= str_repeat(' ', $padding) . "\n";
    file_put_contents($npy, "\x93NUMPY" . chr(1) . chr(0) .
        pack('v', strlen($halfHeader)) . $halfHeader . $halfBytes);
    $halfNpy = PhpTensors\CudaArray::fromNpy($npy);
    if ($halfNpy->dtype() !== 'float16' ||
        $halfNpy->toArray() !== [[1.0, 2.0], [3.0, 4.0]]) {
        throw new RuntimeException('FP16 NumPy import failed');
    }

    try {
        PhpTensors\CudaArray::fromBuffer($bytes, [3, 2]);
    } catch (PhpTensors\InvalidArgumentException $error) {
        echo "invalid buffer size\n";
    }
    file_put_contents($npy, "\x93NUMPY" . chr(1) . chr(0) . pack('v', strlen($header)) .
        str_replace("'<f4'", "'>f4'", $header) . $bytes);
    try {
        PhpTensors\CudaArray::fromNpy($npy);
    } catch (PhpTensors\InvalidArgumentException $error) {
        echo "unsupported endian\n";
    }
} finally {
    unlink($raw);
    unlink($npy);
}
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1)
    [1]=>
    float(2)
  }
  [1]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(4)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1)
    [1]=>
    float(2)
  }
  [1]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(4)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1)
    [1]=>
    float(2)
  }
  [1]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(4)
  }
}
invalid buffer size
unsupported endian