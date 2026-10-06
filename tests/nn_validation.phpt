--TEST--
Cuda NN optional capability and strict inference input validation
--SKIPIF--
<?php
if (!extension_loaded('cuda') || cuda_get_device_count() < 1) die('skip CUDA GPU required');
?>
--FILE--
<?php
use Cuda\NN;
use Cuda\CudaArray as A;
function rejected(callable $fn, string $type): void {
    try { $fn(); throw new Exception('Expected error'); }
    catch (Throwable $e) { if (!$e instanceof $type) throw $e; }
}
var_dump(NN::isAvailable() === cuda_get_backend_info()['cudnn']);
$x = A::ones([1, 2, 4, 4]); $w = A::ones([2, 2, 2, 2]);
rejected(fn() => NN::softmax(A::ones([1, 2, 4, 4], 'float64')), Cuda\InvalidArgumentException::class);
rejected(fn() => NN::softmax(A::ones([2, 4])), Cuda\InvalidArgumentException::class);
rejected(fn() => NN::softmax($x->slice(':, :, ::2, :')), Cuda\InvalidArgumentException::class);
rejected(fn() => NN::conv2d($x, $w, stride: [0, 1]), Cuda\InvalidArgumentException::class);
rejected(fn() => NN::conv2d($x, $w, stride: ['h' => 1, 'w' => 1]), Cuda\InvalidArgumentException::class);
rejected(fn() => NN::conv2d($x, $w, bias: A::ones([1])), Cuda\InvalidArgumentException::class);
rejected(fn() => NN::conv2d($x, $w, groups: 2), Cuda\InvalidArgumentException::class);
rejected(fn() => NN::pool2d($x, [2, 2], mode: 'sum'), Cuda\InvalidArgumentException::class);
rejected(fn() => NN::pool2d($x, [2, 2], padding: [2, 0]), Cuda\InvalidArgumentException::class);
rejected(fn() => NN::pool2d($x, [PHP_INT_MAX, 2]), Cuda\InvalidArgumentException::class);
rejected(fn() => Cuda\Fusion::run(fn() => NN::softmax($x)), Cuda\RuntimeException::class);
if (!NN::isAvailable()) {
    rejected(fn() => NN::conv2d($x, $w, null), Cuda\RuntimeException::class);
    rejected(fn() => NN::pool2d($x, [2, 2]), Cuda\RuntimeException::class);
    rejected(fn() => NN::softmax($x), Cuda\RuntimeException::class);
}
echo "validation passed\n";
?>
--EXPECT--
bool(true)
validation passed
