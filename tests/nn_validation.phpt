--TEST--
Cuda NN automatic backend fallback and strict inference input validation
--SKIPIF--
<?php
if (!extension_loaded('phptensors') || cuda_get_device_count() < 1) die('skip CUDA GPU required');
?>
--FILE--
<?php
use PhpTensors\NN;
use PhpTensors\CudaArray as A;
function check_nn(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}
function rejected(callable $fn, string $type): void {
    try { $fn(); throw new Exception('Expected error'); }
    catch (Throwable $e) { if (!$e instanceof $type) throw $e; }
}
$x = A::ones([1, 2, 4, 4]); $w = A::ones([2, 2, 2, 2]);
check_nn(NN::conv2d($x, $w, A::fromFlatArray([1, -1], [2]))->toArray() === [
    [
        array_fill(0, 3, array_fill(0, 3, 9.0)),
        array_fill(0, 3, array_fill(0, 3, 7.0)),
    ],
], 'convolution fallback result mismatch');
$grouped = NN::conv2d(
    A::fromFlatArray([2, 3], [1, 2, 1, 1]),
    A::fromFlatArray([4, 5], [2, 1, 1, 1]),
    groups: 2
)->toArray();
check_nn($grouped === [[[[8.0]], [[15.0]]]], 'grouped convolution result mismatch');
$dilated = NN::conv2d(
    A::fromFlatArray(range(1, 9), [1, 1, 3, 3]),
    A::ones([1, 1, 2, 2]),
    dilation: [2, 2]
)->toArray();
check_nn($dilated === [[[[20.0]]]], 'dilated convolution result mismatch');
$padded = A::fromFlatArray([2], [1, 1, 1, 1]);
check_nn(NN::pool2d($padded, [2, 2], [1, 1], [1, 1], 'average')->toArray() === [
    [[[2.0, 2.0], [2.0, 2.0]]],
], 'average pooling must exclude padding');
check_nn(NN::pool2d($padded, [2, 2], [1, 1], [1, 1], 'max')->toArray() === [
    [[[2.0, 2.0], [2.0, 2.0]]],
], 'max pooling with padding mismatch');
$softmax = NN::softmax(A::fromFlatArray([1000, 999], [1, 2, 1, 1]))->toArray();
check_nn(abs($softmax[0][0][0][0] - 0.7310585786) < 0.000001 &&
    abs($softmax[0][1][0][0] - 0.2689414214) < 0.000001,
    'channel softmax must be numerically stable');
rejected(fn() => NN::softmax(A::ones([1, 2, 4, 4])->requiresGrad()), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::conv2d(A::ones([1, 2, 4, 4])->requiresGrad(), $w), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::softmax(A::ones([1, 2, 4, 4], 'float64')), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::softmax(A::ones([2, 4])), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::softmax($x->slice(':, :, ::2, :')), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::conv2d($x, $w, stride: [0, 1]), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::conv2d($x, $w, stride: ['h' => 1, 'w' => 1]), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::conv2d($x, $w, bias: A::ones([1])), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::conv2d($x, $w, groups: 2), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::pool2d($x, [2, 2], mode: 'sum'), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::pool2d($x, [2, 2], padding: [2, 0]), PhpTensors\InvalidArgumentException::class);
rejected(fn() => NN::pool2d($x, [PHP_INT_MAX, 2]), PhpTensors\InvalidArgumentException::class);
rejected(fn() => PhpTensors\Fusion::run(fn() => NN::softmax($x)), PhpTensors\RuntimeException::class);
echo "validation passed\n";
?>
--EXPECT--
validation passed
