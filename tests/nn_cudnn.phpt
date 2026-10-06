--TEST--
cuDNN strict FP32 convolution groups, stride, dilation, padding, bias, pooling and channel softmax
--SKIPIF--
<?php
if (!extension_loaded('cuda') || cuda_get_device_count() < 1) die('skip CUDA GPU required');
if (!Cuda\NN::isAvailable()) die('skip cuDNN optional backend unavailable');
?>
--FILE--
<?php
use Cuda\NN;
use Cuda\CudaArray as A;
function close(array $actual, array $expected): void {
    if (count($actual) !== count($expected)) throw new Exception('shape mismatch');
    foreach ($expected as $i => $value) {
        if (is_array($value)) close($actual[$i], $value);
        elseif (!is_finite($actual[$i]) || abs($actual[$i] - $value) > 1e-5 * max(1, abs($value))) throw new Exception('numeric mismatch');
    }
}
$x = A::fromFlatArray(range(1, 16), [1, 1, 4, 4]);
$w = A::ones([1, 1, 2, 2]);
close(NN::conv2d($x, $w, null)->toArray(), [[[[14, 18, 22], [30, 34, 38], [46, 50, 54]]]]);
close(NN::conv2d($x, $w, A::full([1], 2), stride: [2, 2])->toArray(), [[[[16, 24], [48, 56]]]]);
close(NN::conv2d($x, $w, dilation: [2, 2])->toArray(), [[[[24, 28], [40, 44]]]]);
close(NN::conv2d($x, $w, stride: [2, 2], padding: [1, 1])->toArray(),
      [[[[1, 5, 4], [14, 34, 20], [13, 29, 16]]]]);
$grouped = A::ones([1, 2, 4, 4]);
$filters = A::fromFlatArray([1, 2], [2, 1, 1, 1]);
close(NN::conv2d($grouped, $filters, groups: 2)->toArray(),
      [[array_fill(0, 4, array_fill(0, 4, 1)), array_fill(0, 4, array_fill(0, 4, 2))]]);
for ($i = 0; $i < 3; $i++) close(NN::conv2d($x, $w)->toArray(), [[[[14, 18, 22], [30, 34, 38], [46, 50, 54]]]]);
close(NN::pool2d($x, [2, 2])->toArray(), [[[[6, 8], [14, 16]]]]);
close(NN::pool2d($x, [2, 2], mode: 'average')->toArray(), [[[[3.5, 5.5], [11.5, 13.5]]]]);
close(NN::pool2d($x, [2, 2], padding: [1, 1], mode: 'average')->toArray(),
      [[[[1, 2.5, 4], [7, 8.5, 10], [13, 14.5, 16]]]]);
$soft = NN::softmax(A::fromFlatArray([1000, 1001, 1000, 1002], [1, 2, 1, 2]))->toArray();
close($soft, [[[[0.5, 1 / (1 + exp(1))]], [[0.5, exp(1) / (1 + exp(1))]]]]);
echo "cuDNN inference passed\n";
?>
--EXPECT--
cuDNN inference passed
