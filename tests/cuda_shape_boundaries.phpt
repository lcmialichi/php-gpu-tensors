--TEST--
CUDA scalar indexing and oversized dimensions, indices, axes and virtual sizes are checked before narrowing
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;
function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}
function invalid(callable $operation): void
{
    try {
        $operation();
        throw new RuntimeException('Invalid dimension, index or axis accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
}
$scalar = (new CudaArray([7]))->slice('0');
invalid(function () use ($scalar) { $scalar[0] = 9; });
invalid(function () use ($scalar) { $scalar[null] = 9; });
invalid(fn() => $scalar[0]);
check($scalar->item() === 7.0, 'Scalar changed after rejected indexing');
$one = new CudaArray([7]);
$matrix = new CudaArray([[1, 2], [3, 4]]);
foreach ([4294967296, 4294967297, 4294966297, PHP_INT_MAX, PHP_INT_MIN] as $large) {
    invalid(fn() => $one[$large]);
    invalid(function () use ($one, $large) { $one[$large] = 9; });
    invalid(fn() => $one[[0, $large]]);
    check($one->slice([0, $large])->toArray() === ($large > 0 ? [7.0] : []),
          'Python-style slice bound was narrowed before clamping');
    invalid(fn() => $one->reshape([$large]));
    invalid(fn() => $one->sum($large));
    invalid(fn() => $one->argmax($large));
    invalid(fn() => $matrix->transpose([$large, 1]));
    invalid(fn() => $matrix->concat([$matrix], $large));
    invalid(fn() => Fusion::run(fn() => $one->sum($large)));
}
check($one->toArray() === [7.0], 'Rejected writes changed storage');
check($matrix->sum(-1)->toArray() === [3.0, 7.0], 'Valid negative reduction axis');
$wide = $one->broadcastTo([2147483647, 2]);
check($wide->getSize() === 4294967294, 'Virtual size narrowed to int');
check($wide->getStrides() === [0, 0], 'Broadcast strides');
check($wide[2147483646]->toArray() === [7.0, 7.0], 'Large-axis view offset');
invalid(fn() => $wide->flatten());
invalid(fn() => $wide->argmax());
invalid(fn() => Fusion::run(fn() => $wide->argmax()));
invalid(fn() => $one->broadcastTo([0, 2147483647, 2147483647, 3]));
$empty = CudaArray::zeros([0, 2147483647, 2]);
check($empty->getStrides() === [4294967294, 2, 1], 'Empty strides narrowed');
$result = $empty + $one;
check($result->getShape() === $empty->getShape() && $result->getSize() === 0, 'Empty broadcast');
check($result->getStrides() === $empty->getStrides(), 'Empty result strides');
$emptySuffix = CudaArray::zeros([2147483647, 2147483647, 2147483647, 0]);
invalid(fn() => $emptySuffix->sum(3));
invalid(fn() => $emptySuffix->argmax(3));
invalid(fn() => $emptySuffix->var(3));
invalid(fn() => Fusion::run(fn() => $emptySuffix->sum(3)));
check($emptySuffix->sum(0)->getSize() === 0, 'Valid empty reduction rejected');
check($emptySuffix->slice()->getShape() === $emptySuffix->getShape(), 'Empty slice shape');
check($emptySuffix->getSize() === 0, 'Rejected reduction changed empty source');
$byte = new CudaArray([7], 'int8');
$huge = $byte->broadcastTo([1, 2147483647, 2147483647]);
invalid(fn() => $huge->concat([$huge, $huge], 0));
invalid(fn() => Fusion::run(fn() => $huge->concat([$huge, $huge], 0)));
check($byte->toArray() === [7], 'Rejected concat changed backing storage');
$offset = $matrix->slice('1, :')->slice('1')->reshape([1]);
check($offset->toArray() === [4.0], 'Nested offset scalar view lost its footprint');
$singleton = $byte->broadcastTo([1, 1, 1, 1, 1, 2147483647, 2147483647]);
$emptySlice = Fusion::run(fn() => $singleton->slice('1:1,1:1,1:1,1:1,1:1,:,:'));
check($emptySlice->getShape() === [0, 0, 0, 0, 0, 2147483647, 2147483647],
      'Fusion empty slice metadata overflow');
echo "Scalar indexing and oversized arguments rejected transactionally\n";
echo "Virtual size and broadcast strides retain full precision\n";
echo "View footprints and empty reduction/concat boundaries validated\n";
?>
--EXPECT--
Scalar indexing and oversized arguments rejected transactionally
Virtual size and broadcast strides retain full precision
View footprints and empty reduction/concat boundaries validated
