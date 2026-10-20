--TEST--
Tensor factories reject overflowing shapes and preserve valid empty metadata without large allocations
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
use PhpTensors\HostArray;
use PhpTensors\CudaArray;
function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}
function invalid(callable $operation): void
{
    try {
        $operation();
        throw new RuntimeException('Invalid shape accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
}
$classes = [HostArray::class];
if (class_exists(PhpTensors\NN::class, false)) $classes[] = CudaArray::class;
foreach ($classes as $class) {
    foreach ([[], [-1], [1.5], ['1'], array_fill(0, 11, 1)] as $shape) {
        invalid(fn() => $class::zeros($shape));
        invalid(fn() => $class::fromBuffer('', $shape));
    }
    $maximum = $class::ones(array_fill(0, 10, 1));
    check($maximum->getSize() === 1 && $maximum->getNdims() === 10, 'Maximum rank');
    foreach ([[0], [2, 0, 3], [0, 2, 3]] as $shape) {
        $empty = $class::zeros($shape);
        check($empty->getShape() === $shape && $empty->getSize() === 0, 'Empty shape');
        check($empty->toBuffer() === '', 'Empty buffer');
    }
    if (PHP_INT_SIZE >= 8) {
        foreach ([[0, 2147483648], [0, PHP_INT_MAX], [0, 2147483647, 2147483647, 3],
                  [2147483647, 2147483647, 3]] as $shape) {
            invalid(fn() => $class::zeros($shape));
            invalid(fn() => $class::fromBuffer('', $shape));
        }
        invalid(fn() => $class::zeros([2147483647, 2147483647], dtype: 'float64'));
        $shape = [2147483647, 2147483647, 2147483647, 0];
        $empty = $class::zeros($shape);
        check($empty->getShape() === $shape && $empty->getSize() === 0, 'Empty suffix metadata');
        check($empty->toBuffer() === '', 'Empty suffix buffer');
        $restored = unserialize(serialize($empty));
        check($restored->getShape() === $shape && $restored->getSize() === 0, 'Empty metadata round trip');
    }
}
echo "Checked shape, stride and byte limits without large allocations\n";
echo "Maximum rank and empty metadata preserved\n";
?>
--EXPECT--
Checked shape, stride and byte limits without large allocations
Maximum rank and empty metadata preserved
