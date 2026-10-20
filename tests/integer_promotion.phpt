--TEST--
Integer promotion uses exact common types and rejects signed with uint64 in eager and Fusion
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
if (PHP_INT_SIZE !== 8) die('skip 64-bit PHP required');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion as F;
function check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}
function invalid(callable $operation): void {
    try {
        $operation();
        throw new RuntimeException('Inexact integer promotion was accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {
    }
}
$types = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'];
$expectedTypes = [
    'int8'   => ['int8', 'int16', 'int32', 'int64', 'int16', 'int32', 'int64', null],
    'int16'  => ['int16', 'int16', 'int32', 'int64', 'int16', 'int32', 'int64', null],
    'int32'  => ['int32', 'int32', 'int32', 'int64', 'int32', 'int32', 'int64', null],
    'int64'  => ['int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', null],
    'uint8'  => ['int16', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'],
    'uint16' => ['int32', 'int32', 'int32', 'int64', 'uint16', 'uint16', 'uint32', 'uint64'],
    'uint32' => ['int64', 'int64', 'int64', 'int64', 'uint32', 'uint32', 'uint32', 'uint64'],
    'uint64' => [null, null, null, null, 'uint64', 'uint64', 'uint64', 'uint64'],
];
foreach ($types as $leftType) foreach ($types as $index => $rightType) {
    $left = new A([1, 2], $leftType);
    $right = new A([2, 1], $rightType);
    $expectedType = $expectedTypes[$leftType][$index];
    foreach (['add' => [3, 3], 'multiply' => [2, 2], 'maximum' => [2, 2],
              'minimum' => [1, 1], 'eq' => [0, 0], 'lt' => [1, 0]] as $operation => $values) {
        foreach ([false, true] as $fused) {
            $run = fn() => $fused ? F::run(fn() => $left->$operation($right)) : $left->$operation($right);
            if ($expectedType === null) {
                invalid($run);
                continue;
            }
            $output = $run();
            check($output->dtype() === $expectedType && $output->toArray() === $values,
                  "$leftType $operation $rightType fused=$fused promotion");
        }
    }
}
foreach ([false, true] as $fused) {
    $run = fn(callable $operation) => $fused ? F::run($operation) : $operation();
    $unsigned = new A([0, 255], 'uint8');
    $signed = new A([-1, 127], 'int8');
    check($run(fn() => $unsigned + $signed)->toArray() === [-1, 382], 'mixed boundary addition');
    check($run(fn() => $unsigned->gt($signed))->toArray() === [1, 1], 'mixed boundary comparison');
    $largeUnsigned = new A([4294967295], 'uint32');
    $negative = new A([-1], 'int32');
    check($run(fn() => $largeUnsigned + $negative)->toArray() === [4294967294],
          'uint32 mixed addition above signed 32-bit range');
    check($run(fn() => $largeUnsigned->gt($negative))->toArray() === [1],
          'uint32 mixed comparison above signed 32-bit range');
    $division = $run(fn() => (new A([3], 'uint32'))->divide(new A([2], 'int32')));
    check($division->dtype() === 'float64' && $division->toArray() === [1.5],
          'mixed division needs FP64 for promoted int64 inputs');
    $power = $run(fn() => (new A([3], 'int64'))->power(new A([2], 'int64')));
    check($power->dtype() === 'float64' && $power->toArray() === [9.0], 'int64 tensor power');
    $scalarDivision = $run(fn() => (new A([16777217], 'int64')) / 1);
    check($scalarDivision->dtype() === 'float64' && $scalarDivision->toArray() === [16777217.0],
          'int64 scalar division lost precision through FP32');
    $floatingScalar = $run(fn() => (new A([16777217], 'int64')) + 0.5);
    check($floatingScalar->dtype() === 'float64' && $floatingScalar->toArray() === [16777217.5],
          'int64 floating scalar promotion');
    $wide = A::fromBuffer(pack('H*', 'ffffffffffffffff'), [1], 'uint64');
    foreach (['add', 'subtract', 'multiply', 'divide', 'power', 'eq', 'lt', 'maximum', 'minimum'] as $op) {
        invalid(fn() => $run(fn() => $wide->$op(new A([1], 'int64'))));
        invalid(fn() => $run(fn() => $wide->$op(1)));
    }
    invalid(fn() => $run(fn() => 1 + $wide));
    invalid(fn() => $run(fn() => 1 - $wide));
    invalid(fn() => $run(fn() => (new A([127], 'int8')) + 128));
    invalid(fn() => $run(fn() => (new A([1], 'uint8')) + -1));
    $maximum = $run(fn() => (new A([1], 'uint8'))->maximum(-1));
    check($maximum->dtype() === 'int64' && $maximum->toArray() === [1], 'scalar maximum common type');
    $zeroUnsigned = A::zeros([0], 'uint64');
    invalid(fn() => $run(fn() => $zeroUnsigned + A::zeros([0], 'int64')));
    invalid(fn() => $run(fn() => $zeroUnsigned + 1));
    $safe = $run(fn() => new A([2], 'uint64') + new A([3], 'uint32'));
    check($safe->dtype() === 'uint64' && $safe->toArray() === [5], 'unsigned widening');
}
$graph = F::compile(fn($a, $b) => [$a + $b, $a->gt($b)],
                    [new A([1], 'uint8'), new A([1], 'int8')]);
[$sum, $mask] = $graph->run(new A([255], 'uint8'), new A([-1], 'int8'));
check($sum->dtype() === 'int16' && $sum->toArray() === [254] &&
      $mask->dtype() === 'int16' && $mask->toArray() === [1],
      'mixed integer graph replay used capture values or the wrong dtype');
echo "Exact mixed integer promotion checks passed\n";
?>
--EXPECT--
Exact mixed integer promotion checks passed
