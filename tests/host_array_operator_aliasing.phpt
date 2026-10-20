--TEST--
HostArray compound assignments are functional, exception-safe and release replaced operands
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
use PhpTensors\HostArray;
require __DIR__ . '/tensor_operator_aliasing.inc';
exerciseTensorOperators(HostArray::class);
$double = new HostArray([16777217], 'float64');
$double += 2;
check($double->dtype() === 'float64' && $double->at(0) === 16777219.0, 'FP64 assignment lost precision');
$matrix = new HostArray([[1, 2], [3, 4]]);
if (array_key_exists('__contiguous_array_v1', $matrix->__serialize())) {
    $row = $matrix[1];
    $row += 1;
    check($row->toArray() === [4.0, 5.0], 'Offset host assignment');
    check($matrix->toArray() === [[1.0, 2.0], [3.0, 4.0]], 'Host assignment mutated shared storage');
}
foreach ([0, new HostArray([1, 0])] as $divisor) {
    $value = new HostArray([2, 4]);
    $old = $value;
    $reference =& $value;
    try {
        $value /= $divisor;
        throw new RuntimeException('Host division by zero was accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
    check($value === $old && $reference === $old, 'Failed division lost the original object');
    check($value->toArray() === [2.0, 4.0], 'Failed division changed host values');
    unset($reference);
}
$value = HostArray::ones([8192]);
$before = memory_get_usage();
for ($i = 0; $i < 200; $i++) {
    $value += 1;
}
check($value->at(8191) === 201.0, 'Repeated host assignment lost values');
check(memory_get_usage() - $before < 512 * 1024, 'Compound assignment leaked host storage');
echo "Host aliases, references, increments and errors preserve the operator contract\n";
echo "Repeated compound assignment has bounded host memory\n";
?>
--EXPECT--
Host aliases, references, increments and errors preserve the operator contract
Repeated compound assignment has bounded host memory
