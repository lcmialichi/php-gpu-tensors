--TEST--
CUDA-profile nullable parsers and named arguments match stubs without submitting GPU work
--SKIPIF--
<?php
if (!extension_loaded('phptensors') || !class_exists('PhpTensors\\Compiler')) {
    die('skip CUDA build required; no GPU needed');
}
?>
--FILE--
<?php
declare(strict_types=1);
use PhpTensors\CudaArray;

function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}
$compiler = new PhpTensors\Compiler(source: 'extern "C" __global__ void noop() {}', target: null);
check($compiler->kernel(headers: null, parameters: null, name: 'noop') === $compiler, 'Nullable kernel arrays');
check($compiler->header(code: '// declaration') === $compiler, 'Static return type follows fluent compiler');
check(array_key_exists('noop', $compiler->getKernels()), 'Kernel registered without CUDA initialization');
echo "compiler nullable arrays and fluent named calls\n";

$module = new PhpTensors\CompiledModule();
foreach ([null, -1] as $id) {
    check($module->sync(op_id: $id), 'Sync all');
    check($module->isFinished(op_id: $id), 'Query all');
    check($module->wait(timeout_ms: 0, op_id: $id), 'Wait all');
    check($module->getAsyncStatus(op_id: $id) === [], 'All statuses, not operation zero');
}
check($module->sync() && $module->isFinished() && $module->wait(), 'Omitted IDs');
check($module->wait(timeout_ms: 0), 'Named argument fills the nullable operation default');
check($module->getAsyncStatus() === [], 'Omitted status ID');
try {
    $module->launchAsyncBatch(operations: null);
    throw new RuntimeException('Null batch accepted');
} catch (TypeError $expected) {}
echo "module null IDs mean all operations; batches require arrays\n";

foreach ([
    fn() => new CudaArray(data: [[1], [2, 3]], dtype: null),
    fn() => CudaArray::fromFlatArray(values: [], shape: [-1], dtype: null),
    fn() => CudaArray::fromBuffer(bytes: '', shape: [-1], dtype: null),
    fn() => CudaArray::fromFile(path: 'unused', shape: [-1], dtype: null),
    fn() => CudaArray::zeros(shape: [-1], dtype: null),
    fn() => CudaArray::ones(shape: [-1], dtype: null),
    fn() => CudaArray::full(shape: [-1], value: true, dtype: null),
] as $call) {
    try {
        $call();
        throw new RuntimeException('Malformed input accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {}
}
echo "nullable dtype factories reach input validation without GPU allocation\n";
foreach ([
    fn() => cuda_get_device_info(deviceId: -1),
    fn() => cuda_set_device(deviceId: -1),
] as $call) {
    try {
        $call();
        throw new RuntimeException('Invalid device ID accepted');
    } catch (PhpTensors\InvalidArgumentException $expected) {}
}
echo "device helper named arguments use the documented deviceId spelling\n";
?>
--EXPECT--
compiler nullable arrays and fluent named calls
module null IDs mean all operations; batches require arrays
nullable dtype factories reach input validation without GPU allocation
device helper named arguments use the documented deviceId spelling
