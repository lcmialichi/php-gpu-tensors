--TEST--
CUDA tensors, compiled modules and Fusion share ownership and survive rejected resets
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA backend not built');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\Compiler;
use PhpTensors\CudaArray;
use PhpTensors\Fusion;

function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}
function rejectReset(): void
{
    try {
        cuda_device_reset();
        throw new RuntimeException('Reset accepted with an active pool');
    } catch (PhpTensors\RuntimeException $error) {
        check(str_contains($error->getMessage(), 'memory pool'), 'Unexpected reset error');
    }
}

rejectReset();
$a = new CudaArray([[1, 2], [3, 4]]);
$view = $a[1];
$graph = Fusion::compile(fn($x) => ($x * 2 + 1)->sum(), [$a], cudaGraph: true);
check($graph->getStats()['backend'] === 'cuda-graph', 'Native graph was not captured');
check($graph->run($a)->item() === 24.0, 'Initial graph result');
$matrix = CudaArray::ones([128, 128]);
check($matrix->matmul($matrix)->sum()->item() === 2097152.0, 'Initial backend result');

$compiler = new Compiler(source: <<<'CUDA'
extern "C" __global__ void increment(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1.0f;
}
CUDA);
$compiler->kernel('increment', [
    ['name' => 'data', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'n', 'dtype' => 'int32'],
]);
$prototype = $compiler->compile();
$snapshot = serialize($prototype);
$config = ['grid' => [1, 1, 1], 'block' => [4, 1, 1]];
for ($i = 0; $i < 10; $i++) {
    $first = unserialize($snapshot);
    $second = unserialize($snapshot);
    check($first->initialize() && $second->initialize(), 'Module initialization');
    check($graph->run($a)->item() === 24.0, 'Module initialization changed graph context');
    $work = new CudaArray([1, 2, 3, 4]);
    $operation = $first->launchAsync('increment', $config, [$work, 4]);
    rejectReset();
    check($first->sync($operation), 'Pending module operation was lost');
    check($work->toArray() === [2.0, 3.0, 4.0, 5.0], 'First module result');
    unset($first);
    check($graph->run($a)->item() === 24.0, 'Partial module release invalidated graph');
    $second->launchAsync('increment', $config, [$work, 4]);
    unset($second);
    check($work->toArray() === [3.0, 4.0, 5.0, 6.0], 'Module destruction did not finish work');
    check($work->toHost()->toGpu()->toArray() === $work->toArray(), 'Transfers after module release');
    check($view->toArray() === [3.0, 4.0], 'View storage invalidated');
    check($graph->run($a)->item() === 24.0, 'Last module release invalidated graph');
    check($matrix->matmul($matrix)->sum()->item() === 2097152.0, 'Backend context invalidated');
}
rejectReset();
check(($a + 1)->toArray() === [[2.0, 3.0], [4.0, 5.0]], 'Allocation after rejected reset');
check(cuda_synchronize(), 'Owning device synchronization');
echo "Shared primary context survived ten module lifecycles\n";
echo "Reset rejection preserved tensors, views, graphs, backend handles and transfers\n";
?>
--EXPECT--
Shared primary context survived ten module lifecycles
Reset rejection preserved tensors, views, graphs, backend handles and transfers
