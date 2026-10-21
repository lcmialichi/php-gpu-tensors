--TEST--
CUDA initializes after fork and rejects inherited runtime access without touching parent resources
--SKIPIF--
<?php
if (!extension_loaded('phptensors') || !class_exists('PhpTensors\\NN', false)) die('skip CUDA backend unavailable');
if (!function_exists('pcntl_fork')) die('skip pcntl required');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\Compiler;
use PhpTensors\CudaArray;
use PhpTensors\Fusion;
use PhpTensors\HostArray;

function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}
function waitWorker(int $pid): void
{
    check($pid > 0, 'fork failed');
    check(pcntl_waitpid($pid, $status) === $pid, 'wait failed');
    check(pcntl_wifexited($status) && pcntl_wexitstatus($status) === 0, 'worker failed');
}
function rejected(callable $operation): void
{
    try {
        $operation();
    } catch (PhpTensors\RuntimeException $error) {
        check(str_contains($error->getMessage(), 'after fork'), 'wrong ownership error: ' . $error->getMessage());
        return;
    }
    throw new RuntimeException('inherited CUDA runtime accepted');
}

$host = new HostArray([1, 2, 3]);
for ($i = 0; $i < 3; $i++) {
    $pid = pcntl_fork();
    if ($pid === 0) {
        pcntl_alarm(20);
        check($host->toArray() === [1.0, 2.0, 3.0], 'plain CPU storage after fork');
        $tensor = $host->toGpu();
        $graph = Fusion::compile(fn($x) => ($x + 1)->sum(), [$tensor], cudaGraph: true);
        check($graph->run($tensor)->toArray() === [9.0], 'worker CUDA graph');
        $pending = $graph->runAsync($tensor);
        exit(0);
    }
    waitWorker($pid);
}
echo "workers initialize and recycle independently\n";

check(cuda_get_device_count() > 0, 'parent query');
$pid = pcntl_fork();
if ($pid === 0) {
    pcntl_alarm(20);
    rejected(fn() => cuda_get_device_count());
    rejected(fn() => CudaArray::ones([1]));
    exit(0);
}
waitWorker($pid);
echo "queries establish process ownership\n";

$tensor = new CudaArray([1, 2, 3]);
$pinned = new HostArray([4, 5, 6], pinned: true);
$compiler = new Compiler('extern "C" __global__ void noop() {}');
$compiler->kernel('noop', []);
$module = $compiler->compile();
check($module->launch('noop'), 'parent module launch');
$graph = Fusion::compile(fn($x) => ($x + 1)->sum(), [$tensor], cudaGraph: true);
$pending = $graph->runAsync($tensor);
$pid = pcntl_fork();
if ($pid === 0) {
    pcntl_alarm(20);
    foreach ([
        fn() => cuda_get_device_count(),
        fn() => cuda_get_device_info(),
        fn() => cuda_get_current_device(),
        fn() => cuda_set_device(0),
        fn() => cuda_get_driver_version(),
        fn() => cuda_get_runtime_version(),
        fn() => cuda_get_memory_info(),
        fn() => cuda_get_backend_info(),
        fn() => cuda_set_matmul_precision('fp32-strict'),
        fn() => cuda_get_peer_access(0, 0),
        fn() => cuda_get_last_error(),
        fn() => cuda_clear_error(),
        fn() => cuda_synchronize(),
        fn() => $tensor->toArray(),
        fn() => $pinned->toArray(),
        fn() => $compiler->compile(),
        fn() => new Compiler('extern "C" __global__ void another() {}'),
        fn() => $module->sync(),
        fn() => $module->isFinished(),
        fn() => $module->wait(1),
        fn() => $module->getAsyncStatus(),
        fn() => $module->getPendingOperations(),
        fn() => $module->cancelOperation(1),
        fn() => $module->cleanup(),
        fn() => $graph->run($tensor),
        fn() => $pending->wait(),
        fn() => $pending->isFinished(),
    ] as $operation) {
        rejected($operation);
    }
    unset($operation, $pending, $graph, $module, $compiler, $pinned, $tensor);
    gc_collect_cycles();
    check($host->toArray() === [1.0, 2.0, 3.0], 'CPU remains usable in blocked child');
    exit(0);
}
waitWorker($pid);
check($pending->wait()->toArray() === [9.0], 'parent pending work survives child shutdown');
check($graph->run($tensor)->toArray() === [9.0], 'parent graph survives child shutdown');
check($module->launch('noop'), 'parent module survives child shutdown');
check($pinned->toArray() === [4.0, 5.0, 6.0], 'parent pinned storage survives child shutdown');
check(($tensor + 2)->toArray() === [3.0, 4.0, 5.0], 'parent allocator survives child shutdown');
echo "inherited CUDA rejected and safely discarded; parent remains usable\n";

$iteratorPid = -1;
try {
    foreach ($pinned as $value) {
        if ($iteratorPid === -1) {
            $iteratorPid = pcntl_fork();
            if ($iteratorPid === 0) pcntl_alarm(20);
            else {
                waitWorker($iteratorPid);
                break;
            }
        }
    }
    check($iteratorPid > 0, 'inherited pinned iterator accepted');
} catch (PhpTensors\RuntimeException $error) {
    check($iteratorPid === 0 && str_contains($error->getMessage(), 'after fork'), 'iterator ownership error');
    exit(0);
}
echo "active pinned iterators revalidate ownership\n";

$callbackPid = -1;
try {
    $result = Fusion::run(function () use ($tensor, &$callbackPid) {
        $result = $tensor + 1;
        $callbackPid = pcntl_fork();
        if ($callbackPid === 0) pcntl_alarm(20);
        return $result;
    });
    check($callbackPid > 0, 'forked callback returned without ownership validation');
} catch (PhpTensors\RuntimeException $error) {
    check($callbackPid === 0 && str_contains($error->getMessage(), 'after fork'), 'callback ownership error');
    exit(0);
}
waitWorker($callbackPid);
check($result->toArray() === [2.0, 3.0, 4.0], 'parent callback capture preserved');
echo "callback return revalidates ownership before materializing\n";
?>
--EXPECT--
workers initialize and recycle independently
queries establish process ownership
inherited CUDA rejected and safely discarded; parent remains usable
active pinned iterators revalidate ownership
callback return revalidates ownership before materializing
