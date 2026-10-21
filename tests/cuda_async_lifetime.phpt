--TEST--
Async kernel keeps tensor memory alive until stream completion
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (!class_exists('PhpTensors\\NN', false)) die('skip CUDA-enabled build required');
try {
    if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
} catch (PhpTensors\RuntimeException $error) {
    die('skip CUDA device query unavailable: ' . $error->getMessage());
}
?>
--FILE--
<?php
$compiler = new PhpTensors\Compiler(source: 'extern "C" __global__ void touch(float *data) { data[0] += 1.0f; }');
$compiler->kernel('touch', [['name' => 'data', 'type' => 'array', 'dtype' => 'float32']]);
$module = $compiler->compile();
$tensor = new PhpTensors\CudaArray([1.0]);
$weak = WeakReference::create($tensor);
$operation = $module->launchAsync('touch', config: ['grid' => [1, 1, 1], 'block' => [1, 1, 1]], args: [$tensor]);
unset($tensor);
var_dump($weak->get() instanceof PhpTensors\CudaArray);
$module->sync();
var_dump($weak->get() === null);
$next = new PhpTensors\CudaArray([1.0]);
$operation = $module->launchAsync('touch', config: ['grid' => [1, 1, 1], 'block' => [1, 1, 1]], args: [$next]);
usleep(60000);
var_dump($module->sync($operation));
$module->cleanup();
var_dump($next->toArray()[0]);

function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}
$compiler = new PhpTensors\Compiler(source: <<<'CUDA'
extern "C" __global__ void snapshot(float *input, float *output, int count) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && count > 0) output[0] = input[0];
}
extern "C" __global__ void cleanup_probe(float *data, bool enabled) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && enabled) data[0] += 1.0f;
}
CUDA);
$compiler->kernel('snapshot', [
    ['name' => 'input', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'output', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'count', 'dtype' => 'int32'],
]);
$compiler->kernel('cleanup_probe', [
    ['name' => 'data', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'enabled', 'dtype' => 'bool'],
]);
$module = $compiler->compile();
$config = ['grid' => [1, 1, 1], 'block' => [1, 1, 1]];
foreach (['launch', 'launchAsync', 'launchAsyncBatch'] as $method) {
    $input = new PhpTensors\CudaArray([7]);
    $output = new PhpTensors\CudaArray([-1]);
    $earlier = new PhpTensors\CudaArray([-1]);
    $warnings = 0;
    $churn = null;
    set_error_handler(function (int $severity, string $message) use ($input, &$warnings, &$churn): bool {
        if ($severity !== E_WARNING || !str_contains($message, 'could not be converted to int'))
            throw new RuntimeException("Unexpected warning: $message");
        $warnings++;
        $input->__construct([99]);
        $churn = new PhpTensors\CudaArray([123]);
        return true;
    });
    try {
        $args = [$input, $output, new stdClass()];
        if ($method === 'launchAsyncBatch') {
            check($module->launchAsyncBatch([
                ['kernel' => 'snapshot', 'config' => $config, 'args' => [$input, $earlier, 1]],
                ['kernel' => 'snapshot', 'config' => $config, 'args' => $args],
            ]), 'batch submission');
        } else {
            $operation = $module->$method('snapshot', config: $config, args: $args);
            check($operation !== false, "$method submission");
            if ($method === 'launchAsync') check($module->sync(), 'async completion');
        }
    } finally {
        restore_error_handler();
    }
    check($warnings === 1, "$method scalar conversion callback");
    check($input->toArray() === [99.0], "$method replacement binding");
    check($output->toArray() === [7.0], "$method retains submitted allocation");
    if ($method === 'launchAsyncBatch')
        check($earlier->toArray() === [7.0], 'batch retains earlier allocations through later callbacks');

    $output = new PhpTensors\CudaArray([-1]);
    $failure = new LogicException('scalar conversion failed');
    set_error_handler(static function (int $severity, string $message) use ($failure): void {
        if ($severity !== E_WARNING || !str_contains($message, 'could not be converted to int'))
            throw new RuntimeException("Unexpected warning: $message");
        throw $failure;
    });
    try {
        try {
            if ($method === 'launchAsyncBatch')
                $module->launchAsyncBatch([['kernel' => 'snapshot', 'config' => $config,
                    'args' => [$input, $output, new stdClass()]]]);
            else
                $module->$method('snapshot', config: $config, args: [$input, $output, new stdClass()]);
            throw new RuntimeException("$method accepted failed scalar conversion");
        } catch (LogicException $caught) {
            check($caught === $failure, "$method preserves conversion exception");
        }
    } finally {
        restore_error_handler();
    }
    check($module->sync(), "$method failure cleanup");
    check($output->toArray() === [-1.0], "$method must not launch after conversion exception");
}
echo "Native storage snapshots survive callbacks in sync, async and batched launches\n";
echo "Scalar conversion exceptions abort submission and preserve the original exception\n";

final class CleanupProbe
{
    public static int $destroyed = 0;
    public function __construct(private PhpTensors\CompiledModule $module) {}
    public function __destruct()
    {
        self::$destroyed++;
        check($this->module->sync(), 'reentrant cleanup');
    }
}
$cleanup = new PhpTensors\CudaArray([1]);
$module->launchAsync('cleanup_probe', config: $config, args: [$cleanup, new CleanupProbe($module)]);
check($module->sync(), 'outer cleanup');
check(CleanupProbe::$destroyed === 1 && $cleanup->toArray() === [2.0], 'cleanup releases arguments exactly once');
echo "Argument destructors can reenter synchronization without cleaning the same operation twice\n";

foreach (['sync', 'wait', 'isFinished', 'cancelOperation'] as $method) {
    $data = new PhpTensors\CudaArray([1]);
    $id = $module->launchAsync('cleanup_probe', config: $config, args: [$data, true]);
    if ($method === 'isFinished') {
        $deadline = hrtime(true) + 5_000_000_000;
        while (!$module->isFinished($id)) {
            check(hrtime(true) < $deadline, 'completion query deadline');
            usleep(1000);
        }
    } else {
        check($module->$method($id), "$method completion");
    }
    $stats = $module->getStats();
    check($stats['pending_operations'] === 0 && !$stats['has_pending_operations'], "$method pending flags");
    check($data->toArray() === [2.0], "$method output");
    $module->cleanup();
}
echo "All completion methods update pending state consistently\n";

for ($iteration = 0; $iteration < 8; $iteration++) {
    $outputs = [];
    for ($index = 0; $index < 16; $index++) {
        $outputs[] = $data = new PhpTensors\CudaArray([1]);
        $module->launchAsync('cleanup_probe', config: $config, args: [$data, true]);
    }
    unset($data);
    check($module->wait(timeout_ms: 5000), 'global wait');
    foreach ($outputs as $output) check($output->toArray() === [2.0], 'global wait output');
    $stats = $module->getStats();
    check($stats['total_operations'] === 0 && !$stats['has_pending_operations'], 'global wait releases records');
}
echo "Repeated global waits release completed records without accessing freed operations\n";

for ($index = 0; $index < 120; $index++) {
    $data = new PhpTensors\CudaArray([1]);
    $id = $module->launchAsync('cleanup_probe', config: $config, args: [$data, true]);
    check($module->sync($id), 'stream pool pressure completion');
    check($data->toArray() === [2.0], 'stream pool pressure output');
}
$stats = $module->getStats();
check($stats['total_operations'] <= 100 && $stats['stream_pool_size'] <= 100, 'bounded stream ownership');
check($stats['pending_operations'] === 0 && !$stats['has_pending_operations'], 'pool pressure pending flags');
check($module->sync() && $module->getStats()['total_operations'] === 0, 'global sync releases completed records');
echo "Stream pool pressure reclaims owned operations before reusing their streams\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
float(2)
Native storage snapshots survive callbacks in sync, async and batched launches
Scalar conversion exceptions abort submission and preserve the original exception
Argument destructors can reenter synchronization without cleaning the same operation twice
All completion methods update pending state consistently
Repeated global waits release completed records without accessing freed operations
Stream pool pressure reclaims owned operations before reusing their streams