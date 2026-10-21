--TEST--
Custom kernels and Fusion use explicit completion boundaries and retain shared inputs
--SKIPIF--
<?php
if (!extension_loaded('phptensors') || !class_exists('PhpTensors\\NN', false)) die('skip CUDA-enabled build required');
try {
    if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
} catch (PhpTensors\RuntimeException $error) {
    die('skip CUDA device query unavailable: ' . $error->getMessage());
}
?>
--FILE--
<?php
function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}
$input = new PhpTensors\CudaArray([1, 2, 3]);
$graph = PhpTensors\Fusion::compile(static fn($x) => $x * 2 + 1, [$input]);
$compiler = new PhpTensors\Compiler(source:
    'extern "C" __global__ void increment(float *data) { if (threadIdx.x < 3) data[threadIdx.x] += 1.0f; }');
$compiler->kernel('increment', [['name' => 'data', 'type' => 'array', 'dtype' => 'float32']]);
$module = $compiler->compile();
$config = ['grid' => [1, 1, 1], 'block' => [3, 1, 1]];
$id = $module->launchAsync('increment', config: $config, args: [$input]);
check($module->wait($id), 'custom producer completion');
$execution = $graph->runAsync($input);

try {
    $module->launchAsync('increment', config: $config, args: [$input]);
    throw new RuntimeException('Custom launch accepted while Fusion execution is pending');
} catch (PhpTensors\RuntimeException $error) {
    check(str_contains($error->getMessage(), 'pending execution'), 'explicit custom launch guard');
}
try {
    $input[0] = 99;
    throw new RuntimeException('Mutation accepted while Fusion execution is pending');
} catch (PhpTensors\RuntimeException $error) {
    check(str_contains($error->getMessage(), 'pending execution'), 'explicit mutation guard');
}
unset($input);
$module->cleanup();
$output = $execution->wait();
check($output->toArray() === [5.0, 7.0, 9.0], 'Fusion retains shared native storage');
echo "Custom producer completion precedes Fusion and pending Fusion blocks mutation/custom launches\n";

$module->launchAsync('increment', config: $config, args: [$output]);
check($module->wait(timeout_ms: 5000), 'custom consumer completion');
check($output->toArray() === [6.0, 8.0, 10.0], 'custom consumer output');
echo "Fusion completion precedes custom consumers and host reads\n";
?>
--EXPECT--
Custom producer completion precedes Fusion and pending Fusion blocks mutation/custom launches
Fusion completion precedes custom consumers and host reads
