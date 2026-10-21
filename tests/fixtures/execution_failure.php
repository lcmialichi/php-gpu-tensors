<?php
declare(strict_types=1);

use PhpTensors\CudaArray;
use PhpTensors\Fusion;
use PhpTensors\HostArray;

function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}

$input = new CudaArray([1]);
$graph = Fusion::compile(static fn($x) => $x + 1, [$input]);
check($graph->run($input)->toArray() === [2.0], 'healthy graph');
$compiler = new PhpTensors\Compiler(source:
    'extern "C" __global__ void fail(float *data) { if (threadIdx.x == 0) asm("trap;"); }');
$compiler->kernel('fail', [['name' => 'data', 'type' => 'array', 'dtype' => 'float32']]);
$module = $compiler->compile();
$observer = unserialize(serialize($module));
$config = ['grid' => [1, 1, 1], 'block' => [1, 1, 1]];

try {
    if ($method === 'launchAsyncBatch') {
        $module->launchAsyncBatch([['kernel' => 'fail', 'config' => $config, 'args' => [$input]]]);
    } else {
        $launch = in_array($method, ['destroy', 'query'], true) ? 'launchAsync' : $method;
        $id = $module->$launch('fail', config: $config, args: [$input]);
        if ($method === 'destroy') {
            unset($module);
        } elseif ($method === 'query') {
            $deadline = hrtime(true) + 5_000_000_000;
            while (!$module->isFinished($id)) {
                check(hrtime(true) < $deadline, 'failure query deadline');
                usleep(1000);
            }
        } elseif ($method === 'launchAsync') {
            $module->sync($id);
        }
    }
    throw new RuntimeException('Failed kernel was accepted');
} catch (PhpTensors\RuntimeException $error) {
    check(str_contains($error->getMessage(), 'restart the PHP process'), 'explicit restart requirement');
}
echo "Unconfirmed completion disables the CUDA runtime\n";

foreach ([
    static fn() => cuda_get_device_count(),
    static fn() => cuda_device_reset(),
    static fn() => cuda_clear_error(),
    static fn() => $observer->launch('fail', config: $config, args: [$input]),
    static fn() => $observer->sync(),
    static fn() => $graph->run($input),
    static fn() => $graph->runAsync($input),
] as $attempt) {
    try {
        $attempt();
        throw new RuntimeException('CUDA operation accepted after runtime failure');
    } catch (PhpTensors\RuntimeException $error) {
        check(str_contains($error->getMessage(), 'restart the PHP process'), 'runtime remains disabled');
    }
}
echo "Clearing errors and resetting devices cannot revive the failed runtime\n";
check((new HostArray([1, 2]) + 1)->toArray() === [2.0, 3.0], 'CPU operations remain available');
echo "CPU tensors remain usable\n";
unset($module, $observer, $graph, $input);
gc_collect_cycles();
echo "Failed resources can be destroyed without recycling uncertain GPU storage\n";
