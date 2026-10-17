--TEST--
Async kernel keeps tensor memory alive until stream completion
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
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
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
float(2)