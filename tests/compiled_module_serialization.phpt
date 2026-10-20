--TEST--
CompiledModule persists versioned fields and restores metadata transactionally
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

function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}
$compiler = new Compiler(source: <<<'CUDA'
extern "C" __global__ void scale(float *data, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}
extern "C" __global__ void noop() {}
CUDA);
$compiler->kernel('scale', [
    ['name' => 'data', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'factor', 'dtype' => 'float32'],
    ['name' => 'n', 'dtype' => 'int32'],
]);
$compiler->kernel('noop', []);
$module = $compiler->compile();
$payload = $module->__serialize();
check(($payload['__compiled_module_v1'] ?? null) === '1', 'Missing module format version');
check(isset($payload['kernels']['scale']['parameters']), 'Missing portable parameters');
$invalid = [
    array_replace($payload, ['__compiled_module_v1' => '2']),
    array_replace($payload, ['__compiled_module_v1' => 1]),
    array_replace($payload, ['ptx' => 123]),
    array_replace($payload, ['ptx' => 'invalid PTX']),
    array_replace($payload, ['ptx' => ".version\0invalid"]),
    array_replace($payload, ['kernels' => 'invalid']),
    array_replace($payload, ['kernels' => [0 => $payload['kernels']['scale']]]),
    array_replace($payload, ['kernels' => ['scale' => 'invalid']]),
];
foreach (['__compiled_module_v1', 'ptx', 'kernels'] as $key) {
    $bad = $payload;
    unset($bad[$key]);
    $invalid[] = $bad;
}
$bad = $payload;
$bad['kernels']['scale']['name'] = 'wrong';
$invalid[] = $bad;
$bad = $payload;
$bad['kernels']['scale']['parameters'] = array_fill(0, 129, $payload['kernels']['scale']['parameters'][0]);
$invalid[] = $bad;
$bad = $payload;
$bad['kernels']['scale']['parameters'] = [1 => $payload['kernels']['scale']['parameters'][0]];
$invalid[] = $bad;
foreach ([
    ['name' => ''], ['name' => str_repeat('x', 32)], ['name' => "bad\0name"],
    ['dtype' => -1], ['dtype' => PHP_INT_MAX], ['dtype' => 'float32'],
    ['second_dtype' => -1], ['second_dtype' => PHP_INT_MAX],
    ['second_dtype' => $payload['kernels']['scale']['parameters'][1]['second_dtype']],
    ['type' => -1], ['type' => PHP_INT_MAX], ['type' => 'parameter'],
] as $override) {
    $bad = $payload;
    $bad['kernels']['scale']['parameters'][0] =
        array_replace($bad['kernels']['scale']['parameters'][0], $override);
    $invalid[] = $bad;
}
$bad = $payload;
$bad['kernels']['scale']['parameters'][1] = false;
$invalid[] = $bad;
foreach ($invalid as $index => $bad) {
    try {
        $module->__unserialize($bad);
        throw new RuntimeException("Malformed module payload $index was accepted");
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
    check($module->__serialize() === $payload, "Module changed after failed restore ($index)");
}
for ($i = 0; $i < 5; $i++) {
    $module->__unserialize($payload);
    check($module->__serialize() === $payload, 'Repeated module restore');
}
$restored = unserialize(serialize($module));
check($restored->__serialize() === $payload, 'Module metadata round trip');
check($restored->hasKernel('scale') && $restored->hasKernel('noop'), 'Restored kernel names');
echo "Versioned module fields and invalid payload rejection\n";

$values = new CudaArray([1.0, 2.0]);
check($restored->launch('scale', ['grid' => [1, 1, 1], 'block' => [2, 1, 1]],
    [$values, 2.5, 2]), 'Restored scale launch');
check($values->toArray() === [2.5, 5.0], 'Restored scalar and tensor arguments');
check($restored->launch('noop'), 'Restored zero-argument launch');
try {
    $restored->__unserialize($payload);
    throw new RuntimeException('Initialized module accepted metadata replacement');
} catch (PhpTensors\RuntimeException $error) {
    check(str_contains($error->getMessage(), 'initialized'), 'Unexpected initialized module error');
}
$operation = $restored->launchAsync('scale',
    ['grid' => [1, 1, 1], 'block' => [2, 1, 1]], [$values, 2.0, 2]);
try {
    $restored->__unserialize($payload);
    throw new RuntimeException('Active module accepted metadata replacement');
} catch (PhpTensors\RuntimeException $error) {
    check(str_contains($error->getMessage(), 'active'), 'Unexpected active module error');
}
check($restored->sync($operation), 'Preserved async operation');
check($values->toArray() === [5.0, 10.0], 'Module remained usable after rejected restore');
echo "Restored modules execute and reject live replacement\n";
?>
--EXPECT--
Versioned module fields and invalid payload rejection
Restored modules execute and reject live replacement
