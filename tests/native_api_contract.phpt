--TEST--
Native API comparison detects signature drift and HostArray named arguments follow the stubs
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
declare(strict_types=1);
require __DIR__ . '/check_native_api.php';
use PhpTensors\HostArray;

function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}
$extension = new ReflectionExtension('phptensors');
$surface = nativeApiSurface(array_keys($extension->getFunctions()));
check(!class_exists('PhpTensors\\CpuBuild\\CudaArray', false), 'No synthetic public CPU-build class');
check(!class_exists('PhpTensorsTestFixtures\\UnavailableCudaArray', false), 'Test fixtures are not loaded into native consumers');
$class = 'PhpTensors\\HostArray';
$mutations = [
    [[$class, 'final'], false],
    [[$class, 'interfaces', 0], 'ArrayAccess'],
    [[$class, 'interface'], true],
    [[$class, 'parent'], 'stdClass'],
    [[$class, 'methods', 'fromFlatArray', 'visibility'], 'private'],
    [[$class, 'methods', 'fromFlatArray', 'static'], false],
    [[$class, 'methods', 'fromFlatArray', 'final'], true],
    [[$class, 'methods', 'fromFlatArray', 'abstract'], true],
    [[$class, 'methods', 'fromFlatArray', 'required'], 1],
    [[$class, 'methods', 'fromFlatArray', 'returnsReference'], true],
    [[$class, 'methods', 'fromFlatArray', 'returnType'], 'object'],
    [[$class, 'methods', 'fromFlatArray', 'parameters', 0, 'name'], 'renamed'],
    [[$class, 'methods', 'fromFlatArray', 'parameters', 0, 'type'], 'mixed'],
    [[$class, 'methods', 'fromFlatArray', 'parameters', 2, 'type'], 'string'],
    [[$class, 'methods', 'fromFlatArray', 'parameters', 0, 'byReference'], true],
    [[$class, 'methods', 'fromFlatArray', 'parameters', 2, 'optional'], false],
    [[$class, 'methods', 'fromFlatArray', 'parameters', 2, 'hasDefault'], false],
    [[$class, 'methods', 'fromFlatArray', 'parameters', 2, 'default'], null],
    [[$class, 'methods', 'at', 'parameters', 0, 'variadic'], false],
];
foreach ($mutations as [$path, $value]) {
    $changed = $surface;
    $target = &$changed['classes'];
    foreach ($path as $key) $target = &$target[$key];
    $target = $value;
    unset($target);
    $differences = nativeApiDifferences($surface, $changed);
    check(count($differences) === 1, 'A single signature mutation must produce exactly one diagnostic');
    check(str_starts_with($differences[0], 'classes.' . implode('.', $path) . ':'), 'Precise drift path');
}
$changed = $surface;
unset($changed['classes'][$class]['methods']['copy']);
check(count(nativeApiDifferences($surface, $changed)) === 1, 'Missing method');
$changed['classes'][$class]['methods']['copy'] = $surface['classes'][$class]['methods']['copy'];
$changed['classes'][$class]['methods']['unexpected'] = [];
check(count(nativeApiDifferences($surface, $changed)) === 1, 'Unexpected method');
echo "signature, flags, interface, presence and named-parameter drift detected\n";

$host = new HostArray(dtype: null, values: [[1, 2], [3, 4]], pinned: false);
$flat = HostArray::fromFlatArray(dtype: null, shape: [2, 2], values: [1, 2, 3, 4]);
check($host->toArray() === $flat->toArray(), 'Named constructor and factory');
$copy = HostArray::fromBuffer(pinned: false, dtype: null, shape: [2, 2], bytes: $flat->toBuffer());
check($copy->toArray() === $flat->toArray(), 'Named buffer arguments');
check(HostArray::zerosLike(dtype: null, prototype: $host)->sum()->item() === 0.0, 'Named defaults');
check($host->transpose(axis: [1, 0])->get(indices: [0, 1]) === 3.0, 'Named view and indexing');
check($host->add(other: 1)->sum(axis: 0)->toArray() === [6.0, 8.0], 'Named arithmetic/reduction');
check($host->slice(0, 0)->at() === 1.0, 'Zero-required variadic scalar indexing');
check($host->slice(0, 0)->item() === 1.0, 'Exact scalar result');
try {
    $host->add(unexpected: 1);
    throw new RuntimeException('Unknown named argument accepted');
} catch (Error $expected) {
    check(str_contains($expected->getMessage(), 'Unknown named parameter'), 'Unknown name diagnostic');
}
echo "named arguments, nullable defaults and scalar variadics execute as declared\n";
?>
--EXPECT--
signature, flags, interface, presence and named-parameter drift detected
named arguments, nullable defaults and scalar variadics execute as declared
