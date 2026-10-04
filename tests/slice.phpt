--TEST--
Python-style slice expressions and selectors preserve shared layouts, empty tensors and Fusion
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
function check(bool $ok, string $label): void {
    if (!$ok) throw new RuntimeException($label);
}
function values(CudaArray $x, array $expected, string $label): void {
    check($x->toArray() === $expected, $label);
    check($x->toHost()->toArray() === $expected, "$label host");
}
$a = new CudaArray([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]);
values($a->slice(' 1:3 , 1:4:2 '), [[5.0, 7.0], [9.0, 11.0]], 'expression');
values($a->slice([1, 3], [1, 4, 2]), [[5.0, 7.0], [9.0, 11.0]], 'arrays');
values($a->slice(null, '::2'), [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]], 'variadic');
values($a->slice('-2:, -3:-1'), [[5.0, 6.0], [9.0, 10.0]], 'negative bounds');
values($a->slice('-99:99'), $a->toArray(), 'clipping');
values($a->slice(), $a->toArray(), 'all');
values($a->slice(-1), [8.0, 9.0, 10.0, 11.0], 'index drops axis');
values($a->slice('1:2'), [[4.0, 5.0, 6.0, 7.0]], 'range keeps axis');
$scalar = $a->slice('1, 2');
check($scalar->getShape() === [] && $scalar->getNdims() === 0 && $scalar->getStrides() === [], 'scalar metadata');
values($scalar, [6.0], 'scalar');
values($scalar + 1, [7.0], 'scalar arithmetic');
$nested = $a->slice('1:, 1:')->slice(':, 1:')->transpose();
values($nested, [[6.0, 10.0], [7.0, 11.0]], 'nested offsets');
values($nested + $nested, [[12.0, 20.0], [14.0, 22.0]], 'broadcast offsets');
values($nested->sum(1), [16.0, 18.0], 'reduction offsets');
$view = $a->slice('1:2, 1:3');
$view[0][0] = 50;
check($a->toArray()[1][1] === 50.0, 'shared mutation');
$a[1][1] = 5;
unset($view);
$stepped = $a->slice('::2, ::2');
check($stepped[1][1] === 10.0, 'strided dimension read');
$stepped[1][1] = 100;
check($a->toArray()[2][2] === 100.0, 'strided dimension write');
$a[2][2] = 10;
values($stepped([1, 1]), [[8.0, 10.0]], 'legacy invoke after stepped slice');
$stepped[1] = new CudaArray([80, 100]);
check($a->toArray()[2] === [80.0, 9.0, 100.0, 11.0], 'strided row assignment');
$stepped[1] = $stepped[0];
check($a->toArray()[2] === [0.0, 9.0, 2.0, 11.0], 'strided source assignment');
$a[2] = new CudaArray([8, 9, 10, 11]);
unset($stepped);
values($a([1, 2], [1, 2]), [[5.0, 6.0], [9.0, 10.0]], 'legacy inclusive invoke');
echo "selection, offsets, lifetime and compatibility checked\n";

$empty = $a->slice('2:2');
check($empty->getShape() === [0, 4] && $empty->getSize() === 0, 'empty metadata');
values($empty, [], 'empty outer');
$inner = $a->slice(':, 4:4');
values($inner, [[], [], []], 'empty inner');
values($a->slice('99:'), [], 'clipped empty');
values($a->slice('2:1'), [], 'reversed bounds');
values($inner->slice(), [[], [], []], 'nested empty');
values(($empty + 1)->sqrt(), [], 'empty unary scalar');
values($empty + $a->slice('0:1'), [], 'empty broadcast');
values($inner->astype('float64'), [[], [], []], 'empty cast');
values(CudaArray::where($inner, $inner, $inner), [[], [], []], 'empty where');
values($inner->reshape([0, 3])->transpose(), [[], [], []], 'empty reshape transpose');
values($empty->sum(1), [], 'empty output reduction');
values($inner->sum(1), [0.0, 0.0, 0.0], 'empty sum identity');
values($inner->prod(1), [1.0, 1.0, 1.0], 'empty product identity');
check(is_nan($inner->mean()->toArray()[0]), 'empty mean');
values($empty->sum(), [0.0], 'empty global sum');
values($empty->flatten(), [], 'empty flatten');
values($empty->toHost()->toGpu(), [], 'empty host roundtrip');
values(CudaArray::fromBuffer('', [3, 0]), [[], [], []], 'empty import');
values(unserialize(serialize($empty + 1)), [], 'empty serialization');
values(unserialize(serialize($scalar + 1)), [7.0], 'scalar serialization');
values($empty->concat([$a]), $a->toArray(), 'empty concat input');
values($inner->concat([$inner], 1), [[], [], []], 'empty concat result');
values($empty->matmul($a->transpose()), [], 'empty matmul output');
values($inner->matmul($a->transpose()->slice('4:4')), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 'empty contraction');
echo "empty tensors checked\n";

foreach (['float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'] as $dtype) {
    $typed = new CudaArray([[1, 2], [3, 4]], $dtype);
    $expected = $dtype === 'bool' ? [true] : (str_starts_with($dtype, 'float') ? [4.0] : [4]);
    values($typed->slice('1, 1'), $expected, "scalar $dtype");
    check($typed->slice('1, ::2')[0] === ($dtype === 'bool' ? 1.0 : 3.0), "dimension $dtype");
    $write = $typed->slice('1:2, 1:2');
    $write[0][0] = 1;
    values($typed->slice('1, 1'), $dtype === 'bool' ? [true] : (str_starts_with($dtype, 'float') ? [1.0] : [1]), "write $dtype");
    values($typed->slice('1:1'), [], "empty $dtype");
}
$bounds = [-99, -3, -1, 0, 1, 3, 99];
foreach ($bounds as $start) foreach ($bounds as $stop) foreach ([1, 2, 5] as $step) {
    $expected = [];
    $first = min(3, max(0, $start < 0 ? $start + 3 : $start));
    $last = min(3, max(0, $stop < 0 ? $stop + 3 : $stop));
    for ($i = $first; $i < $last; $i += $step) $expected[] = $a->toArray()[$i];
    check($a->slice([$start, $stop, $step])->toArray() === $expected, 'range reference');
}
echo "all dtypes and reference ranges checked\n";
$precise = new CudaArray([[0]], 'int64');
$preciseView = $precise->slice(':, :');
$preciseView[0][0] = 9007199254740993;
values($precise->slice('0, 0'), [9007199254740993], 'int64 assignment precision');
$precise = new CudaArray([[0]], 'float64');
$preciseView = $precise->slice(':, :');
$preciseView[0][0] = 1.000000000000001;
values($precise->slice('0, 0'), [1.000000000000001], 'float64 assignment precision');

values($a->slice([PHP_INT_MIN, PHP_INT_MAX, null]), $a->toArray(), 'integer bound limits');
$expression = '1:, ::2';
$arguments = [&$expression];
values($a->slice(...$arguments), [[4.0, 6.0], [8.0, 10.0]], 'reference expression');
foreach ([':', '1:, ::2', '1, 2', '2:2', ':, 4:4'] as $expression) {
    $graph = Fusion::compile(fn($x) => ($x + 1)->slice($expression) * 2, [$a]);
    $expected = (($a + 1)->slice($expression) * 2)->toArray();
    values($graph->run($a), $expected, "compiled $expression");
    values($graph->runAsync($a)->wait(), $expected, "async $expression");
    values(Fusion::run(fn() => ($a + 1)->slice($expression) * 2), $expected, "scoped $expression");
}
$graph = Fusion::compile(fn($x) => $x->slice('1:, 1:')->sum(0), [$a]);
check($graph->getStats()['viewAliases'] === 1, 'native slice alias');
values($graph->run($a), [14.0, 16.0, 18.0], 'native alias');
values($graph->runAsync($a)->wait(), [14.0, 16.0, 18.0], 'async alias');
$other = $a + 10;
$first = $graph->runAsync($a);
$second = $graph->runAsync($other);
values($second->wait(), [34.0, 36.0, 38.0], 'replaced inputs');
$retainedOutput = $first->wait();
values($graph->run($other), [34.0, 36.0, 38.0], 'replaced sync inputs');
values($retainedOutput, [14.0, 16.0, 18.0], 'independent outputs');
$graph = Fusion::compile(fn($x) => $x->slice('1:, 1:')->slice(':, ::2')->transpose()->matmul($x->slice('1:, 1:')->slice(':, ::2')), [$a]);
$expected = $a->slice('1:, 1:')->slice(':, ::2')->transpose()->matmul($a->slice('1:, 1:')->slice(':, ::2'))->toArray();
values($graph->runAsync($a)->wait(), $expected, 'nested matmul layouts');
$graph = Fusion::compile(fn($x) => $x->slice(':, 4:4')->sum(1), [$a]);
values($graph->runAsync($a)->wait(), [0.0, 0.0, 0.0], 'empty native alias');
$graph = Fusion::compile(fn($x) => $x->slice('2:2')->sum(), [$a]);
values($graph->run($a), [0.0], 'empty global fused');
$graph = Fusion::compile(fn($x) => $x->slice('2:2') + 1, [$a], cudaGraph: true);
values($graph->run($a), [], 'empty cuda graph');
values($graph->run($a), [], 'empty cuda graph replay');
$graph = Fusion::compile(fn($x) => $x->slice('1, 2') + 1, [$a], cudaGraph: true);
values($graph->runAsync($a)->wait(), [7.0], 'scalar cuda graph');
echo "Fusion scoped, compiled, async and CUDA Graph checked\n";

foreach (['', '1:2:3:4', '1,,2', 'foo', '1.5:2', '::0', '::-1', '1:2, :, :',
          "1:\0", '9999999999999999999999999999', '3', '-4'] as $expression) {
    try { $a->slice($expression); throw new RuntimeException("accepted $expression"); }
    catch (Cuda\InvalidArgumentException $error) {}
}
foreach ([[1], [1, 2, 3, 4], ['start' => 1, 'stop' => 2], [1.0, 2]] as $range) {
    try { $a->slice($range); throw new RuntimeException('accepted invalid range'); }
    catch (Cuda\InvalidArgumentException $error) {}
}
try { $empty->slice(0); throw new RuntimeException('indexed empty axis'); }
catch (Cuda\InvalidArgumentException $error) {}
try { $inner[0] = 1; throw new RuntimeException('wrote empty storage'); }
catch (Cuda\InvalidArgumentException $error) {}
foreach (['min', 'max', 'argmin', 'argmax'] as $method) {
    try { $inner->$method(1); throw new RuntimeException('accepted empty extrema'); }
    catch (Cuda\InvalidArgumentException $error) {}
    try { Fusion::compile(fn($x) => $x->slice(':, 4:4')->$method(1), [$a]); throw new RuntimeException('captured empty extrema'); }
    catch (Cuda\InvalidArgumentException $error) {}
}
$retained = $a->slice('1:, ::2');
unset($a);
values($retained, [[4.0, 6.0], [8.0, 10.0]], 'retained base');
echo "invalid selectors and empty extrema rejected\n";
?>
--EXPECT--
selection, offsets, lifetime and compatibility checked
empty tensors checked
all dtypes and reference ranges checked
Fusion scoped, compiled, async and CUDA Graph checked
invalid selectors and empty extrema rejected
