--TEST--
Backend dispatch: large CUB reductions, coalesced axes, padded GEMM and concurrent Fusion streams
--SKIPIF--
<?php
if (!extension_loaded('cuda') || cuda_get_device_count() < 1) die('skip CUDA GPU required');
?>
--FILE--
<?php
use Cuda\CudaArray as A;
use Cuda\Fusion;
function check(bool $ok) { if (!$ok) throw new RuntimeException('check failed'); }
$before = cuda_get_backend_info();
$x = A::ones([100000]);
check($x->sum()->toArray() === [100000.0]);
check($x->mean()->toArray() === [1.0]);
check($x->prod()->toArray() === [1.0]);
check(A::ones([8192], 'int64')->sum()->toArray() === [8192]);
check(A::full([8192], -4000000000, 'int64')->max()->toArray() === [-4000000000]);
check(A::full([8192], 4000000000, 'int64')->min()->toArray() === [4000000000]);
$values = array_fill(0, 8192, 1.0);
$values[3000] = $values[6000] = 10.0;
$values[1000] = $values[7000] = -10.0;
$values[500] = NAN;
$indices = A::fromFlatArray($values, [8192]);
check($indices->argMax()->toArray() === [3000]);
check($indices->argMin()->toArray() === [1000]);
check(A::full([8192], NAN)->argMax()->toArray() === [0]);
check(A::full([8192], -INF)->argMax()->toArray() === [0]);
foreach (['float32', 'float64', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'bool'] as $dtype) {
    $typed = A::fromFlatArray(array_fill(0, 4096, 1), [4096], $dtype);
    $sum = $dtype === 'bool' ? 1 : (in_array($dtype, ['int8', 'uint8'], true) ? 0 : 4096);
    check((float)$typed->sum()->toArray()[0] === (float)$sum);
    check((float)$typed->mean()->toArray()[0] === 1.0);
    check((float)$typed->prod()->toArray()[0] === 1.0);
    check((float)$typed->min()->toArray()[0] === 1.0);
    check((float)$typed->max()->toArray()[0] === 1.0);
    check($typed->argMin()->toArray() === [0] && $typed->argMax()->toArray() === [0]);
}
$axis = A::fromFlatArray(range(1, 4096), [64, 64]);
$column = $axis->sum(0)->toArray();
check($column[0] === 129088.0 && $column[63] === 133120.0);
$column = $axis->slice('::2, ::2')->sum(0)->toArray();
check($column[0] === 63520.0 && $column[31] === 65504.0);
check(cuda_get_backend_info()['cubReductionCalls'] >= $before['cubReductionCalls'] + 9);
check(cuda_get_backend_info()['coalescedReductionCalls'] > $before['coalescedReductionCalls']);
echo "reductions and first-index ties passed\n";

$flat = array_map(fn($i) => (float)(($i * 7) % 13 - 6), range(0, 64 * 130 - 1));
$padded = A::fromFlatArray($flat, [64, 130]);
$a = $padded->slice(':, 1:129');
$b = $a->transpose();
$product = $a->matmul($b)->toArray();
foreach ([[0, 0], [0, 63], [37, 15], [63, 63]] as [$r, $c]) {
    $expected = 0.0;
    for ($k = 1; $k < 129; $k++) $expected += $flat[$r * 130 + $k] * $flat[$c * 130 + $k];
    check($product[$r][$c] === $expected);
}
$info = cuda_get_backend_info();
check($info['cublas'] ? in_array($info['lastMatmul'], $info['cublasLt'] ? ['cublasLt', 'cublas'] : ['cublas'], true)
                     : $info['lastMatmul'] === 'builtin');
$aligned = A::ones([64, 128]);
check($aligned->matmul($aligned->transpose())->toArray()[0][0] === 128.0);
check($a->matmul($b)->toArray() === $product);
$dot = A::ones([1, 100000])->matmul(A::ones([100000, 1]));
check($dot->toArray() === [[100000.0]]);
if ($info['cublas']) check(cuda_get_backend_info()['lastMatmul'] === 'cublasDot');
$skinny = A::ones([16, 1024])->matmul(A::ones([1024, 16]));
check($skinny->toArray() === array_fill(0, 16, array_fill(0, 16, 1024.0)));
$skinnyBackend = cuda_get_backend_info()['lastMatmul'];
check($skinnyBackend === ($info['cublas'] ? 'cublas' : 'builtin'));
echo "padded GEMM and dot dispatch passed\n";

$plan = Fusion::compile(fn($a) => ($a + 1)->sum(), [$x]);
$first = $plan->runAsync($x);
$second = $plan->runAsync(A::full([100000], 2));
check($first->wait()->toArray() === [200000.0]);
check($second->wait()->toArray() === [300000.0]);
unset($first, $second);
for ($i = 0; $i < 70; $i++) {
    $execution = $plan->runAsync($x);
    check($execution->wait()->toArray() === [200000.0]);
    unset($execution);
}
echo "concurrent CUB workspaces passed\n";
?>
--EXPECT--
reductions and first-index ties passed
padded GEMM and dot dispatch passed
concurrent CUB workspaces passed
