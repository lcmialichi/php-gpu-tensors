--TEST--
CUDA Graph updates kernel pointers for multi-segment replay and preserves retained outputs
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;
function check(bool $ok): void { if (!$ok) throw new RuntimeException('check failed'); }
$a = new CudaArray([1, 2, 3]);
$graph = Fusion::compile(function ($x) {
    for ($i = 0; $i < 160; $i++) $x = $x + 1;
    return $x;
}, [$a], cudaGraph: true);
check($graph->getStats()['backend'] === 'cuda-graph');
check($graph->getStats()['fusedKernels'] > 2);
$first = $graph->run($a);
$b = new CudaArray([10, 20, 30]);
$second = $graph->run($b);
check($first->toArray() === [161.0, 162.0, 163.0]);
check($second->toArray() === [170.0, 180.0, 190.0]);
check($graph->getStats()['graphLaunches'] === 2);
check($graph->getStats()['bufferReuses'] > 0);
echo "graph pointers updated and scratch reused\n";
$pending = $graph->runAsync($a);
try { $graph->runAsync($b); echo "MISSED\n"; }
catch (Cuda\RuntimeException $error) { echo "overlapping graph replay rejected\n"; }
check($pending->wait()->toArray() === $first->toArray());
for ($i = 0; $i < 25; $i++) {
    check($graph->run($b)->toArray() === [170.0, 180.0, 190.0]);
}
check($first->toArray() === [161.0, 162.0, 163.0]);
echo "retained outputs stable\n";
$grouped = Fusion::compile(fn($x) => [$x + 1, $x - 1], [$a], cudaGraph: true);
check($grouped->getStats()['fusedKernels'] === 1);
$outputs = $grouped->runAsync($a)->wait();
check($outputs[0]->toArray() === [2.0, 3.0, 4.0]);
check($outputs[1]->toArray() === [0.0, 1.0, 2.0]);
echo "graph multi-output\n";
$matrixA = CudaArray::ones([64, 64]);
$matrixB = CudaArray::ones([64, 64]);
$matmulGraph = Fusion::compile(fn($x, $y) => ($x->matmul($y) + 1) * 2,
    [$matrixA, $matrixB], cudaGraph: true);
check($matmulGraph->getStats()['backend'] === 'cuda-graph');
check($matmulGraph->getStats()['cudaGraphCompatible']);
$matrixFirstResult = $matmulGraph->run($matrixA, $matrixB);
$matrixSecondResult = $matmulGraph->run($matrixA * 2, $matrixB * 3);
$matrixFirst = $matrixFirstResult->toArray();
$matrixSecond = $matrixSecondResult->toArray();
check($matrixFirst[0][0] === 130.0 && $matrixFirst[63][63] === 130.0);
check($matrixSecond[0][0] === 770.0 && $matrixSecond[63][63] === 770.0);
echo "graph native matmul and pointer updates\n";
$vector = CudaArray::ones([8192]);
$reductionGraph = Fusion::compile(fn($x) => $x->sum() + 1, [$vector], cudaGraph: true);
check($reductionGraph->getStats()['backend'] === 'cuda-graph');
check($reductionGraph->run($vector)->toArray() === [8193.0]);
$twice = $vector * 2;
check($reductionGraph->run($twice)->toArray() === [16385.0]);
echo "graph native CUB reduction and pointer updates\n";
$pending = $reductionGraph->runAsync($vector);
check($pending->wait()->toArray() === [8193.0]);
echo "graph native async reduction\n";
?>
--EXPECT--
graph pointers updated and scratch reused
overlapping graph replay rejected
retained outputs stable
graph multi-output
graph native matmul and pointer updates
graph native CUB reduction and pointer updates
graph native async reduction
