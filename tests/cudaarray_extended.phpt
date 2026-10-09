--TEST--
CudaArray shape views, elementwise bounds, logical/statistical reductions and item
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use Cuda\CudaArray;
use Cuda\Fusion;

$input = new CudaArray([[1.0, 0.0], [3.0, 4.0]]);
echo json_encode($input->maximum(0)->toArray()), "\n";
echo json_encode($input->minimum(0)->toArray()), "\n";
echo json_encode($input->clamp(0, 2)->toArray()), "\n";
echo json_encode($input->all(1)->toArray()), "\n";
echo json_encode($input->any(1)->toArray()), "\n";
printf("%.3f %.3f\n", $input->var()->item(), $input->std()->item());
echo json_encode($input->var(0)->toArray()), "\n";
echo json_encode((new CudaArray([[7.0], [8.0]]))->squeeze(1)->getShape()), "\n";
echo json_encode($input->unsqueeze(1)->getShape()), "\n";
echo json_encode((new CudaArray([[1.0, 2.0]]))->broadcastTo([2, 2])->toArray()), "\n";
echo json_encode((new CudaArray([[1.0, 2.0], [3.0, 4.0]]))
  ->transpose()->unsqueeze(0)->squeeze(0)->toArray()), "\n";
var_dump(CudaArray::full([1], 42, 'int64')->item());

$source = new CudaArray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]);
$indices = CudaArray::fromFlatArray([2, 0, 1, 1], [2, 2], 'int32');
echo json_encode($source->gather($indices, 1)->toArray()), "\n";
$base = CudaArray::zeros([2, 3]);
$updates = new CudaArray([[1.0, 2.0], [3.0, 4.0]]);
echo json_encode($base->scatterAdd($indices, $updates, 1)->toArray()), "\n";

$graph = Fusion::compile(fn($x) => $x->clamp(0, 2)->maximum(0.5)->var(), [$input]);
printf("%.3f\n", $graph->run($input)->item());
$logicalGraph = Fusion::compile(fn($x) => [$x->all(1), $x->any(1)], [$input]);
[$all, $any] = $logicalGraph->run($input);
echo json_encode($all->toArray()), "\n";
echo json_encode($any->toArray()), "\n";
$broadcastGraph = Fusion::compile(fn($x) => $x->broadcastTo([2, 2])->sum(0), [new CudaArray([1.0, 2.0])]);
echo json_encode($broadcastGraph->run(new CudaArray([1.0, 2.0]))->toArray()), "\n";

function fails(callable $callback): void {
    try { $callback(); echo "MISSED\n"; }
    catch (Throwable $error) { echo get_class($error), "\n"; }
}
$badIndices = CudaArray::fromFlatArray([3, 0], [1, 2], 'int32');
fails(fn() => $source->gather($badIndices, 1));
fails(fn() => $source->gather($indices, 3));
fails(fn() => $input->var(1, 2));
fails(fn() => Fusion::compile(fn($x) => $x->gather($indices, 1), [$source]));
fails(fn() => Fusion::compile(fn($x) => $x->scatterAdd($indices, $updates, 1), [$base]));

$tracked = (new CudaArray([-1.0, 0.5, 3.0]))->requiresGrad();
$tracked->clamp(0, 2)->sum()->backward();
echo json_encode($tracked->grad()->toArray()), "\n";
$gatherTracked = (new CudaArray([[1.0, 2.0, 3.0]]))->requiresGrad();
$gatherIndices = CudaArray::fromFlatArray([2, 0], [1, 2], 'int32');
$gatherTracked->gather($gatherIndices, 1)->sum()->backward();
echo json_encode($gatherTracked->grad()->toArray()), "\n";
$broadcastTracked = (new CudaArray([[1.0, 2.0, 3.0]]))->requiresGrad();
$broadcastTracked->broadcastTo([2, 3])->sum()->backward();
echo json_encode($broadcastTracked->grad()->toArray()), "\n";
$updateTracked = (new CudaArray([[5.0, 6.0]]))->requiresGrad();
CudaArray::zeros([1, 3])->requiresGrad()
  ->scatterAdd($gatherIndices, $updateTracked, 1)->sum()->backward();
echo json_encode($updateTracked->grad()->toArray()), "\n";
?>
--EXPECT--
[[1,0],[3,4]]
[[0,0],[0,0]]
[[1,0],[2,2]]
[false,true]
[true,true]
2.500 1.581
[1,4]
[2]
[2,1,2]
[[1,2],[1,2]]
[[1,3],[2,4]]
int(42)
[[30,10],[50,50]]
[[2,0,1],[0,7,0]]
0.422
[false,true]
[true,true]
[2,4]
Cuda\InvalidArgumentException
Cuda\InvalidArgumentException
Cuda\InvalidArgumentException
Cuda\InvalidArgumentException
Cuda\InvalidArgumentException
[0,1,0]
[[1,0,1]]
[[2,2,2]]
[[1,1]]
