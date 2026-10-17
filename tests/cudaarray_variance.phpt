--TEST--
CudaArray variance and standard deviation preserve dtype, layouts, autograd and Fusion
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;

$values = CudaArray::fromFlatArray([1.0, 2.0, 3.0, 4.0], [4], 'float64');
printf("%.6f %.6f\n", $values->var()->item(), $values->std()->item());
printf("%.6f %.6f\n", $values->var(null, 1)->item(), $values->std(null, 1)->item());

$view = (new CudaArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))->transpose();
printf("%.6f\n", $view->var()->item());
$axisVariance = $view->var(0)->toArray();
printf("%.6f %.6f\n", $axisVariance[0], $axisVariance[1]);
$axisVariance = $view->var(1)->toArray();
printf("%.6f %.6f %.6f\n", $axisVariance[0], $axisVariance[1], $axisVariance[2]);

$long = CudaArray::ones([4096]);
printf("%.6f %.6f %.6f %.6f\n",
    $long->var()->item(), $long->std()->item(),
    $long->var(0)->item(), $long->std(0)->item());

$integers = CudaArray::fromFlatArray([1, 2, 3], [3], 'int32');
printf("%.6f %s\n", $integers->var()->item(), $integers->var()->dtype());
$largeView = CudaArray::ones([64, 128])->transpose();
$axisVariance = $largeView->var(0)->toArray();
printf("%.6f %.6f\n", $largeView->var()->item(), $axisVariance[0]);

$nonFinite = CudaArray::fromFlatArray([1.0, INF], [2], 'float64');
var_dump(is_nan($nonFinite->var()->item()), is_nan($nonFinite->std()->item()));
$empty = CudaArray::zeros([0, 3]);
echo json_encode($empty->var(1)->getShape()), "\n";
function fails(callable $callback): void {
    try { $callback(); echo "MISSED\n"; }
    catch (Throwable $error) { echo get_class($error), "\n"; }
}
fails(fn() => $empty->var());
fails(fn() => $empty->std());

$trackedVariance = (new CudaArray([1.0, 2.0, 3.0]))->requiresGrad();
$trackedVariance->var()->backward();
$gradient = $trackedVariance->grad()->toArray();
printf("%.6f %.6f %.6f\n", $gradient[0], $gradient[1], $gradient[2]);
$trackedStd = (new CudaArray([1.0, 2.0, 3.0]))->requiresGrad();
$trackedStd->std()->backward();
$gradient = $trackedStd->grad()->toArray();
printf("%.6f %.6f %.6f\n", $gradient[0], $gradient[1], $gradient[2]);

$graph = Fusion::compile(fn($x) => $x->var(), [$values]);
printf("%.6f\n", $graph->run($values)->item());
?>
--EXPECT--
1.250000 1.118034
1.666667 1.290994
2.916667
0.666667 0.666667
2.250000 2.250000 2.250000
0.000000 0.000000 0.000000 0.000000
0.666667 float64
0.000000 0.000000
bool(true)
bool(true)
[0]
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
-0.666667 0.000000 0.666667
-0.408248 0.000000 0.408248
1.250000
