--TEST--
CudaArray softmax is stable across axes, layouts, dtypes, autograd and Fusion
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;

function rejected(callable $callback): void {
    try { $callback(); echo "MISSED\n"; }
    catch (Throwable $error) { echo get_class($error), "\n"; }
}

$logits = CudaArray::fromFlatArray([1000.0, 1001.0, 1002.0], [3]);
$probabilities = $logits->softmax();
foreach ($probabilities->toArray() as $value) printf("%.6f ", $value);
echo "\n";
var_dump($probabilities->toArray() === $logits->softmax(-1)->toArray());

$matrix = CudaArray::fromFlatArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
$axisZero = $matrix->softmax(0)->toArray();
printf("%.6f %.6f %.6f\n", $axisZero[0][0], $axisZero[0][1], $axisZero[0][2]);
printf("%.6f %.6f %.6f\n", $axisZero[1][0], $axisZero[1][1], $axisZero[1][2]);
$transposed = $matrix->transpose()->softmax(-2)->toArray();
printf("%.6f %.6f %.6f\n", $transposed[0][0], $transposed[1][0], $transposed[2][0]);

$integers = CudaArray::fromFlatArray([0, 1, 2], [3], 'int32');
$integerProbabilities = $integers->softmax();
printf("%s %.6f\n", $integerProbabilities->dtype(), $integerProbabilities->toArray()[2]);
$doubleProbabilities = CudaArray::fromFlatArray([0.0, 1.0], [2], 'float64')->softmax();
echo $doubleProbabilities->dtype(), "\n";

$tracked = CudaArray::fromFlatArray([1.0, 2.0, 3.0], [3])->requiresGrad();
$weights = CudaArray::fromFlatArray([1.0, 2.0, 3.0], [3]);
($tracked->softmax() * $weights)->sum()->backward();
$gradient = $tracked->grad()->toArray();
printf("%.6f %.6f %.6f\n", $gradient[0], $gradient[1], $gradient[2]);

$fused = Fusion::run(fn() => $logits->softmax());
var_dump($fused->toArray() === $probabilities->toArray());
$compiled = Fusion::compile(fn($x) => $x->softmax(), [$logits]);
var_dump($compiled->run($logits)->toArray() === $probabilities->toArray());
$asyncGraph = Fusion::compile(fn($x) => $x->softmax(), [$logits], true);
var_dump(
    $asyncGraph->getStats()['cudaGraphCompatible'] === true &&
    $asyncGraph->runAsync($logits)->wait()->toArray() === $probabilities->toArray()
);
$emptyRows = CudaArray::zeros([0, 3]);
echo json_encode($emptyRows->softmax()->getShape()), "\n";
rejected(fn() => $matrix->softmax(2));
rejected(fn() => CudaArray::zeros([2, 0])->softmax());
?>
--EXPECT--
0.090031 0.244728 0.665241 
bool(true)
0.047426 0.047426 0.047426
0.952574 0.952574 0.952574
0.090031 0.244728 0.665241
float64 0.665241
float64
-0.141817 -0.140770 0.282587
bool(true)
bool(true)
bool(true)
[0,3]
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
