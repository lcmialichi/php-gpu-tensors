--TEST--
CudaArray layerNorm supports explicit axes, affine parameters, autograd and Fusion
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

$input = CudaArray::fromFlatArray([1.0, 2.0, 3.0, 4.0], [2, 2]);
$normalized = $input->layerNorm([1])->toArray();
printf("%.5f %.5f\n", $normalized[0][0], $normalized[0][1]);
$constant = CudaArray::ones([2, 3])->layerNorm([1])->toArray();
var_dump($constant === [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);

$scale = CudaArray::fromFlatArray([2.0, 3.0], [2]);
$bias = CudaArray::fromFlatArray([0.5, -0.5], [2]);
$affine = $input->layerNorm([1], 0.00001, $scale, $bias)->toArray();
printf("%.5f %.5f\n", $affine[0][0], $affine[0][1]);

$permutedAxes = CudaArray::fromFlatArray([1.0, 2.0, 3.0, 4.0], [2, 1, 2]);
$permutedScale = CudaArray::fromFlatArray([1.0, 2.0, 3.0, 4.0], [2, 2]);
$permuted = $permutedAxes->layerNorm([2, 0], 0.00001, $permutedScale)->toArray();
printf("%.5f\n", $permuted[1][0][0]);
$strided = CudaArray::fromFlatArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    ->transpose()->layerNorm([0])->toArray();
printf("%.5f\n", $strided[0][0]);

$integerOutput = CudaArray::fromFlatArray([1, 2, 3, 4], [2, 2], 'int32')->layerNorm([1]);
echo $integerOutput->dtype(), "\n";
$doubleOutput = CudaArray::fromFlatArray([1.0, 2.0], [2], 'float64')->layerNorm([0]);
echo $doubleOutput->dtype(), "\n";
$promotedOutput = $input->layerNorm([1], 0.00001, CudaArray::ones([2], 'float64'));
echo $promotedOutput->dtype(), "\n";

$fused = Fusion::run(fn() => $input->layerNorm([1], 0.00001, $scale, $bias));
$fusedValues = $fused->toArray();
var_dump(abs($fusedValues[0][0] - $affine[0][0]) < 0.00001);
$compiled = Fusion::compile(
    fn($x, $s, $b) => $x->layerNorm([1], 0.00001, $s, $b),
    [$input, $scale, $bias]
);
$compiledValues = $compiled->run($input, $scale, $bias)->toArray();
var_dump(abs($compiledValues[0][0] - $affine[0][0]) < 0.00001);
$asyncGraph = Fusion::compile(
    fn($x, $s, $b) => $x->layerNorm([1], 0.00001, $s, $b),
    [$input, $scale, $bias],
    true
);
$asyncValues = $asyncGraph->runAsync($input, $scale, $bias)->wait()->toArray();
var_dump(
    $asyncGraph->getStats()['cudaGraphCompatible'] === true &&
    abs($asyncValues[0][0] - $affine[0][0]) < 0.00001
);

$tracked = CudaArray::fromFlatArray([1.0, 2.0, 4.0, 2.0, 5.0, 9.0], [2, 3])->requiresGrad();
$trackedScale = CudaArray::fromFlatArray([1.0, 2.0, 3.0], [3])->requiresGrad();
$trackedBias = CudaArray::zeros([3])->requiresGrad();
$tracked->layerNorm([1], 0.00001, $trackedScale, $trackedBias)->sum()->backward();
var_dump(count($tracked->grad()->toArray()) === 2);
var_dump(count($trackedScale->grad()->toArray()) === 3);
var_dump($trackedBias->grad()->toArray() === [2.0, 2.0, 2.0]);

rejected(fn() => $input->layerNorm([1, -1]));
rejected(fn() => $input->layerNorm([2]));
rejected(fn() => $input->layerNorm([1], 0.0));
rejected(fn() => $input->layerNorm([1], NAN));
rejected(fn() => $input->layerNorm([1], 0.00001, CudaArray::ones([3])));
rejected(fn() => $input->layerNorm([1], 0.00001, CudaArray::ones([2], 'int32')));
rejected(fn() => CudaArray::zeros([2, 0])->layerNorm([1]));
?>
--EXPECT--
-0.99998 0.99998
bool(true)
-1.49996 2.49994
0.89442
-1.22474
float64
float64
float64
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
