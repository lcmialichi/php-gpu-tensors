--TEST--
CudaArray diagonal, repeat, tile, flip and roll transforms and gradients
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;
use PhpTensors\Fusion;

function check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}

$matrix = A::fromFlatArray([1, 2, 3, 4, 5, 6], [2, 3]);
check($matrix->diagonal()->toArray() === [1.0, 5.0],
    'main diagonal mismatch');
check($matrix->diagonal(1)->toArray() === [2.0, 6.0],
    'positive diagonal offset mismatch');
check($matrix->diagonal(-1)->toArray() === [4.0],
    'negative diagonal offset mismatch');
check($matrix->diagonal(0, 1, 0)->toArray() === [1.0, 5.0],
    'diagonal axis order mismatch');
$volume = A::fromFlatArray(range(0, 23), [2, 3, 4]);
check($volume->diagonal(1, 0, 2)->getShape() === [3, 2] &&
    $volume->diagonal(1, 0, 2)->toArray() === [[1.0, 14.0], [5.0, 18.0], [9.0, 22.0]],
    'N-D diagonal mismatch');

$square = A::fromFlatArray([1, 2, 3, 4], [2, 2]);
check($square->repeat(2)->toArray() === [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    'flattened repeat mismatch');
check($square->repeat(2, 0)->toArray() ===
    [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]],
    'axis repeat mismatch');
check($square->tile([2, 1])->toArray() ===
    [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]],
    'tile repetitions mismatch');
check($square->tile(2)->toArray() ===
    [[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0],
     [1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]],
    'scalar tile mismatch');

check($square->flip()->toArray() === [[4.0, 3.0], [2.0, 1.0]],
    'all-axis flip mismatch');
check($square->flip(0)->toArray() === [[3.0, 4.0], [1.0, 2.0]],
    'single-axis flip mismatch');
check($square->roll(1)->toArray() === [[4.0, 1.0], [2.0, 3.0]],
    'flattened roll mismatch');
check($square->roll(1, 1)->toArray() === [[2.0, 1.0], [4.0, 3.0]],
    'axis roll mismatch');
check($square->roll([1, -1], [0, 1])->toArray() === [[4.0, 3.0], [2.0, 1.0]],
    'multi-axis roll mismatch');
$strided = $square->transpose();
check($strided->diagonal()->toArray() === [1.0, 4.0] &&
    $strided->repeat(2, 1)->toArray() ===
        [[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0]] &&
    $strided->flip(1)->toArray() === [[3.0, 1.0], [4.0, 2.0]] &&
    $strided->roll(1, 0)->toArray() === [[2.0, 4.0], [1.0, 3.0]],
    'strided transform mismatch');
check(A::fromFlatArray([1.5, 2.5], [2], 'float16')
    ->flip()->toArray() === [2.5, 1.5],
    'low-precision transform mismatch');
check(A::zeros([0, 3])->flip()->getSize() === 0 &&
    A::zeros([0, 3])->roll(2, 1)->getSize() === 0,
    'empty transform mismatch');

$diagonalInput = A::zeros([2, 3])->requiresGrad();
$diagonalInput->diagonal()->sum()->backward();
check($diagonalInput->grad()->toArray() === [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    'diagonal gradient mismatch');
$repeatInput = A::fromFlatArray([1, 2], [2])->requiresGrad();
$repeatInput->repeat(3)->sum()->backward();
check($repeatInput->grad()->toArray() === [3.0, 3.0],
    'repeat gradient mismatch');
$tileInput = A::fromFlatArray([1, 2, 3, 4], [2, 2])->requiresGrad();
$tileInput->tile([2, 1])->sum()->backward();
check($tileInput->grad()->toArray() === [[2.0, 2.0], [2.0, 2.0]],
    'tile gradient mismatch');
$flipInput = A::fromFlatArray([1, 2, 3, 4], [2, 2])->requiresGrad();
$weights = A::fromFlatArray([1, 2, 3, 4], [2, 2]);
$flipInput->flip(0)->multiply($weights)->sum()->backward();
check($flipInput->grad()->toArray() === [[3.0, 4.0], [1.0, 2.0]],
    'flip gradient mapping mismatch');
$rollInput = A::fromFlatArray([1, 2, 3, 4], [2, 2])->requiresGrad();
$rollInput->roll(1, 1)->multiply($weights)->sum()->backward();
check($rollInput->grad()->toArray() === [[2.0, 1.0], [4.0, 3.0]],
    'roll gradient mapping mismatch');

function fused_transform_gradient(A $input, callable $loss): array {
    [, $gradient] = Fusion::run(function () use ($input, $loss) {
        $value = $loss($input);
        $value->backward();
        return [$value, $input->grad()];
    });
    return $gradient->toArray();
}

$fusedDiagonalInput = A::zeros([2, 2])->requiresGrad();
check(fused_transform_gradient($fusedDiagonalInput,
    fn($x) => $x->diagonal()->sum()) === [[1.0, 0.0], [0.0, 1.0]],
    'Fusion diagonal gradient mismatch');
$fusedRepeatInput = A::fromFlatArray([1, 2], [2])->requiresGrad();
check(fused_transform_gradient($fusedRepeatInput,
    fn($x) => $x->repeat(2)->sum()) === [2.0, 2.0],
    'Fusion repeat gradient mismatch');
$fusedTileInput = A::fromFlatArray([1, 2, 3, 4], [2, 2])->requiresGrad();
check(fused_transform_gradient($fusedTileInput,
    fn($x) => $x->tile([2, 1])->sum()) === [[2.0, 2.0], [2.0, 2.0]],
    'Fusion tile gradient mismatch');
$fusedFlipInput = A::fromFlatArray([1, 2, 3, 4], [2, 2])->requiresGrad();
check(fused_transform_gradient($fusedFlipInput,
    fn($x) => $x->flip(0)->multiply($weights)->sum()) ===
        [[3.0, 4.0], [1.0, 2.0]],
    'Fusion flip gradient mismatch');
$fusedRollInput = A::fromFlatArray([1, 2, 3, 4], [2, 2])->requiresGrad();
check(fused_transform_gradient($fusedRollInput,
    fn($x) => $x->roll(1, 1)->multiply($weights)->sum()) ===
        [[2.0, 1.0], [4.0, 3.0]],
    'Fusion roll gradient mismatch');

$transformGraph = Fusion::compile(fn($x) => [
    $x->diagonal(),
    $x->repeat(2),
    $x->tile([2, 1]),
    $x->flip(),
    $x->roll(1, 1),
], [$strided]);
[$fusedDiagonal, $fusedRepeat, $fusedTile, $fusedFlip, $fusedRoll] =
    $transformGraph->run($strided);
check($fusedDiagonal->toArray() === [1.0, 4.0] &&
    $fusedRepeat->toArray() === [1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 4.0, 4.0] &&
    $fusedTile->toArray() === [[1.0, 3.0], [2.0, 4.0], [1.0, 3.0], [2.0, 4.0]] &&
    $fusedFlip->toArray() === [[4.0, 2.0], [3.0, 1.0]] &&
    $fusedRoll->toArray() === [[3.0, 1.0], [4.0, 2.0]],
    'Fusion transform results mismatch');

$chainedGraph = Fusion::compile(
    fn($x) => $x->flip(0)->roll(1, 1)->tile([1, 1]) + $x->repeat(1, 0),
    [$strided]);
$chainedExpected = $strided->flip(0)->roll(1, 1)->tile([1, 1]) +
    $strided->repeat(1, 0);
check($chainedGraph->getStats()['fusedKernels'] === 1 &&
    $chainedGraph->getStats()['intermediateBuffers'] === 0 &&
    $chainedGraph->run($strided)->toArray() === $chainedExpected->toArray(),
    'Fusion did not inline chained transforms into one kernel');
$asyncResult = $chainedGraph->runAsync($strided)->wait();
check($asyncResult->toArray() === $chainedExpected->toArray(),
    'asynchronous Fusion transform result mismatch');
$graphCapture = Fusion::compile(
    fn($x) => $x->flip(0)->roll(1, 1)->tile([1, 1]) + $x->repeat(1, 0),
    [$strided], true);
check($graphCapture->run($strided)->toArray() === $chainedExpected->toArray(),
    'CUDA Graph transform result mismatch');

$diagonalGraph = Fusion::compile(
    fn($x) => $x->diagonal() + $x->diagonal(1),
    [$square]);
check($diagonalGraph->run($square)->toArray() === [3.0, 6.0],
    'Fusion diagonal broadcasting mismatch');

foreach ([
    fn() => $square->diagonal(0, 1, 1),
    fn() => $square->repeat(-1),
    fn() => $square->tile([-1, 1]),
    fn() => $square->flip(2),
    fn() => $square->roll([1, 2], [0]),
] as $invalid) {
    try {
        $invalid();
        throw new RuntimeException('invalid transform arguments were accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
}

echo "CudaArray index transforms checked\n";
?>
--EXPECT--
CudaArray index transforms checked
