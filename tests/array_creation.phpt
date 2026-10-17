--TEST--
CudaArray factories, ranges and reproducible uniform/normal random creation
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray as A;

function check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}

check(A::eye(3)->toArray() === [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    'square eye mismatch');
check(A::eye(2, 4, 1, 'int32')->toArray() === [[0, 1, 0, 0], [0, 0, 1, 0]],
    'rectangular offset eye mismatch');
check(A::eye(3, 2, -1, 'float16')->toArray() === [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    'negative diagonal eye mismatch');
check(A::identity(2, 'int16')->toArray() === [[1, 0], [0, 1]],
    'identity mismatch');
check(A::arange(5)->toArray() === [0, 1, 2, 3, 4] &&
    A::arange(5)->dtype() === 'int64', 'single-argument integer arange mismatch');
check(A::arange(9007199254740993, 9007199254740996)->toArray() ===
    [9007199254740993, 9007199254740994, 9007199254740995],
    'integer arange lost precision above 2^53');
check(A::arange(-8000000000000000000, 8000000000000000000, 5000000000000000000)
    ->toArray() === [-8000000000000000000, -3000000000000000000,
        2000000000000000000, 7000000000000000000],
    'integer arange overflowed intermediate multiplication');
check(A::arange(5, -1, -2)->toArray() === [5, 3, 1],
    'descending arange mismatch');
check(A::arange(0, 3, -1)->getSize() === 0,
    'opposite-direction arange should be empty');
check(A::arange(2.5)->dtype() === 'float64' &&
    A::arange(2.5)->toArray() === [0.0, 1.0, 2.0],
    'floating-point arange inference mismatch');
check(A::linspace(0, 1, 5)->dtype() === 'float64' &&
    A::linspace(0, 1, 5)->toArray() === [0.0, 0.25, 0.5, 0.75, 1.0],
    'linspace endpoint mismatch');
check(A::linspace(0, 1, 4, false)->toArray() === [0.0, 0.25, 0.5, 0.75],
    'linspace endpoint=false mismatch');
check(A::linspace(3, 9, 1)->toArray() === [3.0] &&
    A::linspace(3, 9, 0)->getSize() === 0,
    'linspace empty/singleton behavior mismatch');
check(A::linspace(0, 2, 3, true, 'int32')->toArray() === [0, 1, 2],
    'linspace integer dtype mismatch');

$prototype = A::fromFlatArray([1.5, 2.25, 3.0], [1, 3], 'float16');
$zeros = A::zerosLike($prototype);
$ones = A::onesLike($prototype, 'float64');
$full = A::fullLike($prototype, 7.5);
check($zeros->getShape() === [1, 3] && $zeros->dtype() === 'float16' &&
    $zeros->toArray() === [[0.0, 0.0, 0.0]], 'zerosLike mismatch');
check($ones->dtype() === 'float64' && $ones->toArray() === [[1.0, 1.0, 1.0]],
    'onesLike dtype override mismatch');
check($full->dtype() === 'float16' && $full->toArray() === [[7.5, 7.5, 7.5]],
    'fullLike mismatch');

$uniform = A::rand([128], -2.0, 3.0, 'float32', 1234);
$uniformAgain = A::rand([128], -2.0, 3.0, 'float32', 1234);
$uniformOther = A::rand([128], -2.0, 3.0, 'float32', 1235);
check($uniform->toBuffer() === $uniformAgain->toBuffer(), 'uniform seed is not reproducible');
check($uniform->toBuffer() !== $uniformOther->toBuffer(), 'different uniform seeds matched');
foreach ($uniform->toArray() as $value)
    check($value >= -2.0 && $value <= 3.0, 'uniform value outside requested range');
$unseededA = A::rand([128]);
$unseededB = A::rand([128]);
check($unseededA->toBuffer() !== $unseededB->toBuffer(),
    'unseeded uniform calls should generate new values');

$normal = A::randn([8192], 'float64', 4321);
$normalAgain = A::randn([8192], 'float64', 4321);
check($normal->toBuffer() === $normalAgain->toBuffer(), 'normal seed is not reproducible');
$samples = $normal->toArray();
$mean = array_sum($samples) / count($samples);
$variance = 0.0;
foreach ($samples as $sample) $variance += ($sample - $mean) ** 2;
$variance /= count($samples);
check(abs($mean) < 0.05 && $variance > 0.9 && $variance < 1.1,
    'normal random distribution is outside tolerance');
$halfNormal = A::randn([8], 'float16', 9);
check($halfNormal->dtype() === 'float16', 'randn low-precision dtype mismatch');
check(A::randn([0])->getSize() === 0, 'empty randn mismatch');
$unseededNormalA = A::randn([128]);
$unseededNormalB = A::randn([128]);
check($unseededNormalA->toBuffer() !== $unseededNormalB->toBuffer(),
    'unseeded normal calls should generate new values');

try {
    A::randn([4], 'int32');
    throw new RuntimeException('randn accepted an integer dtype');
} catch (PhpTensors\InvalidArgumentException $error) {
}
try {
    A::rand([4], 0, 1, 'float32', -1);
    throw new RuntimeException('rand accepted a negative seed');
} catch (PhpTensors\InvalidArgumentException $error) {
}
try {
    A::eye(-1);
    throw new RuntimeException('eye accepted a negative dimension');
} catch (PhpTensors\InvalidArgumentException $error) {
}
try {
    A::arange(0, 5, 0);
    throw new RuntimeException('arange accepted a zero step');
} catch (PhpTensors\InvalidArgumentException $error) {
}
try {
    A::arange(0, 1, 1, 'uint8');
    A::arange(-1, 1, 1, 'uint8');
    throw new RuntimeException('arange accepted values outside dtype range');
} catch (PhpTensors\InvalidArgumentException $error) {
}
try {
    A::linspace(0, 1, -1);
    throw new RuntimeException('linspace accepted a negative count');
} catch (PhpTensors\InvalidArgumentException $error) {
}
try {
    A::zeros([2, 'invalid']);
    throw new RuntimeException('zeros accepted an invalid shape');
} catch (PhpTensors\InvalidArgumentException $error) {
}

echo "CudaArray creation APIs checked\n";
?>
--EXPECT--
CudaArray creation APIs checked
