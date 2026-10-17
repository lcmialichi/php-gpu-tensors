--TEST--
CudaArray matmul accelerated layouts and broadcast
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
$row = PhpTensors\CudaArray::ones([1, 100000]);
$column = PhpTensors\CudaArray::ones([100000, 1]);
var_dump($row->matmul($column)->toArray()[0][0]);

$transposed = PhpTensors\CudaArray::ones([64, 128])->transpose();
$matrix = PhpTensors\CudaArray::ones([64, 128]);
$viewResult = $transposed->matmul($matrix)->toArray();
var_dump(count($viewResult), count($viewResult[0]), $viewResult[0][0], $viewResult[127][127]);

$batch = PhpTensors\CudaArray::ones([8, 64, 64]);
$batchedResult = $batch->matmul($batch)->toArray();
var_dump(count($batchedResult), $batchedResult[0][0][0], $batchedResult[7][63][63]);

$broadcast = PhpTensors\CudaArray::ones([1, 64, 64]);
$broadcastResult = $broadcast->matmul($batch)->toArray();
var_dump(count($broadcastResult), $broadcastResult[0][0][0], $broadcastResult[7][63][63]);

$multi = PhpTensors\CudaArray::ones([2, 4, 64, 64]);
$multiResult = $multi->matmul($multi)->toArray();
var_dump(count($multiResult), count($multiResult[0]), $multiResult[1][3][63][63]);

$leftValues = $rightValues = [];
for ($row = 0; $row < 64; $row++) {
	for ($col = 0; $col < 128; $col++) {
		$leftValues[] = (float)(($row + $col * 2) % 7);
		$rightValues[] = (float)(($row * 3 + $col) % 5);
	}
}
$left = PhpTensors\CudaArray::fromBuffer(pack('g*', ...$leftValues), [64, 128]);
$right = PhpTensors\CudaArray::fromBuffer(pack('g*', ...$rightValues), [64, 128]);
$product = $left->transpose()->matmul($right)->toArray();
$correct = true;
foreach ([[0, 0], [0, 127], [127, 0], [127, 127], [19, 73]] as [$row, $col]) {
	$expected = 0;
	for ($inner = 0; $inner < 64; $inner++) {
		$expected += $leftValues[$inner * 128 + $row] * $rightValues[$inner * 128 + $col];
	}
	$correct = $correct && $product[$row][$col] === (float)$expected;
}
var_dump($correct);
?>
--EXPECT--
float(100000)
int(128)
int(128)
float(64)
float(64)
int(8)
float(64)
float(64)
int(8)
float(64)
float(64)
int(2)
int(4)
float(64)
bool(true)