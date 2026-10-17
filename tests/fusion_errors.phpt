--TEST--
Fusion validates capture, replay contracts and restores eager execution after failures
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
use PhpTensors\CudaArray;
use PhpTensors\Fusion;
$a = new CudaArray([1, 2]);
function fails(callable $callback): void {
    try { $callback(); echo "MISSED\n"; }
    catch (Throwable $e) { echo get_class($e), "\n"; }
}
fails(fn() => Fusion::run(fn() => Fusion::run(fn() => $a + 1)));
fails(fn() => Fusion::run(function () use ($a) { $a[0] = 9; return $a; }));
fails(fn() => Fusion::compile(fn($x) => $x->toArray(), [$a]));
fails(fn() => Fusion::run(fn() => 123));
fails(fn() => Fusion::run(fn() => $a + new CudaArray([1, 2, 3])));
$escaped = null;
fails(function () use ($a, &$escaped) {
    Fusion::run(function () use ($a, &$escaped) {
        $escaped = $a * 2;
        throw new LogicException('abort');
    });
});
fails(fn() => $escaped->toArray());
$graph = Fusion::compile(fn($x) => $x * 2, [$a]);
fails(fn() => $graph->run());
fails(fn() => $graph->run(new CudaArray([1, 2, 3])));
fails(fn() => $graph->run(new CudaArray([1, 2], 'float64')));
$fiber = new Fiber(fn() => Fusion::run(function () use ($a) {
    Fiber::suspend();
    return $a + 1;
}));
fails(fn() => $fiber->start());
var_dump(($a + 1)->toArray(), Fusion::run(fn() => $a * 2)->toArray());
?>
--EXPECT--
PhpTensors\RuntimeException
PhpTensors\RuntimeException
PhpTensors\RuntimeException
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
LogicException
PhpTensors\RuntimeException
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
PhpTensors\InvalidArgumentException
FiberError
array(2) {
  [0]=>
  float(2)
  [1]=>
  float(3)
}
array(2) {
  [0]=>
  float(2)
  [1]=>
  float(4)
}
