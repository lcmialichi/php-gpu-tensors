--TEST--
Fusion PTX cache ignores pointers, specializes layouts and evicts at its bounded capacity
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
Fusion::clearCache();
$a = new CudaArray([[1, 2], [3, 4]]);
$b = new CudaArray([[5, 6], [7, 8]]);
$first = Fusion::compile(fn($x) => $x * 2 + 1, [$a]);
$second = Fusion::compile(fn($x) => $x * 2 + 1, [$b]);
check(!$first->getStats()['cacheHit']);
check($second->getStats()['cacheHit']);
check(Fusion::getCacheStats()['compilations'] === 1);
$layout = Fusion::compile(fn($x) => $x * 2 + 1, [$a->transpose()]);
check(!$layout->getStats()['cacheHit']);
$before = Fusion::getCacheStats()['compilations'];
Fusion::run(fn() => $a * 2 + 1);
Fusion::run(fn() => $b * 2 + 1);
check(Fusion::getCacheStats()['compilations'] === $before);
echo "cache keys specialize layouts, not pointers\n";
for ($i = 0; $i < 20; $i++) {
    $graph = Fusion::compile(fn($x) => $x + $i, [$a]);
    check($graph->run($a)->toArray()[0][0] === 1.0 + $i);
}
$stats = Fusion::getCacheStats();
check($stats['entries'] <= $stats['maxEntries']);
check($stats['bytes'] <= $stats['maxBytes']);
check($stats['evictions'] > 0);
echo "cache bounded\n";
Fusion::clearCache();
check($first->run($b)->toArray() === [[11.0, 13.0], [15.0, 17.0]]);
check(Fusion::getCacheStats()['entries'] === 0);
echo "compiled graphs survive cache clear\n";
?>
--EXPECT--
cache keys specialize layouts, not pointers
cache bounded
compiled graphs survive cache clear
