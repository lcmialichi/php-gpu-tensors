--TEST--
Fusion private streams retain inputs and support concurrent executions and repeated waits
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
$b = new CudaArray([4, 5, 6]);
$graph = Fusion::compile(fn($x, $y) => ($x * $y + 1)->sqrt(), [$a, $b]);
$first = $graph->runAsync($a, $b);
$second = $graph->runAsync($b, $b);
check($graph->getStats()['pending'] === 2);
check(is_bool($first->isFinished()));
try { $a[0] = 9; echo "MISSED\n"; }
catch (Cuda\RuntimeException $error) { echo "mutation blocked\n"; }
try { cuda_device_reset(); echo "MISSED\n"; }
catch (Cuda\RuntimeException $error) { echo "reset blocked\n"; }
$expectedFirst = ($a * $b + 1)->sqrt()->toArray();
$expectedSecond = ($b * $b + 1)->sqrt()->toArray();
unset($a, $b);
$result = $first->wait();
check($result->toArray() === $expectedFirst);
check($first->wait() === $result);
check($first->isFinished());
check($second->wait()->toArray() === $expectedSecond);
check($graph->getStats()['pending'] === 0);
check($graph->getStats()['executions'] === 2);
echo "concurrent replay completed\n";
$a = new CudaArray([2, 3, 4]);
$pending = $graph->runAsync($a, $a);
unset($graph, $pending);
$a[0] = 7;
check($a->toArray()[0] === 7.0);
echo "destruction waited\n";
$native = Fusion::compile(fn($x) => ($x + 1)->sum() * 2, [$a], cudaGraph: true);
check($native->getStats()['backend'] === 'stream');
check($native->getStats()['incompatibility'] === null);
check($native->getStats()['cudaGraphIncompatibility'] === 'native-graph-not-supported');
check($native->runAsync($a)->wait()->toArray() === [34.0]);
echo "native async completed\n";
check($native->run($a)->toArray() === [34.0]);
echo "native replay preserved\n";
$power = Fusion::compile(fn($x) => $x->power(2) + 1, [$a]);
try { $power->runAsync($a); echo "MISSED\n"; }
catch (Cuda\RuntimeException $error) { echo "unsupported power async rejected\n"; }
?>
--EXPECT--
mutation blocked
reset blocked
concurrent replay completed
destruction waited
native async completed
native replay preserved
unsupported power async rejected
