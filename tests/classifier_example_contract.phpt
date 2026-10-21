--TEST--
MNIST training example is standalone, importable and needs no configuration
--FILE--
<?php
ob_start();
$root = is_dir(__DIR__ . '/../examples') ? dirname(__DIR__) : dirname(__DIR__, 2);
$includedBefore = count(get_included_files());
require $root . '/examples/08_gpu_classifier.php';
if (ob_get_clean() !== '' || (new ReflectionFunction('trainMnist'))->getNumberOfParameters() !== 0) {
    throw new RuntimeException('Import ran training or the example needs configuration.');
}
if (count(get_included_files()) !== $includedBefore + 1) {
    throw new RuntimeException('The example includes external code.');
}
if (preg_match('/\beval\s*\(/', file_get_contents($root . '/examples/08_gpu_classifier.php'))) {
    throw new RuntimeException('The example generates callbacks.');
}
echo "standalone import; no configuration, includes or eval\n";
try {
    readMnist('invalid');
    throw new RuntimeException('Invalid split was accepted.');
} catch (InvalidArgumentException $error) {
    echo $error->getMessage(), "\n";
}
?>
--EXPECT--
standalone import; no configuration, includes or eval
MNIST split must be train or test.
