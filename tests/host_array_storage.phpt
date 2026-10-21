--TEST--
HostArray allocation footprint matches native dtype widths rather than double backing
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (getenv('USE_ZEND_ALLOC') === '0') die('skip Zend allocation accounting required');
?>
--FILE--
<?php
$count = 1024 * 1024;
foreach (['float32' => 4, 'float64' => 8, 'float16' => 2, 'bfloat16' => 2,
          'int8' => 1, 'int16' => 2, 'int32' => 4, 'int64' => 8,
          'uint8' => 1, 'uint16' => 2, 'uint32' => 4, 'uint64' => 8, 'bool' => 1] as $dtype => $width) {
    $before = memory_get_usage(false);
    $host = PhpTensors\HostArray::zeros([$count], $dtype);
    $allocated = memory_get_usage(false) - $before;
    $expected = $count * $width;
    if ($allocated < $expected || $allocated > $expected + 65536) {
        throw new RuntimeException("$dtype: expected $expected bytes plus bounded metadata, allocated $allocated");
    }
    if ($host->getElementSize() !== $width || strlen($host->toBuffer()) !== $expected) {
        throw new RuntimeException("$dtype: physical and exported element widths differ");
    }
    unset($host);
}
echo "13 dtypes allocate their native width; FP32 uses four bytes per element, FP64 eight\n";
?>
--EXPECT--
13 dtypes allocate their native width; FP32 uses four bytes per element, FP64 eight
