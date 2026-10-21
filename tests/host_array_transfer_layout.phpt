--TEST--
HostArray transfer staging preserves exact dtype bytes and logical views without PHP scalar conversion
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
use PhpTensors\HostArray;
function check(bool $ok, string $message): void
{
    if (!$ok) throw new RuntimeException($message);
}
$fixtures = require __DIR__ . '/fixtures/transfer_contract.php';
check(count($fixtures['patterns']) === 13, 'all storage dtypes');
foreach ($fixtures['patterns'] as $dtype => $bytes) {
    $width = intdiv(strlen($bytes), 8);
    check(strlen($bytes) === $width * 8, "$dtype eight complete elements");
    $source = HostArray::fromBuffer($bytes, [2, 4], $dtype);
    $retained = $copies = [];
    foreach ($fixtures['layouts'] as $name => [$shape, $indices, $makeView]) {
        $view = $makeView($source);
        $expected = '';
        foreach ($indices as $index) $expected .= substr($bytes, $index * $width, $width);
        check($view->getShape() === $shape && $view->dtype() === $dtype, "$dtype $name metadata");
        check($view->toBuffer() === $expected, "$dtype $name exact logical bytes");
        check($view->toHost() === $view, "$dtype $name same-backend identity");
        $copy = $view->copy();
        check($copy !== $view && $copy->toBuffer() === $expected, "$dtype $name independent copy");
        $copies[] = [$copy, $expected];
        if ($view !== $source) $retained[] = [$view, $expected];
    }
    $source->__construct([[0, 0, 0, 0], [0, 0, 0, 0]], $dtype);
    unset($source, $view, $copy);
    foreach (array_merge($retained, $copies) as [$value, $expected]) {
        check($value->toBuffer() === $expected, "$dtype view/copy survives replacement and release");
    }
    unset($retained, $copies, $value);

    $large = HostArray::fromBuffer(str_repeat($bytes, 8192), [8192, 8], $dtype);
    $expected = '';
    for ($i = 0; $i < 8; $i++) $expected .= str_repeat(substr($bytes, $i * $width, $width), 8192);
    check($large->transpose()->toBuffer() === $expected, "$dtype large dense strided staging");
    check($large->slice(':, 3')->toBuffer() === str_repeat(substr($bytes, 3 * $width, $width), 8192),
        "$dtype large sparse offset staging");
    unset($large);
}
echo "All 13 dtypes preserve raw bytes across nine layouts, copies and retained storage\n";
echo "Large transposed and sparse views pack exact bytes without scalar round trips\n";
?>
--EXPECT--
All 13 dtypes preserve raw bytes across nine layouts, copies and retained storage
Large transposed and sparse views pack exact bytes without scalar round trips
