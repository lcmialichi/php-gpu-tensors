--TEST--
HostArray sum, mean, min and max cover every storage dtype, axes, exact integers and empty identities
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
$dtypes = ['float32', 'float64', 'float16', 'bfloat16', 'int8', 'int16', 'int32',
    'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'];
foreach ($dtypes as $dtype) {
    $base = new HostArray([[1, 2], [3, 4]], $dtype);
    $floating = str_contains($dtype, 'float');
    foreach (['sum', 'mean', 'min', 'max'] as $method) {
        $outDtype = in_array($dtype, ['float16', 'bfloat16'], true) ? 'float32' : $dtype;
        if ($method === 'mean' && !$floating) $outDtype = 'float64';
        $number = $dtype === 'bool' ? 1 : match ($method) {
            'sum' => 10, 'mean' => 2.5, 'min' => 1, 'max' => 4,
        };
        $expected = $outDtype === 'bool' ? true : (str_contains($outDtype, 'float') ? (float)$number : $number);
        $result = $base->$method();
        check($result->dtype() === $outDtype && $result->getShape() === [1], "$dtype $method global metadata");
        check($result->item() === $expected, "$dtype $method global result");
        check($base->transpose()->$method()->item() === $expected, "$dtype $method strided global result");
        $axis0 = $base->$method(0);
        $axis1 = $base->$method(-1);
        check($axis0->getShape() === [2] && $axis1->getShape() === [2], "$dtype $method axis shape");
        if ($dtype !== 'bool') {
            $expected0 = match ($method) { 'sum' => [4, 6], 'mean' => [2, 3], 'min' => [1, 2], 'max' => [3, 4] };
            $expected1 = match ($method) { 'sum' => [3, 7], 'mean' => [1.5, 3.5], 'min' => [1, 3], 'max' => [2, 4] };
            check($axis0->toBuffer() === (new HostArray($expected0, $outDtype))->toBuffer(), "$dtype $method axis0 values");
            check($axis1->toBuffer() === (new HostArray($expected1, $outDtype))->toBuffer(), "$dtype $method axis1 values");
        }
        $expanded = $base->slice(1, null)->broadcastTo([3, 2]);
        check($expanded->$method(1)->getShape() === [3], "$dtype $method zero-stride input");
        foreach ([-3, 2, 0.5, '0', true, PHP_INT_MAX] as $invalid) {
            try {
                $base->$method($invalid);
                throw new RuntimeException('Invalid reduction axis accepted');
            } catch (PhpTensors\InvalidArgumentException $expectedError) {}
        }
    }
    $empty = HostArray::zeros([2, 0, 3], $dtype);
    check($empty->sum(1)->getShape() === [2, 3], "$dtype empty sum output shape");
    check($empty->sum(1)->toBuffer() === HostArray::zeros([2, 3], $empty->sum()->dtype())->toBuffer(), "$dtype sum identity");
    check($empty->min(0)->getShape() === [0, 3] && $empty->max(0)->getSize() === 0, "$dtype no output groups");
    check(is_nan($empty->mean()->item()) && is_nan($empty->mean(1)->at(1, 2)), "$dtype empty mean NaN");
    foreach (['min', 'max'] as $method) foreach ([null, 1] as $axis) {
        try {
            $empty->$method($axis);
            throw new RuntimeException('Empty min/max accepted without identity');
        } catch (PhpTensors\InvalidArgumentException $expectedError) {}
    }
}
echo "13 dtypes cover global/axis/strided reductions, dtype promotion and empty identities\n";

check((new HostArray([PHP_INT_MAX, 1], 'int64'))->sum()->item() === PHP_INT_MIN, 'int64 sum wraps without signed C overflow');
check((new HostArray([127, 1], 'int8'))->sum()->item() === -128, 'int8 sum wraps at dtype width');
$uint = HostArray::fromBuffer(hex2bin('ffffffffffffffff0100000000000000'), [2], 'uint64');
check($uint->sum()->item() === 0, 'uint64 sum wraps exactly');
check($uint->max()->item() === '18446744073709551615', 'uint64 max preserves exact export');
check($uint->min()->item() === 1, 'uint64 min preserves exact ordering');
$signed = new HostArray([PHP_INT_MIN, PHP_INT_MAX], 'int64');
check($signed->min()->item() === PHP_INT_MIN && $signed->max()->item() === PHP_INT_MAX, 'signed extrema do not round through double');
$tiny = new HostArray([1e-300, 2e-300], 'float64');
check(abs($tiny->sum()->item() / 3e-300 - 1) < 1e-14, 'FP64 tiny sum precision');
check(abs($tiny->mean()->item() / 1.5e-300 - 1) < 1e-14, 'FP64 tiny mean precision');
check(is_nan((new HostArray([NAN, 1]))->sum()->item()), 'sum propagates NaN');
check(is_nan((new HostArray([INF, -INF]))->mean()->item()), 'mean preserves IEEE singularity');
check((new HostArray([false, false], 'bool'))->sum()->item() === false, 'boolean sum identity');
check((new HostArray([false, true], 'bool'))->mean()->item() === 0.5, 'boolean mean uses FP64');
echo "Integer overflow is defined modulo dtype width; exact extrema and FP64/IEEE precision are preserved\n";
?>
--EXPECT--
13 dtypes cover global/axis/strided reductions, dtype promotion and empty identities
Integer overflow is defined modulo dtype width; exact extrema and FP64/IEEE precision are preserved
