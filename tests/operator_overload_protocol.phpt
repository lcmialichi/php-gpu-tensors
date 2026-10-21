--TEST--
Custom operator protocol preserves dispatch, operand order, unsupported boundaries and exceptions
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
declare(strict_types=1);
use PhpTensors\OperatorOverload;
use PhpTensors\OperatorOverloadable;
require __DIR__ . '/tensor_operator_aliasing.inc';

abstract class ProtocolValue extends OperatorOverload
{
    public static array $calls = [];
    public static ?Throwable $failure = null;
    public function __construct(public readonly float $value) {}
    private static function number(mixed $value): float
    {
        if ($value instanceof self) return $value->value;
        if (is_int($value) || is_float($value)) return (float)$value;
        throw new DomainException('This example supports only ProtocolValue and numeric scalars');
    }
    private function apply(string $hook, mixed $left, mixed $right): static
    {
        self::$calls[] = [$this, $hook, $left, $right];
        if (self::$failure !== null) throw self::$failure;
        $a = self::number($left);
        $b = self::number($right);
        return new static(match ($hook) {
            '__add' => $a + $b,
            '__sub' => $a - $b,
            '__mul' => $a * $b,
            '__div' => $a / $b,
            '__pow' => $a ** $b,
            '__mod' => fmod($a, $b),
        });
    }
    public function __add(mixed $left, mixed $right): static { return $this->apply(__FUNCTION__, $left, $right); }
    public function __sub(mixed $left, mixed $right): static { return $this->apply(__FUNCTION__, $left, $right); }
    public function __mul(mixed $left, mixed $right): static { return $this->apply(__FUNCTION__, $left, $right); }
    public function __div(mixed $left, mixed $right): static { return $this->apply(__FUNCTION__, $left, $right); }
    public function __pow(mixed $left, mixed $right): static { return $this->apply(__FUNCTION__, $left, $right); }
    public function __mod(mixed $left, mixed $right): static { return $this->apply(__FUNCTION__, $left, $right); }
}
final class ProtocolLeft extends ProtocolValue {}
final class ProtocolRight extends ProtocolValue {}

$operations = [
    ['__add', fn($a, $b) => $a + $b, function (&$a, $b) { return $a += $b; }, 12.0],
    ['__sub', fn($a, $b) => $a - $b, function (&$a, $b) { return $a -= $b; }, 8.0],
    ['__mul', fn($a, $b) => $a * $b, function (&$a, $b) { return $a *= $b; }, 20.0],
    ['__div', fn($a, $b) => $a / $b, function (&$a, $b) { return $a /= $b; }, 5.0],
    ['__pow', fn($a, $b) => $a ** $b, function (&$a, $b) { return $a **= $b; }, 100.0],
    ['__mod', fn($a, $b) => $a % $b, function (&$a, $b) { return $a %= $b; }, 0.0],
];
$left = new ProtocolLeft(10);
$right = new ProtocolRight(2);
check($left instanceof OperatorOverloadable, 'Base class supplies marker automatically');
foreach ($operations as [$hook, $binary, $assign, $expected]) {
    foreach ([[$left, 2, $left], [10, $right, $right], [$left, $right, $left]] as [$a, $b, $receiver]) {
        ProtocolValue::$calls = [];
        $result = $binary($a, $b);
        check($result instanceof ProtocolValue && $result::class === $receiver::class, "$hook receiver type");
        check($result->value === $expected, "$hook arithmetic");
        check(ProtocolValue::$calls === [[$receiver, $hook, $a, $b]], "$hook original operands and one callback");
        check($left->value === 10.0 && $right->value === 2.0, "$hook immutable inputs");

        ProtocolValue::$calls = [];
        $target = $a;
        $reference =& $target;
        $result = $assign($target, $b);
        check($target === $result && $reference === $target, "$hook compound reference binding");
        check($target->value === $expected && $left->value === 10.0 && $right->value === 2.0, "$hook compound value");
        check(ProtocolValue::$calls === [[$receiver, $hook, $a, $b]], "$hook compound dispatch");
        unset($reference, $target, $result);

        $error = new LogicException("$hook exact callback exception");
        ProtocolValue::$failure = $error;
        ProtocolValue::$calls = [];
        $target = $a;
        $reference =& $target;
        try {
            $assign($target, $b);
            throw new RuntimeException("$hook swallowed exception");
        } catch (LogicException $caught) {
            check($caught === $error, "$hook exception identity");
        }
        check($target === $a && $reference === $a, "$hook failed assignment binding");
        check(ProtocolValue::$calls === [[$receiver, $hook, $a, $b]], "$hook must not retry another receiver after failure");
        ProtocolValue::$calls = [];
        try {
            $binary($a, $b);
            throw new RuntimeException("$hook swallowed binary exception");
        } catch (LogicException $caught) {
            check($caught === $error, "$hook binary exception identity");
        }
        check(ProtocolValue::$calls === [[$receiver, $hook, $a, $b]], "$hook binary dispatch must not retry after failure");
        ProtocolValue::$failure = null;
        unset($reference, $target);
    }
}
echo "six operators and compounds preserve scalar order, mixed-class dispatch and exact exceptions\n";

class MissingProtocol extends OperatorOverload {}
$missing = new MissingProtocol();
foreach ($operations as [$hook, $binary]) {
    foreach ([[$missing, 1], [1, $missing], [$missing, $right]] as [$a, $b]) {
        ProtocolValue::$calls = [];
        try {
            $binary($a, $b);
            throw new RuntimeException("$hook missing hook accepted");
        } catch (PhpTensors\RuntimeException $caught) {
            check(str_contains($caught->getMessage(), 'MissingProtocol::' . $hook), "$hook missing-hook diagnostic");
        }
        check(ProtocolValue::$calls === [], "$hook must not fall back to the right class after a left hook exception");
    }
}
class MarkerOnly implements OperatorOverloadable
{
    public static int $calls = 0;
    public function __add(mixed $left, mixed $right): mixed { self::$calls++; return 123; }
}
$marker = new MarkerOnly();
foreach ([fn() => $marker + 1, fn() => 1 + $marker] as $call) {
    try {
        $call();
        throw new RuntimeException('Marker installed an operator handler');
    } catch (TypeError $expected) {}
}
check(MarkerOnly::$calls === 0, 'Marker alone must not call hook-named methods');
try {
    $left + new stdClass();
    throw new RuntimeException('Custom operand rejection swallowed');
} catch (DomainException $expected) {
    check($expected->getMessage() === 'This example supports only ProtocolValue and numeric scalars', 'Custom operand domain preserved');
}
$host = new PhpTensors\HostArray([2]);
ProtocolValue::$calls = [];
try {
    $host + $left;
    throw new RuntimeException('Native host accepted an unrelated custom class');
} catch (PhpTensors\InvalidArgumentException $expected) {}
check(ProtocolValue::$calls === [], 'Native rejection must not fall back to a custom right operand');
try {
    $left + $host;
    throw new RuntimeException('Custom domain restriction bypassed for a native tensor');
} catch (DomainException $expected) {}
check(ProtocolValue::$calls === [[$left, '__add', $left, $host]], 'Custom left class controls native interoperability');
class SideEffectProtocol extends OperatorOverload
{
    public int $value = 0;
    public function __construct(private Throwable $failure) {}
    public function __add(mixed $left, mixed $right): mixed
    {
        $this->value++;
        throw $this->failure;
    }
}
$error = new LogicException('Userland side effect');
$mutable = new SideEffectProtocol($error);
$alias = $mutable;
try {
    $mutable += 1;
    throw new RuntimeException('Side-effect callback exception swallowed');
} catch (LogicException $caught) {
    check($caught === $error, 'Side-effect exception identity');
}
check($mutable === $alias && $alias->value === 1, 'Binding safety does not roll back arbitrary userland mutations');
echo "marker-only objects do not overload; missing hooks and mixed operand errors are explicit\n";

$unsupported = [
    fn($a) => $a & 1, fn($a) => 1 & $a,
    fn($a) => $a | 1, fn($a) => 1 | $a,
    fn($a) => $a ^ 1, fn($a) => 1 ^ $a,
    fn($a) => $a << 1, fn($a) => 1 << $a,
    fn($a) => $a >> 1, fn($a) => 1 >> $a,
    fn($a) => ~$a,
    function (&$a) { return $a &= 1; },
    function (&$a) { return $a |= 1; },
    function (&$a) { return $a ^= 1; },
    function (&$a) { return $a <<= 1; },
    function (&$a) { return $a >>= 1; },
];
foreach ($unsupported as $call) {
    ProtocolValue::$calls = [];
    $target = $left;
    $reference =& $target;
    try {
        $call($target);
        throw new RuntimeException('Unsupported opcode accepted');
    } catch (TypeError $expected) {}
    check($target === $left && $reference === $left, 'Unsupported opcode preserves binding');
    check(ProtocolValue::$calls === [], 'Unsupported opcode must not call arithmetic hooks');
    unset($reference, $target);
}
foreach ([fn() => $left . 'x', fn() => 'x' . $left] as $call) {
    try {
        $call();
        throw new RuntimeException('Non-stringable object concatenated');
    } catch (Error $expected) {
        check(str_contains($expected->getMessage(), 'could not be converted to string'), 'Standard PHP string conversion error');
    }
}
class PrintableProtocol extends ProtocolValue
{
    public function __toString(): string { return 'printable'; }
}
ProtocolValue::$calls = [];
$printable = new PrintableProtocol(1);
check($printable . '!' === 'printable!' && '!' . $printable === '!printable', 'Concatenation uses ordinary string conversion');
check(!($left == $right) && $left == clone $left && $left !== clone $left, 'Standard object equality and identity');
check(!$left === false && (bool)$left === true, 'Standard object truthiness');
check(ProtocolValue::$calls === [], 'Comparison, concatenation and logical conversion do not call hooks');
echo "unsupported opcodes reject safely; strings, comparisons and truthiness keep PHP semantics\n";

foreach ([
    [fn($a) => +$a, 1, 10.0],
    [fn($a) => -$a, -1, -10.0],
] as [$unary, $multiplier, $expected]) {
    ProtocolValue::$calls = [];
    check($unary($left)->value === $expected, 'Unary sign arithmetic');
    check(ProtocolValue::$calls === [[$left, '__mul', $left, $multiplier]], 'Unary signs multiply by integer one or minus one');
}
foreach ([
    [function (&$a) { return $a++; }, '__add', false, 11.0],
    [function (&$a) { return ++$a; }, '__add', true, 11.0],
    [function (&$a) { return $a--; }, '__sub', false, 9.0],
    [function (&$a) { return --$a; }, '__sub', true, 9.0],
] as [$change, $hook, $prefix, $expected]) {
    ProtocolValue::$calls = [];
    $target = $left;
    $reference =& $target;
    $result = $change($target);
    check($result === ($prefix ? $target : $left), 'Prefix/postfix return identity');
    check($target->value === $expected && $reference === $target && $left->value === 10.0, 'Increment/decrement aliasing');
    check(ProtocolValue::$calls === [[$left, $hook, $left, 1]], 'Increment/decrement dispatch with integer one');
    unset($reference, $target, $result);

    $error = new LogicException('Increment/decrement failure');
    ProtocolValue::$failure = $error;
    ProtocolValue::$calls = [];
    $target = $left;
    $reference =& $target;
    try {
        $change($target);
        throw new RuntimeException('Increment/decrement callback failure swallowed');
    } catch (LogicException $caught) {
        check($caught === $error, 'Increment/decrement exception identity');
    }
    check($target === $left && $reference === $left, 'Failed increment/decrement preserves bindings');
    check(ProtocolValue::$calls === [[$left, $hook, $left, 1]], 'Failed increment/decrement invokes exactly one hook');
    ProtocolValue::$failure = null;
    unset($reference, $target);
}
$clone = clone $left;
check(($clone + 1)->value === 11.0 && $clone !== $left, 'Clones retain the custom operator handler');
class ObservedCloneProtocol extends ProtocolValue
{
    public int $clones = 0;
    public function __clone() { $this->clones++; }
}
$source = new ObservedCloneProtocol(3);
$clone = clone $source;
check($clone->clones === 1 && $source->clones === 0 && ($clone * 2)->value === 6.0, 'User clone hook and member copy preserved');
class ThrowingCloneProtocol extends ProtocolValue
{
    public static ?WeakReference $copy = null;
    public static ?Throwable $error = null;
    public function __clone()
    {
        self::$copy = WeakReference::create($this);
        throw self::$error;
    }
}
$source = new ThrowingCloneProtocol(3);
$error = new LogicException('Clone hook failure');
ThrowingCloneProtocol::$error = $error;
try {
    $copy = clone $source;
    throw new RuntimeException('Clone hook exception swallowed');
} catch (LogicException $caught) {
    check($caught === $error, 'Clone exception identity');
}
check(ThrowingCloneProtocol::$copy->get() === null, 'Failed clone released exactly once');
check(($source + 1)->value === 4.0, 'Failed clone does not damage original operator dispatch');
ThrowingCloneProtocol::$error = null;
ProtocolValue::$calls = [];
echo "unary signs and pre/post increments preserve hooks, integer operands and aliases\n";
?>
--EXPECT--
six operators and compounds preserve scalar order, mixed-class dispatch and exact exceptions
marker-only objects do not overload; missing hooks and mixed operand errors are explicit
unsupported opcodes reject safely; strings, comparisons and truthiness keep PHP semantics
unary signs and pre/post increments preserve hooks, integer operands and aliases
