--TEST--
Custom operator hooks are optional and compound results preserve values, references and exceptions
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
use PhpTensors\OperatorOverload;
require __DIR__ . '/tensor_operator_aliasing.inc';

class ScalarValue extends OperatorOverload
{
    public static int $created = 0;
    public static int $destroyed = 0;
    public static ?Throwable $failure = null;
    public function __construct(public readonly float $value) { self::$created++; }
    public function __destruct() { self::$destroyed++; }
    private static function number(mixed $value): float
    {
        return $value instanceof self ? $value->value : $value;
    }
    public function __add(mixed $left, mixed $right): mixed
    {
        if (self::$failure !== null) throw self::$failure;
        return new static(self::number($left) + self::number($right));
    }
    public function __sub(mixed $left, mixed $right): mixed
    {
        return new static(self::number($left) - self::number($right));
    }
    public function __mul(mixed $left, mixed $right): mixed
    {
        return new static(self::number($left) * self::number($right));
    }
    public function __div(mixed $left, mixed $right): mixed
    {
        return new static(self::number($left) / self::number($right));
    }
    public function __pow(mixed $left, mixed $right): mixed
    {
        return new static(self::number($left) ** self::number($right));
    }
    public function __mod(mixed $left, mixed $right): mixed
    {
        return new static(fmod(self::number($left), self::number($right)));
    }
}

$value = new ScalarValue(4);
$old = $value;
$reference =& $value;
$assigned = ($value += 2);
check($assigned === $value && $reference === $value && $old->value === 4.0, 'Custom rebinding');
check($value->value === 6.0 && $value !== $old, 'Custom addition');
$value -= 1;
$value *= 2;
$value /= 5;
$value **= 3;
$value %= 3;
check($value->value === 2.0, 'Custom compound arithmetic');
$post = $value++;
$pre = ++$value;
check($post->value === 2.0 && $pre === $value && $value->value === 4.0, 'Custom increments');
$post = $value--;
$pre = --$value;
check($post->value === 4.0 && $pre === $value && $value->value === 2.0, 'Custom decrements');
$reversed = 10;
$reversed -= $value;
check($reversed->value === 8.0, 'Custom original operand order');
unset($old, $assigned, $post, $pre, $reversed, $reference);

$error = new LogicException('original callback failure');
ScalarValue::$failure = $error;
$original = $value;
$reference =& $value;
try {
    $value += 1;
    throw new RuntimeException('Throwing hook was accepted');
} catch (LogicException $caught) {
    check($caught === $error, 'The callback exception was replaced');
}
check($value === $original && $reference === $original, 'Throwing hook changed the binding');
ScalarValue::$failure = null;
unset($original, $reference);
echo "Custom arithmetic, PHP references and original callback exceptions are preserved\n";

class AddOnly extends OperatorOverload
{
    public function __add(mixed $left, mixed $right): mixed { return $this; }
}
$identity = new AddOnly();
$same = $identity;
$identity += 1;
check($identity === $same, 'Returning the original operand destroyed it');
foreach (['__sub', '__mul', '__div', '__pow', '__mod'] as $method) {
    try {
        $identity->$method($identity, 1);
        throw new RuntimeException('An unsupported hook was accepted');
    } catch (PhpTensors\RuntimeException $caught) {
        check(str_contains($caught->getMessage(), $method), 'Unsupported hook diagnostic');
    }
}
try {
    $identity -= 1;
    throw new RuntimeException('Unsupported compound operation was accepted');
} catch (PhpTensors\RuntimeException $caught) {
}
check($identity === $same, 'Unsupported compound operation destroyed its operand');
class ArbitraryResult extends OperatorOverload
{
    public function __construct(public mixed $output) {}
    public function &__add(mixed $left, mixed $right): mixed { return $this->output; }
}
foreach ([null, false, 0, 0.0, '', []] as $expected) {
    $output = new ArbitraryResult($expected);
    $output += 1;
    check($output === $expected, 'Falsy callback result was treated as a failure');
}
$source = new ArbitraryResult(new ScalarValue(7));
$output = $source;
$output += 1;
$output = new ScalarValue(9);
check($source->output->value === 7.0, 'A hook reference escaped into the result binding');
$typed = new class {
    public ArbitraryResult $value;
};
$typed->value = new ArbitraryResult('not an object');
$original = $typed->value;
try {
    $typed->value += 1;
    throw new RuntimeException('Typed property accepted an incompatible result');
} catch (TypeError $caught) {
}
check($typed->value === $original, 'Typed property failure changed its operand');
unset($source, $output, $typed, $original, $identity, $same);
echo "Partial implementations, identity results and by-reference callbacks work\n";

class ObservedValue extends ScalarValue
{
    public static ?Closure $observer = null;
    public function __destruct()
    {
        parent::__destruct();
        if (self::$observer !== null) (self::$observer)($this->value);
    }
}
$observed = [];
$target = new ObservedValue(1);
ObservedValue::$observer = static function () use (&$target, &$observed): void {
    $observed[] = $target->value;
};
$target += 1;
check($observed === [2.0], 'Destructor saw a stale result binding');
ObservedValue::$observer = static function (float $old) use ($error): void {
    if ($old === 2.0) throw $error;
};
try {
    $target += 1;
    throw new RuntimeException('Destructor exception was swallowed');
} catch (LogicException $caught) {
    check($caught === $error, 'Destructor exception was replaced');
}
check($target instanceof ObservedValue && $target->value === 3.0, 'Destructor exception corrupted result');
ObservedValue::$observer = null;
unset($target, $value);
check(ScalarValue::$created === ScalarValue::$destroyed, 'Custom results were leaked or destroyed twice');
$beforeCreated = ScalarValue::$created;
$beforeDestroyed = ScalarValue::$destroyed;
$value = new ScalarValue(0);
for ($i = 0; $i < 1000; $i++) $value += 1;
check($value->value === 1000.0, 'Repeated custom assignment');
unset($value);
check(ScalarValue::$created - $beforeCreated === ScalarValue::$destroyed - $beforeDestroyed,
    'Repeated custom assignment leaked operands');
echo "Result commits precede destructors and release each replaced object exactly once\n";
?>
--EXPECT--
Custom arithmetic, PHP references and original callback exceptions are preserved
Partial implementations, identity results and by-reference callbacks work
Result commits precede destructors and release each replaced object exactly once
