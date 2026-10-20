<?php
function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}

function exerciseTensorOperators(string $class): void
{
    $operations = [
        ['add', function (&$left, $right) { return $left += $right; }, [4.0, 6.0], [4.0, 8.0]],
        ['subtract', function (&$left, $right) { return $left -= $right; }, [0.0, 2.0], [0.0, 0.0]],
        ['multiply', function (&$left, $right) { return $left *= $right; }, [4.0, 8.0], [4.0, 16.0]],
        ['divide', function (&$left, $right) { return $left /= $right; }, [1.0, 2.0], [1.0, 1.0]],
    ];
    foreach ($operations as [$name, $assign, $scalarExpected, $selfExpected]) {
        $left = new $class([2, 4]);
        $old = $left;
        $reference =& $left;
        $result = $assign($left, 2);
        check($left->toArray() === $scalarExpected, "$name compound values");
        check($result === $left && $reference === $left && $left !== $old, "$name rebinding");
        check($old->toArray() === [2.0, 4.0], "$name changed a surviving object alias");
        unset($old, $reference, $result);

        $left = new $class([2, 4]);
        $weak = WeakReference::create($left);
        $assign($left, 2);
        check($weak->get() === null, "$name leaked the replaced operand");

        $left = new $class([2, 4]);
        $assign($left, $left);
        check($left->toArray() === $selfExpected, "$name object self-alias");

        $left = new $class([2, 4]);
        $holder = new class($left) {
            public object $value;
            public function __construct(object $value) { $this->value = $value; }
        };
        $propertyReference =& $holder->value;
        $assign($propertyReference, 2);
        check($holder->value->toArray() === $scalarExpected, "$name typed property reference");
        check($left->toArray() === [2.0, 4.0], "$name mutated the property alias");
        $array = [new $class([2, 4])];
        $assign($array[0], 2);
        check($array[0]->toArray() === $scalarExpected, "$name array element");
        unset($propertyReference, $holder, $array);
    }
    foreach ([
        [function (&$value) { $value += $value; }, [4.0, 8.0]],
        [function (&$value) { $value -= $value; }, [0.0, 0.0]],
        [function (&$value) { $value *= $value; }, [4.0, 16.0]],
        [function (&$value) { $value /= $value; }, [1.0, 1.0]],
    ] as [$assign, $expected]) {
        $value = new $class([2, 4]);
        $reference =& $value;
        $assign($value);
        check($value->toArray() === $expected && $reference === $value, 'Identical operand zvals');
        unset($reference);
    }
    $right = new $class([2, 4]);
    $scalar = 8;
    $scalar -= $right;
    check($scalar->toArray() === [6.0, 4.0], 'Reversed subtraction assignment');
    $scalar = 8;
    $scalar /= $right;
    check($scalar->toArray() === [4.0, 2.0], 'Reversed division assignment');
    check($right->toArray() === [2.0, 4.0], 'Reversed assignment mutated the right operand');

    $value = new $class([2, 4]);
    $old = $value;
    $post = $value++;
    check($post === $old && $value->toArray() === [3.0, 5.0], 'Post-increment');
    $pre = ++$value;
    check($pre === $value && $value->toArray() === [4.0, 6.0], 'Pre-increment');
    $post = $value--;
    check($post->toArray() === [4.0, 6.0] && $value->toArray() === [3.0, 5.0], 'Post-decrement');
    $pre = --$value;
    check($pre === $value && $value->toArray() === [2.0, 4.0], 'Pre-decrement');
    check($old->toArray() === [2.0, 4.0], 'Increment mutated an existing object alias');

    $value = new $class([2, 4]);
    $original = $value;
    $reference =& $value;
    foreach ([new $class([1, 2, 3]), new stdClass()] as $invalid) {
        try {
            $value += $invalid;
            throw new RuntimeException('Invalid compound assignment was accepted');
        } catch (PhpTensors\InvalidArgumentException $error) {
        }
        check($value === $original && $reference === $original, 'Failure changed the binding');
        check($value->toArray() === [2.0, 4.0], 'Failure changed the operand values');
    }
    unset($reference);
}
