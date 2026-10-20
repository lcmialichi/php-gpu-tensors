<?php
function check(bool $condition, string $message): void
{
    if (!$condition) throw new RuntimeException($message);
}

class TensorLifecycleCapture
{
    public $tensor;
    public static $captured;

    public function __wakeup(): void
    {
        self::$captured = $this->tensor;
    }
}

function uninitializedTensor(string $class): object
{
    // A nested callback can retain the native object before its restore fails.
    $capture = TensorLifecycleCapture::class;
    $payload = sprintf(
        'O:%d:"%s":1:{s:5:"probe";O:%d:"%s":1:{s:6:"tensor";r:1;}}',
        strlen($class), $class, strlen($capture), $capture
    );
    TensorLifecycleCapture::$captured = null;
    try {
        unserialize($payload);
        throw new RuntimeException('An incomplete tensor payload was accepted');
    } catch (PhpTensors\InvalidArgumentException $error) {
    }
    $tensor = TensorLifecycleCapture::$captured;
    TensorLifecycleCapture::$captured = null;
    check($tensor instanceof $class, 'The failed restore did not retain the native object');
    return $tensor;
}

function expectUninitialized(callable $operation): void
{
    try {
        $operation();
        throw new RuntimeException('An uninitialized tensor was accepted');
    } catch (PhpTensors\RuntimeException $error) {
        check(stripos($error->getMessage(), 'uninitialized') !== false,
            'The failure must identify the uninitialized tensor');
    }
}
