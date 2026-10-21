<?php

namespace PhpTensorsTestFixtures;

/**
 * @internal Reflection-only contract for the CPU build's CUDA availability guard.
 */
final class UnavailableCudaArray implements \PhpTensors\OperatorOverloadable
{
    public function __construct(mixed ...$arguments) {}
    public function __call(string $name, array $arguments): mixed {}
    public static function __callStatic(string $name, array $arguments): mixed {}
}
