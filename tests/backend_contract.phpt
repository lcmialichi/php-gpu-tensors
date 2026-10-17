--TEST--
Tensor and backend contracts support direct user-defined implementations
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
use PhpTensors\BackendInterface;
use PhpTensors\Number;
use PhpTensors\Tensor;

final class CustomTensor implements Tensor
{
    public function getBackend(): string { return 'custom'; }
    public function getShape(): array { return [2]; }
    public function getSize(): int { return 2; }
    public function dtype(): string { return 'float32'; }
    public function toArray(): array { return [3.0, 4.0]; }
}

final class CustomBackend implements BackendInterface
{
    public function getName(): string { return 'custom'; }
    public function isAvailable(): bool { return true; }
    public function supports(string $operation, array $context = []): bool
    {
        return $operation === 'identity';
    }
    public function createTensor(array $values, array $shape, ?string $dtype = 'float32'): Tensor
    {
        return new CustomTensor();
    }
    public function execute(string $operation, array $inputs, array $arguments = []): Tensor
    {
        return new CustomTensor();
    }
}

final class CustomNumberTensor extends Number implements Tensor
{
    public function __construct(public float $value) {}
    public function getBackend(): string { return 'custom-number'; }
    public function getShape(): array { return []; }
    public function getSize(): int { return 1; }
    public function dtype(): string { return 'float64'; }
    public function toArray(): array { return [$this->value]; }
    public function __add($left, $right): static
    {
        return new static(self::numberValue($left) + self::numberValue($right));
    }
    public function __sub($left, $right): static
    {
        return new static(self::numberValue($left) - self::numberValue($right));
    }
    public function __mul($left, $right): static
    {
        return new static(self::numberValue($left) * self::numberValue($right));
    }
    public function __div($left, $right): static
    {
        return new static(self::numberValue($left) / self::numberValue($right));
    }
    public function __pow($left, $right): static
    {
        return new static(self::numberValue($left) ** self::numberValue($right));
    }
    public function __mod($left, $right): static
    {
        return new static(fmod(self::numberValue($left), self::numberValue($right)));
    }
    public function __inc(): static { return new static($this->value + 1); }
    public function __dec(): static { return new static($this->value - 1); }
    private static function numberValue(mixed $value): float
    {
        return $value instanceof self ? $value->value : (float)$value;
    }
}

$backend = new CustomBackend();
$tensor = $backend->createTensor([3.0, 4.0], [2]);
var_dump($tensor instanceof Tensor);
var_dump($tensor->getBackend());
var_dump($backend->supports('identity'));
var_dump($backend->execute('identity', [$tensor]) instanceof Tensor);
var_dump(class_exists('PhpTensors\\BackendManager'));
$number = new CustomNumberTensor(4.0) + 3.0;
var_dump($number->toArray());
?>
--EXPECT--
bool(true)
string(6) "custom"
bool(true)
bool(true)
bool(false)
array(1) {
  [0]=>
  float(7)
}
