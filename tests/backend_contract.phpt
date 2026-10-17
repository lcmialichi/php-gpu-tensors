--TEST--
PhpTensors backend interfaces can be implemented by userland classes
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
?>
--FILE--
<?php
use PhpTensors\BackendInterface;
use PhpTensors\BackendManager;
use PhpTensors\Tensor;
use PhpTensors\Numbers;
use PhpTensors\InvalidArgumentException;
use PhpTensors\RuntimeException;
use PhpTensors\Cuda\CudaArray;

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
    public bool $available = true;
    public function getName(): string { return 'custom'; }
    public function isAvailable(): bool { return $this->available; }
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

$backend = new CustomBackend();
$managerRejected = false;
try {
    BackendManager::register(new stdClass());
} catch (InvalidArgumentException) {
    $managerRejected = true;
}
BackendManager::register($backend);
$custom = $backend->createTensor([3.0, 4.0], [2]);
var_dump($custom instanceof Tensor);
var_dump($backend->supports('identity'));
var_dump($custom->getBackend());
var_dump($managerRejected);
var_dump(BackendManager::available() === ['custom']);
var_dump(BackendManager::get('custom') === $backend);
$factoryTensor = BackendManager::create('custom', [3.0, 4.0], [2]);
var_dump($factoryTensor instanceof Tensor);
var_dump(BackendManager::forTensor($factoryTensor) === $backend);
var_dump(BackendManager::unregister('custom'));
var_dump(BackendManager::available() === []);
$unregisteredRejected = false;
try {
    BackendManager::get('custom');
} catch (InvalidArgumentException) {
    $unregisteredRejected = true;
}
var_dump($unregisteredRejected);
var_dump(BackendManager::forTensor($factoryTensor) === $backend);
var_dump(BackendManager::unregister('custom'));
$backend->available = false;
var_dump(CudaArray::ones([1]) instanceof Tensor);
var_dump(CudaArray::ones([1]) instanceof Numbers);
var_dump(\Cuda\CudaArray::ones([1]) instanceof Tensor);
var_dump(\PhpTensors\Cuda\Tensor::ones([1]) instanceof Tensor);
?>
--EXPECT--
bool(true)
bool(true)
string(6) "custom"
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(false)
bool(true)
bool(true)
bool(true)
bool(true)
