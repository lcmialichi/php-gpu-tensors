--TEST--
CPU-only backend tensor creation, arithmetic, transfers, and explicit CUDA errors
--SKIPIF--
<?php
if (!extension_loaded('phptensors')) die('skip phptensors extension unavailable');
if (class_exists('PhpTensors\\Cuda\\NN', false)) die('skip CUDA-enabled build');
?>
--FILE--
<?php
use PhpTensors\BackendManager;
use PhpTensors\BackendInterface;
use PhpTensors\HostArray;
use PhpTensors\InvalidArgumentException;
use PhpTensors\RuntimeException;
use PhpTensors\Tensor;

final class CpuCustomTensor implements Tensor
{
    public function getBackend(): string { return 'custom-cpu'; }
    public function getShape(): array { return [1]; }
    public function getSize(): int { return 1; }
    public function dtype(): string { return 'float64'; }
    public function toArray(): array { return [7.0]; }
}

final class CpuCustomBackend implements BackendInterface
{
    public function getName(): string { return 'custom-cpu'; }
    public function isAvailable(): bool { return true; }
    public function supports(string $operation, array $context = []): bool
    {
        return $operation === 'identity';
    }
    public function createTensor(array $values, array $shape, ?string $dtype = 'float32'): Tensor
    {
        return new CpuCustomTensor();
    }
    public function execute(string $operation, array $inputs, array $arguments = []): Tensor
    {
        return new CpuCustomTensor();
    }
}

var_dump(BackendManager::available() === ['cpu']);
$tensor = BackendManager::create('cpu', [1, 2, 3, 4], [2, 2]);
var_dump($tensor instanceof Tensor);
var_dump($tensor->getBackend());
var_dump($tensor->getShape());
var_dump($tensor->dtype());
var_dump($tensor->toArray());
var_dump(BackendManager::forTensor($tensor) === BackendManager::get('cpu'));
BackendManager::register(new CpuCustomBackend());
var_dump(BackendManager::available() === ['cpu', 'custom-cpu']);
$customTensor = BackendManager::create('custom-cpu', [7], [1]);
var_dump($customTensor->toArray());
var_dump(BackendManager::unregister('custom-cpu'));
var_dump(BackendManager::forTensor($customTensor) instanceof CpuCustomBackend);

$added = BackendManager::get('cpu')->execute('add', [$tensor, 2]);
var_dump($added->toArray());
var_dump($tensor->subtract(1)->multiply(2)->divide(2)->toArray());
var_dump($tensor->multiply($tensor)->toArray());

$host = new HostArray([[1, 2], [3, 4]], 'float64');
var_dump($host->getBackend(), $host->getShape(), $host->getNdims(), $host->getSize());
var_dump($host->getDtype(), $host->getElementSize(), $host->count(), $host->isPinned());
var_dump($host->get([1, 0]), $host->at(0, 1));
var_dump(HostArray::fromBuffer($host->toBuffer(), [2, 2], 'float64')->toArray());
var_dump(HostArray::zeros([2], 'float64')->toArray());
var_dump(HostArray::ones([2], 'float32')->toArray());

$serialized = serialize($host);
$restored = unserialize($serialized);
var_dump($restored instanceof HostArray, $restored->toArray(), $restored->dtype());

foreach ([
    static fn() => new HostArray([[1], [2, 3]]),
    static fn() => HostArray::fromFlatArray([1, 2], [3]),
    static fn() => $tensor->divide(0),
    static fn() => $tensor->add(new HostArray([1, 2])),
] as $invalidOperation) {
    try {
        $invalidOperation();
        var_dump(false);
    } catch (InvalidArgumentException) {
        var_dump(true);
    }
}

try {
    new HostArray([1], 'int32');
    var_dump(false);
} catch (InvalidArgumentException) {
    var_dump(true);
}

try {
    new HostArray([1], 'float32', true);
    var_dump(false);
} catch (RuntimeException) {
    var_dump(true);
}

try {
    new PhpTensors\Cuda\CudaArray([1]);
    var_dump(false);
} catch (RuntimeException) {
    var_dump(true);
}

try {
    PhpTensors\Cuda\CudaArray::ones([1]);
    var_dump(false);
} catch (RuntimeException) {
    var_dump(true);
}
?>
--EXPECT--
bool(true)
bool(true)
string(3) "cpu"
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(2)
}
string(7) "float32"
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1)
    [1]=>
    float(2)
  }
  [1]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(4)
  }
}
bool(true)
bool(true)
array(1) {
  [0]=>
  float(7)
}
bool(true)
bool(true)
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(4)
  }
  [1]=>
  array(2) {
    [0]=>
    float(5)
    [1]=>
    float(6)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(0)
    [1]=>
    float(1)
  }
  [1]=>
  array(2) {
    [0]=>
    float(2)
    [1]=>
    float(3)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1)
    [1]=>
    float(4)
  }
  [1]=>
  array(2) {
    [0]=>
    float(9)
    [1]=>
    float(16)
  }
}
string(3) "cpu"
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(2)
}
int(2)
int(4)
string(7) "float64"
int(8)
int(2)
bool(false)
float(3)
float(2)
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1)
    [1]=>
    float(2)
  }
  [1]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(4)
  }
}
array(2) {
  [0]=>
  float(0)
  [1]=>
  float(0)
}
array(2) {
  [0]=>
  float(1)
  [1]=>
  float(1)
}
bool(true)
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(1)
    [1]=>
    float(2)
  }
  [1]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(4)
  }
}
string(7) "float64"
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
