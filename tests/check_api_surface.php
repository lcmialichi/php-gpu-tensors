<?php
declare(strict_types=1);

$functionsBeforeStubs = get_defined_functions()['user'];

$stubDirectory = __DIR__ . '/../stubs';
if (!is_dir($stubDirectory)) {
    $stubDirectory = __DIR__ . '/../../stubs';
}
if (!is_dir($stubDirectory)) {
    fwrite(STDERR, "Unable to locate the PHP API stubs.\n");
    exit(1);
}

require $stubDirectory . '/cuda.stub.php';
require $stubDirectory . '/cuda_methods.stub.php';

function apiType(?ReflectionType $type): ?string
{
    return $type === null ? null : (string)$type;
}

function apiParameters(ReflectionFunctionAbstract $function): array
{
    return array_map(static function (ReflectionParameter $parameter): array {
        return [
            'name' => $parameter->getName(),
            'type' => apiType($parameter->getType()),
            'byReference' => $parameter->isPassedByReference(),
            'variadic' => $parameter->isVariadic(),
            'optional' => $parameter->isOptional(),
            'hasDefault' => $parameter->isDefaultValueAvailable(),
            'default' => $parameter->isDefaultValueAvailable() ? $parameter->getDefaultValue() : null,
        ];
    }, $function->getParameters());
}

function publicApiSurface(array $stubFunctions): array
{
    $classes = [];
    foreach (get_declared_classes() as $className) {
        if (!str_starts_with($className, 'PhpTensors\\') &&
            !str_starts_with($className, 'PhpTensors\\')) {
            continue;
        }

        $class = new ReflectionClass($className);
        $methods = [];
        foreach ($class->getMethods(ReflectionMethod::IS_PUBLIC) as $method) {
            if ($method->getDeclaringClass()->getName() !== $className) {
                continue;
            }

            $methods[$method->getName()] = [
                'static' => $method->isStatic(),
                'abstract' => $method->isAbstract(),
                'returnsReference' => $method->returnsReference(),
                'parameters' => apiParameters($method),
                'returnType' => apiType($method->getReturnType()),
            ];
        }
        ksort($methods);

        $parent = $class->getParentClass();
        $classes[$className] = [
            'parent' => $parent ? $parent->getName() : null,
            'abstract' => $class->isAbstract(),
            'final' => $class->isFinal(),
            'methods' => $methods,
        ];
    }
    ksort($classes);

    $functions = [];
    foreach ($stubFunctions as $functionName) {
        $function = new ReflectionFunction($functionName);
        $functions[$functionName] = [
            'parameters' => apiParameters($function),
            'returnType' => apiType($function->getReturnType()),
        ];
    }
    ksort($functions);

    $interfaces = [];
    foreach (get_declared_interfaces() as $interfaceName) {
        if (!str_starts_with($interfaceName, 'PhpTensors\\')) {
            continue;
        }

        $interface = new ReflectionClass($interfaceName);
        $methods = [];
        foreach ($interface->getMethods() as $method) {
            $methods[$method->getName()] = [
                'static' => $method->isStatic(),
                'abstract' => $method->isAbstract(),
                'returnsReference' => $method->returnsReference(),
                'parameters' => apiParameters($method),
                'returnType' => apiType($method->getReturnType()),
            ];
        }
        ksort($methods);
        $interfaces[$interfaceName] = ['methods' => $methods];
    }
    ksort($interfaces);

    $aliases = [];
    foreach (get_declared_classes() as $className) {
        if (str_starts_with($className, 'PhpTensors\\')) {
            $aliases[$className] = (new ReflectionClass($className))->getName();
        }
    }
    ksort($aliases);

    return [
        'classes' => $classes,
        'interfaces' => $interfaces,
        'aliases' => $aliases,
        'functions' => $functions,
    ];
}

$normalizeSelfTypes = static function (array $surface): array {
    foreach ($surface['classes'] as $className => &$class) {
        foreach ($class['methods'] as &$method) {
            if ($method['returnType'] === $className) {
                $method['returnType'] = 'self';
            }

            foreach ($method['parameters'] as &$parameter) {
                if ($parameter['type'] === $className) {
                    $parameter['type'] = 'self';
                }
            }
            unset($parameter);
        }
        unset($method);
    }
    unset($class);

    foreach ($surface['interfaces'] as $interfaceName => &$interface) {
        foreach ($interface['methods'] as &$method) {
            if ($method['returnType'] === $interfaceName) {
                $method['returnType'] = 'self';
            }
            foreach ($method['parameters'] as &$parameter) {
                if ($parameter['type'] === $interfaceName) {
                    $parameter['type'] = 'self';
                }
            }
            unset($parameter);
        }
        unset($method);
    }
    unset($interface);

    return $surface;
};

$baselinePath = __DIR__ . '/api_surface.json';
$stubFunctions = array_values(array_diff(get_defined_functions()['user'], $functionsBeforeStubs));
sort($stubFunctions);
$actual = json_encode(
    $normalizeSelfTypes(publicApiSurface($stubFunctions)),
    JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR
) . PHP_EOL;

if (($argv[1] ?? '') === '--update') {
    if (file_put_contents($baselinePath, $actual) === false) {
        fwrite(STDERR, "Unable to update API baseline.\n");
        exit(1);
    }
    echo "Updated frozen API baseline.\n";
    exit(0);
}

if (isset($argv[1])) {
    fwrite(STDERR, "Usage: php tests/check_api_surface.php [--update]\n");
    exit(2);
}

$expected = file_get_contents($baselinePath);
if ($expected === false) {
    fwrite(STDERR, "Unable to read frozen PHP API baseline.\n");
    exit(1);
}

$expectedSurface = json_decode($expected, true, 512, JSON_THROW_ON_ERROR);
$expected = json_encode(
    $normalizeSelfTypes($expectedSurface),
    JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR
) . PHP_EOL;

if (!hash_equals($expected, $actual)) {
    fwrite(STDERR, "Public PHP API differs from tests/api_surface.json. Review the change and update the baseline only when approved.\n");
    exit(1);
}

echo "Frozen PHP API surface matches the baseline.\n";