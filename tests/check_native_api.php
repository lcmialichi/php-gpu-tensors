<?php
declare(strict_types=1);

function nativeApiType(?ReflectionType $type, ?string $scope): ?string
{
    if ($type === null) {
        return null;
    }
    if ($type instanceof ReflectionUnionType) {
        $types = array_map(static fn(ReflectionType $part): string => nativeApiType($part, $scope), $type->getTypes());
        sort($types);
        return implode('|', $types);
    }
    if (!$type instanceof ReflectionNamedType) {
        throw new RuntimeException('Unsupported API type: ' . $type);
    }
    $name = $type->getName();
    if ($name === 'self') {
        $name = $scope;
    }
    if ($type->allowsNull() && $name !== 'mixed' && $name !== 'null') {
        $types = [$name, 'null'];
        sort($types);
        return implode('|', $types);
    }
    return $name;
}

function nativeApiSignature(ReflectionFunctionAbstract $function): array
{
    $scope = $function instanceof ReflectionMethod ? $function->getDeclaringClass()->getName() : null;
    return [
        'required' => $function->getNumberOfRequiredParameters(),
        'returnsReference' => $function->returnsReference(),
        'returnType' => nativeApiType($function->getReturnType(), $scope),
        'parameters' => array_map(static fn(ReflectionParameter $parameter): array => [
            'name' => $parameter->getName(),
            'type' => nativeApiType($parameter->getType(), $scope),
            'byReference' => $parameter->isPassedByReference(),
            'variadic' => $parameter->isVariadic(),
            'optional' => $parameter->isOptional(),
            'hasDefault' => $parameter->isDefaultValueAvailable(),
            'default' => $parameter->isDefaultValueAvailable() ? $parameter->getDefaultValue() : null,
        ], $function->getParameters()),
    ];
}

function nativeApiSurface(array $functions, array $fixtureClasses = []): array
{
    $classes = [];
    $aliases = [];
    foreach (array_merge(get_declared_classes(), get_declared_interfaces()) as $name) {
        if (!str_starts_with($name, 'PhpTensors\\') && !in_array($name, $fixtureClasses, true)) {
            continue;
        }
        $class = new ReflectionClass($name);
        if (strcasecmp($name, $class->getName()) !== 0) {
            $aliases[$name] = $class->getName();
            continue;
        }
        $methods = [];
        foreach ($class->getMethods() as $method) {
            if ($method->getDeclaringClass()->getName() !== $name) {
                continue;
            }
            $methods[$method->getName()] = [
                'visibility' => $method->isPublic() ? 'public' : ($method->isProtected() ? 'protected' : 'private'),
                'static' => $method->isStatic(),
                'final' => $method->isFinal(),
                'abstract' => $method->isAbstract(),
            ] + nativeApiSignature($method);
        }
        ksort($methods);
        $interfaces = $class->getInterfaceNames();
        sort($interfaces);
        $parent = $class->getParentClass();
        $classes[$name] = [
            'interface' => $class->isInterface(),
            'parent' => $parent ? $parent->getName() : null,
            'interfaces' => $interfaces,
            'abstract' => $class->isAbstract(),
            'final' => $class->isFinal(),
            'methods' => $methods,
        ];
    }
    ksort($classes);
    ksort($aliases);
    $signatures = [];
    foreach ($functions as $name) {
        $signatures[$name] = nativeApiSignature(new ReflectionFunction($name));
    }
    ksort($signatures);
    return ['classes' => $classes, 'aliases' => $aliases, 'functions' => $signatures];
}

function nativeApiDifferences(mixed $expected, mixed $actual, string $path = ''): array
{
    if (!is_array($expected) || !is_array($actual)) {
        return $expected === $actual ? [] : [
            $path . ': expected ' . json_encode($expected, JSON_THROW_ON_ERROR | JSON_PRESERVE_ZERO_FRACTION)
                . ', native ' . json_encode($actual, JSON_THROW_ON_ERROR | JSON_PRESERVE_ZERO_FRACTION),
        ];
    }
    $differences = [];
    foreach ($expected as $key => $value) {
        $child = $path === '' ? (string)$key : $path . '.' . $key;
        if (!array_key_exists($key, $actual)) {
            $differences[] = $child . ': missing from native API';
        } else {
            array_push($differences, ...nativeApiDifferences($value, $actual[$key], $child));
        }
    }
    foreach (array_diff_key($actual, $expected) as $key => $_) {
        $differences[] = $path . '.' . $key . ': not declared in stubs';
    }
    return $differences;
}

if (realpath($_SERVER['SCRIPT_FILENAME']) !== __FILE__) {
    return;
}

$mode = $argv[1] ?? '';
if ($mode === '--dump-stubs') {
    if (count($argv) !== 3 || !in_array($argv[2], ['--cpu-only', '--cuda'], true)) {
        fwrite(STDERR, "Stub reflection needs an explicit --cpu-only or --cuda profile.\n");
        exit(2);
    }
    $before = get_defined_functions()['user'];
    require __DIR__ . '/../stubs/cuda.stub.php';
    require __DIR__ . '/../stubs/cuda_methods.stub.php';
    require __DIR__ . '/fixtures/cpu_cuda_array.stub.php';
    $fixtureClass = PhpTensorsTestFixtures\UnavailableCudaArray::class;
    $surface = nativeApiSurface(
        array_values(array_diff(get_defined_functions()['user'], $before)),
        [$fixtureClass]
    );
    if (($argv[2] ?? '') === '--cpu-only') {
        foreach (['NN', 'Fusion', 'FusionGraph', 'FusionExecution', 'Compiler', 'CompiledModule', 'Optimizer', 'CudaArray'] as $class) {
            unset($surface['classes']['PhpTensors\\' . $class]);
        }
        $surface['classes']['PhpTensors\\CudaArray'] = $surface['classes'][$fixtureClass];
        $surface['functions'] = [];
    }
    unset($surface['classes'][$fixtureClass]);
    ksort($surface['classes']);
    echo json_encode($surface, JSON_THROW_ON_ERROR | JSON_PRESERVE_ZERO_FRACTION);
    exit(0);
}

if (!in_array($mode, ['--cpu-only', '--cuda'], true) || count($argv) !== 2) {
    fwrite(STDERR, "Usage: php -n -d extension=... tests/check_native_api.php --cpu-only|--cuda\n");
    exit(2);
}
if (!extension_loaded('phptensors')) {
    fwrite(STDERR, "Native API comparison requires the loaded phptensors extension.\n");
    exit(1);
}
$process = proc_open([PHP_BINARY, '-n', __FILE__, '--dump-stubs', $mode],
    [0 => ['pipe', 'r'], 1 => ['pipe', 'w'], 2 => STDERR], $pipes);
if (!is_resource($process)) {
    throw new RuntimeException('Unable to start isolated stub reflection.');
}
fclose($pipes[0]);
$json = stream_get_contents($pipes[1]);
fclose($pipes[1]);
$status = proc_close($process);
if ($status !== 0 || $json === false) {
    throw new RuntimeException('Isolated stub reflection failed.');
}
$expected = json_decode($json, true, 512, JSON_THROW_ON_ERROR);
$extension = new ReflectionExtension('phptensors');
$actual = nativeApiSurface(array_keys($extension->getFunctions()));
$differences = nativeApiDifferences($expected, $actual);
if ($differences !== []) {
    fwrite(STDERR, "Native/stub API mismatch ($mode):\n" . implode("\n", $differences) . "\n");
    exit(1);
}
echo 'Native/stub API matches (' . $mode . ', ' . count($actual['classes']) . " classes/interfaces).\n";
