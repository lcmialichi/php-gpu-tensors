<?php
declare(strict_types=1);

if (extension_loaded('phptensors')) {
    throw new RuntimeException('Generate arginfo without loading phptensors (use php -n).');
}
if (count($argv) > 2 || (isset($argv[1]) && $argv[1] !== '--check')) {
    fwrite(STDERR, "Usage: php -n tests/generate_arginfo.php [--check]\n");
    exit(2);
}
$before = get_defined_functions()['user'];
require __DIR__ . '/../stubs/cuda.stub.php';
require __DIR__ . '/../stubs/cuda_methods.stub.php';
require __DIR__ . '/fixtures/cpu_cuda_array.stub.php';

function arginfoName(string $class, string $method): string
{
    return 'arginfo_api_' . str_replace('\\', '_', $class) . '_' . $method;
}

function arginfoClass(string $name, ?string $scope): string
{
    return str_replace('\\', '\\\\', $name === 'self' ? $scope : $name);
}

function arginfoMask(ReflectionType $type, ?string $scope): array
{
    $builtins = [
        'int' => 'MAY_BE_LONG', 'float' => 'MAY_BE_DOUBLE', 'bool' => 'MAY_BE_BOOL',
        'string' => 'MAY_BE_STRING', 'array' => 'MAY_BE_ARRAY', 'object' => 'MAY_BE_OBJECT',
        'callable' => 'MAY_BE_CALLABLE', 'null' => 'MAY_BE_NULL', 'mixed' => 'MAY_BE_ANY',
    ];
    $parts = $type instanceof ReflectionUnionType ? $type->getTypes() : [$type];
    $mask = [];
    $class = null;
    foreach ($parts as $part) {
        if (!$part instanceof ReflectionNamedType) {
            throw new RuntimeException('Unsupported arginfo type: ' . $part);
        }
        if (isset($builtins[$part->getName()])) {
            $mask[] = $builtins[$part->getName()];
        } elseif (!$part->isBuiltin()) {
            if ($class !== null) {
                throw new RuntimeException('Multiple object classes require a Zend union type: ' . $type);
            }
            $class = arginfoClass($part->getName(), $scope);
        } else {
            throw new RuntimeException('Unsupported arginfo mask type: ' . $type);
        }
    }
    if ($type->allowsNull() && !in_array('MAY_BE_ANY', $mask, true)) {
        $mask[] = 'MAY_BE_NULL';
    }
    return [$class, implode(' | ', array_unique($mask))];
}

function arginfoDefault(ReflectionParameter $parameter): string
{
    if (!$parameter->isDefaultValueAvailable()) {
        return 'NULL';
    }
    $value = $parameter->isDefaultValueConstant()
        ? $parameter->getDefaultValueConstantName()
        : var_export($parameter->getDefaultValue(), true);
    return json_encode($value, JSON_THROW_ON_ERROR | JSON_UNESCAPED_SLASHES);
}

function generateArginfo(ReflectionFunctionAbstract $function, string $name): string
{
    $scope = $function instanceof ReflectionMethod ? $function->getDeclaringClass()->getName() : null;
    $required = $function->getNumberOfRequiredParameters();
    $reference = (int)$function->returnsReference();
    $type = $function->getReturnType();
    $codes = [
        'int' => 'IS_LONG', 'float' => 'IS_DOUBLE', 'bool' => '_IS_BOOL',
        'string' => 'IS_STRING', 'array' => 'IS_ARRAY', 'object' => 'IS_OBJECT',
        'callable' => 'IS_CALLABLE', 'mixed' => 'IS_MIXED', 'void' => 'IS_VOID', 'static' => 'IS_STATIC',
    ];
    if ($type === null) {
        $text = "ZEND_BEGIN_ARG_INFO_EX($name, 0, $reference, $required)\n";
    } elseif ($type instanceof ReflectionUnionType) {
        [$class, $mask] = arginfoMask($type, $scope);
        if ($class !== null) {
            throw new RuntimeException('Object return union requires explicit support: ' . $type);
        }
        $text = "ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX($name, $reference, $required, $mask)\n";
    } elseif ($type instanceof ReflectionNamedType) {
        $nullable = (int)$type->allowsNull();
        if (isset($codes[$type->getName()])) {
            $code = $codes[$type->getName()];
            $text = "ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX($name, $reference, $required, $code, $nullable)\n";
        } else {
            $class = arginfoClass($type->getName(), $scope);
            $text = "ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX($name, $reference, $required, $class, $nullable)\n";
        }
    } else {
        throw new RuntimeException('Unsupported return type: ' . $type);
    }
    foreach ($function->getParameters() as $parameter) {
        $parameterName = $parameter->getName();
        $type = $parameter->getType();
        $reference = (int)$parameter->isPassedByReference();
        $variadic = (int)$parameter->isVariadic();
        $default = arginfoDefault($parameter);
        if ($type === null) {
            throw new RuntimeException('Stub parameter needs an explicit type: ' . $function->getName() . '::$' . $parameterName);
        }
        [$class, $mask] = arginfoMask($type, $scope);
        if ($variadic) {
            if ($class !== null) {
                if ($mask !== '') {
                    throw new RuntimeException('Unsupported variadic object union: ' . $function->getName());
                }
                $text .= "    ZEND_ARG_VARIADIC_OBJ_INFO($reference, $parameterName, $class, 0)\n";
            } else {
                $text .= "    { \"$parameterName\", ZEND_TYPE_INIT_MASK($mask | _ZEND_ARG_INFO_FLAGS($reference, 1, 0)), NULL },\n";
            }
        } elseif ($class !== null) {
            $text .= "    ZEND_ARG_OBJ_TYPE_MASK($reference, $parameterName, $class, " . ($mask === '' ? '0' : $mask) . ", $default)\n";
        } else {
            $text .= "    ZEND_ARG_TYPE_MASK($reference, $parameterName, $mask, $default)\n";
        }
    }
    return $text . "ZEND_END_ARG_INFO()\n\n";
}

$text = "/* Generated from PHP API stubs by tests/generate_arginfo.php. Do not edit. */\n"
    . "#ifndef PHPTENSORS_API_ARGINFO_H\n#define PHPTENSORS_API_ARGINFO_H\n\n#include \"php.h\"\n\n";
$classes = array_merge(get_declared_classes(), get_declared_interfaces());
sort($classes);
foreach ($classes as $name) {
    if (!str_starts_with($name, 'PhpTensors\\')
        && $name !== PhpTensorsTestFixtures\UnavailableCudaArray::class) {
        continue;
    }
    $class = new ReflectionClass($name);
    $methods = $class->getMethods();
    usort($methods, static fn(ReflectionMethod $a, ReflectionMethod $b): int => strcmp($a->getName(), $b->getName()));
    foreach ($methods as $method) {
        if ($method->getDeclaringClass()->getName() === $name) {
            $text .= generateArginfo($method, arginfoName($name, $method->getName()));
        }
    }
}
$functions = array_diff(get_defined_functions()['user'], $before);
sort($functions);
foreach ($functions as $name) {
    $text .= generateArginfo(new ReflectionFunction($name), 'arginfo_api_' . $name);
}
$text .= "#endif\n";
$path = __DIR__ . '/../src/api_arginfo.h';
if (($argv[1] ?? '') === '--check') {
    if (!is_file($path) || file_get_contents($path) !== $text) {
        fwrite(STDERR, "Generated arginfo is stale. Run php -n tests/generate_arginfo.php.\n");
        exit(1);
    }
    echo "Generated arginfo matches the authoritative stubs.\n";
} else {
    if (file_put_contents($path, $text) !== strlen($text)) {
        throw new RuntimeException('Unable to write generated arginfo.');
    }
    echo "Generated native signature metadata.\n";
}
