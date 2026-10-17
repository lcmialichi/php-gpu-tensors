<?php
declare(strict_types=1);

$package = json_decode(
    file_get_contents(__DIR__ . '/../composer.json'),
    true,
    512,
    JSON_THROW_ON_ERROR
);

$configureOptions = $package['php-ext']['configure-options'] ?? [];
$cudaOption = null;
foreach ($configureOptions as $option) {
    if (($option['name'] ?? null) === 'with-cuda') {
        $cudaOption = $option;
        break;
    }
}

if (
    ($package['name'] ?? null) !== 'lcmialichi/php-gpu-tensors' ||
    ($package['type'] ?? null) !== 'php-ext' ||
    ($package['php-ext']['extension-name'] ?? null) !== 'phptensors' ||
    ($package['php-ext']['support-zts'] ?? null) !== true ||
    ($package['php-ext']['os-families'] ?? null) !== ['linux'] ||
    ($cudaOption['needs-value'] ?? false) !== true
) {
    fwrite(STDERR, "PIE package metadata is incomplete or inconsistent.\n");
    exit(1);
}

echo "PIE package metadata is valid.\n";