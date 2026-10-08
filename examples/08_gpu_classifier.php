<?php
declare(strict_types=1);

require_once __DIR__ . '/Support/ModelConfiguration.php';
require_once __DIR__ . '/Support/functions.php';
require_once __DIR__ . '/Support/model_functions.php';
require_once __DIR__ . '/Support/training_functions.php';
require_once __DIR__ . '/Support/ClassifierRunner.php';

if (realpath($_SERVER['SCRIPT_FILENAME'] ?? '') === __FILE__) {
    try {
        exit(runClassifier());
    } catch (Throwable $error) {
        fwrite(STDERR, C_RED . C_BLD . 'Error: ' . C_RST . $error->getMessage() . PHP_EOL);
        exit(2);
    }
}
