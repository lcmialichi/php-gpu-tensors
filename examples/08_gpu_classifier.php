<?php
declare(strict_types=1);

require_once __DIR__ . '/Support/ModelConfiguration.php';
require_once __DIR__ . '/Support/functions.php';
require_once __DIR__ . '/Support/model_functions.php';
require_once __DIR__ . '/Support/training_functions.php';
require_once __DIR__ . '/Support/ClassifierRunner.php';

$config = [
    'dataset' => 'mnist',
    'hiddenLayers' => [512, 256],
    'activation' => 'relu',
    'epochs' => 40,
    'batchSize' => 128,
    'optimizer' => 'adam',
    'learningRate' => null,
    'seed' => 2026,
];

if (realpath($_SERVER['SCRIPT_FILENAME'] ?? '') === __FILE__) {
    try {
        exit(runClassifier($config));
    } catch (Throwable $error) {
        fwrite(STDERR, C_RED . C_BLD . 'Error: ' . C_RST . $error->getMessage() . PHP_EOL);
        exit(2);
    }
}
