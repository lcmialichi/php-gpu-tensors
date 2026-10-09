<?php
declare(strict_types=1);

use Cuda\CudaArray;

/**
 * -------------------------------------------------------------------------
 * Creates model parameters and optimizer tensors from packed float buffers.
 * -------------------------------------------------------------------------
 *
 * @param list<string> $buffers
 * @return array{
 *   params:list<CudaArray>,
 *   firstMoments:list<CudaArray>,
 *   secondMoments:list<CudaArray>,
 *   beta1Power:?CudaArray,
 *   beta2Power:?CudaArray
 * }
 */
function createTrainState(ModelConfiguration $configuration, array $buffers): array
{
    $state = [
        'params' => [],
        'firstMoments' => [],
        'secondMoments' => [],
        'beta1Power' => null,
        'beta2Power' => null,
    ];
    for ($layer = 0; $layer < modelLayerCount($configuration); $layer++) {
        $inputUnits = $configuration->dimensions[$layer];
        $outputUnits = $configuration->dimensions[$layer + 1];
        $state['params'][] = CudaArray::fromBuffer($buffers[2 * $layer], [$inputUnits, $outputUnits]);
        $state['params'][] = CudaArray::fromBuffer($buffers[2 * $layer + 1], [1, $outputUnits]);
    }
    $optimizerState = createModelOptimizer($configuration)->initState($state['params']);
    if (modelUsesAdam($configuration)) {
        $state['firstMoments'] = $optimizerState['firstMoment'];
        $state['secondMoments'] = $optimizerState['secondMoment'];
        $state['beta1Power'] = $optimizerState['beta1Power'];
        $state['beta2Power'] = $optimizerState['beta2Power'];
    } else {
        $state['firstMoments'] = $optimizerState['velocity'];
    }
    return $state;
}

/**
 * -------------------------------------------------------------------------
 * Packs one training batch and the current optimizer state for graph replay.
 * -------------------------------------------------------------------------
 *
 * @param array{params:list<CudaArray>,firstMoments:list<CudaArray>,secondMoments:list<CudaArray>,beta1Power:?CudaArray,beta2Power:?CudaArray} $state
 * @return list<CudaArray>
 */
function trainStateInputs(
    array $state,
    CudaArray $features,
    CudaArray $transposedFeatures,
    CudaArray $labels,
    CudaArray $learningRate
): array {
    $inputs = [$features, $transposedFeatures, $labels, $learningRate];
    if ($state['beta1Power'] !== null) {
        $inputs[] = $state['beta1Power'];
        $inputs[] = $state['beta2Power'];
    }
    return array_merge($inputs, $state['params'], $state['firstMoments'], $state['secondMoments']);
}

/**
 * -------------------------------------------------------------------------
 * Replaces optimizer state with tensors returned by a compiled train step.
 * -------------------------------------------------------------------------
 *
 * @param array{params:list<CudaArray>,firstMoments:list<CudaArray>,secondMoments:list<CudaArray>,beta1Power:?CudaArray,beta2Power:?CudaArray} $state
 * @param list<CudaArray> $outputs
 * @return list<CudaArray> Additional outputs such as the batch loss.
 */
function applyTrainOutputs(array &$state, array $outputs, ModelConfiguration $configuration): array
{
    $outputs = array_values($outputs);
    $parameterCount = modelParameterCount($configuration);
    $offset = 0;
    $state['params'] = array_slice($outputs, $offset, $parameterCount);
    $offset += $parameterCount;
    $state['firstMoments'] = array_slice($outputs, $offset, $parameterCount);
    $offset += $parameterCount;
    if (modelUsesAdam($configuration)) {
        $state['secondMoments'] = array_slice($outputs, $offset, $parameterCount);
        $offset += $parameterCount;
        $state['beta1Power'] = $outputs[$offset];
        $state['beta2Power'] = $outputs[$offset + 1];
        $offset += 2;
    }
    return array_slice($outputs, $offset);
}

/**
 * -------------------------------------------------------------------------
 * Adds a self-test result and prints it with the matching terminal color.
 * -------------------------------------------------------------------------
 *
 * @param list<array{status:string,name:string,detail:string}> $checks
 */
function addCheck(array &$checks, string $status, string $name, string $detail): void
{
    $checks[] = ['status' => $status, 'name' => $name, 'detail' => $detail];
    $color = match ($status) {
        'PASS' => C_GRN,
        'FAIL' => C_RED,
        'WARN' => C_YLW,
        default => C_RST,
    };
    printf("  [%s%-4s%s] %s: %s\n", $color, $status, C_RST, $name, $detail);
}

/**
 * -------------------------------------------------------------------------
 * Counts failed self-test records.
 * -------------------------------------------------------------------------
 *
 * @param list<array{status:string,name:string,detail:string}> $checks
 */
function checkFailureCount(array $checks): int
{
    return count(array_filter($checks, static fn(array $check): bool => $check['status'] === 'FAIL'));
}

/**
 * -------------------------------------------------------------------------
 * Counts warning self-test records.
 * -------------------------------------------------------------------------
 *
 * @param list<array{status:string,name:string,detail:string}> $checks
 */
function checkWarningCount(array $checks): int
{
    return count(array_filter($checks, static fn(array $check): bool => $check['status'] === 'WARN'));
}

/**
 * -------------------------------------------------------------------------
 * Returns a reproducible Gaussian random-number generator closure.
 * -------------------------------------------------------------------------
 */
function gaussianGenerator(int $seed): Closure
{
    mt_srand($seed);
    $spare = null;
    return static function () use (&$spare): float {
        if ($spare !== null) {
            $value = $spare;
            $spare = null;
            return $value;
        }
        $radius = sqrt(-2.0 * log((mt_rand() + 0.5) / (mt_getrandmax() + 1.0)));
        $angle = 2.0 * M_PI * ((mt_rand() + 0.5) / (mt_getrandmax() + 1.0));
        $spare = $radius * sin($angle);
        return $radius * cos($angle);
    };
}
