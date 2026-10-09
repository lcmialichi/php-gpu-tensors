<?php
declare(strict_types=1);

use Cuda\CudaArray;
use Cuda\Optimizer;

/**
 * -------------------------------------------------------------------------
 * Returns the number of affine layers in the configured network.
 * -------------------------------------------------------------------------
 */
function modelLayerCount(ModelConfiguration $configuration): int
{
    return count($configuration->dimensions) - 1;
}

/**
 * -------------------------------------------------------------------------
 * Returns the number of weight and bias tensors in the model.
 * -------------------------------------------------------------------------
 */
function modelParameterCount(ModelConfiguration $configuration): int
{
    return 2 * modelLayerCount($configuration);
}

/**
 * -------------------------------------------------------------------------
 * Reports whether the configured optimizer keeps Adam moment estimates.
 * -------------------------------------------------------------------------
 */
function modelUsesAdam(ModelConfiguration $configuration): bool
{
    return $configuration->optimizer === 'adam';
}

function createModelOptimizer(ModelConfiguration $configuration): Optimizer
{
    if (modelUsesAdam($configuration)) {
        return Optimizer::adamW(
            learningRate: 0.001,
            beta1: $configuration->beta1,
            beta2: $configuration->beta2,
            weightDecay: $configuration->weightDecay
        );
    }
    return Optimizer::sgd(
        learningRate: 0.02,
        momentum: $configuration->momentum,
        weightDecay: $configuration->weightDecay
    );
}

/**
 * -------------------------------------------------------------------------
 * Counts scalar weights and biases across all model layers.
 * -------------------------------------------------------------------------
 */
function modelParameterElements(ModelConfiguration $configuration): int
{
    $total = 0;
    for ($layer = 0; $layer < modelLayerCount($configuration); $layer++) {
        $inputUnits = $configuration->dimensions[$layer];
        $outputUnits = $configuration->dimensions[$layer + 1];
        $total += $inputUnits * $outputUnits + $outputUnits;
    }
    return $total;
}

/**
 * -------------------------------------------------------------------------
 * Estimates multiply-add work for one forward and backward training step.
 * -------------------------------------------------------------------------
 */
function modelStepFlops(ModelConfiguration $configuration, int $rows): float
{
    $flops = 0.0;
    for ($layer = 0; $layer < modelLayerCount($configuration); $layer++) {
        $product = (float)$rows * $configuration->dimensions[$layer] * $configuration->dimensions[$layer + 1];
        $flops += 2.0 * $product * ($layer > 0 ? 3 : 2);
    }
    return $flops;
}

/**
 * -------------------------------------------------------------------------
 * Computes the positional callback arity needed to capture one train step.
 * -------------------------------------------------------------------------
 */
function modelTrainArity(ModelConfiguration $configuration): int
{
    $parameters = modelParameterCount($configuration);
    return modelUsesAdam($configuration) ? 6 + 3 * $parameters : 4 + 2 * $parameters;
}

/**
 * -------------------------------------------------------------------------
 * Computes the positional callback arity needed by evaluation plans.
 * -------------------------------------------------------------------------
 */
function modelEvaluationArity(ModelConfiguration $configuration): int
{
    return 2 + modelParameterCount($configuration);
}

/**
 * -------------------------------------------------------------------------
 * Returns the expected native boundaries for the captured train-step graph.
 * -------------------------------------------------------------------------
 */
function modelExpectedBoundaries(ModelConfiguration $configuration, bool $withLoss): int
{
    $layers = modelLayerCount($configuration);
    return 4 * $layers + 4 + ($withLoss ? 1 : 0);
}

/**
 * -------------------------------------------------------------------------
 * Creates a closure with positional parameters for Fusion capture.
 * -------------------------------------------------------------------------
 */
function fusionPositionalCallback(Closure $body, int $arity): Closure
{
    $parameters = [];
    for ($index = 0; $index < $arity; $index++) {
        $parameters[] = '$input' . $index;
    }
    $arguments = implode(', ', $parameters);
    $code = 'return static function (' . $arguments . ') use ($body) { return $body([' . $arguments . ']); };';
    return eval($code);
}

/**
 * -------------------------------------------------------------------------
 * Creates float32 weight buffers using He/Xavier initialization.
 * -------------------------------------------------------------------------
 *
 * @return list<string>
 */
function initializeModelBuffers(ModelConfiguration $configuration, int $seed): array
{
    $gaussian = gaussianGenerator($seed);
    $buffers = [];
    for ($layer = 0; $layer < modelLayerCount($configuration); $layer++) {
        $inputUnits = $configuration->dimensions[$layer];
        $outputUnits = $configuration->dimensions[$layer + 1];
        $scale = sqrt(($layer === modelLayerCount($configuration) - 1 ? 1.0 : 2.0) / $inputUnits);
        $weights = [];
        for ($index = 0, $count = $inputUnits * $outputUnits; $index < $count; $index++) {
            $weights[] = $gaussian() * $scale;
        }
        $buffers[] = packFloats($weights);
        $buffers[] = str_repeat("\0", 4 * $outputUnits);
    }
    return $buffers;
}

/**
 * -------------------------------------------------------------------------
 * Applies the configured hidden-layer activation to pre-activation values.
 * -------------------------------------------------------------------------
 */
function activateLayer(CudaArray $values, string $activation, CudaArray $zero): CudaArray
{
    if ($activation === 'leaky') {
        return CudaArray::where($values->gt(0), $values, $values * LEAKY_SLOPE);
    }
    return CudaArray::where($values->gt(0), $values, $zero);
}

/**
 * -------------------------------------------------------------------------
 * Applies the derivative of the configured activation to a gradient tensor.
 * -------------------------------------------------------------------------
 */
/**
 * -------------------------------------------------------------------------
 * Builds the forward, backward, optimizer, and optional loss graph for a batch.
 * -------------------------------------------------------------------------
 *
 * @param array{zero:CudaArray,one:CudaArray,upper:CudaArray,lower:CudaArray} $constants
 * @return Closure(list<CudaArray>): list<CudaArray>
 */
function trainingBody(
    ModelConfiguration $configuration,
    int $rows,
    bool $withLoss,
    array $constants
): Closure {
    $dimensions = $configuration->dimensions;
    $layers = modelLayerCount($configuration);
    $parameterCount = modelParameterCount($configuration);
    $adam = modelUsesAdam($configuration);
    $activation = $configuration->activation;
    $weightDecay = $configuration->weightDecay;
    $clip = $configuration->gradientClip;
    $labelSmoothing = $configuration->labelSmoothing;
    $classes = $dimensions[$layers];
    $zero = $constants['zero'];
    $one = $constants['one'];
    $upper = $constants['upper'];
    $lower = $constants['lower'];
    $optimizer = createModelOptimizer($configuration);
    $weightDecayMask = [];
    for ($index = 0; $index < $parameterCount; $index++) {
        $weightDecayMask[] = $index % 2 === 0;
    }

    /**
     * @param list<CudaArray> $inputs Batch tensors, parameters, and optimizer state.
     * @return list<CudaArray> Updated parameters/state and optional scalar loss.
     */
    return static function (array $inputs) use (
        $dimensions, $layers, $parameterCount, $adam, $activation, $clip,
        $labelSmoothing, $classes, $zero, $one, $upper, $lower, $rows, $withLoss,
        $optimizer, $weightDecayMask
    ): array {
        $features = $inputs[0];
        $transposedFeatures = $inputs[1];
        $labels = $inputs[2];
        $learningRate = $inputs[3];
        $offset = 4;
        $beta1Power = null;
        $beta2Power = null;
        if ($adam) {
            $beta1Power = $inputs[$offset];
            $beta2Power = $inputs[$offset + 1];
            $offset += 2;
        }
        $parameters = array_slice($inputs, $offset, $parameterCount);
        $offset += $parameterCount;
        $firstMoments = array_slice($inputs, $offset, $parameterCount);
        $offset += $parameterCount;
        $secondMoments = $adam ? array_slice($inputs, $offset, $parameterCount) : [];
        $optimizerState = $adam
            ? [
                'firstMoment' => $firstMoments,
                'secondMoment' => $secondMoments,
                'beta1Power' => $beta1Power,
                'beta2Power' => $beta2Power,
            ]
            : ['velocity' => $firstMoments];

        $optimizer->zeroGrad($parameters);
        foreach ($parameters as $parameter) {
            $parameter->requiresGrad();
        }

        /** @var list<CudaArray> $activations */
        $activations = [$features];
        $logits = $features;
        for ($layer = 0; $layer < $layers; $layer++) {
            $preActivation = $activations[$layer]->matmul($parameters[2 * $layer])
                + $parameters[2 * $layer + 1];
            if ($layer < $layers - 1) {
                $activations[$layer + 1] = activateLayer($preActivation, $activation, $zero);
            } else {
                $logits = $preActivation;
            }
        }

        /** @var CudaArray $shifted */
        $shifted = $logits - $logits->max(1)->reshape([$rows, 1]);
        $exponentials = $shifted->exp();
        $denominator = $exponentials->sum(1)->reshape([$rows, 1]);
        /** @var CudaArray $targets */
        $targets = $labelSmoothing > 0.0
            ? $labels * (1.0 - $labelSmoothing) + ($labelSmoothing / $classes)
            : $labels;
        $targetLogit = ($shifted * $targets)->sum(1)->reshape([$rows, 1]);
        $loss = ($denominator->log() - $targetLogit)->sum(0);
        $loss->backward($one / $rows);
        /** @var list<CudaArray> $gradients */
        $gradients = array_map(static fn(CudaArray $parameter): CudaArray => $parameter->grad(), $parameters);
        foreach ($gradients as $index => $parameterGradient) {
            $parameterGradient = CudaArray::where($parameterGradient->gt($clip), $upper, $parameterGradient);
            $parameterGradient = CudaArray::where($parameterGradient->lt(-$clip), $lower, $parameterGradient);
            $gradients[$index] = $parameterGradient;
        }
        $updated = $optimizer->step($parameters, $gradients, $optimizerState, $learningRate, $weightDecayMask);
        $updatedParameters = $updated['parameters'];
        $updatedState = $updated['state'];

        $updatedFirstMoments = $adam ? $updatedState['firstMoment'] : $updatedState['velocity'];
        $outputs = array_merge($updatedParameters, $updatedFirstMoments);
        if ($adam) {
            $outputs = array_merge(
                $outputs,
                $updatedState['secondMoment'],
                [$updatedState['beta1Power'], $updatedState['beta2Power']]
            );
        }
        if ($withLoss) {
            $outputs[] = $loss;
        }
        return array_values($outputs);
    };
}

/**
 * -------------------------------------------------------------------------
 * Builds the forward and cross-entropy evaluation graph for a fixed row count.
 * -------------------------------------------------------------------------
 *
 * @param array{zero:CudaArray,one:CudaArray,upper:CudaArray,lower:CudaArray} $constants
 * @return Closure(list<CudaArray>): array{CudaArray,CudaArray,CudaArray}
 */
function evaluationBody(ModelConfiguration $configuration, int $rows, array $constants): Closure
{
    $layers = modelLayerCount($configuration);
    $activation = $configuration->activation;
    $zero = $constants['zero'];

    /**
     * @param list<CudaArray> $inputs Features, labels, and parameters.
     * @return array{CudaArray,CudaArray,CudaArray} Predictions, loss, and logits.
     */
    return static function (array $inputs) use ($layers, $activation, $zero, $rows): array {
        $features = $inputs[0];
        $labels = $inputs[1];
        $parameters = array_slice($inputs, 2);
        $activations = $features;
        $logits = $features;
        for ($layer = 0; $layer < $layers; $layer++) {
            $preActivation = $activations->matmul($parameters[2 * $layer])
                + $parameters[2 * $layer + 1];
            if ($layer < $layers - 1) {
                $activations = activateLayer($preActivation, $activation, $zero);
            } else {
                $logits = $preActivation;
            }
        }
        /** @var CudaArray $shifted */
        $shifted = $logits - $logits->max(1)->reshape([$rows, 1]);
        $denominator = $shifted->exp()->sum(1)->reshape([$rows, 1]);
        $targetLogit = ($shifted * $labels)->sum(1)->reshape([$rows, 1]);
        $loss = ($denominator->log() - $targetLogit)->sum(0);
        return [$logits->argMax(1), $loss, $logits];
    };
}

/**
 * -------------------------------------------------------------------------
 * Builds the softmax probability graph used for single-row inference.
 * -------------------------------------------------------------------------
 *
 * @param array{zero:CudaArray,one:CudaArray,upper:CudaArray,lower:CudaArray} $constants
 * @return Closure(list<CudaArray>): array{CudaArray}
 */
function probabilityBody(ModelConfiguration $configuration, array $constants): Closure
{
    $layers = modelLayerCount($configuration);
    $activation = $configuration->activation;
    $zero = $constants['zero'];

    /**
     * @param list<CudaArray> $inputs Features and parameters.
     * @return array{CudaArray} Class probabilities.
     */
    return static function (array $inputs) use ($layers, $activation, $zero): array {
        $activations = $inputs[0];
        $parameters = array_slice($inputs, 1);
        $logits = $activations;
        for ($layer = 0; $layer < $layers; $layer++) {
            $preActivation = $activations->matmul($parameters[2 * $layer])
                + $parameters[2 * $layer + 1];
            if ($layer < $layers - 1) {
                $activations = activateLayer($preActivation, $activation, $zero);
            } else {
                $logits = $preActivation;
            }
        }
        $exponentials = ($logits - $logits->max(1)->reshape([1, 1]))->exp();
        $probabilities = $exponentials / $exponentials->sum(1)->reshape([1, 1]);
        return [$probabilities];
    };
}
