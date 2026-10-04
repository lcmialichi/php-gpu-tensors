<?php
declare(strict_types=1);

use Cuda\CudaArray;
use Cuda\Fusion;

final class DatasetManager
{
    public string $trainXBuffer = '';
    public string $trainYBuffer = '';
    public string $testXBuffer = '';
    public array $testTargets = [];
    public int $trainSamples = 0;
    public int $testSamples = 0;

    public function __construct(
        private string $datasetPath,
        private int $numClasses = 10,
        private int $inputFeatures = 64
    ) {}

    public function loadDataset(float $trainRatio = 0.8): void
    {
        if ($trainRatio <= 0 || $trainRatio >= 1) {
            throw new InvalidArgumentException('Training ratio must be between zero and one.');
        }
        if (!is_file($this->datasetPath)) {
            echo "Downloading Optdigits from UCI...\n";
            $context = stream_context_create(['http' => ['timeout' => 30]]);
            $data = file_get_contents(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
                false,
                $context
            );
            if ($data === false || file_put_contents($this->datasetPath, $data) !== strlen($data)) {
                throw new RuntimeException('Unable to download and save the dataset.');
            }
        }
        $file = fopen($this->datasetPath, 'rb');
        if ($file === false) {
            throw new RuntimeException('Unable to open the dataset.');
        }
        $samples = [];
        try {
            while (($line = fgets($file)) !== false) {
                $line = trim($line);
                if ($line === '') continue;
                $values = explode(',', $line);
                if (count($values) !== $this->inputFeatures + 1) {
                    throw new RuntimeException('Invalid Optdigits row width.');
                }
                $label = array_pop($values);
                if ($label === '' || strspn($label, '0123456789') !== strlen($label) || (int)$label >= $this->numClasses) {
                    throw new RuntimeException('Invalid Optdigits class label.');
                }
                $pixels = [];
                foreach ($values as $value) {
                    if ($value === '' || strspn($value, '0123456789') !== strlen($value) || (int)$value > 16) {
                        throw new RuntimeException('Invalid Optdigits pixel.');
                    }
                    $pixels[] = (int)$value / 16.0;
                }
                $target = (int)$label;
                $oneHot = array_fill(0, $this->numClasses, 0.0);
                $oneHot[$target] = 1.0;
                $samples[] = [pack('g*', ...$pixels), pack('g*', ...$oneHot), $target];
            }
            if (!feof($file)) {
                throw new RuntimeException('Unable to read the complete dataset.');
            }
        } finally {
            fclose($file);
        }
        if (count($samples) < 2) {
            throw new RuntimeException('The dataset must contain at least two samples.');
        }
        mt_srand(1337);
        shuffle($samples);
        $this->trainSamples = max(1, min(count($samples) - 1, (int)(count($samples) * $trainRatio)));
        $this->testSamples = count($samples) - $this->trainSamples;
        $this->trainXBuffer = $this->trainYBuffer = $this->testXBuffer = '';
        $this->testTargets = [];
        foreach ($samples as $index => [$pixels, $oneHot, $target]) {
            if ($index < $this->trainSamples) {
                $this->trainXBuffer .= $pixels;
                $this->trainYBuffer .= $oneHot;
            } else {
                $this->testXBuffer .= $pixels;
                $this->testTargets[] = $target;
            }
        }
        printf("Loaded %d training and %d test samples.\n", $this->trainSamples, $this->testSamples);
    }
}

final class NeuralNetwork
{
    private CudaArray $W1;
    private CudaArray $b1;
    private CudaArray $W2;
    private CudaArray $b2;
    private CudaArray $zero;
    private CudaArray $lower;
    private CudaArray $upper;

    public function __construct(
        private int $inputFeatures = 64,
        private int $hiddenNodes = 64,
        private int $numClasses = 10
    ) {
        $this->zero = CudaArray::zeros([1]);
        $this->lower = CudaArray::full([1], -5.0);
        $this->upper = CudaArray::full([1], 5.0);
        mt_srand(2026);
        $weights = static function (array $shape): CudaArray {
            $values = [];
            for ($i = 0, $count = array_product($shape); $i < $count; $i++) {
                $values[] = (mt_rand() / mt_getrandmax() * 2 - 1) * 0.1;
            }
            return CudaArray::fromBuffer(pack('g*', ...$values), $shape);
        };
        $this->W1 = $weights([$inputFeatures, $hiddenNodes]);
        $this->b1 = CudaArray::zeros([1, $hiddenNodes]);
        $this->W2 = $weights([$hiddenNodes, $numClasses]);
        $this->b2 = CudaArray::zeros([1, $numClasses]);
    }

    public function parameters(): array
    {
        return [$this->W1, $this->b1, $this->W2, $this->b2];
    }

    public function trainingExpression(int $rows, float $learningRate, bool $withLoss): Closure
    {
        if ($rows < 1 || !is_finite($learningRate) || $learningRate <= 0) {
            throw new InvalidArgumentException('Invalid training batch or learning rate.');
        }
        $zero = $this->zero;
        $lower = $this->lower;
        $upper = $this->upper;
        $hidden = $this->hiddenNodes;
        $classes = $this->numClasses;
        return static function ($x, $xt, $y, $w1, $b1, $w2, $b2) use (
            $rows, $learningRate, $withLoss, $zero, $lower, $upper, $hidden, $classes
        ): array {
            $z1 = $x->matmul($w1) + $b1;
            $a1 = CudaArray::where($z1->gt(0), $z1, $zero);
            $logits = $a1->matmul($w2) + $b2;
            $shifted = $logits - $logits->max(1)->reshape([$rows, 1]);
            $exponentials = $shifted->exp();
            $denominator = $exponentials->sum(1)->reshape([$rows, 1]);
            $dz2 = ($exponentials / $denominator - $y) * (1.0 / $rows);
            $dw2 = $a1->transpose()->matmul($dz2);
            $db2 = $dz2->sum(0)->reshape([1, $classes]);
            $da1 = $dz2->matmul($w2->transpose());
            $dz1 = CudaArray::where($a1->gt(0), $da1, $zero);
            $dw1 = $xt->matmul($dz1);
            $db1 = $dz1->sum(0)->reshape([1, $hidden]);
            $update = static function ($weight, $gradient) use ($learningRate, $lower, $upper): CudaArray {
                $clipped = CudaArray::where($gradient->gt(5), $upper, $gradient);
                $clipped = CudaArray::where($clipped->lt(-5), $lower, $clipped);
                return CudaArray::where($gradient->eq($gradient), $weight - $clipped * $learningRate, $weight);
            };
            $result = [
                $update($w1, $dw1), $update($b1, $db1),
                $update($w2, $dw2), $update($b2, $db2),
            ];
            if ($withLoss) {
                // Log-sum-exp avoids a probability clamp that changes the loss.
                $targetLogit = ($shifted * $y)->sum(1)->reshape([$rows, 1]);
                $result[] = ($denominator->log() - $targetLogit)->sum(0);
            }
            return $result;
        };
    }

    public function train(DatasetManager $dataset, int $epochs, int $batchSize, float $learningRate, bool $profile = false): void
    {
        if ($epochs < 1 || $batchSize < 1 || $dataset->trainSamples < 1 ||
            !is_finite($learningRate) || $learningRate <= 0) {
            throw new InvalidArgumentException('Training requires positive epochs, batch size, sample count and learning rate.');
        }
        $batches = [];
        $start = hrtime(true);
        for ($offset = 0; $offset < $dataset->trainSamples; $offset += $batchSize) {
            $rows = min($batchSize, $dataset->trainSamples - $offset);
            $x = CudaArray::fromBuffer(
                substr($dataset->trainXBuffer, $offset * $this->inputFeatures * 4, $rows * $this->inputFeatures * 4),
                [$rows, $this->inputFeatures]
            );
            $y = CudaArray::fromBuffer(
                substr($dataset->trainYBuffer, $offset * $this->numClasses * 4, $rows * $this->numClasses * 4),
                [$rows, $this->numClasses]
            );
            $batches[] = [$x, $x->transpose(), $y, $rows];
        }
        printf("One-time batch upload: %.2f ms\n", (hrtime(true) - $start) / 1e6);
        $dataset->trainXBuffer = $dataset->trainYBuffer = '';

        $plans = [];
        $start = hrtime(true);
        foreach ($batches as [$x, $xt, $y, $rows]) {
            if (isset($plans[$rows])) continue;
            $inputs = [$x, $xt, $y, ...$this->parameters()];
            $plans[$rows] = [
                Fusion::compile($this->trainingExpression($rows, $learningRate, false), $inputs),
                Fusion::compile($this->trainingExpression($rows, $learningRate, true), $inputs),
            ];
            $stats = $plans[$rows][0]->getStats();
            if ($profile) {
                $plans[$rows][0]->setProfiling(true);
                $plans[$rows][1]->setProfiling(true);
            }
            if ($stats['fusedKernels'] < 1) {
                throw new RuntimeException('Training did not produce any fused kernels.');
            }
            printf("Batch %d plan: %d fused kernels, %d native boundaries, %s backend.\n",
                $rows, $stats['fusedKernels'], $stats['boundaries'], $stats['backend']);
        }
        printf("Plan compilation: %.2f ms\n", (hrtime(true) - $start) / 1e6);
        echo "Training with compiled Fusion replay (no custom kernels)...\n";
        $start = hrtime(true);
        $blockStart = $start;
        $blockEpoch = 0;
        $steps = 0;
        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            $report = $epoch % 50 === 0 || $epoch === $epochs - 1;
            $loss = 0.0;
            foreach ($batches as [$x, $xt, $y, $rows]) {
                $outputs = $plans[$rows][(int)$report]->run($x, $xt, $y, ...$this->parameters());
                [$this->W1, $this->b1, $this->W2, $this->b2] = $outputs;
                if ($report) $loss += $outputs[4][0];
                unset($outputs);
                $steps++;
            }
            if ($report) {
                $loss /= $dataset->trainSamples;
                if (!is_finite($loss)) {
                    throw new RuntimeException('Training diverged: non-finite cross-entropy.');
                }
                printf("Epoch %4d: cross-entropy %.5f\n", $epoch + 1, $loss);
                $now = hrtime(true);
                printf("  Block %d-%d: %.3f ms/step\n", $blockEpoch + 1, $epoch + 1,
                    ($now - $blockStart) / 1e6 / (($epoch + 1 - $blockEpoch) * count($batches)));
                $blockStart = $now;
                $blockEpoch = $epoch + 1;
            }
        }
        $seconds = (hrtime(true) - $start) / 1e9;
        printf("Training: %.2f s, %.2f ms/step, %.0f samples/s (%d steps).\n",
            $seconds, $seconds * 1000 / $steps, $dataset->trainSamples * $epochs / $seconds, $steps);
        if ($profile) {
            foreach ($plans as $rows => $pair) {
                foreach ($pair as $withLoss => $plan) {
                    echo json_encode(['batch' => $rows, 'withLoss' => (bool)$withLoss,
                        'stats' => $plan->getStats()], JSON_THROW_ON_ERROR), PHP_EOL;
                }
            }
        }
    }

    public function evaluate(DatasetManager $dataset): float
    {
        $x = CudaArray::fromBuffer($dataset->testXBuffer, [$dataset->testSamples, $this->inputFeatures]);
        $zero = $this->zero;
        $plan = Fusion::compile(static function ($x, $w1, $b1, $w2, $b2) use ($zero) {
            $z1 = $x->matmul($w1) + $b1;
            return (CudaArray::where($z1->gt(0), $z1, $zero)->matmul($w2) + $b2)->argMax(1);
        }, [$x, ...$this->parameters()]);
        $predictions = $plan->run($x, ...$this->parameters())->toArray();
        $correct = 0;
        foreach ($predictions as $index => $prediction) {
            if ((int)$prediction === $dataset->testTargets[$index]) $correct++;
        }
        $accuracy = 100.0 * $correct / $dataset->testSamples;
        printf("Test accuracy: %.2f%% (%d/%d).\n", $accuracy, $correct, $dataset->testSamples);
        if ($accuracy < 80.0) {
            throw new RuntimeException(sprintf('Accuracy below the 80%% training check: %.2f%%.', $accuracy));
        }
        return $accuracy;
    }

    public function saveModel(string $path, string $version): void
    {
        $data = serialize(['version' => $version, 'parameters' => $this->parameters()]);
        if (file_put_contents($path, $data) !== strlen($data)) {
            throw new RuntimeException('Unable to save trained parameters.');
        }
        echo "Model saved: $path\n";
    }

    public function loadModel(string $path, string $version): void
    {
        $data = file_get_contents($path);
        if ($data === false) throw new RuntimeException('Unable to read the saved model.');
        $model = unserialize($data, ['allowed_classes' => [CudaArray::class]]);
        $parameters = is_array($model) ? ($model['parameters'] ?? null) : null;
        if (($model['version'] ?? null) !== $version || !is_array($parameters) || count($parameters) !== 4) {
            throw new RuntimeException('Saved model version or parameters are incompatible.');
        }
        $shapes = [[$this->inputFeatures, $this->hiddenNodes], [1, $this->hiddenNodes],
            [$this->hiddenNodes, $this->numClasses], [1, $this->numClasses]];
        foreach (array_values($parameters) as $index => $parameter) {
            if (!$parameter instanceof CudaArray || $parameter->dtype() !== 'float32' ||
                $parameter->getShape() !== $shapes[$index]) {
                throw new RuntimeException('Saved parameter dtype or shape is incompatible.');
            }
        }
        [$this->W1, $this->b1, $this->W2, $this->b2] = array_values($parameters);
    }
}

function runFusedTraining(): void
{
    if (!extension_loaded('cuda') || cuda_get_device_count() < 1) {
        throw new RuntimeException('Training requires the CUDA extension and a visible NVIDIA GPU.');
    }
    $options = getopt('', ['epochs:', 'batch-size:', 'learning-rate:', 'load-model', 'no-save', 'profile']);
    $epochs = filter_var($options['epochs'] ?? 1000, FILTER_VALIDATE_INT);
    $batchSize = filter_var($options['batch-size'] ?? 256, FILTER_VALIDATE_INT);
    $learningRate = filter_var($options['learning-rate'] ?? 0.05, FILTER_VALIDATE_FLOAT);
    if ($epochs === false || $epochs < 1 || $batchSize === false || $batchSize < 1 ||
        $learningRate === false || !is_finite($learningRate) || $learningRate <= 0) {
        throw new InvalidArgumentException('epochs, batch-size and learning-rate must be positive numbers.');
    }
    $modelPath = __DIR__ . DIRECTORY_SEPARATOR . 'trained_model_stable.dat';
    $modelVersion = 'fusion-softmax-ce-v2-64x64x10';
    $dataset = new DatasetManager(__DIR__ . DIRECTORY_SEPARATOR . 'optdigits.csv');
    $dataset->loadDataset();
    $network = new NeuralNetwork();
    if (array_key_exists('load-model', $options)) {
        $network->loadModel($modelPath, $modelVersion);
    } else {
        $network->train($dataset, $epochs, $batchSize, $learningRate, array_key_exists('profile', $options));
    }
    $network->evaluate($dataset);
    if (!array_key_exists('load-model', $options) && !array_key_exists('no-save', $options)) {
        $network->saveModel($modelPath, $modelVersion);
    }
    echo "Real Fusion training check passed.\n";
}

if (realpath($_SERVER['SCRIPT_FILENAME'] ?? '') === __FILE__) {
    try {
        runFusedTraining();
    } catch (Throwable $error) {
        fwrite(STDERR, 'Error: ' . $error->getMessage() . PHP_EOL);
        exit(1);
    }
}
