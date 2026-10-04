<?php
declare(strict_types=1);

/**
 * gpu_classifier.php: A pure-PHP Deep Learning Multilayer Perceptron (MLP) Classifier.
 * 
 * This script serves as a complete example of using the `php-cuda-ext` extension to train 
 * neural networks directly on the GPU without relying on Python or external machine learning 
 * libraries. It demonstrates tensor operations, Just-In-Time (JIT) CUDA kernel fusion, 
 * mathematical optimization (AdamW/SGD), and pure PHP data orchestration.
 *
 * Supported Datasets:
 *   --dataset=mnist    Handwritten digits (60,000 train / 10,000 test, 28x28)
 *   --dataset=fashion  Fashion-MNIST clothing images (same layout)
 *   --dataset=csv      Any numeric CSV (--csv=file --header --label-col=-1)
 */

use Cuda\CudaArray;
use Cuda\Fusion;

// ANSI Color Codes for CLI output
const C_RST = "\033[0m";
const C_BLD = "\033[1m";
const C_RED = "\033[31m";
const C_GRN = "\033[32m";
const C_YLW = "\033[33m";
const C_BLU = "\033[34m";
const C_MAG = "\033[35m";
const C_CYN = "\033[36m";

const LEAKY_SLOPE = 0.01;
const MODEL_MAGIC = "PGTMLP1\n";

const IMAGE_DATASETS = [
    'mnist' => [
        'mean' => 0.1307,
        'std' => 0.3081,
        'classes' => ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'mirrors' => [
            'https://storage.googleapis.com/cvdf-datasets/mnist/',
            'https://ossci-datasets.s3.amazonaws.com/mnist/',
            'https://yann.lecun.com/exdb/mnist/',
        ],
    ],
    'fashion' => [
        'mean' => 0.2860,
        'std' => 0.3530,
        'classes' => ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'mirrors' => [
            'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
        ],
    ],
];

const IDX_FILES = [
    'trainImages' => 'train-images-idx3-ubyte.gz',
    'trainLabels' => 'train-labels-idx1-ubyte.gz',
    'testImages' => 't10k-images-idx3-ubyte.gz',
    'testLabels' => 't10k-labels-idx1-ubyte.gz',
];

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

final class Rng
{
    private ?float $spare = null;

    public function __construct(int $seed)
    {
        mt_srand($seed);
    }

    public function uniform(): float
    {
        return (mt_rand() + 0.5) / (mt_getrandmax() + 1.0);
    }

    public function gauss(): float
    {
        if ($this->spare !== null) {
            $value = $this->spare;
            $this->spare = null;
            return $value;
        }
        $radius = sqrt(-2.0 * log($this->uniform()));
        $angle = 2.0 * M_PI * $this->uniform();
        $this->spare = $radius * sin($angle);
        return $radius * cos($angle);
    }
}

function packFloats(array $values): string
{
    $out = '';
    foreach (array_chunk($values, 4096) as $chunk) {
        $out .= pack('g*', ...$chunk);
    }
    return $out;
}

function scalarTensor(float $value): CudaArray
{
    return CudaArray::fromBuffer(pack('g', $value), [1]);
}

function flattenValues($value): array
{
    if (!is_array($value)) {
        return [(float)$value];
    }
    $out = [];
    array_walk_recursive($value, static function ($item) use (&$out): void {
        $out[] = (float)$item;
    });
    return $out;
}

function compareTensors(CudaArray $a, CudaArray $b, float $atol, float $rtol): array
{
    $fa = flattenValues($a->toArray());
    $fb = flattenValues($b->toArray());
    if (count($fa) !== count($fb)) {
        return [INF, false];
    }
    $worst = 0.0;
    $ok = true;
    foreach ($fa as $i => $va) {
        $vb = $fb[$i];
        if (is_nan($va) || is_nan($vb)) {
            if (is_nan($va) !== is_nan($vb)) {
                $ok = false;
            }
            continue;
        }
        $diff = abs($va - $vb);
        if ($diff > $atol + $rtol * abs($vb)) {
            $ok = false;
        }
        if ($diff > $worst) {
            $worst = $diff;
        }
    }
    return [$worst, $ok];
}

final class Checks
{
    public array $items = [];

    public function add(string $status, string $name, string $detail): void
    {
        $this->items[] = ['status' => $status, 'name' => $name, 'detail' => $detail];
        $color = match ($status) {
            'PASS' => C_GRN,
            'FAIL' => C_RED,
            'WARN' => C_YLW,
            default => C_RST
        };
        printf("  [%s%-4s%s] %s: %s\n", $color, $status, C_RST, $name, $detail);
    }

    public function failures(): int
    {
        return count(array_filter($this->items, static fn(array $i): bool => $i['status'] === 'FAIL'));
    }

    public function warnings(): int
    {
        return count(array_filter($this->items, static fn(array $i): bool => $i['status'] === 'WARN'));
    }
}

// ---------------------------------------------------------------------------
// Image datasets: download, IDX parsing, normalization, augmentation
// ---------------------------------------------------------------------------

function normalizationTable(float $mean, float $std): array
{
    $table = [];
    for ($b = 0; $b < 256; $b++) {
        $table[chr($b)] = pack('g', ($b / 255.0 - $mean) / $std);
    }
    return $table;
}

function gunzipFile(string $path): string
{
    $data = file_get_contents($path);
    if ($data === false) {
        throw new RuntimeException("Unable to read $path.");
    }
    if (function_exists('gzdecode')) {
        $raw = @gzdecode($data);
        if ($raw === false) {
            throw new RuntimeException("$path is not a valid gzip file (delete it and run again).");
        }
        return $raw;
    }
    $process = proc_open(['gzip', '-dc', $path], [1 => ['pipe', 'w'], 2 => ['pipe', 'w']], $pipes);
    if (!is_resource($process)) {
        throw new RuntimeException('PHP has no zlib support and the gzip command is unavailable.');
    }
    $raw = stream_get_contents($pipes[1]);
    $error = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    if (proc_close($process) !== 0 || $raw === false) {
        throw new RuntimeException("gzip failed for $path: $error");
    }
    return $raw;
}

function ensureFile(string $dir, string $file, array $mirrors, ?string $override): string
{
    $path = $dir . DIRECTORY_SEPARATOR . $file;
    if (is_file($path) && filesize($path) > 0) {
        return $path;
    }
    if (!is_dir($dir) && !mkdir($dir, 0775, true) && !is_dir($dir)) {
        throw new RuntimeException("Unable to create $dir.");
    }
    $bases = $override !== null ? array_merge([$override], $mirrors) : $mirrors;
    $context = stream_context_create(['http' => [
        'timeout' => 60,
        'follow_location' => 1,
        'max_redirects' => 5,
        'header' => "User-Agent: php-gpu-tensors\r\n",
    ]]);
    $errors = [];
    foreach ($bases as $base) {
        $url = rtrim($base, '/') . '/' . $file;
        $tmp = $path . '.part';
        printf("  downloading %s\n", $url);
        if (@copy($url, $tmp, $context)) {
            $head = (string)@file_get_contents($tmp, false, null, 0, 2);
            if ($head === "\x1f\x8b") {
                rename($tmp, $path);
                return $path;
            }
            $errors[] = "$url: the response is not a gzip file";
        } else {
            $last = error_get_last();
            $errors[] = $url . ': ' . ($last['message'] ?? 'download failed');
        }
        @unlink($tmp);
    }
    throw new RuntimeException(
        "Could not download $file. Tried:\n  " . implode("\n  ", $errors)
        . "\nDownload the four IDX .gz files by hand into $dir, or pass --url-base=<mirror directory>."
    );
}

function parseIdx(string $raw, string $what): array
{
    if (strlen($raw) < 8) {
        throw new RuntimeException("$what: file too short.");
    }
    $magic = unpack('N', substr($raw, 0, 4))[1];
    if (($magic >> 16) !== 0 || (($magic >> 8) & 0xFF) !== 0x08) {
        throw new RuntimeException(sprintf('%s: not an unsigned-byte IDX file (magic 0x%08X).', $what, $magic));
    }
    $count = $magic & 0xFF;
    $dims = [];
    for ($i = 0; $i < $count; $i++) {
        $dims[] = unpack('N', substr($raw, 4 + 4 * $i, 4))[1];
    }
    $offset = 4 + 4 * $count;
    if (strlen($raw) - $offset !== array_product($dims)) {
        throw new RuntimeException("$what: the size does not match the header (truncated download?).");
    }
    return [$dims, $offset];
}

function shiftImage(string $row, int $h, int $w, int $dx, int $dy, string $bg): string
{
    if ($dx === 0 && $dy === 0) {
        return $row;
    }
    $stride = 4 * $w;
    if ($dy !== 0) {
        $cut = abs($dy) * $stride;
        $row = $dy > 0
            ? str_repeat($bg, $w * $dy) . substr($row, 0, $h * $stride - $cut)
            : substr($row, $cut) . str_repeat($bg, $w * -$dy);
    }
    if ($dx !== 0) {
        $pad = str_repeat($bg, abs($dx));
        $padBytes = 4 * abs($dx);
        $keep = $stride - $padBytes;
        $out = '';
        for ($y = 0; $y < $h; $y++) {
            $line = substr($row, $y * $stride, $stride);
            $out .= $dx > 0 ? $pad . substr($line, 0, $keep) : substr($line, $padBytes) . $pad;
        }
        $row = $out;
    }
    return $row;
}

function loadImageDataset(string $name, string $dir, ?string $override, int $valCount, int $seed): array
{
    $def = IMAGE_DATASETS[$name];
    $table = normalizationTable($def['mean'], $def['std']);
    $raw = [];
    foreach (IDX_FILES as $key => $file) {
        $raw[$key] = gunzipFile(ensureFile($dir, $file, $def['mirrors'], $override));
    }
    [$trainDims, $trainOffset] = parseIdx($raw['trainImages'], 'train images');
    [$trainLabelDims, $trainLabelOffset] = parseIdx($raw['trainLabels'], 'train labels');
    [$testDims, $testOffset] = parseIdx($raw['testImages'], 'test images');
    [$testLabelDims, $testLabelOffset] = parseIdx($raw['testLabels'], 'test labels');
    if (count($trainDims) !== 3 || count($testDims) !== 3 || $trainDims[1] !== $testDims[1] || $trainDims[2] !== $testDims[2]) {
        throw new RuntimeException('Unexpected image dimensions in the IDX files.');
    }
    if ($trainDims[0] !== $trainLabelDims[0] || $testDims[0] !== $testLabelDims[0]) {
        throw new RuntimeException('Image and label counts differ.');
    }
    [$trainCount, $h, $w] = $trainDims;
    $testCount = $testDims[0];
    $pixels = $h * $w;
    $classes = count($def['classes']);

    $labelsAll = array_values(unpack('C*', substr($raw['trainLabels'], $trainLabelOffset)));
    $testLabels = array_values(unpack('C*', substr($raw['testLabels'], $testLabelOffset)));
    foreach ([$labelsAll, $testLabels] as $set) {
        if ($set !== [] && max($set) >= $classes) {
            throw new RuntimeException('A label is outside the expected class range.');
        }
    }
    if ($valCount < 1 || $valCount >= $trainCount) {
        throw new InvalidArgumentException('--val must be between 1 and the training set size minus one.');
    }

    $trainRowsAll = [];
    for ($i = 0; $i < $trainCount; $i++) {
        $trainRowsAll[] = strtr(substr($raw['trainImages'], $trainOffset + $i * $pixels, $pixels), $table);
    }
    $testRows = [];
    for ($i = 0; $i < $testCount; $i++) {
        $testRows[] = strtr(substr($raw['testImages'], $testOffset + $i * $pixels, $pixels), $table);
    }
    unset($raw);

    $order = range(0, $trainCount - 1);
    mt_srand($seed);
    shuffle($order);
    $train = ['rows' => [], 'labels' => []];
    $val = ['rows' => [], 'labels' => []];
    foreach ($order as $position => $index) {
        $target = $position < $valCount ? 'val' : 'train';
        $$target['rows'][] = $trainRowsAll[$index];
        $$target['labels'][] = $labelsAll[$index];
    }
    unset($trainRowsAll);

    return [
        'name' => $name,
        'features' => $pixels,
        'classes' => $classes,
        'classNames' => $def['classes'],
        'image' => ['h' => $h, 'w' => $w, 'mean' => $def['mean'], 'std' => $def['std'], 'bg' => $table["\0"], 'table' => $table],
        'train' => $train,
        'val' => $val,
        'test' => ['rows' => $testRows, 'labels' => $testLabels],
    ];
}

// ---------------------------------------------------------------------------
// CSV datasets
// ---------------------------------------------------------------------------

function loadCsvDataset(string $path, bool $header, int $labelCol, float $valFraction, float $testFraction, int $seed): array
{
    $file = @fopen($path, 'rb');
    if ($file === false) {
        throw new RuntimeException("Unable to open $path.");
    }
    $features = [];
    $labels = [];
    $labelIndex = null;
    $width = null;
    $line = 0;
    try {
        while (($fields = fgetcsv($file, 0, ',', '"', '\\')) !== false) {
            $line++;
            if ($fields === [null] || $fields === []) {
                continue;
            }
            if ($header && $line === 1) {
                continue;
            }
            if ($width === null) {
                $width = count($fields);
                if ($width < 2) {
                    throw new RuntimeException('The CSV needs at least one feature column and a label column.');
                }
                $labelIndex = $labelCol >= 0 ? $labelCol : $width + $labelCol;
                if ($labelIndex < 0 || $labelIndex >= $width) {
                    throw new InvalidArgumentException('--label-col is outside the CSV width.');
                }
            }
            if (count($fields) !== $width) {
                throw new RuntimeException("CSV line $line has " . count($fields) . " fields, expected $width.");
            }
            $row = [];
            foreach ($fields as $i => $value) {
                if ($i === $labelIndex) {
                    $labels[] = trim((string)$value);
                    continue;
                }
                if (!is_numeric($value)) {
                    throw new RuntimeException("CSV line $line, column " . ($i + 1) . ': non-numeric feature value "' . $value . '".');
                }
                $row[] = (float)$value;
            }
            $features[] = $row;
        }
    } finally {
        fclose($file);
    }
    $n = count($features);
    if ($n < 50) {
        throw new RuntimeException('The CSV needs at least 50 data rows.');
    }
    $names = array_values(array_unique($labels));
    $allNumeric = count(array_filter($names, 'is_numeric')) === count($names);
    sort($names, $allNumeric ? SORT_NUMERIC : SORT_STRING);
    $classIndex = array_flip(array_map('strval', $names));
    $classes = count($names);
    if ($classes < 2) {
        throw new RuntimeException('The CSV has a single class.');
    }
    $featureCount = count($features[0]);

    $order = range(0, $n - 1);
    mt_srand($seed);
    shuffle($order);
    $nTest = max(1, (int)floor($n * $testFraction));
    $nVal = max(1, (int)floor($n * $valFraction));
    if ($nTest + $nVal >= $n) {
        throw new InvalidArgumentException('Validation and test fractions leave no training data.');
    }
    $trainIdx = array_slice($order, $nTest + $nVal);

    $mean = array_fill(0, $featureCount, 0.0);
    foreach ($trainIdx as $i) {
        foreach ($features[$i] as $j => $v) {
            $mean[$j] += $v;
        }
    }
    foreach ($mean as $j => $sum) {
        $mean[$j] = $sum / count($trainIdx);
    }
    $std = array_fill(0, $featureCount, 0.0);
    foreach ($trainIdx as $i) {
        foreach ($features[$i] as $j => $v) {
            $std[$j] += ($v - $mean[$j]) ** 2;
        }
    }
    foreach ($std as $j => $sum) {
        $s = sqrt($sum / count($trainIdx));
        $std[$j] = $s < 1e-9 ? 1.0 : $s;
    }

    $split = ['test' => array_slice($order, 0, $nTest), 'val' => array_slice($order, $nTest, $nVal), 'train' => $trainIdx];
    $out = [];
    foreach ($split as $key => $indices) {
        $rows = [];
        $ys = [];
        foreach ($indices as $i) {
            $row = [];
            foreach ($features[$i] as $j => $v) {
                $row[] = ($v - $mean[$j]) / $std[$j];
            }
            $rows[] = pack('g*', ...$row);
            $ys[] = $classIndex[$labels[$i]];
        }
        $out[$key] = ['rows' => $rows, 'labels' => $ys];
    }
    return [
        'name' => 'csv:' . basename($path),
        'features' => $featureCount,
        'classes' => $classes,
        'classNames' => array_map('strval', $names),
        'image' => null,
        'train' => $out['train'],
        'val' => $out['val'],
        'test' => $out['test'],
    ];
}

// ---------------------------------------------------------------------------
// Batches and evaluation tensors
// ---------------------------------------------------------------------------

function oneHotTemplates(int $classes): array
{
    $templates = [];
    for ($k = 0; $k < $classes; $k++) {
        $row = array_fill(0, $classes, 0.0);
        $row[$k] = 1.0;
        $templates[$k] = pack('g*', ...$row);
    }
    return $templates;
}

function buildBatches(array $data, int $batchSize, int $seed, int $shift): array
{
    $rows = $data['train']['rows'];
    $labels = $data['train']['labels'];
    $n = count($rows);
    $templates = oneHotTemplates($data['classes']);
    $image = $data['image'];
    $order = range(0, $n - 1);
    mt_srand($seed);
    shuffle($order);
    $batches = [];
    for ($offset = 0; $offset < $n; $offset += $batchSize) {
        $slice = array_slice($order, $offset, $batchSize);
        $count = count($slice);
        $xs = '';
        $ys = '';
        foreach ($slice as $index) {
            $row = $rows[$index];
            if ($shift > 0 && $image !== null) {
                $row = shiftImage($row, $image['h'], $image['w'], mt_rand(-$shift, $shift), mt_rand(-$shift, $shift), $image['bg']);
            }
            $xs .= $row;
            $ys .= $templates[$labels[$index]];
        }
        $x = CudaArray::fromBuffer($xs, [$count, $data['features']]);
        $y = CudaArray::fromBuffer($ys, [$count, $data['classes']]);
        $batches[] = [$x, $x->transpose(), $y, $count];
    }
    return $batches;
}

function evalTensors(array $split, int $features, int $classes): array
{
    $templates = oneHotTemplates($classes);
    $n = count($split['rows']);
    $ys = '';
    foreach ($split['labels'] as $label) {
        $ys .= $templates[$label];
    }
    return [
        CudaArray::fromBuffer(implode('', $split['rows']), [$n, $features]),
        CudaArray::fromBuffer($ys, [$n, $classes]),
        $split['labels'],
    ];
}

// ---------------------------------------------------------------------------
// Model description and the compiled step expressions
// ---------------------------------------------------------------------------

final class ModelSpec
{
    public function __construct(
        public array $dims,
        public string $activation,
        public string $optimizer,
        public float $weightDecay,
        public float $clip,
        public float $labelSmoothing,
        public float $momentum = 0.9,
        public float $beta1 = 0.9,
        public float $beta2 = 0.999
    ) {
    }

    public function layers(): int
    {
        return count($this->dims) - 1;
    }

    public function paramCount(): int
    {
        return 2 * $this->layers();
    }

    public function isAdam(): bool
    {
        return $this->optimizer === 'adam';
    }

    public function parameterElements(): int
    {
        $total = 0;
        for ($l = 0; $l < $this->layers(); $l++) {
            $total += $this->dims[$l] * $this->dims[$l + 1] + $this->dims[$l + 1];
        }
        return $total;
    }

    public function stepFlops(int $rows): float
    {
        $flops = 0.0;
        for ($l = 0; $l < $this->layers(); $l++) {
            $product = (float)$rows * $this->dims[$l] * $this->dims[$l + 1];
            $flops += 2.0 * $product * ($l > 0 ? 3 : 2);
        }
        return $flops;
    }

    public function trainArity(): int
    {
        $np = $this->paramCount();
        return $this->isAdam() ? 4 + 2 + 3 * $np : 4 + 2 * $np;
    }

    public function evalArity(): int
    {
        return 2 + $this->paramCount();
    }

    public function expectedBoundaries(bool $withLoss): int
    {
        $layers = $this->layers();
        return (3 * $layers - 1) + ($layers + 2) + ($withLoss ? 2 : 0);
    }
}

final class TrainState
{
    public array $params = [];
    public array $m = [];
    public array $v = [];
    public ?CudaArray $b1t = null;
    public ?CudaArray $b2t = null;

    public static function create(ModelSpec $spec, array $buffers): self
    {
        $state = new self();
        for ($l = 0; $l < $spec->layers(); $l++) {
            $in = $spec->dims[$l];
            $out = $spec->dims[$l + 1];
            $state->params[] = CudaArray::fromBuffer($buffers[2 * $l], [$in, $out]);
            $state->params[] = CudaArray::fromBuffer($buffers[2 * $l + 1], [1, $out]);
        }
        foreach ($state->params as $param) {
            $shape = $param->getShape();
            $state->m[] = CudaArray::zeros($shape);
            if ($spec->isAdam()) {
                $state->v[] = CudaArray::zeros($shape);
            }
        }
        if ($spec->isAdam()) {
            $state->b1t = scalarTensor(1.0);
            $state->b2t = scalarTensor(1.0);
        }
        return $state;
    }

    public function inputs(CudaArray $x, CudaArray $xt, CudaArray $y, CudaArray $lr): array
    {
        $in = [$x, $xt, $y, $lr];
        if ($this->b1t !== null) {
            $in[] = $this->b1t;
            $in[] = $this->b2t;
        }
        foreach ($this->params as $t) {
            $in[] = $t;
        }
        foreach ($this->m as $t) {
            $in[] = $t;
        }
        foreach ($this->v as $t) {
            $in[] = $t;
        }
        return $in;
    }

    public function apply(array $outputs, ModelSpec $spec): array
    {
        $o = array_values($outputs);
        $np = $spec->paramCount();
        $c = 0;
        $this->params = array_slice($o, $c, $np);
        $c += $np;
        $this->m = array_slice($o, $c, $np);
        $c += $np;
        if ($spec->isAdam()) {
            $this->v = array_slice($o, $c, $np);
            $c += $np;
            $this->b1t = $o[$c];
            $this->b2t = $o[$c + 1];
            $c += 2;
        }
        return array_slice($o, $c);
    }
}

final class Net
{
    public static function initBuffers(ModelSpec $spec, int $seed): array
    {
        $rng = new Rng($seed);
        $buffers = [];
        $layers = $spec->layers();
        for ($l = 0; $l < $layers; $l++) {
            $in = $spec->dims[$l];
            $out = $spec->dims[$l + 1];
            $scale = sqrt(($l === $layers - 1 ? 1.0 : 2.0) / $in);
            $w = [];
            for ($i = 0, $n = $in * $out; $i < $n; $i++) {
                $w[] = $rng->gauss() * $scale;
            }
            $buffers[] = packFloats($w);
            $buffers[] = str_repeat("\0", 4 * $out);
        }
        return $buffers;
    }

    public static function positional(Closure $body, int $arity): Closure
    {
        $names = [];
        for ($i = 0; $i < $arity; $i++) {
            $names[] = '$t' . $i;
        }
        $list = implode(', ', $names);
        $code = 'return static function (' . $list . ') use ($body) { return $body([' . $list . ']); };';
        return eval($code);
    }

    private static function activate($z, string $kind, $zero)
    {
        if ($kind === 'leaky') {
            return CudaArray::where($z->gt(0), $z, $z * LEAKY_SLOPE);
        }
        return CudaArray::where($z->gt(0), $z, $zero);
    }

    private static function activateGrad($z, $da, string $kind, $zero)
    {
        if ($kind === 'leaky') {
            return CudaArray::where($z->gt(0), $da, $da * LEAKY_SLOPE);
        }
        return CudaArray::where($z->gt(0), $da, $zero);
    }

    public static function trainBody(ModelSpec $spec, int $rows, bool $withLoss, array $k): Closure
    {
        $dims = $spec->dims;
        $layers = $spec->layers();
        $np = $spec->paramCount();
        $adam = $spec->isAdam();
        $kind = $spec->activation;
        $wd = $spec->weightDecay;
        $clip = $spec->clip;
        $ls = $spec->labelSmoothing;
        $mu = $spec->momentum;
        $beta1 = $spec->beta1;
        $beta2 = $spec->beta2;
        $classes = $dims[$layers];
        $zero = $k['zero'];
        $one = $k['one'];
        $upper = $k['upper'];
        $lower = $k['lower'];

        return static function (array $in) use (
            $dims, $layers, $np, $adam, $kind, $wd, $clip, $ls, $mu, $beta1, $beta2,
            $classes, $zero, $one, $upper, $lower, $rows, $withLoss
        ): array {
            $x = $in[0];
            $xt = $in[1];
            $y = $in[2];
            $lr = $in[3];
            $c = 4;
            $b1t = null;
            $b2t = null;
            if ($adam) {
                $b1t = $in[$c];
                $b2t = $in[$c + 1];
                $c += 2;
            }
            $p = array_slice($in, $c, $np);
            $c += $np;
            $m = array_slice($in, $c, $np);
            $c += $np;
            $v = $adam ? array_slice($in, $c, $np) : [];

            // Forward pass
            $acts = [$x];
            $zs = [];
            $logits = null;
            for ($l = 0; $l < $layers; $l++) {
                $z = $acts[$l]->matmul($p[2 * $l]) + $p[2 * $l + 1];
                if ($l < $layers - 1) {
                    $zs[$l] = $z;
                    $acts[$l + 1] = self::activate($z, $kind, $zero);
                } else {
                    $logits = $z;
                }
            }

            // Numerically stable softmax cross-entropy
            $shifted = $logits - $logits->max(1)->reshape([$rows, 1]);
            $exponentials = $shifted->exp();
            $denominator = $exponentials->sum(1)->reshape([$rows, 1]);
            $target = $ls > 0.0 ? $y * (1.0 - $ls) + ($ls / $classes) : $y;
            $dz = ($exponentials / $denominator - $target) * (1.0 / $rows);

            // Backward pass
            $grads = array_fill(0, $np, null);
            for ($l = $layers - 1; $l >= 0; $l--) {
                $inputT = $l === 0 ? $xt : $acts[$l]->transpose();
                $grads[2 * $l] = $inputT->matmul($dz);
                $grads[2 * $l + 1] = $dz->sum(0)->reshape([1, $dims[$l + 1]]);
                if ($l > 0) {
                    $da = $dz->matmul($p[2 * $l]->transpose());
                    $dz = self::activateGrad($zs[$l - 1], $da, $kind, $zero);
                }
            }

            // Optimizer
            $b1tn = null;
            $b2tn = null;
            $corr1 = null;
            $corr2 = null;
            if ($adam) {
                $b1tn = $b1t * $beta1;
                $b2tn = $b2t * $beta2;
                $corr1 = $one - $b1tn;
                $corr2 = $one - $b2tn;
            }
            $newP = [];
            $newM = [];
            $newV = [];
            for ($i = 0; $i < $np; $i++) {
                $g = $grads[$i];
                $ok = $g->eq($g);
                $g = CudaArray::where($g->gt($clip), $upper, $g);
                $g = CudaArray::where($g->lt(-$clip), $lower, $g);
                if ($adam) {
                    $mn = $m[$i] * $beta1 + $g * (1.0 - $beta1);
                    $vn = $v[$i] * $beta2 + ($g * $g) * (1.0 - $beta2);
                    $direction = ($mn / $corr1) / (($vn / $corr2)->sqrt() + 1e-8);
                } else {
                    $mn = $m[$i] * $mu + $g;
                    $direction = $mn;
                }
                if ($wd > 0.0 && $i % 2 === 0) {
                    $direction = $direction + $p[$i] * $wd;
                }
                $pn = $p[$i] - $direction * $lr;
                $newP[$i] = CudaArray::where($ok, $pn, $p[$i]);
                $newM[$i] = CudaArray::where($ok, $mn, $m[$i]);
                if ($adam) {
                    $newV[$i] = CudaArray::where($ok, $vn, $v[$i]);
                }
            }

            $out = array_merge($newP, $newM);
            if ($adam) {
                $out = array_merge($out, $newV, [$b1tn, $b2tn]);
            }
            if ($withLoss) {
                $targetLogit = ($shifted * $y)->sum(1)->reshape([$rows, 1]);
                $out[] = ($denominator->log() - $targetLogit)->sum(0);
            }
            return array_values($out);
        };
    }

    public static function evalBody(ModelSpec $spec, int $rows, array $k): Closure
    {
        $layers = $spec->layers();
        $kind = $spec->activation;
        $zero = $k['zero'];

        return static function (array $in) use ($layers, $kind, $zero, $rows): array {
            $x = $in[0];
            $y = $in[1];
            $p = array_slice($in, 2);
            $a = $x;
            $logits = null;
            for ($l = 0; $l < $layers; $l++) {
                $z = $a->matmul($p[2 * $l]) + $p[2 * $l + 1];
                if ($l < $layers - 1) {
                    $a = self::activate($z, $kind, $zero);
                } else {
                    $logits = $z;
                }
            }
            $shifted = $logits - $logits->max(1)->reshape([$rows, 1]);
            $denominator = $shifted->exp()->sum(1)->reshape([$rows, 1]);
            $targetLogit = ($shifted * $y)->sum(1)->reshape([$rows, 1]);
            $lossSum = ($denominator->log() - $targetLogit)->sum(0);
            return [$logits->argMax(1), $lossSum, $logits];
        };
    }

    public static function probsBody(ModelSpec $spec, int $rows, array $k): Closure
    {
        $layers = $spec->layers();
        $kind = $spec->activation;
        $zero = $k['zero'];

        return static function (array $in) use ($layers, $kind, $zero, $rows): array {
            $a = $in[0];
            $p = array_slice($in, 1);
            $logits = null;
            for ($l = 0; $l < $layers; $l++) {
                $z = $a->matmul($p[2 * $l]) + $p[2 * $l + 1];
                if ($l < $layers - 1) {
                    $a = self::activate($z, $kind, $zero);
                } else {
                    $logits = $z;
                }
            }
            $exponentials = ($logits - $logits->max(1)->reshape([$rows, 1]))->exp();
            return [$exponentials / $exponentials->sum(1)->reshape([$rows, 1])];
        };
    }
}

// ---------------------------------------------------------------------------
// Evaluation, reporting, schedules
// ---------------------------------------------------------------------------

function evaluateSet($plan, CudaArray $x, CudaArray $y, array $params, array $targets): array
{
    $out = $plan->run($x, $y, ...$params);
    $predictions = $out[0]->toArray();
    $lossSum = $out[1][0];
    $correct = 0;
    foreach ($predictions as $i => $prediction) {
        if ((int)$prediction === $targets[$i]) {
            $correct++;
        }
    }
    $n = count($targets);
    return [
        'loss' => $lossSum / $n,
        'accuracy' => 100.0 * $correct / $n,
        'predictions' => $predictions,
        'logits' => $out[2],
    ];
}

function classificationReport(array $predictions, array $targets, int $classes): array
{
    $cm = array_fill(0, $classes, array_fill(0, $classes, 0));
    foreach ($targets as $i => $truth) {
        $p = (int)$predictions[$i];
        if ($p >= 0 && $p < $classes) {
            $cm[$truth][$p]++;
        }
    }
    $perClass = [];
    $f1Sum = 0.0;
    $correct = 0;
    for ($k = 0; $k < $classes; $k++) {
        $tp = $cm[$k][$k];
        $fn = array_sum($cm[$k]) - $tp;
        $fp = 0;
        for ($r = 0; $r < $classes; $r++) {
            $fp += $cm[$r][$k];
        }
        $fp -= $tp;
        $precision = ($tp + $fp) > 0 ? $tp / ($tp + $fp) : 0.0;
        $recall = ($tp + $fn) > 0 ? $tp / ($tp + $fn) : 0.0;
        $f1 = ($precision + $recall) > 0 ? 2 * $precision * $recall / ($precision + $recall) : 0.0;
        $f1Sum += $f1;
        $correct += $tp;
        $perClass[$k] = ['support' => $tp + $fn, 'precision' => $precision, 'recall' => $recall, 'f1' => $f1];
    }
    $pairs = [];
    for ($t = 0; $t < $classes; $t++) {
        for ($p = 0; $p < $classes; $p++) {
            if ($t !== $p && $cm[$t][$p] > 0) {
                $pairs[] = ['truth' => $t, 'predicted' => $p, 'count' => $cm[$t][$p]];
            }
        }
    }
    usort($pairs, static fn(array $a, array $b): int => $b['count'] <=> $a['count']);
    return [
        'accuracy' => 100.0 * $correct / max(1, count($targets)),
        'macroF1' => $f1Sum / $classes,
        'perClass' => $perClass,
        'confusion' => $cm,
        'topConfusions' => array_slice($pairs, 0, 5),
    ];
}

function cpuForward(ModelSpec $spec, array $params, array $packedRows): array
{
    $layers = $spec->layers();
    $weights = [];
    $biases = [];
    for ($l = 0; $l < $layers; $l++) {
        $weights[$l] = $params[2 * $l]->toArray();
        $biases[$l] = $params[2 * $l + 1]->toArray()[0];
    }
    $out = [];
    foreach ($packedRows as $packed) {
        $a = array_values(unpack('g*', $packed));
        for ($l = 0; $l < $layers; $l++) {
            $in = $spec->dims[$l];
            $outDim = $spec->dims[$l + 1];
            $z = $biases[$l];
            for ($i = 0; $i < $in; $i++) {
                $ai = $a[$i];
                if ($ai == 0.0) {
                    continue;
                }
                $wRow = $weights[$l][$i];
                for ($j = 0; $j < $outDim; $j++) {
                    $z[$j] += $ai * $wRow[$j];
                }
            }
            if ($l < $layers - 1) {
                foreach ($z as $j => $zv) {
                    $z[$j] = $zv > 0 ? $zv : ($spec->activation === 'leaky' ? LEAKY_SLOPE * $zv : 0.0);
                }
            }
            $a = $z;
        }
        $out[] = $a;
    }
    return $out;
}

function scheduleLr(int $epoch, int $epochs, int $warmup, float $base, float $floorRatio = 0.05): float
{
    if ($warmup > 0 && $epoch < $warmup) {
        return $base * ($epoch + 1) / $warmup;
    }
    $span = max(1, $epochs - $warmup - 1);
    $t = min(1.0, max(0.0, ($epoch - $warmup) / $span));
    return $base * ($floorRatio + (1.0 - $floorRatio) * 0.5 * (1.0 + cos(M_PI * $t)));
}

// ---------------------------------------------------------------------------
// Model file
// ---------------------------------------------------------------------------

function saveModel(string $path, ModelSpec $spec, array $params, array $data): void
{
    $image = $data['image'];
    $header = json_encode([
        'format' => 'php-gpu-tensors-mlp',
        'version' => 1,
        'dataset' => $data['name'],
        'dims' => $spec->dims,
        'activation' => $spec->activation,
        'classNames' => $data['classNames'],
        'image' => $image === null ? null : ['h' => $image['h'], 'w' => $image['w'], 'mean' => $image['mean'], 'std' => $image['std']],
        'createdAt' => date(DATE_ATOM),
    ], JSON_THROW_ON_ERROR);
    $out = MODEL_MAGIC . $header . "\n";
    foreach ($params as $param) {
        $out .= $param->toHost()->toBuffer();
    }
    if (file_put_contents($path, $out) !== strlen($out)) {
        throw new RuntimeException("Unable to write the model file $path.");
    }
}

function loadModelFile(string $path): array
{
    $data = @file_get_contents($path);
    if ($data === false) {
        throw new RuntimeException("Unable to read the model file $path.");
    }
    if (!str_starts_with($data, MODEL_MAGIC)) {
        throw new RuntimeException('This is not a php-gpu-tensors MLP model file.');
    }
    $start = strlen(MODEL_MAGIC);
    $newline = strpos($data, "\n", $start);
    if ($newline === false) {
        throw new RuntimeException('Corrupt model file header.');
    }
    $header = json_decode(substr($data, $start, $newline - $start), true, 512, JSON_THROW_ON_ERROR);
    $dims = $header['dims'] ?? null;
    if (!is_array($dims) || count($dims) < 2) {
        throw new RuntimeException('The model header has no valid layer sizes.');
    }
    $pos = $newline + 1;
    $buffers = [];
    for ($l = 0; $l < count($dims) - 1; $l++) {
        foreach ([4 * $dims[$l] * $dims[$l + 1], 4 * $dims[$l + 1]] as $bytes) {
            $chunk = substr($data, $pos, $bytes);
            if (strlen($chunk) !== $bytes) {
                throw new RuntimeException('The model file is truncated.');
            }
            $buffers[] = $chunk;
            $pos += $bytes;
        }
    }
    return [$header, $buffers];
}

// ---------------------------------------------------------------------------
// Inference helpers: ASCII rendering, PGM input, preprocessing
// ---------------------------------------------------------------------------

function asciiImage(string $packedRow, int $h, int $w, float $mean, float $std): string
{
    $ramp = ' .:-=+*#%@';
    $values = array_values(unpack('g*', $packedRow));
    $out = '';
    for ($y = 0; $y < $h; $y++) {
        for ($x = 0; $x < $w; $x++) {
            $intensity = min(1.0, max(0.0, $values[$y * $w + $x] * $std + $mean));
            $out .= str_repeat($ramp[(int)round($intensity * 9)], 2);
        }
        $out .= "\n";
    }
    return $out;
}

function readPgm(string $path): array
{
    $data = @file_get_contents($path);
    if ($data === false) {
        throw new RuntimeException("Unable to read $path.");
    }
    $magic = substr($data, 0, 2);
    if ($magic !== 'P2' && $magic !== 'P5') {
        throw new RuntimeException('Only PGM files (P2 or P5) are supported. Convert with: convert in.png -colorspace Gray out.pgm');
    }
    $pos = 2;
    $length = strlen($data);
    $readToken = static function () use ($data, &$pos, $length): string {
        while ($pos < $length) {
            $ch = $data[$pos];
            if ($ch === '#') {
                while ($pos < $length && $data[$pos] !== "\n") {
                    $pos++;
                }
            } elseif (ctype_space($ch)) {
                $pos++;
            } else {
                break;
            }
        }
        $start = $pos;
        while ($pos < $length && !ctype_space($data[$pos]) && $data[$pos] !== '#') {
            $pos++;
        }
        return substr($data, $start, $pos - $start);
    };
    $w = (int)$readToken();
    $h = (int)$readToken();
    $max = (int)$readToken();
    if ($w < 1 || $h < 1 || $max < 1 || $max > 65535) {
        throw new RuntimeException('Invalid PGM header.');
    }
    $pixels = [];
    if ($magic === 'P5') {
        $pos++;
        $bytes = $max > 255 ? 2 : 1;
        if ($length - $pos < $w * $h * $bytes) {
            throw new RuntimeException('The PGM file is truncated.');
        }
        for ($i = 0; $i < $w * $h; $i++) {
            $v = $bytes === 1 ? ord($data[$pos + $i]) : (ord($data[$pos + 2 * $i]) << 8) | ord($data[$pos + 2 * $i + 1]);
            $pixels[] = (int)round(255 * $v / $max);
        }
    } else {
        for ($i = 0; $i < $w * $h; $i++) {
            $token = $readToken();
            if ($token === '') {
                throw new RuntimeException('The PGM file is truncated.');
            }
            $pixels[] = (int)round(255 * (int)$token / $max);
        }
    }
    return [$w, $h, $pixels];
}

function resizeBox(array $pixels, int $w, int $h, int $tw, int $th): array
{
    $out = [];
    for ($ty = 0; $ty < $th; $ty++) {
        $y0 = (int)floor($ty * $h / $th);
        $y1 = max($y0 + 1, (int)floor(($ty + 1) * $h / $th));
        for ($tx = 0; $tx < $tw; $tx++) {
            $x0 = (int)floor($tx * $w / $tw);
            $x1 = max($x0 + 1, (int)floor(($tx + 1) * $w / $tw));
            $sum = 0;
            $count = 0;
            for ($y = $y0; $y < min($y1, $h); $y++) {
                for ($x = $x0; $x < min($x1, $w); $x++) {
                    $sum += $pixels[$y * $w + $x];
                    $count++;
                }
            }
            $out[] = (int)round($sum / max(1, $count));
        }
    }
    return $out;
}

function prepareDrawing(array $pixels, int $w, int $h, int $canvasW, int $canvasH, bool $invert, bool $raw): array
{
    if ($invert) {
        $pixels = array_map(static fn(int $p): int => 255 - $p, $pixels);
    }
    if ($raw) {
        return resizeBox($pixels, $w, $h, $canvasW, $canvasH);
    }
    $threshold = 40;
    $minX = $w;
    $minY = $h;
    $maxX = -1;
    $maxY = -1;
    for ($y = 0; $y < $h; $y++) {
        for ($x = 0; $x < $w; $x++) {
            if ($pixels[$y * $w + $x] > $threshold) {
                $minX = min($minX, $x);
                $maxX = max($maxX, $x);
                $minY = min($minY, $y);
                $maxY = max($maxY, $y);
            }
        }
    }
    if ($maxX < 0) {
        throw new RuntimeException('The image looks blank (try --invert for dark ink on a light background).');
    }
    $cropW = $maxX - $minX + 1;
    $cropH = $maxY - $minY + 1;
    $crop = [];
    for ($y = $minY; $y <= $maxY; $y++) {
        for ($x = $minX; $x <= $maxX; $x++) {
            $crop[] = $pixels[$y * $w + $x];
        }
    }
    $box = (int)round(min($canvasW, $canvasH) * 20 / 28);
    $scale = $box / max($cropW, $cropH);
    $newW = max(1, (int)round($cropW * $scale));
    $newH = max(1, (int)round($cropH * $scale));
    $small = resizeBox($crop, $cropW, $cropH, $newW, $newH);
    $canvas = array_fill(0, $canvasW * $canvasH, 0);
    $offX = intdiv($canvasW - $newW, 2);
    $offY = intdiv($canvasH - $newH, 2);
    for ($y = 0; $y < $newH; $y++) {
        for ($x = 0; $x < $newW; $x++) {
            $canvas[($offY + $y) * $canvasW + $offX + $x] = $small[$y * $newW + $x];
        }
    }
    return $canvas;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

function optInt(array $o, string $name, int $default, int $min = 1): int
{
    $value = filter_var($o[$name] ?? $default, FILTER_VALIDATE_INT);
    if ($value === false || $value < $min) {
        throw new InvalidArgumentException("--$name must be an integer >= $min.");
    }
    return $value;
}

function optFloat(array $o, string $name, float $default, float $min = 0.0, float $max = INF): float
{
    $value = filter_var($o[$name] ?? $default, FILTER_VALIDATE_FLOAT);
    if ($value === false || !is_finite($value) || $value < $min || $value > $max) {
        throw new InvalidArgumentException("--$name must be a number between $min and $max.");
    }
    return (float)$value;
}

function usage(): string
{
    return <<<'TXT'
gpu_classifier.php: deep MLP classifier on real data with php-gpu-tensors (pure PHP)

Data:      --dataset=mnist|fashion|csv  --data-dir=<script_dir>/training/data  --url-base=<mirror>  --val=5000
           CSV: --csv=file.csv --header --label-col=-1 --val-fraction=0.15 --test-fraction=0.15
Model:     --layers=512,256  --activation=relu|leaky
Training:  --epochs=40 --batch-size=128 --optimizer=adam|sgd --lr=<auto> --weight-decay=<auto>
           --clip=5 --label-smoothing=0 --warmup=2 --shift=2 --permutations=4 --seed=2026
Reporting: --report-every=1 --patience=8 --min-accuracy=0 --json=report.json --profile
Model I/O: --model=<file> (default <script_dir>/training/<dataset>_mlp.pgt)  --no-save  --load
Inference: --predict-index=N (test sample)  --predict-file=digit.pgm [--invert] [--raw]
Other:     --no-selftest  --help
TXT;
}

function parseConfig(): array
{
    $o = getopt('', [
        'dataset:', 'data-dir:', 'url-base:', 'val:', 'csv:', 'label-col:', 'val-fraction:', 'test-fraction:',
        'layers:', 'activation:', 'epochs:', 'batch-size:', 'optimizer:', 'lr:', 'weight-decay:', 'clip:',
        'label-smoothing:', 'warmup:', 'shift:', 'permutations:', 'seed:', 'report-every:', 'patience:',
        'min-accuracy:', 'json:', 'model:', 'predict-index:', 'predict-file:',
        'header', 'no-save', 'load', 'invert', 'raw', 'no-selftest', 'profile', 'help',
    ]);
    
    if (array_key_exists('help', $o)) {
        echo usage(), PHP_EOL;
        exit(0);
    }
    
    $dataset = strtolower((string)($o['dataset'] ?? 'mnist'));
    if (!in_array($dataset, ['mnist', 'fashion', 'csv'], true)) {
        throw new InvalidArgumentException('--dataset must be mnist, fashion or csv.');
    }
    if ($dataset === 'csv' && !isset($o['csv'])) {
        throw new InvalidArgumentException('--dataset=csv needs --csv=<file>.');
    }
    
    $optimizer = strtolower((string)($o['optimizer'] ?? 'adam'));
    if (!in_array($optimizer, ['adam', 'sgd'], true)) {
        throw new InvalidArgumentException('--optimizer must be adam or sgd.');
    }
    
    $activation = strtolower((string)($o['activation'] ?? 'relu'));
    if (!in_array($activation, ['relu', 'leaky'], true)) {
        throw new InvalidArgumentException('--activation must be relu or leaky.');
    }
    
    $hidden = [];
    foreach (explode(',', (string)($o['layers'] ?? '512,256')) as $part) {
        $part = trim($part);
        if ($part === '') {
            continue;
        }
        $h = filter_var($part, FILTER_VALIDATE_INT);
        if ($h === false || $h < 1) {
            throw new InvalidArgumentException('--layers must be a comma-separated list of positive integers.');
        }
        $hidden[] = $h;
    }
    
    if ($hidden === []) {
        throw new InvalidArgumentException('--layers needs at least one hidden layer.');
    }
    
    $adam = $optimizer === 'adam';

    // Cria e define o diretório 'training' base
    $trainingDir = __DIR__ . DIRECTORY_SEPARATOR . 'training';
    if (!is_dir($trainingDir) && !mkdir($trainingDir, 0775, true) && !is_dir($trainingDir)) {
        throw new RuntimeException("Não foi possível criar o diretório $trainingDir");
    }

    // Ancorando diretórios de dados e do modelo ao novo diretório 'training'
    $dataDir = (string)($o['data-dir'] ?? ($trainingDir . DIRECTORY_SEPARATOR . 'data'));
    
    return [
        'dataset' => $dataset,
        'dataDir' => $dataset === 'csv' ? $dataDir : $dataDir . DIRECTORY_SEPARATOR . $dataset,
        'urlBase' => isset($o['url-base']) ? (string)$o['url-base'] : null,
        'val' => optInt($o, 'val', 5000, 1),
        'csv' => isset($o['csv']) ? (string)$o['csv'] : null,
        'header' => array_key_exists('header', $o),
        'labelCol' => optInt($o, 'label-col', -1, -1000000),
        'valFraction' => optFloat($o, 'val-fraction', 0.15, 0.01, 0.5),
        'testFraction' => optFloat($o, 'test-fraction', 0.15, 0.01, 0.5),
        'hidden' => $hidden,
        'activation' => $activation,
        'epochs' => optInt($o, 'epochs', 40, 1),
        'batchSize' => optInt($o, 'batch-size', 128, 1),
        'optimizer' => $optimizer,
        'lr' => optFloat($o, 'lr', $adam ? 0.001 : 0.02, 1e-8, 10.0),
        'weightDecay' => optFloat($o, 'weight-decay', $adam ? 0.01 : 0.0005, 0.0, 10.0),
        'clip' => optFloat($o, 'clip', 5.0, 1e-3, 1e6),
        'labelSmoothing' => optFloat($o, 'label-smoothing', 0.0, 0.0, 0.5),
        'warmup' => optInt($o, 'warmup', 2, 0),
        'shift' => optInt($o, 'shift', 2, 0),
        'permutations' => optInt($o, 'permutations', 4, 1),
        'seed' => optInt($o, 'seed', 2026, 0),
        'reportEvery' => optInt($o, 'report-every', 1, 1),
        'patience' => optInt($o, 'patience', 8, 1),
        'minAccuracy' => optFloat($o, 'min-accuracy', 0.0, 0.0, 100.0),
        'json' => isset($o['json']) ? (string)$o['json'] : null,
        'model' => (string)($o['model'] ?? ($trainingDir . DIRECTORY_SEPARATOR . ($dataset === 'csv' ? 'csv' : $dataset) . '_mlp.pgt')),
        'save' => !array_key_exists('no-save', $o),
        'load' => array_key_exists('load', $o),
        'predictIndex' => isset($o['predict-index']) ? optInt($o, 'predict-index', 0, 0) : null,
        'predictFile' => isset($o['predict-file']) ? (string)$o['predict-file'] : null,
        'invert' => array_key_exists('invert', $o),
        'raw' => array_key_exists('raw', $o),
        'selftest' => !array_key_exists('no-selftest', $o),
        'profile' => array_key_exists('profile', $o),
    ];
}

function loadData(array $cfg): array
{
    if ($cfg['dataset'] === 'csv') {
        return loadCsvDataset($cfg['csv'], $cfg['header'], $cfg['labelCol'], $cfg['valFraction'],
            $cfg['testFraction'], $cfg['seed']);
    }
    return loadImageDataset($cfg['dataset'], $cfg['dataDir'], $cfg['urlBase'], $cfg['val'], $cfg['seed']);
}

// ---------------------------------------------------------------------------
// Main Pipeline
// ---------------------------------------------------------------------------

function main(): int
{
    $cfg = parseConfig();
    if (!extension_loaded('cuda') || cuda_get_device_count() < 1) {
        throw new RuntimeException('This tool requires the CUDA extension and a visible NVIDIA GPU.');
    }
    @ini_set('memory_limit', '4G');
    $checks = new Checks();
    $wallStart = hrtime(true);
    
    // Header
    printf("\n" . C_BLD . C_MAG . "=== PHP-CUDA-EXT MLP Classifier ===" . C_RST . "\n");
    printf(C_CYN . "PHP %s (%s) | cuda extension %s | %d GPU(s)" . C_RST . "\n\n", 
        PHP_VERSION, PHP_ZTS ? 'ZTS' : 'NTS', phpversion('cuda') ?: 'unknown', cuda_get_device_count());

    // 1. Data Loading --------------------------------------------------------
    $start = hrtime(true);
    echo C_BLD . "[1/4] Data Loading" . C_RST . "\n";
    echo "  Loading dataset " . C_CYN . $cfg['dataset'] . C_RST . "...\n";
    $data = loadData($cfg);
    $nTrain = count($data['train']['rows']);
    $nVal = count($data['val']['rows']);
    $nTest = count($data['test']['rows']);
    printf("  %s: %d train / %d val / %d test | %d features, %d classes (%.2f s)\n", $data['name'], $nTrain, $nVal,
        $nTest, $data['features'], $data['classes'], (hrtime(true) - $start) / 1e9);

    // 2. Model Initialization -----------------------------------------------
    echo C_BLD . "\n[2/4] Model & JIT Compilation" . C_RST . "\n";
    if ($cfg['load']) {
        [$header, $buffers] = loadModelFile($cfg['model']);
        $dims = $header['dims'];
        $activation = (string)($header['activation'] ?? 'relu');
        if ($dims[0] !== $data['features'] || $dims[count($dims) - 1] !== $data['classes']) {
            throw new RuntimeException(sprintf('The model expects %d features and %d classes, the dataset has %d and %d.',
                $dims[0], $dims[count($dims) - 1], $data['features'], $data['classes']));
        }
        printf("  Loaded model %s (%s)\n", $cfg['model'], $header['dataset'] ?? 'unknown dataset');
    } else {
        $dims = array_merge([$data['features']], $cfg['hidden'], [$data['classes']]);
        $activation = $cfg['activation'];
    }
    $spec = new ModelSpec($dims, $activation, $cfg['optimizer'], $cfg['weightDecay'], $cfg['clip'], $cfg['labelSmoothing']);
    printf("  Architecture: %s | %d weight layers | %s parameters | %s%s\n", implode('-', $dims), $spec->layers(),
        number_format($spec->parameterElements()), $activation,
        $cfg['load'] ? '' : ', ' . ($spec->isAdam() ? 'AdamW' : 'SGD+momentum'));

    if (!$cfg['load']) {
        $buffers = Net::initBuffers($spec, $cfg['seed'] + 1);
    }
    $consts = [
        'zero' => CudaArray::zeros([1]),
        'one' => CudaArray::full([1], 1.0),
        'upper' => CudaArray::full([1], $cfg['clip']),
        'lower' => CudaArray::full([1], -$cfg['clip']),
    ];

    // 3. Tensor Upload ------------------------------------------------------
    $start = hrtime(true);
    [$valX, $valY, $valTargets] = evalTensors($data['val'], $data['features'], $data['classes']);
    [$testX, $testY, $testTargets] = evalTensors($data['test'], $data['features'], $data['classes']);
    $cpuSample = array_slice($data['val']['rows'], 0, min(32, $nVal));
    $predictRows = $data['test']['rows'];
    $perms = [];
    if (!$cfg['load']) {
        $trainMb = $nTrain * $data['features'] * 4 / 1048576;
        printf("  Uploading %d epoch permutation(s) (~%.0f MB each)%s...\n",
            $cfg['permutations'], $trainMb, ($cfg['shift'] > 0 && $data['image'] !== null) ? ", shift aug up to {$cfg['shift']}px" : '');
        for ($p = 0; $p < $cfg['permutations']; $p++) {
            $perms[$p] = buildBatches($data, $cfg['batchSize'], $cfg['seed'] + 100 + $p, $cfg['shift']);
        }
    }
    $data['train'] = ['rows' => [], 'labels' => []];
    $data['val']['rows'] = [];
    $data['test']['rows'] = $cfg['predictIndex'] !== null ? $predictRows : [];
    unset($predictRows);

    // 4. JIT Compilation (Fusion Plans) -------------------------------------
    $state0 = TrainState::create($spec, $buffers);
    $plans = [];
    if (!$cfg['load']) {
        $lrExample = scalarTensor($cfg['lr']);
        $firstByRows = [];
        foreach ($perms[0] as [$x, $xt, $y, $rows]) {
            if (!isset($firstByRows[$rows])) {
                $firstByRows[$rows] = [$x, $xt, $y];
            }
        }
        foreach ($firstByRows as $rows => [$x, $xt, $y]) {
            $inputs = $state0->inputs($x, $xt, $y, $lrExample);
            $plans[$rows] = [
                Fusion::compile(Net::positional(Net::trainBody($spec, $rows, false, $consts), $spec->trainArity()), $inputs),
                Fusion::compile(Net::positional(Net::trainBody($spec, $rows, true, $consts), $spec->trainArity()), $inputs),
            ];
            if ($cfg['profile']) {
                $plans[$rows][0]->setProfiling(true);
                $plans[$rows][1]->setProfiling(true);
            }
        }
    }
    $evalPlans = [];
    foreach ([[$nVal, $valX, $valY], [$nTest, $testX, $testY]] as [$n, $ex, $ey]) {
        if (!isset($evalPlans[$n])) {
            $evalPlans[$n] = Fusion::compile(Net::positional(Net::evalBody($spec, $n, $consts), $spec->evalArity()),
                [$ex, $ey, ...$state0->params]);
        }
    }
    printf("  Compilation & Upload done in %.2f s\n", (hrtime(true) - $start) / 1e9);

    // 5. Self-tests (Verification) ------------------------------------------
    if ($cfg['selftest']) {
        echo "\n" . C_BLD . "[3/4] Self-Tests" . C_RST . "\n";
        foreach ($plans as $rows => $pair) {
            foreach ([false, true] as $withLoss) {
                $stats = $pair[(int)$withLoss]->getStats();
                $name = sprintf('plan batch=%d%s', $rows, $withLoss ? ' (with loss)' : '');
                if ($stats['fusedKernels'] < 1) {
                    $checks->add('FAIL', $name, 'no fused kernels were produced');
                    continue;
                }
                $expected = $spec->expectedBoundaries($withLoss);
                $checks->add($stats['boundaries'] === $expected ? 'PASS' : 'WARN', $name,
                    sprintf('%d fused kernels, %d boundaries (expected %d)', $stats['fusedKernels'], $stats['boundaries'], $expected));
            }
        }
        $initial = evaluateSet($evalPlans[$nVal], $valX, $valY, $state0->params, $valTargets);
        $cpu = cpuForward($spec, $state0->params, $cpuSample);
        $gpu = $initial['logits']->toArray();
        $worst = 0.0;
        $scale = 1.0;
        foreach ($cpu as $r => $row) {
            foreach ($row as $c => $value) {
                $worst = max($worst, abs($gpu[$r][$c] - $value));
                $scale = max($scale, abs($value));
            }
        }
        $checks->add($worst <= 2e-3 * $scale ? 'PASS' : 'FAIL', 'gpu-vs-cpu forward',
            sprintf('max |diff| %.3e over %d samples (logit scale %.2f)', $worst, count($cpu), $scale));

        if (!$cfg['load']) {
            [$x, $xt, $y, $rows] = $perms[0][0];
            $stepLr = scalarTensor($cfg['lr'] * 0.1);
            $inputs = TrainState::create($spec, $buffers)->inputs($x, $xt, $y, $stepLr);
            $eagerBody = Net::trainBody($spec, $rows, true, $consts);
            $eagerOut = array_values($eagerBody($inputs));
            $fusedOut = array_values($plans[$rows][1]->run(...$inputs));
            if (count($eagerOut) !== count($fusedOut)) {
                $checks->add('FAIL', 'eager-vs-fusion step', sprintf('output count differs (%d vs %d)', count($eagerOut), count($fusedOut)));
            } else {
                $worst = 0.0;
                $allOk = true;
                foreach ($eagerOut as $i => $tensor) {
                    [$diff, $ok] = compareTensors($tensor, $fusedOut[$i], 1e-5, 2e-3);
                    $worst = max($worst, $diff);
                    $allOk = $allOk && $ok;
                }
                $checks->add($allOk ? 'PASS' : 'FAIL', 'eager-vs-fusion step',
                    sprintf('%d output tensors, max |diff| %.3e', count($eagerOut), $worst));
            }
            unset($eagerOut, $fusedOut, $inputs);
        }
    }

    // 6. Training Loop ------------------------------------------------------
    $state = TrainState::create($spec, $buffers);
    $initialVal = evaluateSet($evalPlans[$nVal], $valX, $valY, $state->params, $valTargets);
    $history = [];
    $epochsRun = 0;
    $steps = 0;
    $trainSeconds = 0.0;
    $plainSeconds = 0.0;
    $plainSteps = 0;
    $bestEpoch = 0;
    $bestLoss = INF;
    if (!$cfg['load']) {
        echo "\n" . C_BLD . "[4/4] Training Loop" . C_RST . "\n";
        printf("  Initial validation: loss %.4f, acc %.2f%%\n\n", $initialVal['loss'], $initialVal['accuracy']);
        $bestParams = null;
        $bad = 0;
        for ($epoch = 0; $epoch < $cfg['epochs']; $epoch++) {
            $report = $epoch % $cfg['reportEvery'] === 0 || $epoch === $cfg['epochs'] - 1;
            $lr = scheduleLr($epoch, $cfg['epochs'], $cfg['warmup'], $cfg['lr']);
            $lrT = scalarTensor($lr);
            $batches = $perms[$epoch % $cfg['permutations']];
            $lossSum = 0.0;
            $t0 = hrtime(true);
            foreach ($batches as [$x, $xt, $y, $rows]) {
                // Execute fused plan
                $tail = $state->apply($plans[$rows][(int)$report]->run(...$state->inputs($x, $xt, $y, $lrT)), $spec);
                if ($report) {
                    $lossSum += $tail[0][0];
                }
                $steps++;
            }
            $dt = (hrtime(true) - $t0) / 1e9;
            $trainSeconds += $dt;
            $epochsRun = $epoch + 1;
            if (!$report) {
                $plainSeconds += $dt;
                $plainSteps += count($batches);
                continue;
            }
            $trainLoss = $lossSum / $nTrain;
            if (!is_finite($trainLoss)) {
                throw new RuntimeException('Training diverged: non-finite cross-entropy.');
            }
            $val = evaluateSet($evalPlans[$nVal], $valX, $valY, $state->params, $valTargets);
            $history[] = ['epoch' => $epoch + 1, 'trainLoss' => $trainLoss, 'valLoss' => $val['loss'], 'valAccuracy' => $val['accuracy'], 'lr' => $lr];
            
            // Format metrics
            $accColor = $val['accuracy'] >= 85.0 ? C_GRN : C_YLW; // Highlights good accuracy
            printf("  " . C_BLD . "Epoch %3d" . C_RST . ": train CE %.4f | val CE %.4f, acc %s%6.2f%%%s | lr %.5f | %.2f ms/step\n", 
                $epoch + 1, $trainLoss, $val['loss'], $accColor, $val['accuracy'], C_RST, $lr, $dt * 1000 / count($batches));
            
            // Early stopping condition
            if ($val['loss'] < $bestLoss - 1e-6) {
                $bestLoss = $val['loss'];
                $bestEpoch = $epoch + 1;
                $bestParams = $state->params; 
                $bad = 0;
            } elseif (++$bad >= $cfg['patience']) {
                printf(C_YLW . "  Early stopping: no validation improvement for %d reports (best epoch %d)." . C_RST . "\n", $cfg['patience'], $bestEpoch);
                break;
            }
        }
        if ($bestParams !== null) {
            $state->params = $bestParams;
        }
        $msStep = $trainSeconds * 1000 / max(1, $steps);
        $steadyMs = $plainSteps > 0 ? $plainSeconds * 1000 / $plainSteps : $msStep;
        $samplesPerSecond = $nTrain * $epochsRun / max(1e-9, $trainSeconds);
        $gflops = $spec->stepFlops($cfg['batchSize']) / ($steadyMs / 1000.0) / 1e9;
        
        printf("\n  " . C_BLD . "Training Stats:" . C_RST . " %.2f s, %d epochs, %d steps, %.3f ms/step\n", $trainSeconds, $epochsRun, $steps, $msStep);
        printf("  " . C_BLD . "Performance:" . C_RST . " %.0f samples/s, ~" . C_CYN . "%.1f GFLOP/s" . C_RST . " matmul work\n", $samplesPerSecond, $gflops);
    }

    // 7. Final Evaluation ---------------------------------------------------
    $finalVal = evaluateSet($evalPlans[$nVal], $valX, $valY, $state->params, $valTargets);
    $test = evaluateSet($evalPlans[$nTest], $testX, $testY, $state->params, $testTargets);
    $report = classificationReport($test['predictions'], $testTargets, $data['classes']);
    
    echo "\n" . C_BLD . C_MAG . "=== Final Evaluation ===" . C_RST . "\n";
    printf("  Validation: loss %.4f, accuracy %s%.2f%%%s.\n", $finalVal['loss'], C_GRN, $finalVal['accuracy'], C_RST);
    printf("  Test Set:   loss %.4f, accuracy %s%.2f%%%s (%d samples), macro-F1 %s%.4f%s\n\n", 
        $test['loss'], C_GRN, $test['accuracy'], C_RST, $nTest, C_GRN, $report['macroF1'], C_RST);
        
    echo C_BLD . C_CYN . "  class                  support  precision  recall     F1" . C_RST . "\n";
    foreach ($report['perClass'] as $k => $row) {
        printf("  %-20s %8d  %9.3f  %6.3f  %5.3f\n", substr($data['classNames'][$k], 0, 20), $row['support'],
            $row['precision'], $row['recall'], $row['f1']);
    }

    if ($report['topConfusions'] !== []) {
        echo "\n" . C_BLD . "  Top Confusions (truth -> predicted):" . C_RST . "\n";
        foreach ($report['topConfusions'] as $pair) {
            printf("    %-14s -> %-14s %s%d%s\n", substr($data['classNames'][$pair['truth']], 0, 14),
                substr($data['classNames'][$pair['predicted']], 0, 14), C_YLW, $pair['count'], C_RST);
        }
    }

    if ($cfg['selftest'] && !$cfg['load']) {
        $checks->add($finalVal['loss'] < $initialVal['loss'] ? 'PASS' : 'FAIL', 'validation loss improved',
            sprintf('%.4f -> %.4f', $initialVal['loss'], $finalVal['loss']));
    }
    if ($cfg['minAccuracy'] > 0.0) {
        $checks->add($test['accuracy'] >= $cfg['minAccuracy'] ? 'PASS' : 'FAIL', 'test accuracy threshold',
            sprintf('%.2f%% vs. required %.2f%%', $test['accuracy'], $cfg['minAccuracy']));
    }

    // 8. Save Model ---------------------------------------------------------
    if (!$cfg['load'] && $cfg['save']) {
        saveModel($cfg['model'], $spec, $state->params, $data);
        printf("\n  " . C_GRN . "✔ Model saved:" . C_RST . " %s (%.1f KB)\n", $cfg['model'], filesize($cfg['model']) / 1024);
    }

    // 9. Inference Showcase -------------------------------------------------
    if ($cfg['predictIndex'] !== null || $cfg['predictFile'] !== null) {
        echo "\n" . C_BLD . "--- Inference Mode ---" . C_RST . "\n";
        $probsPlan = null;
        $classify = static function (string $row) use (&$probsPlan, $spec, $consts, $state): array {
            $x = CudaArray::fromBuffer($row, [1, $spec->dims[0]]);
            if ($probsPlan === null) {
                // Compile inference probability graph on the fly
                $probsPlan = Fusion::compile(Net::positional(Net::probsBody($spec, 1, $consts), 1 + $spec->paramCount()),
                    [$x, ...$state->params]);
            }
            return $probsPlan->run($x, ...$state->params)[0]->toArray()[0];
        };
        $show = static function (array $probs, ?int $truth) use ($data): void {
            $order = array_keys($probs);
            usort($order, static fn(int $a, int $b): int => $probs[$b] <=> $probs[$a]);
            printf("  Prediction: %s%s (%.1f%%)%s%s\n", 
                C_BLD, $data['classNames'][$order[0]], 100 * $probs[$order[0]], C_RST,
                $truth === null ? '' : ($truth === $order[0] ? C_GRN . '  [correct]' . C_RST : C_RED . '  [truth: ' . $data['classNames'][$truth] . ']' . C_RST));
            foreach (array_slice($order, 0, 3) as $k) {
                printf("    %-20s %6.2f%%\n", $data['classNames'][$k], 100 * $probs[$k]);
            }
        };
        if ($cfg['predictIndex'] !== null) {
            $index = $cfg['predictIndex'];
            if ($index >= $nTest) {
                throw new InvalidArgumentException("--predict-index must be below $nTest.");
            }
            $row = $data['test']['rows'][$index];
            $image = $data['image'];
            echo "  Test sample #$index:\n";
            if ($image !== null) {
                echo asciiImage($row, $image['h'], $image['w'], $image['mean'], $image['std']);
            }
            $show($classify($row), $testTargets[$index]);
        }
        if ($cfg['predictFile'] !== null) {
            $image = $data['image'];
            if ($image === null) {
                throw new InvalidArgumentException('--predict-file only works with the image datasets.');
            }
            [$w, $h, $pixels] = readPgm($cfg['predictFile']);
            $canvas = prepareDrawing($pixels, $w, $h, $image['w'], $image['h'], $cfg['invert'], $cfg['raw']);
            $row = strtr(implode('', array_map('chr', $canvas)), $image['table']);
            echo "  Model input from {$cfg['predictFile']}:\n", asciiImage($row, $image['h'], $image['w'], $image['mean'], $image['std']);
            $show($classify($row), null);
        }
    }

    if ($cfg['json'] !== null) {
        // Garantindo que se houver um arquivo json, seja salvo relativo ao diretório training
        $jsonPath = (string)$cfg['json'];
        if (!str_starts_with($jsonPath, '/') && !preg_match('/^[a-zA-Z]:\\\\/', $jsonPath)) { // Se não for caminho absoluto
            $jsonPath = __DIR__ . DIRECTORY_SEPARATOR . 'training' . DIRECTORY_SEPARATOR . $jsonPath;
        }

        $payload = [
            'schema' => 1,
            'generatedAt' => date(DATE_ATOM),
            'environment' => ['php' => PHP_VERSION, 'zts' => (bool)PHP_ZTS, 'extension' => phpversion('cuda') ?: null],
            'configuration' => $cfg,
            'dataset' => ['name' => $data['name'], 'train' => $nTrain, 'val' => $nVal, 'test' => $nTest,
                'features' => $data['features'], 'classes' => $data['classes']],
            'model' => ['dims' => $dims, 'parameters' => $spec->parameterElements()],
            'training' => $cfg['load'] ? null : ['epochsRun' => $epochsRun, 'steps' => $steps, 'seconds' => $trainSeconds,
                'bestEpoch' => $bestEpoch, 'history' => $history],
            'validation' => ['loss' => $finalVal['loss'], 'accuracy' => $finalVal['accuracy']],
            'test' => ['loss' => $test['loss'], 'accuracy' => $test['accuracy'], 'macroF1' => $report['macroF1'],
                'perClass' => $report['perClass'], 'topConfusions' => $report['topConfusions']],
            'checks' => $checks->items,
            'wallSeconds' => (hrtime(true) - $wallStart) / 1e9,
        ];
        file_put_contents($jsonPath, json_encode($payload, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR));
    }
    
    printf("\nExecution finished in %.1f s: %d failure(s), %d warning(s).\n", (hrtime(true) - $wallStart) / 1e9, $checks->failures(), $checks->warnings());
    return $checks->failures() > 0 ? 1 : 0;
}

if (realpath($_SERVER['SCRIPT_FILENAME'] ?? '') === __FILE__) {
    try {
        exit(main());
    } catch (Throwable $error) {
        fwrite(STDERR, C_RED . C_BLD . 'Error: ' . C_RST . $error->getMessage() . PHP_EOL);
        exit(2);
    }
}