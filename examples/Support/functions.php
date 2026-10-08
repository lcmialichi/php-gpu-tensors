<?php
declare(strict_types=1);

use Cuda\CudaArray;
use Cuda\FusionGraph;

const C_RST = "\033[0m";

const C_BLD = "\033[1m";

const C_RED = "\033[31m";

const C_GRN = "\033[32m";

const C_YLW = "\033[33m";

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

/**
 * -------------------------------------------------------------------------
 * Packs a list of numeric values into little-endian float32 bytes.
 * -------------------------------------------------------------------------
 *
 * @param list<float|int> $values
 */
function packFloats(array $values): string
{
    $out = '';
    foreach (array_chunk($values, 4096) as $chunk) {
        $out .= pack('g*', ...$chunk);
    }
    return $out;
}

/**
 * -------------------------------------------------------------------------
 * Creates a one-element float32 tensor from a PHP scalar.
 * -------------------------------------------------------------------------
 */
function scalarTensor(float $value): CudaArray
{
    return CudaArray::fromBuffer(pack('g', $value), [1]);
}

/**
 * -------------------------------------------------------------------------
 * Flattens nested PHP values into floats for comparison and reporting.
 * -------------------------------------------------------------------------
 *
 * @return list<float>
 */
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

/**
 * -------------------------------------------------------------------------
 * Compares two tensors using absolute and relative error tolerances.
 * -------------------------------------------------------------------------
 *
 * @return array{float,bool} Maximum absolute difference and tolerance result.
 */
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

/**
 * -------------------------------------------------------------------------
 * Precomputes normalized float32 pixels for every possible byte value.
 * -------------------------------------------------------------------------
 *
 * @return array<string,string>
 */
function normalizationTable(float $mean, float $std): array
{
    $table = [];
    for ($b = 0; $b < 256; $b++) {
        $table[chr($b)] = pack('g', ($b / 255.0 - $mean) / $std);
    }
    return $table;
}

/**
 * -------------------------------------------------------------------------
 * Reads and decompresses a gzip file, surfacing corrupt or unreadable input.
 * -------------------------------------------------------------------------
 */
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

/**
 * -------------------------------------------------------------------------
 * Finds or downloads a valid dataset file from the configured mirrors.
 * -------------------------------------------------------------------------
 *
 * @param list<string> $mirrors
 */
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

/**
 * -------------------------------------------------------------------------
 * Validates an IDX byte stream and returns its dimensions and data offset.
 * -------------------------------------------------------------------------
 *
 * @return array{0:list<int>,1:int}
 */
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

/**
 * -------------------------------------------------------------------------
 * Applies an integer translation to a packed normalized image row.
 * -------------------------------------------------------------------------
 */
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

/**
 * -------------------------------------------------------------------------
 * Downloads, normalizes, and splits an IDX image classification dataset.
 * -------------------------------------------------------------------------
 *
 * @return array{
 *   name:string,features:int,classes:int,classNames:list<string>,
 *   image:array{h:int,w:int,mean:float,std:float,bg:string,table:array<string,string>},
 *   train:array{rows:list<string>,labels:list<int>},
 *   val:array{rows:list<string>,labels:list<int>},
 *   test:array{rows:list<string>,labels:list<int>}
 * }
 */
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
        if ($position < $valCount) {
            $val['rows'][] = $trainRowsAll[$index];
            $val['labels'][] = $labelsAll[$index];
        } else {
            $train['rows'][] = $trainRowsAll[$index];
            $train['labels'][] = $labelsAll[$index];
        }
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

/**
 * -------------------------------------------------------------------------
 * Parses, standardizes, and splits a numeric CSV classification dataset.
 * -------------------------------------------------------------------------
 *
 * @return array{
 *   name:string,features:int,classes:int,classNames:list<string>,image:null,
 *   train:array{rows:list<string>,labels:list<int>},
 *   val:array{rows:list<string>,labels:list<int>},
 *   test:array{rows:list<string>,labels:list<int>}
 * }
 */
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

/**
 * -------------------------------------------------------------------------
 * Builds packed float32 one-hot vectors indexed by class label.
 * -------------------------------------------------------------------------
 *
 * @return list<string>
 */
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

/**
 * -------------------------------------------------------------------------
 * Shuffles the training set and uploads each requested batch to the GPU.
 * -------------------------------------------------------------------------
 *
 * @param array{train:array{rows:list<string>,labels:list<int>},classes:int,features:int,image:?array{h:int,w:int,bg:string}} $data
 * @return list<array{CudaArray,CudaArray,CudaArray,int}>
 */
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

/**
 * -------------------------------------------------------------------------
 * Converts a dataset split into GPU feature and one-hot label tensors.
 * -------------------------------------------------------------------------
 *
 * @param array{rows:list<string>,labels:list<int>} $split
 * @return array{CudaArray,CudaArray,list<int>}
 */
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

/**
 * -------------------------------------------------------------------------
 * Runs a stored evaluation plan and calculates loss and accuracy.
 * -------------------------------------------------------------------------
 *
 * @param list<CudaArray> $params
 * @param list<int> $targets
 * @return array{loss:float,accuracy:float,predictions:list<int>,logits:CudaArray}
 */
function evaluateSet(FusionGraph $plan, CudaArray $x, CudaArray $y, array $params, array $targets): array
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

/**
 * -------------------------------------------------------------------------
 * Calculates per-class metrics and the largest confusion pairs.
 * -------------------------------------------------------------------------
 *
 * @param list<int> $predictions
 * @param list<int> $targets
 * @return array{
 *   macroF1:float,
 *   perClass:list<array{support:int,precision:float,recall:float,f1:float}>,
 *   topConfusions:list<array{truth:int,predicted:int,count:int}>
 * }
 */
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
        'macroF1' => $f1Sum / $classes,
        'perClass' => $perClass,
        'topConfusions' => array_slice($pairs, 0, 5),
    ];
}

/**
 * -------------------------------------------------------------------------
 * Runs the configured network on CPU for a small correctness sample.
 * -------------------------------------------------------------------------
 *
 * @param list<CudaArray> $params
 * @param list<string> $packedRows
 * @return list<list<float>>
 */
function cpuForward(ModelConfiguration $configuration, array $params, array $packedRows): array
{
    $layers = modelLayerCount($configuration);
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
            $in = $configuration->dimensions[$l];
            $outDim = $configuration->dimensions[$l + 1];
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
                    $z[$j] = $zv > 0 ? $zv : ($configuration->activation === 'leaky' ? LEAKY_SLOPE * $zv : 0.0);
                }
            }
            $a = $z;
        }
        $out[] = $a;
    }
    return $out;
}

/**
 * -------------------------------------------------------------------------
 * Applies linear warmup followed by cosine learning-rate decay.
 * -------------------------------------------------------------------------
 */
function scheduleLr(int $epoch, int $epochs, int $warmup, float $base, float $floorRatio = 0.05): float
{
    if ($warmup > 0 && $epoch < $warmup) {
        return $base * ($epoch + 1) / $warmup;
    }
    $span = max(1, $epochs - $warmup - 1);
    $t = min(1.0, max(0.0, ($epoch - $warmup) / $span));
    return $base * ($floorRatio + (1.0 - $floorRatio) * 0.5 * (1.0 + cos(M_PI * $t)));
}

/**
 * -------------------------------------------------------------------------
 * Saves model metadata and packed parameters to the portable model format.
 * -------------------------------------------------------------------------
 *
 * @param list<CudaArray> $params
 */
function saveModel(string $path, ModelConfiguration $configuration, array $params, array $data): void
{
    $image = $data['image'];
    $header = json_encode([
        'format' => 'php-gpu-tensors-mlp',
        'version' => 1,
        'dataset' => $data['name'],
        'dims' => $configuration->dimensions,
        'activation' => $configuration->activation,
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

/**
 * -------------------------------------------------------------------------
 * Loads and validates model metadata and packed parameter buffers.
 * -------------------------------------------------------------------------
 *
 * @return array{0:array<string,mixed>,1:list<string>}
 */
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

/**
 * -------------------------------------------------------------------------
 * Renders a packed grayscale image as a terminal-compatible text preview.
 * -------------------------------------------------------------------------
 */
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

/**
 * -------------------------------------------------------------------------
 * Parses ASCII or binary PGM input into width, height, and grayscale pixels.
 * -------------------------------------------------------------------------
 *
 * @return array{0:int,1:int,2:list<int>}
 */
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

/**
 * -------------------------------------------------------------------------
 * Resizes grayscale pixels using area averaging into the target dimensions.
 * -------------------------------------------------------------------------
 *
 * @param list<int> $pixels
 * @return list<int>
 */
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

/**
 * -------------------------------------------------------------------------
 * Crops and scales a drawing to the configured image input dimensions.
 * -------------------------------------------------------------------------
 *
 * @param list<int> $pixels
 * @return list<int>
 */
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

/**
 * -------------------------------------------------------------------------
 * Parses and validates an integer command-line option.
 * -------------------------------------------------------------------------
 */
function optInt(array $o, string $name, int $default, int $min = 1): int
{
    $value = filter_var($o[$name] ?? $default, FILTER_VALIDATE_INT);
    if ($value === false || $value < $min) {
        throw new InvalidArgumentException("--$name must be an integer >= $min.");
    }
    return $value;
}

/**
 * -------------------------------------------------------------------------
 * Parses and validates a floating-point command-line option.
 * -------------------------------------------------------------------------
 */
function optFloat(array $o, string $name, float $default, float $min = 0.0, float $max = INF): float
{
    $value = filter_var($o[$name] ?? $default, FILTER_VALIDATE_FLOAT);
    if ($value === false || !is_finite($value) || $value < $min || $value > $max) {
        throw new InvalidArgumentException("--$name must be a number between $min and $max.");
    }
    return (float)$value;
}

/**
 * -------------------------------------------------------------------------
 * Returns the command-line usage text for the classifier example.
 * -------------------------------------------------------------------------
 */
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
            --profile prints Fusion kernel/boundary and execution timing diagnostics
Model I/O: --model=<file> (default <script_dir>/training/<dataset>_mlp.pgt)  --no-save  --load
Inference: --predict-index=N (test sample)  --predict-file=digit.pgm [--invert] [--raw]
Other:     --no-selftest  --help
TXT;
}

/**
 * -------------------------------------------------------------------------
 * Parses command-line arguments into the classifier run configuration.
 * -------------------------------------------------------------------------
 *
 * @return array<string,mixed>
 */
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

    $trainingDir = dirname(__DIR__) . DIRECTORY_SEPARATOR . 'training';
    if (!is_dir($trainingDir) && !mkdir($trainingDir, 0775, true) && !is_dir($trainingDir)) {
        throw new RuntimeException("Unable to create the training directory $trainingDir.");
    }

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

/**
 * -------------------------------------------------------------------------
 * Loads either an image dataset or a CSV dataset from the parsed options.
 * -------------------------------------------------------------------------
 *
 * @param array<string,mixed> $cfg
 * @return array{
 *   name:string,features:int,classes:int,classNames:list<string>,
 *   image:null|array{h:int,w:int,mean:float,std:float,bg:string,table:array<string,string>},
 *   train:array{rows:list<string>,labels:list<int>},
 *   val:array{rows:list<string>,labels:list<int>},
 *   test:array{rows:list<string>,labels:list<int>}
 * }
 */
function loadData(array $cfg): array
{
    if ($cfg['dataset'] === 'csv') {
        return loadCsvDataset($cfg['csv'], $cfg['header'], $cfg['labelCol'], $cfg['valFraction'],
            $cfg['testFraction'], $cfg['seed']);
    }
    return loadImageDataset($cfg['dataset'], $cfg['dataDir'], $cfg['urlBase'], $cfg['val'], $cfg['seed']);
}
