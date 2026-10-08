<?php
declare(strict_types=1);

/**
 * -------------------------------------------------------------------------
 * Stores the architecture and optimizer settings shared by model functions.
 * -------------------------------------------------------------------------
 */
final class ModelConfiguration
{
    /** @param list<int> $dimensions Input, hidden, and output layer widths. */
    public function __construct(
        public readonly array $dimensions,
        public readonly string $activation,
        public readonly string $optimizer,
        public readonly float $weightDecay,
        public readonly float $gradientClip,
        public readonly float $labelSmoothing,
        public readonly float $momentum = 0.9,
        public readonly float $beta1 = 0.9,
        public readonly float $beta2 = 0.999
    ) {
    }
}
