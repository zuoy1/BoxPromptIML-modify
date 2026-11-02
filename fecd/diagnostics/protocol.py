"""Fixed stage-3 choices, deliberately independent of diagnostic labels."""
from fecd.pseudo.cache_schema import canonical_hash

PROTOCOL = {
    'version': 'fecd-stage3-audit-v1', 'seed': 20261029,
    'prediction_threshold': 0.5, 'gt_threshold': 127,
    'high_confidence': 0.90, 'low_variance': 0.01,
    'box_erosion': 3, 'box_ring': 5,
    'coverages': [0.1, 0.25, 0.5, 0.75, 0.9],
    'bootstrap_resamples': 2000,
    'micro_histogram_bins': 4096,
    'micro_definition': 'uniform reliability bins; all within-bin scores tied',
    'macro_definition': 'exact per-image metrics; image-weighted macro, source-group bootstrap',
    'area_strata': [0.01, 0.1, 0.3],
    'trusted_threshold': 0.9,
    'primary': 'S4 minus S0 exact macro AUROC on high-consensus predicted-positive pixels',
    'secondary': 'S4 minus S0 exact macro risk at 50% predicted-positive coverage',
    'go': 'primary CI low > 0; secondary CI high < 0; non-worse dataset risk for datasets >=20 images and >=50 candidate errors',
    'conditional_go': 'both primary and secondary favorable point estimates; CI crosses zero or domain reversal with explicit follow-up',
    'no_go': 'otherwise, or severe domain confounding; do not enter stage4',
    'jpeg_sensitivity': 'Q75/Q95 recompression on fixed first 8 hash-ordered teacher_diag images per dataset; diagnostic only, never change Q90 or statistics',
    'micro_sensitivity': 'exact pooled metrics on fixed first 8 hash-ordered teacher_diag images per dataset',
    'minimum_coverage_pixels': 10,
}
PROTOCOL_HASH = canonical_hash(PROTOCOL)
