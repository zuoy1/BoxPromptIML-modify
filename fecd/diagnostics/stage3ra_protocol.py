"""Stage-3R-A developmental decisions frozen before opening new_dev labels."""
from fecd.diagnostics.protocol import PROTOCOL as STAGE3_PROTOCOL
from fecd.pseudo.cache_schema import canonical_hash

METHODS = ('base_margin', 'mean_margin', 'variance', 'threshold', 'c_sam', 'q_a')
PROTOCOL = {
    'version': 'fecd-stage3ra-development-v1', 'seed': 20261029,
    'allowed_gt_partition': 'new_dev', 'confirm_gt_allowed': False,
    'methods': list(METHODS),
    'score_definitions': {
        'base_margin': '2*abs(base_mu-0.5)', 'mean_margin': '2*abs(mu-0.5)',
        'variance': '1-clamp(4*var,0,1)',
        'threshold': '1-4*agreement*(1-agreement)',
        'c_sam': '(1-clamp(4*var,0,1))*(1-4*agreement*(1-agreement))*sqrt(2*abs(mu-0.5))',
        'q_a': 'c_sam/(1+abs(mu-base_mu))',
    },
    **{key: STAGE3_PROTOCOL[key] for key in (
        'prediction_threshold', 'gt_threshold', 'high_confidence', 'low_variance',
        'box_erosion', 'box_ring', 'coverages', 'bootstrap_resamples',
        'micro_histogram_bins', 'minimum_coverage_pixels')},
    'primary': 'q_a-c_sam high_pos exact per-image AUROC',
    'secondary': 'q_a-c_sam pos exact per-image risk@0.5',
    'negative_guard': 'q_a-c_sam neg exact per-image risk@0.5',
    'minimum_primary_effect': .02, 'maximum_secondary_effect': -.002,
    'negative_noninferiority_margin': .001,
    'minimum_domain_groups': 20, 'minimum_domain_images': 20,
    'minimum_domain_errors': 50,
    'random_budget': 'analytic uniform random selection expectation, no Monte Carlo draw',
    'count_policy': 'coverage*n expected pixels; fractional boundary ties, not integer realized masks',
    'realized_budget': {
        'count': 'Python round(coverage*n), shared for random/c_sam/q_a',
        'seed': 'first 8 bytes big-endian SHA256("20261029:{sample_key}:{sign}")',
        'rng': 'NumPy default_rng(seed).permutation(n); shared permutation tie priorities',
        'score_order': 'np.lexsort((priority,-score)); random uses the same permutation',
        'role': 'descriptive actual integer selected counts; fractional-tie primary unchanged',
    },
    'conditional_edges': {'mean_margin': [0., .5, .8, .9, .95, .99, 1.],
                          'variance': [0., .0001, .001, .01, .05, .2501]},
    'conditional_policy': 'left closed right open; final bin includes upper edge; descriptive, not causal',
    'composition_coverage': .5,
    'scope': 'development only; no confirmation, training, SAM inference or alternative formulas',
}
PROTOCOL_HASH = canonical_hash(PROTOCOL)
