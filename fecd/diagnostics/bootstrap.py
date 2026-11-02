"""Paired, source-group cluster bootstrap for per-image metric differences."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(method_values, baseline_values, groups, seed=20261029,
                     n_resamples=2000):
    """Resample source groups and retain every valid paired image in each draw.

    The estimand is the image-weighted macro difference, not a pixel-weighted
    difference or a group-equal mean. Missing metrics are removed as pairs.
    """
    if not (len(method_values) == len(baseline_values) == len(groups)):
        raise ValueError('Paired metrics and source groups must have equal length')
    if not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError('n_resamples must be an integer >= 2')
    differences, labels = [], []
    for method, baseline, group in zip(method_values, baseline_values, groups):
        if method is None or baseline is None:
            continue
        if np.isfinite(method) and np.isfinite(baseline):
            if group is None or str(group) == '':
                raise ValueError('Each valid pair requires an explicit source group')
            differences.append(float(method) - float(baseline))
            labels.append(str(group))
    differences = np.asarray(differences, dtype=np.float64)
    unique_groups, indices = np.unique(labels, return_inverse=True)
    output = {'difference': float(differences.mean()) if len(differences) else None,
              'ci95': None, 'n_images': len(differences),
              'n_groups': len(unique_groups),
              'excluded_pairs': len(groups) - len(differences),
              'n_resamples': n_resamples, 'seed': seed, 'reason': None,
              'estimand': 'image-weighted mean paired difference; whole source-group resampling'}
    if len(unique_groups) < 2:
        output['reason'] = 'fewer than two source groups with finite paired metrics'
        return output
    group_sums = np.bincount(indices, weights=differences)
    group_counts = np.bincount(indices)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    # Chunk draws so bootstrap memory remains bounded for large image lists.
    for begin in range(0, n_resamples, 100):
        end = min(begin + 100, n_resamples)
        draws = rng.integers(0, len(unique_groups), (end - begin, len(unique_groups)))
        estimates[begin:end] = group_sums[draws].sum(1) / group_counts[draws].sum(1)
    output['ci95'] = np.quantile(estimates, [.025, .975]).tolist()
    return output
