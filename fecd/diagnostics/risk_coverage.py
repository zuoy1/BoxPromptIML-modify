"""Coverage-matched risk with expected fractional selection within score ties."""
from __future__ import annotations

import numpy as np

from fecd.diagnostics.metrics import _exact_groups, _histogram_groups

DEFAULT_COVERAGES = (.1, .25, .5, .75, .9)


def _risk_from_groups(counts, errors, coverages):
    coverages = tuple(float(x) for x in coverages)
    if any(not np.isfinite(x) or not 0 < x <= 1 for x in coverages):
        raise ValueError('Coverage must be finite in (0, 1]')
    counts = counts[::-1].astype(np.float64)
    errors = errors[::-1].astype(np.float64)
    n = float(counts.sum())
    cumulative_n = np.r_[0., np.cumsum(counts)]
    cumulative_error = np.r_[0., np.cumsum(errors)]
    output = {'n': int(n), 'aurc': None, 'coverage': {},
              'curve': {'coverage': [], 'risk': []},
              'reason': None if n else 'empty candidate set',
              'tie_policy': 'fractional expected selection within each tie group'}
    for coverage in coverages:
        selected = coverage * n
        error_count = float(np.interp(selected, cumulative_n, cumulative_error))
        risk = error_count / selected if selected else None
        output['coverage'][f'{coverage:g}'] = {
            'coverage': coverage if n else None, 'selected_count': selected,
            'risk': risk, 'precision': 1 - risk if risk is not None else None,
        }
    if not n:
        return output
    group_risk = errors / counts
    before = cumulative_n[:-1]
    after = cumulative_n[1:]
    correction = cumulative_error[:-1] - group_risk * before
    integral = group_risk * counts
    valid = before > 0
    # Between endpoints of a tie group, expected cumulative error is affine in
    # selected count. Integrating its ratio analytically avoids trapezoid bias.
    integral[valid] += correction[valid] * np.log(after[valid] / before[valid])
    output['aurc'] = float(integral.sum() / n)
    output['curve'] = {'coverage': (after / n).tolist(),
                       'risk': (cumulative_error[1:] / after).tolist()}
    return output


def risk_coverage(error, reliability, coverages=DEFAULT_COVERAGES):
    """Exact reliability ordering; conditional coverage refers to given candidates."""
    return _risk_from_groups(*_exact_groups(error, reliability), coverages)


def risk_coverage_from_hist(hist, coverages=DEFAULT_COVERAGES):
    """Pooled risk–coverage using the same tie rule as histogram detection."""
    return _risk_from_groups(*_histogram_groups(hist), coverages)
