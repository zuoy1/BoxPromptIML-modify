"""Fixed residual-region differences, with no correctness interpretation."""
from __future__ import annotations

import numpy as np

from fecd.forensic_reference.matching import MIN_REFERENCES


QUANTILE_LEVELS = tuple(i / 20 for i in range(1, 20))
SCALE_EPSILON = .005
MAD_EPSILON = .001


def _descriptor(patch):
    values = np.asarray(patch['quantiles'], dtype=np.float64)
    mad = float(patch['mad'])
    if (values.shape != (len(QUANTILE_LEVELS),) or not np.isfinite(values).all()
            or (np.abs(values) > 1).any() or (np.diff(values) < 0).any()
            or not np.isfinite(mad) or mad < 0 or mad > 1):
        raise ValueError('Invalid finite ordered residual quantiles or nonnegative MAD')
    return values, mad


def compare_regions(candidate, references):
    """Median reference-wise standardized 19-quantile signed-residual distance.

    The quantile mean absolute difference approximates a central-distribution
    Wasserstein-1 distance; it is not the full empirical W1. MAD scaling fixes
    units without converting distance to anomaly probability or score direction.
    """
    values, mad = _descriptor(candidate)
    descriptors = [_descriptor(reference) for reference in references]
    output = {'status': 'not_applicable', 'n_references': len(references),
              'standardized_quantile_distance': None, 'raw_quantile_distance': None,
              'log_mad_ratio': None, 'reference_values': [],
              'leave_one_out_range': None}
    if len(references) < MIN_REFERENCES:
        output['reason'] = 'fewer_than_three_nonoverlapping_matched_references'
        return output
    for reference, (ref_values, ref_mad) in zip(references, descriptors):
        distance = float(np.mean(np.abs(values-ref_values)))
        output['reference_values'].append({
            'id': reference['id'], 'raw_quantile_distance': distance,
            'standardized_quantile_distance': distance / (SCALE_EPSILON+(mad+ref_mad)/2),
            'log_mad_ratio': float(abs(np.log((mad+MAD_EPSILON)/(ref_mad+MAD_EPSILON))))})
    for name in ('standardized_quantile_distance', 'raw_quantile_distance', 'log_mad_ratio'):
        output[name] = float(np.median([value[name] for value in output['reference_values']]))
    primary = [value['standardized_quantile_distance'] for value in output['reference_values']]
    leave_one_out = [float(np.median(primary[:i]+primary[i+1:])) for i in range(len(primary))]
    output.update(status='applicable', reason=None,
                  leave_one_out_range=[min(leave_one_out), max(leave_one_out)])
    return output
