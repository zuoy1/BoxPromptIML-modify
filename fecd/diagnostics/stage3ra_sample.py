"""Array-only development diagnosis on fixed K8 prediction candidate sets."""
from __future__ import annotations

import numpy as np

from fecd.confidence.stage3ra_scores import teacher_scores
from fecd.diagnostics.metrics import reliability_histogram
from fecd.diagnostics.risk_coverage import risk_coverage
from fecd.diagnostics.sample_audit import _confusion, _distribution, _metric_record, region_masks
from fecd.diagnostics.stage3ra_protocol import METHODS, PROTOCOL
from fecd.diagnostics.stage3ra_budget import realized_budgets


def _budget(error, score, candidate, random=False):
    errors = error[candidate]
    n = int(errors.size)
    result = {}
    curve = None if random else risk_coverage(errors, score[candidate], PROTOCOL['coverages'])
    for coverage in PROTOCOL['coverages']:
        key = f'{coverage:g}'
        selected = coverage * n
        risk = float(errors.mean()) if n else None
        if not random:
            risk = curve['coverage'][key]['risk']
        count_errors = selected * risk if risk is not None else 0.
        sufficient = selected >= PROTOCOL['minimum_coverage_pixels']
        result[key] = {
            'n_candidates': n, 'selected_count': selected, 'error_count': count_errors,
            'risk': risk if sufficient else None,
            'precision': 1 - risk if sufficient else None,
            'conditional_coverage': coverage if n else None,
            'full_coverage': selected / candidate.size,
            'reason': None if sufficient else 'fewer than 10 expected selected pixels',
            'count_policy': 'expected fractional selection, not realized integer pixels',
        }
    return result


def _selection_weights(score, candidate, coverage):
    values = score[candidate]
    weights = np.zeros(score.shape, np.float64)
    if not values.size:
        return weights
    selected = coverage * len(values)
    boundary = np.sort(values)[::-1][min(int(np.ceil(selected)) - 1, len(values) - 1)]
    above = candidate & (score > boundary)
    tied = candidate & (score == boundary)
    weights[above] = 1
    weights[tied] = (selected - int(above.sum())) / int(tied.sum())
    return weights


def _composition(error, scores, masks):
    composition, balanced = {}, {}
    boxes = ('box_inside', 'box_edge', 'box_ring', 'box_far')
    coverage = PROTOCOL['composition_coverage']
    for sign in ('pos', 'neg'):
        composition[sign], balanced[sign] = {}, {}
        for method in ('c_sam', 'q_a'):
            weights = _selection_weights(scores[method], masks[sign], coverage)
            parts = {}
            balanced_n, balanced_error = 0., 0.
            for box in boxes:
                candidate = masks[sign] & masks[box]
                chosen = weights[candidate]
                selected, errors = float(chosen.sum()), float((chosen * error[candidate]).sum())
                parts[box] = {'n_candidates': int(candidate.sum()),
                              'selected_count': selected, 'error_count': errors,
                              'precision': 1 - errors / selected if selected else None}
                local_weights = _selection_weights(scores[method], candidate, coverage)
                balanced_n += float(local_weights.sum())
                balanced_error += float((local_weights * error).sum())
            composition[sign][method] = parts
            balanced[sign][method] = {
                'selected_count': balanced_n, 'error_count': balanced_error,
                'risk': balanced_error / balanced_n if balanced_n >= 10 else None,
                'reason': None if balanced_n >= 10 else 'fewer than 10 expected selected pixels',
            }
    return composition, balanced


def _conditioned(error, scores, variance, masks):
    result = {}
    variables = {'mean_margin': scores['mean_margin'], 'variance': variance}
    for region in ('pos', 'high_pos', 'neg'):
        result[region] = {}
        for name, value in variables.items():
            edges = PROTOCOL['conditional_edges'][name]
            bins = {}
            for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                final = i == len(edges) - 2
                label = f'[{left:g},{right:g}' + (']' if final else ')')
                candidate = masks[region] & (value >= left) & ((value <= right) if final else (value < right))
                bins[label] = {method: _metric_record(error[candidate], scores[method][candidate])
                               for method in ('c_sam', 'q_a', 'mean_margin', 'variance')}
            result[region][name] = bins
    return result


def audit_sample(mu, var, agreement, base_mu, box_masks, gt, sample_key=None):
    """Return JSON-ready per-image diagnostics and additive histograms; no IO."""
    arrays = [np.asarray(x) for x in (mu, var, agreement, base_mu, gt)]
    if any(value.shape != (224, 224) for value in arrays):
        raise ValueError('All diagnostic arrays must use the frozen 224x224 grid')
    if not np.isin(gt, (0, 1)).all():
        raise ValueError('Diagnostic labels must be binary')
    boxes = np.asarray(box_masks)
    if boxes.ndim != 3 or boxes.shape[1:] != (224, 224) or not np.isin(boxes, (0, 1)).all():
        raise ValueError('Box masks must be binary [N,224,224]')
    scores = teacher_scores(mu, var, agreement, base_mu)
    pred, gt = np.asarray(mu) >= PROTOCOL['prediction_threshold'], np.asarray(gt, bool)
    error = pred != gt
    masks = region_masks(pred, scores['c_sam'], var, boxes)
    for name in ('box_inside', 'box_edge', 'box_ring', 'box_far'):
        masks['pos_' + name] = pred & masks[name]
        masks['neg_' + name] = ~pred & masks[name]
    fraction = float(gt.mean())
    result = {
        'gt_area_fraction': fraction,
        'area_stratum': '<1%' if fraction < .01 else '1-10%' if fraction < .1 else '10-30%' if fraction < .3 else '>=30%',
        'segmentation': {'K1': _confusion(np.asarray(base_mu) >= .5, gt), 'K8': _confusion(pred, gt)},
        'regions': {}, 'region_counts': {}, 'score_distributions': {},
        'budgets': {}, 'base_mean_difference': _distribution(abs(np.asarray(mu)-base_mu)),
    }
    for name, candidate in {'TP': pred & gt, 'FP': pred & ~gt, 'TN': ~pred & ~gt, 'FN': ~pred & gt}.items():
        result['score_distributions'][name] = {
            method: _distribution(score[candidate]) for method, score in scores.items()}
    histograms = {}
    for region, candidate in masks.items():
        result['region_counts'][region] = {
            'n': int(candidate.sum()), 'errors': int(error[candidate].sum()),
            'fp': int((candidate & pred & ~gt).sum()), 'fn': int((candidate & ~pred & gt).sum())}
        result['regions'][region] = {}
        for method in METHODS:
            result['regions'][region][method] = _metric_record(error[candidate], scores[method][candidate])
            histograms[(region, method)] = reliability_histogram(
                error[candidate], scores[method][candidate], PROTOCOL['micro_histogram_bins'])
    for sign in ('pos', 'neg'):
        result['budgets'][sign] = {
            'random': _budget(error, scores['c_sam'], masks[sign], random=True),
            **{method: _budget(error, scores[method], masks[sign]) for method in ('c_sam', 'q_a')}}
    result['selection_composition'], result['region_balanced'] = _composition(error, scores, masks)
    result['conditioned_metrics'] = _conditioned(error, scores, np.asarray(var), masks)
    result['realized_budgets'] = realized_budgets(error, scores, masks, sample_key or 'array-only')
    if sample_key is not None:
        result['sample_key'] = sample_key
    return result, histograms
