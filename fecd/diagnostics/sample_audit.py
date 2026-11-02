"""Array-only teacher diagnosis; filesystem and label loading belong to the CLI."""
from __future__ import annotations

import cv2
import numpy as np

from fecd.confidence.teacher_scores import rule_confidence, sam_confidence
from fecd.diagnostics.metrics import error_detection_metrics, reliability_histogram
from fecd.diagnostics.protocol import PROTOCOL
from fecd.diagnostics.risk_coverage import risk_coverage


def confidence_methods(c_sam, normalized, pred):
    methods = {'S0': c_sam}
    for method, feature in (('S1', 'noise'), ('S2', 'jpeg'), ('S3', 'fft'),
                            ('S4', 'r'), ('S_HP', 'hp')):
        positive, negative = rule_confidence(c_sam, normalized[feature])
        methods[method] = np.where(pred, positive, negative).astype(np.float32)
    methods['S5'] = np.where(pred, normalized['r'], 1 - normalized['r']).astype(np.float32)
    return methods


def region_masks(pred, confidence, var, box_masks):
    box = np.asarray(box_masks).astype(bool).any(axis=0).astype(np.uint8)
    erode = PROTOCOL['box_erosion']
    ring = PROTOCOL['box_ring']
    inside = cv2.erode(box, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    dilated = cv2.dilate(box, np.ones((2 * ring + 1, 2 * ring + 1), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    box = box.astype(bool)
    lowvar = var <= PROTOCOL['low_variance']
    high = (confidence >= PROTOCOL['high_confidence']) & lowvar
    masks = {'all': np.ones_like(pred, bool), 'pos': pred, 'neg': ~pred,
             'high_pos': pred & high, 'high_neg': ~pred & high,
             'lowvar_pos': pred & lowvar, 'lowvar_neg': ~pred & lowvar,
             'box_inside': inside, 'box_edge': box & ~inside,
             'box_ring': dilated & ~box, 'box_far': ~dilated}
    for name in ('box_inside', 'box_edge', 'box_ring'):
        masks['pos_' + name] = pred & masks[name]
    return masks


def _confusion(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else None}


def _distribution(value):
    if not value.size:
        return {'n': 0, 'mean': None, 'q10': None, 'q50': None, 'q90': None}
    quantiles = np.quantile(value, [.1, .5, .9])
    return {'n': int(value.size), 'mean': float(value.mean()),
            **dict(zip(('q10', 'q50', 'q90'), map(float, quantiles)))}


def _metric_record(error, score):
    metrics = error_detection_metrics(error, score)
    risk = risk_coverage(error, score, PROTOCOL['coverages'])
    risk.pop('curve')
    for entry in risk['coverage'].values():
        entry['reason'] = None
        if entry['selected_count'] < PROTOCOL['minimum_coverage_pixels']:
            entry.update(risk=None, precision=None,
                         reason=f"fewer than {PROTOCOL['minimum_coverage_pixels']} expected selected pixels")
    return {**metrics, **{k: v for k, v in risk.items() if k != 'n'}}


def _trusted(error, score, candidate):
    selected = candidate & (score >= PROTOCOL['trusted_threshold'])
    count = int(selected.sum())
    candidate_count = int(candidate.sum())
    return {'selected_count': count, 'error_count': int(error[selected].sum()),
            'precision': float(1 - error[selected].mean()) if count else None,
            'conditional_coverage': count / candidate_count if candidate_count else None,
            'full_coverage': count / candidate.size}


def audit_sample(mu, var, agreement, k1_mu, box_masks, normalized, gt):
    arrays = [np.asarray(x) for x in (mu, var, agreement, k1_mu, gt)]
    if any(value.shape != (224, 224) for value in arrays):
        raise ValueError('Audit arrays must all use the 224x224 SAM grid')
    if not np.isin(gt, (0, 1)).all():
        raise ValueError('Diagnostic label array must be binary')
    gt = np.asarray(gt, bool)
    pred = np.asarray(mu) >= PROTOCOL['prediction_threshold']
    confidence = sam_confidence(mu, var, agreement)
    methods = confidence_methods(confidence, normalized, pred)
    regions = region_masks(pred, confidence, var, box_masks)
    error = pred != gt
    fp, fn = pred & ~gt, ~pred & gt
    classes = {'TP': pred & gt, 'FP': fp, 'TN': ~pred & ~gt, 'FN': fn}
    fraction = float(gt.mean())
    area_stratum = ('<1%' if fraction < .01 else '1-10%' if fraction < .1
                    else '10-30%' if fraction < .3 else '>=30%')
    result = {'gt_area_fraction': fraction, 'area_stratum': area_stratum,
              'segmentation': {'K1': _confusion(k1_mu >= .5, gt), 'K8': _confusion(pred, gt)},
              'score_distributions': {name: _distribution(confidence[mask]) for name, mask in classes.items()},
              'evidence_distributions': {
                  name: {feature: _distribution(normalized[feature][mask])
                         for feature in ('hp', 'noise', 'jpeg', 'fft', 'r')}
                  for name, mask in classes.items()},
              'stable_counts': {}, 'region_counts': {}, 'regions': {}, 'trusted': {}}
    for name in ('high_pos', 'high_neg', 'lowvar_pos', 'lowvar_neg'):
        mask = regions[name]
        result['stable_counts'][name] = {'n': int(mask.sum()), 'errors': int(error[mask].sum()),
                                         'fp': int(fp[mask].sum()), 'fn': int(fn[mask].sum())}
    for name, score in methods.items():
        result['trusted'][name] = {'pos': _trusted(error, score, pred),
                                  'neg': _trusted(error, score, ~pred)}
    histograms = {}
    for region, mask in regions.items():
        result['region_counts'][region] = {'n': int(mask.sum()), 'errors': int(error[mask].sum()),
                                           'fp': int(fp[mask].sum()), 'fn': int(fn[mask].sum())}
        result['regions'][region] = {}
        for method, score in methods.items():
            result['regions'][region][method] = _metric_record(error[mask], score[mask])
            histograms[(region, method)] = reliability_histogram(
                error[mask], score[mask], PROTOCOL['micro_histogram_bins'])
    return result, histograms
