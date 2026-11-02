"""Objective case selection and array-based diagnostic visualization helpers."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fecd.confidence.teacher_scores import rule_confidence, sam_confidence
from fecd.diagnostics.metrics import error_detection_metrics, metrics_from_hist, reliability_histogram
from fecd.diagnostics.protocol import PROTOCOL, PROTOCOL_HASH
from fecd.diagnostics.risk_coverage import risk_coverage, risk_coverage_from_hist
from fecd.diagnostics.sample_audit import confidence_methods, region_masks
from fecd.diagnostics.visualizations import save_diagnostic

FIXED_IDS = (
    ('historical_animal', 'Tp_D_CND_M_N_ani00018_sec00096_00138'),
    ('historical_zero', 'Tp_S_CNN_M_N_sec00091_sec00091_10355'),
    ('historical_high_coverage', 'Tp_D_NNN_L_N_ani00010_nat00099_10217'),
    ('historical_max_variance', 'NC2016_2255'),
    ('historical_k1_k8_difference', 'NC2016_3660'),
)


def fractional_selection(reliability, candidates, coverage):
    """Equal weights within a boundary tie; total weight equals coverage*n."""
    score, mask = np.asarray(reliability), np.asarray(candidates, bool)
    if score.shape != mask.shape or not 0 < coverage <= 1 or not np.isfinite(score).all():
        raise ValueError('Expected finite same-shaped arrays and coverage in (0,1]')
    result = np.zeros(score.shape, np.float32)
    if not mask.any():
        return result
    unique, counts = np.unique(score[mask], return_counts=True)
    unique, counts = unique[::-1], counts[::-1]
    cumulative = np.cumsum(counts)
    target = coverage * int(mask.sum())
    index = min(int(np.searchsorted(cumulative, target, side='left')), len(unique) - 1)
    before = cumulative[index - 1] if index else 0
    result[mask & (score > unique[index])] = 1.
    result[mask & (score == unique[index])] = (target - before) / counts[index]
    return result


def select_cases(per_image, split_records):
    selected = []
    for case, sample_id in FIXED_IDS:
        row = next((r for r in split_records if r['sample_id'] == sample_id), None)
        selected.append({'case': case, 'sample_key': row['sample_key'] if row else None,
                         'reason': 'fixed historical identity; GT intentionally unavailable for train_fit'})
    objectives = [('high_confidence_fp', lambda r: r['stable_counts']['high_pos']['fp']),
                  ('high_confidence_fn', lambda r: r['stable_counts']['high_neg']['fn']),
                  ('natural_texture_alarm', lambda r: r['evidence_distributions']['TN']['r']['mean']),
                  ('jpeg_natural_alarm', lambda r: r['evidence_distributions']['TN']['jpeg']['mean'])]
    for case, key in objectives:
        eligible = [r for r in per_image if key(r) is not None and key(r) > 0]
        chosen = max(eligible, key=lambda r: (key(r), r['sample_key'])) if eligible else None
        selected.append({'case': case, 'sample_key': chosen['sample_key'] if chosen else None,
                         'objective_value': float(key(chosen)) if chosen else None,
                         'reason': 'maximum predeclared high-consensus count or mean evidence in true-negative natural pixels'})
    eligible = []
    for row in per_image:
        scores = row['regions']['pos']
        base, rule = scores['S0'], scores['S4']
        before, after = base['coverage']['0.5']['risk'], rule['coverage']['0.5']['risk']
        if base['n'] >= 50 and base['error_count'] >= 50 and before is not None and after is not None:
            eligible.append((before - after, row))
    for case, positive in (('evidence_success', True), ('evidence_failure', False)):
        candidates = [(delta, row) for delta, row in eligible if (delta > 0 if positive else delta <= 0)]
        chosen = (max(candidates, key=lambda x: (x[0], x[1]['sample_key'])) if positive
                  else min(candidates, key=lambda x: (x[0], x[1]['sample_key']))) if candidates else None
        selected.append({'case': case, 'sample_key': chosen[1]['sample_key'] if chosen else None,
                         'risk_reduction_at_50pct': chosen[0] if chosen else None,
                         'reason': ('largest matched-50%-coverage risk improvement' if positive else 'largest matched-50%-coverage risk deterioration or no improvement')
                         if chosen else 'no eligible ' + ('success' if positive else 'failure')})
    return selected


def visualize_cases(per_image_path, split_records, load_sample, output_dir):
    """The callback alone loads labels; this module receives arrays and metadata."""
    records = [json.loads(line) for line in Path(per_image_path).read_text().splitlines() if line]
    selected = select_cases(records, split_records)
    by_key = {r['sample_key']: r for r in split_records}
    output_dir = Path(output_dir)
    for selection in selected:
        if not selection['sample_key']:
            continue
        row = by_key[selection['sample_key']]
        sample = load_sample(row, include_rgb=True)
        k8, normalized = sample['k8'], sample['normalized']
        mu, var, agreement = (k8[k].astype(np.float32) for k in ('mu', 'var', 'agreement'))
        pred = mu >= .5
        confidence = sam_confidence(mu, var, agreement)
        methods = confidence_methods(confidence, normalized, pred)
        difference = fractional_selection(methods['S4'], pred, .5) - fractional_selection(confidence, pred, .5)
        maps = {'K1 mask': (sample['k1']['mu'] >= .5).astype(float), 'K8 mu': mu, 'K8 var': var,
                'c_sam': confidence, **{name: normalized[name] for name in ('hp', 'noise', 'jpeg', 'fft', 'r')},
                'c_pos': rule_confidence(confidence, normalized['r'])[0], 'selection50 S4 minus S0': difference}
        gt = sample['gt']
        if gt is not None:
            maps.update({'GT diagnostic only': gt.astype(float), 'FP': (pred & ~gt).astype(float),
                         'FN': (~pred & gt).astype(float)})
        selection.update({'sample_id': row['sample_id'], 'dataset': row['dataset'], 'subset': row['subset'],
                          'gt_included': gt is not None, 'original_format': sample['evidence_metadata']['original_format'],
                          'original_jpeg_quantization': sample['evidence_metadata']['original_jpeg_quantization']})
        caption = f"{selection['case']} | {row['dataset']}/{row['sample_id']} | {selection['reason']}"
        if gt is None:
            caption += ' | train_fit: GT intentionally unavailable'
        selection['visualization'] = save_diagnostic(output_dir / 'cases' / (selection['case'] + '.png'),
                                                     sample['rgb224'], sample['boxes224'], maps, caption,
                                                     contains_gt=gt is not None)
    output = {'protocol_hash': PROTOCOL_HASH, 'cases': selected,
              'interpretation': 'post-hoc explanatory ranking; not feature or threshold selection; texture is a TN evidence proxy, not semantic proof'}
    (output_dir / 'cases_selection.json').write_text(json.dumps(output, indent=2) + '\n')
    return output


def micro_sensitivity(split_records, load_sample, output_dir):
    selected = []
    for dataset in sorted({r['dataset'] for r in split_records}):
        rows = [r for r in split_records if r['dataset'] == dataset and r['subset'] == 'teacher_diag']
        selected.extend(sorted(rows, key=lambda r: r['sample_key'])[:8])
    pools = defaultdict(lambda: [[], []])
    for row in selected:
        sample = load_sample(row)
        k8, gt = sample['k8'], sample['gt']
        mu, var, agreement = (k8[k].astype(np.float32) for k in ('mu', 'var', 'agreement'))
        pred = mu >= .5
        confidence = sam_confidence(mu, var, agreement)
        methods = confidence_methods(confidence, sample['normalized'], pred)
        masks = region_masks(pred, confidence, var, k8['box_masks'])
        errors = pred != gt
        for region in ('pos', 'high_pos', 'neg'):
            for method, scores in methods.items():
                for dataset in ('ALL', row['dataset']):
                    data = pools[(dataset, region, method)]
                    data[0].append(errors[masks[region]])
                    data[1].append(scores[masks[region]])
    results = []
    for (dataset, region, method), (errors, scores) in sorted(pools.items()):
        error, score = np.concatenate(errors), np.concatenate(scores)
        histogram = reliability_histogram(error, score, PROTOCOL['micro_histogram_bins'])
        exact = error_detection_metrics(error, score)
        exact_risk = risk_coverage(error, score, PROTOCOL['coverages'])
        exact_risk.pop('curve')
        binned = metrics_from_hist(histogram)
        binned_risk = risk_coverage_from_hist(histogram, PROTOCOL['coverages'])
        binned_risk.pop('curve')
        for risk in (exact_risk, binned_risk):
            for coverage in risk['coverage'].values():
                if coverage['selected_count'] < PROTOCOL['minimum_coverage_pixels']:
                    coverage.update(risk=None, precision=None,
                                    reason='fewer than 10 expected selected pixels')
        exact.update(exact_risk)
        binned.update(binned_risk)
        delta = {metric: binned[metric] - exact[metric]
                 if exact[metric] is not None and binned[metric] is not None else None
                 for metric in ('auroc', 'auprc', 'aurc')}
        results.append({'dataset': dataset, 'region': region, 'method': method,
                        'exact': exact, 'histogram': binned, 'histogram_minus_exact': delta})
    output = {'protocol_hash': PROTOCOL_HASH, 'selection': 'first8 hash-ordered teacher_diag per dataset',
              'sample_keys': [r['sample_key'] for r in selected], 'results': results}
    (Path(output_dir) / 'micro_sensitivity.json').write_text(json.dumps(output, indent=2) + '\n')
    return output
