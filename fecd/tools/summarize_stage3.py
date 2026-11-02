"""Aggregate diagnostic outputs without opening images, GT, or SAM records."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from fecd.diagnostics.bootstrap import paired_bootstrap
from fecd.diagnostics.artifact_validation import validate_run
from fecd.diagnostics.metrics import metrics_from_hist
from fecd.diagnostics.protocol import PROTOCOL, PROTOCOL_HASH
from fecd.diagnostics.risk_coverage import risk_coverage_from_hist
from fecd.diagnostics.visualizations import save_risk_curves


def _mean(values):
    valid = [float(x) for x in values if x is not None and np.isfinite(x)]
    return {'mean': float(np.mean(valid)) if valid else None,
            'n_valid': len(valid), 'n_total': len(values)}


def _metric(row, region, method, metric):
    value = row.get('regions', {}).get(region, {}).get(method, {})
    if metric.startswith('risk@'):
        return value.get('coverage', {}).get(metric[5:], {}).get('risk')
    return value.get(metric)


def _macro(rows):
    keys = sorted({(region, method) for row in rows
                   for region, methods in row['regions'].items() for method in methods})
    result = {}
    for region, method in keys:
        values = [row['regions'].get(region, {}).get(method, {}) for row in rows]
        output = {key: _mean([value.get(key) for value in values])
                  for key in ('auroc', 'auprc', 'error_prevalence', 'aurc')}
        output['n_candidates'] = int(sum(value.get('n', 0) for value in values))
        output['n_errors'] = int(sum(value.get('error_count', 0) for value in values))
        output['coverage'] = {
            f'{coverage:g}': {key: _mean([value.get('coverage', {}).get(f'{coverage:g}', {}).get(key)
                                        for value in values])
                              for key in ('risk', 'precision')}
            for coverage in PROTOCOL['coverages']}
        result.setdefault(region, {})[method] = output
    return result


def _segmentation(rows):
    result = {}
    for teacher in ('K1', 'K8'):
        confusion = {key: int(sum(row['segmentation'][teacher][key] for row in rows))
                     for key in ('tp', 'fp', 'tn', 'fn')}
        tp, fp, fn = confusion['tp'], confusion['fp'], confusion['fn']
        f1_denominator, iou_denominator = 2 * tp + fp + fn, tp + fp + fn
        result[teacher] = {
            'confusion': confusion,
            'micro': {'f1': 2 * tp / f1_denominator if f1_denominator else None,
                      'iou': tp / iou_denominator if iou_denominator else None},
            'macro': {key: _mean([row['segmentation'][teacher].get(key) for row in rows])
                      for key in ('f1', 'iou')},
        }
    return result


def _trusted(rows):
    methods = sorted({method for row in rows for method in row.get('trusted', {})})
    output = {}
    for method in methods:
        output[method] = {}
        for region in ('pos', 'neg'):
            values = [row.get('trusted', {}).get(method, {}).get(region, {}) for row in rows]
            selected = sum(value.get('selected_count', 0) for value in values)
            correct = sum(value.get('selected_count', 0) * value['precision']
                          for value in values if value.get('precision') is not None)
            candidates = sum(row['regions'].get(region, {}).get(method, {}).get('n', 0) for row in rows)
            all_pixels = sum(row['regions'].get('all', {}).get('S0', {}).get('n', 0) for row in rows)
            output[method][region] = {
                'macro': {key: _mean([value.get(key) for value in values])
                          for key in ('precision', 'conditional_coverage', 'full_coverage')},
                'micro': {'selected_count': float(selected),
                          'precision': correct / selected if selected else None,
                          'conditional_coverage': selected / candidates if candidates else None,
                          'full_coverage': selected / all_pixels if all_pixels else None},
            }
    return output


def _bootstrap(rows):
    groups = [f"{row['dataset']}:{row['source_group_id']}" for row in rows]
    comparisons = {}
    for region in ('pos', 'high_pos', 'neg'):
        comparisons[region] = {}
        for method in ('S1', 'S2', 'S3', 'S4', 'S5', 'S_HP'):
            comparisons[region][method] = {}
            for metric in ('auroc', 'auprc', 'aurc', 'risk@0.5'):
                comparisons[region][method][metric] = paired_bootstrap(
                    [_metric(row, region, method, metric) for row in rows],
                    [_metric(row, region, 'S0', metric) for row in rows], groups,
                    seed=PROTOCOL['seed'], n_resamples=PROTOCOL['bootstrap_resamples'])
    return comparisons


def _diagnostic_counts(rows):
    stable = {}
    regions = sorted({region for row in rows for region in row.get('stable_counts', {})})
    for region in regions:
        values = [row.get('stable_counts', {}).get(region, {}) for row in rows]
        stable[region] = {key: int(sum(value.get(key, 0) for value in values))
                          for key in ('n', 'errors', 'fp', 'fn')}
        stable[region]['images_with_errors'] = sum(value.get('errors', 0) > 0 for value in values)
        count = stable[region]['n']
        stable[region]['error_fraction'] = stable[region]['errors'] / count if count else None
    distributions = {}
    for label in ('TP', 'FP', 'TN', 'FN'):
        values = [row.get('score_distributions', {}).get(label, {}) for row in rows]
        count = sum(value.get('n', 0) for value in values)
        total = sum(value.get('n', 0) * value['mean'] for value in values if value.get('mean') is not None)
        distributions[label] = {'n': int(count), 'pixel_weighted_mean': total / count if count else None,
                                'macro_image_statistics': {key: _mean([value.get(key) for value in values])
                                                           for key in ('mean', 'q10', 'q50', 'q90')}}
    return {'stable_counts': stable, 'sam_score_distributions': distributions}


def decide(summaries):
    """Apply frozen quantitative criteria; domain-confounding review is separate."""
    comparisons = summaries['ALL']['paired_differences']
    primary = comparisons['high_pos']['S4']['auroc']
    secondary = comparisons['pos']['S4']['risk@0.5']
    eligibility = {}
    for dataset, summary in summaries.items():
        if dataset == 'ALL':
            continue
        candidates = summary['macro'].get('pos', {}).get('S0', {})
        eligible = summary['n_images'] >= 20 and candidates.get('n_errors', 0) >= 50
        delta = summary['paired_differences']['pos']['S4']['risk@0.5']['difference']
        eligibility[dataset] = {'eligible': eligible, 'n_images': summary['n_images'],
                                'pos_errors': candidates.get('n_errors', 0),
                                'risk50_difference': delta,
                                'nonworse': delta is not None and delta <= 0}
    favorable = (primary['difference'] is not None and primary['difference'] > 0
                 and secondary['difference'] is not None and secondary['difference'] < 0)
    interval_pass = (primary['ci95'] is not None and primary['ci95'][0] > 0
                     and secondary['ci95'] is not None and secondary['ci95'][1] < 0)
    domain_pass = all(value['nonworse'] for value in eligibility.values() if value['eligible'])
    decision = ('GO' if favorable and interval_pass and domain_pass else
                'CONDITIONAL GO' if favorable else 'NO-GO')
    followups = []
    if favorable and not interval_pass:
        followups.append('Replicate primary and matched-coverage endpoints on additional independent source groups; preserve frozen formulas.')
    for dataset, value in eligibility.items():
        if value['eligible'] and not value['nonworse']:
            followups.append(f'Investigate {dataset} matched-coverage reversal with frozen source/codec/texture stratification before any stage-4 work.')
        elif not value['eligible']:
            followups.append(f'{dataset} has insufficient diagnostic images or positive-candidate errors for the cross-dataset gate.')
    return {'quantitative_decision': decision, 'primary': primary, 'secondary': secondary,
            'dataset_gate': eligibility, 'followups': followups,
            'domain_review_required': 'Severe source/codec/texture confounding can downgrade this quantitative decision to NO-GO; review belongs in the stage3 report.'}


def summarize(input_dir, output=None):
    input_dir = Path(input_dir)
    rows = [json.loads(line) for line in (input_dir / 'per_image.jsonl').read_text().splitlines() if line]
    if not rows:
        raise ValueError('No diagnostic image metrics to summarize')
    if len({row['sample_key'] for row in rows}) != len(rows):
        raise ValueError('Duplicate diagnostic sample keys')
    index = json.loads((input_dir / 'hist_index.json').read_text())
    if isinstance(index, dict):
        index = index['records']
    with np.load(input_dir / 'histograms.npz', allow_pickle=False) as histograms:
        validation = validate_run(input_dir, rows, index, histograms)
    datasets = sorted({row['dataset'] for row in rows})
    summaries = {}
    for dataset in ['ALL'] + datasets:
        subset = rows if dataset == 'ALL' else [row for row in rows if row['dataset'] == dataset]
        summaries[dataset] = {'n_images': len(subset),
                              'n_groups': len({(row['dataset'], row['source_group_id']) for row in subset}),
                              'macro': _macro(subset), 'segmentation': _segmentation(subset),
                              'trusted': _trusted(subset), 'paired_differences': _bootstrap(subset),
                              **_diagnostic_counts(subset),
                              'micro': {}, 'area_strata': {}}
    curves = defaultdict(dict)
    with np.load(input_dir / 'histograms.npz', allow_pickle=False) as histograms:
        for record in index:
            histogram = histograms[record['key']]
            result = metrics_from_hist(histogram)
            result.update(risk_coverage_from_hist(histogram))
            for item in result['coverage'].values():
                if item['selected_count'] < PROTOCOL['minimum_coverage_pixels']:
                    item.update(risk=None, precision=None, reason='fewer than 10 expected selected pixels')
            dataset, area, region, method = (record[key] for key in ('dataset', 'area_stratum', 'region', 'method'))
            target = (summaries[dataset]['micro'] if area == 'ALL' else
                      summaries[dataset]['area_strata'].setdefault(area, {}))
            curve = result.pop('curve')
            target.setdefault(region, {})[method] = result
            if area == 'ALL' and region in ('pos', 'high_pos', 'neg'):
                curves[(dataset, region)][method] = curve
    plots = []
    for (dataset, region), values in curves.items():
        path = input_dir / 'risk_curves' / f'{dataset}_{region}.png'
        save_risk_curves(path, values)
        plots.append(str(path))
    result = {'protocol_hash': PROTOCOL_HASH, 'protocol': PROTOCOL,
              'artifact_validation': validation,
              'definitions': {'macro': 'equal image weight; N/A excluded per metric',
                              'micro': 'pooled pixels using 4096 uniform reliability bins',
                              'uncertainty': 'paired whole-source-group bootstrap, image-weighted macro differences'},
              'datasets': summaries, 'decision': decide(summaries), 'risk_curve_paths': plots}
    output = Path(output) if output else input_dir / 'summary.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('fecd/assets/stage3_diagnostics'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.input_dir, args.output)
    print(json.dumps({'decision': result['decision'], 'datasets': {
        dataset: {'n_images': summary['n_images'], 'n_groups': summary['n_groups']}
        for dataset, summary in result['datasets'].items()}}, indent=2))


if __name__ == '__main__':
    main()
