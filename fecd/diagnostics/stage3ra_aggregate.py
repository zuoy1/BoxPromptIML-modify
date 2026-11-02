"""Aggregate frozen development records without reading image or label files."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from fecd.diagnostics.metrics import metrics_from_hist
from fecd.diagnostics.risk_coverage import risk_coverage_from_hist
from fecd.diagnostics.stage3ra_inference import comparisons, decide, domain_equal, paired_comparison
from fecd.diagnostics.visualizations import save_risk_curves

COVERAGES = (.1, .25, .5, .75, .9)


def mean_record(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {'mean': float(np.mean(finite)) if finite else None,
            'n_valid': len(finite), 'n_total': len(values)}


def _metric_records(values):
    output = {metric: mean_record([value.get(metric) for value in values])
              for metric in ('auroc', 'auprc', 'aurc', 'error_prevalence')}
    output.update(n_candidates=sum(value.get('n', 0) for value in values),
                  n_errors=sum(value.get('error_count', 0) for value in values))
    output['na_reasons'] = {metric: dict(Counter(value.get(metric+'_reason') or 'unspecified'
                                               for value in values if value.get(metric) is None))
                            for metric in ('auroc', 'auprc')}
    output['coverage'] = {f'{coverage:g}': {
        metric: mean_record([value.get('coverage', {}).get(f'{coverage:g}', {}).get(metric)
                             for value in values]) for metric in ('risk', 'precision')}
        for coverage in COVERAGES}
    return output


def _macro(rows):
    keys = sorted({(region, method) for row in rows
                   for region, methods in row['regions'].items() for method in methods})
    output = {}
    for region, method in keys:
        values = [row['regions'].get(region, {}).get(method, {}) for row in rows]
        output.setdefault(region, {})[method] = _metric_records(values)
    return output


def _group_equal(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['dataset'], row['source_group_id'])].append(row)
    macros = [_macro(values) for values in grouped.values()]
    output = {}
    for region, methods in _macro(rows).items():
        output[region] = {}
        for method in methods:
            summaries = [macro[region][method] for macro in macros]
            record = {metric: mean_record([value[metric]['mean'] for value in summaries])
                      for metric in ('auroc', 'auprc', 'aurc', 'error_prevalence')}
            record['coverage'] = {f'{coverage:g}': {
                metric: mean_record([value['coverage'][f'{coverage:g}'][metric]['mean']
                                     for value in summaries]) for metric in ('risk', 'precision')}
                for coverage in COVERAGES}
            output[region][method] = record
    return output


def _budgets(rows, field='budgets'):
    output = {}
    for region in ('pos', 'neg'):
        output[region] = {}
        for method in ('random', 'c_sam', 'q_a'):
            output[region][method] = {}
            for coverage in COVERAGES:
                key = f'{coverage:g}'
                values = [row.get(field, {}).get(region, {}).get(method, {}).get(key, {})
                          for row in rows]
                selected = sum(value.get('selected_count', 0) for value in values)
                errors = sum(value.get('error_count', 0) for value in values)
                candidates = sum(row['regions'].get(region, {}).get('c_sam', {}).get('n', 0)
                                 for row in rows)
                pixels = len(rows) * 224 * 224
                output[region][method][key] = {
                    'macro': {metric: mean_record([value.get(metric) for value in values])
                              for metric in ('risk', 'precision', 'conditional_coverage', 'full_coverage')},
                    'micro': {'selected_count': selected, 'error_count': errors,
                              'risk': errors/selected if selected else None,
                              'precision': 1-errors/selected if selected else None,
                              'conditional_coverage': selected/candidates if candidates else None,
                              'full_coverage': selected/pixels if pixels else None}}
    return output


def _conditional(rows):
    output = {}
    keys = sorted({(region, feature, label, method) for row in rows
                   for region, features in row.get('conditioned_metrics', {}).items()
                   for feature, labels in features.items() for label, methods in labels.items()
                   for method in methods})
    for region, feature, label, method in keys:
        values = [row.get('conditioned_metrics', {}).get(region, {}).get(feature, {})
                  .get(label, {}).get(method, {}) for row in rows]
        output.setdefault(region, {}).setdefault(feature, {}).setdefault(label, {})[method] = _metric_records(values)
    return output


def _conditional_differences(rows):
    output = {}
    for region, features in _conditional(rows).items():
        output[region] = {}
        for feature, labels in features.items():
            output[region][feature] = {}
            for label in labels:
                pairs = []
                for row in rows:
                    methods = row.get('conditioned_metrics', {}).get(region, {}).get(feature, {}).get(label, {})
                    if methods:
                        pairs.append({**row, 'regions': {'condition': methods}})
                output[region][feature][label] = {
                    metric: paired_comparison(pairs, 'condition', 'q_a', 'c_sam', metric)
                    for metric in ('auroc', 'risk@0.5')}
    return output


def _composition(rows):
    output = {}
    for region in ('pos', 'neg'):
        output[region] = {}
        for method in ('c_sam', 'q_a'):
            values = [row.get('selection_composition', {}).get(region, {}).get(method, {})
                      for row in rows]
            strata = sorted({name for value in values for name in value})
            totals = {name: {key: sum(value.get(name, {}).get(key, 0) for value in values)
                             for key in ('selected_count', 'error_count')}
                      for name in strata}
            balanced = [row.get('region_balanced', {}).get(region, {}).get(method, {})
                        for row in rows]
            output[region][method] = {'global50_box_composition': totals,
                                      'balanced50_macro_risk': mean_record([value.get('risk') for value in balanced])}
    for sign in ('pos', 'neg'):
        proxy = [{**row, 'regions': {sign: {
            method: {'coverage': {'0.5': {'risk': row.get('region_balanced', {}).get(sign, {}).get(method, {}).get('risk')}}}
            for method in ('c_sam', 'q_a')}}} for row in rows]
        output[sign]['balanced50_paired_difference'] = paired_comparison(proxy, sign, 'q_a', 'c_sam', 'risk@0.5')
    return output


def validate_run(assets, rows):
    from fecd.diagnostics.stage3ra_protocol import METHODS, PROTOCOL_HASH
    run = json.loads((assets/'stage3ra_run_manifest.json').read_text())
    split = json.loads((assets/'stage3ra_split_manifest.json').read_text())
    expected = {row['sample_key'] for row in split['records'] if row['subset'] == 'new_dev'}
    keys = [row['sample_key'] for row in rows]
    membership = hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest()
    if (not rows or len(set(keys)) != len(keys) or set(keys) != expected
            or any(row.get('subset', 'new_dev') != 'new_dev' for row in rows)
            or run.get('status') != 'passed' or run.get('requested') != len(rows)
            or run.get('evaluated') != len(rows) or run.get('protocol_hash') != PROTOCOL_HASH
            or run.get('split_hash') != split.get('manifest_sha256')
            or run.get('evaluated_sample_keys_sha256') != membership):
        raise ValueError('Incomplete, mixed-protocol, or non-development diagnostic records')
    for row in rows:
        for region, methods in row['regions'].items():
            if set(methods) != set(METHODS):
                raise ValueError(f'Missing or unexpected methods within {region}')
            totals = {(value['n'], value['error_count']) for value in methods.values()}
            if len(totals) != 1:
                raise ValueError(f'Method candidate counts differ within {region}')
            n, errors = next(iter(totals))
            if n < 0 or errors < 0 or errors > n:
                raise ValueError('Invalid per-image candidate or error count')
    return {'status': 'passed', 'images': len(rows), 'membership_sha256': membership,
            'protocol_hash': PROTOCOL_HASH, 'split_hash': split['manifest_sha256']}


def _micro(assets, rows, datasets):
    index_path = assets/'stage3ra_hist_index.json'
    if not index_path.exists():
        return {'available': False, 'reason': 'no pooled histograms supplied'}
    index = json.loads(index_path.read_text())
    if isinstance(index, dict):
        index = index['records']
    expected = defaultdict(lambda: np.zeros(2, np.int64))
    for row in rows:
        for dataset in ('ALL', row['dataset']):
            areas = ['ALL'] if row.get('area_stratum', 'ALL') == 'ALL' else ['ALL', row['area_stratum']]
            for area in areas:
                for region, methods in row['regions'].items():
                    for method, value in methods.items():
                        expected[(dataset, area, region, method)] += [value['n'], value['error_count']]
    keys = [record['key'] for record in index]
    scopes = [tuple(record[name] for name in ('dataset', 'area_stratum', 'region', 'method'))
              for record in index]
    with np.load(assets/'stage3ra_histograms.npz', allow_pickle=False) as histograms:
        if (len(set(keys)) != len(keys) or set(keys) != set(histograms.files)
                or len(set(scopes)) != len(scopes) or set(scopes) != set(expected)):
            raise ValueError('Pooled histogram scopes or keys disagree with per-image records')
        for record, scope in zip(index, scopes):
            histogram = histograms[record['key']]
            if histogram.shape != (2, 4096) or not np.array_equal(histogram.sum(axis=1), expected[scope]):
                raise ValueError('Pooled histogram counts disagree with per-image records')
            result = metrics_from_hist(histogram)
            result.update(risk_coverage_from_hist(histogram))
            result.pop('curve')
            target = datasets[record['dataset']]
            target = target.setdefault('micro', {}) if record['area_stratum'] == 'ALL' else target.setdefault('area_strata', {}).setdefault(record['area_stratum'], {})
            target.setdefault(record['region'], {})[record['method']] = result
    return {'available': True, 'scopes': len(index),
            'definition': '4096 uniform reliability bins; within-bin ties; descriptive approximation only'}


def summarize(assets_dir, output=None):
    assets = Path(assets_dir)
    rows = [json.loads(line) for line in (assets/'stage3ra_per_image.jsonl').read_text().splitlines() if line]
    validation = validate_run(assets, rows)
    datasets = {}
    plots = []
    for dataset in ['ALL']+sorted({row['dataset'] for row in rows}):
        subset = rows if dataset == 'ALL' else [row for row in rows if row['dataset'] == dataset]
        macro = _macro(subset)
        datasets[dataset] = {'n_images': len(subset),
                             'n_groups': len({(row['dataset'], row['source_group_id']) for row in subset}),
                             'macro': macro, 'group_equal': _group_equal(subset),
                             'paired_differences': comparisons(subset), 'budgets': _budgets(subset),
                             'realized_budgets': _budgets(subset, 'realized_budgets'),
                             'conditioned_metrics': _conditional(subset),
                             'conditional_paired_differences': _conditional_differences(subset),
                             'selection_composition': _composition(subset)}
        for region in ('pos', 'high_pos', 'neg'):
            if region not in macro:
                continue
            curves = {method: {'coverage': [c for c in COVERAGES if values['coverage'][f'{c:g}']['risk']['mean'] is not None],
                               'risk': [values['coverage'][f'{c:g}']['risk']['mean'] for c in COVERAGES
                                        if values['coverage'][f'{c:g}']['risk']['mean'] is not None]}
                      for method, values in macro[region].items()}
            path = assets/'stage3ra_risk_curves'/f'{dataset}_{region}_macro.png'
            plots.append(save_risk_curves(path, curves))
    pooled = _micro(assets, rows, datasets)
    result = {'schema_version': 'fecd-stage3ra-development-summary-v1',
              **validation, 'datasets': datasets, 'decision': decide(datasets),
              'domain_equal_sensitivity': domain_equal(datasets),
              'pooled_micro': pooled, 'risk_curve_paths': plots,
              'definitions': {'macro': 'equal image weight; exact per-image metrics; N/A excluded per metric',
                              'group_equal': 'equal source group weight after paired valid-image aggregation',
                              'budget': 'expected fractional selection within ties; random is exact uniform-selection expectation',
                              'curves': 'five fixed coverage macro points; connecting segments are visual aids'}}
    destination = Path(output) if output else assets/'stage3ra_dev_summary.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
