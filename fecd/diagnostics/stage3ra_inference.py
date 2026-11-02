"""Paired development inference; no label files or trainable models are opened."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from fecd.diagnostics.bootstrap import paired_bootstrap


def metric_value(row, region, method, metric):
    value = row.get('regions', {}).get(region, {}).get(method, {})
    if metric.startswith('risk@'):
        return value.get('coverage', {}).get(metric[5:], {}).get('risk')
    return value.get(metric)


def paired_comparison(rows, region, method, baseline, metric):
    groups = [f"{row['dataset']}:{row['source_group_id']}" for row in rows]
    candidate = [metric_value(row, region, method, metric) for row in rows]
    reference = [metric_value(row, region, baseline, metric) for row in rows]
    image = paired_bootstrap(candidate, reference, groups, seed=20261029, n_resamples=2000)
    group_pairs = defaultdict(list)
    for group, first, second in zip(groups, candidate, reference):
        if first is not None and second is not None and np.isfinite([first, second]).all():
            group_pairs[group].append((float(first), float(second)))
    means = {group: np.mean(values, axis=0) for group, values in group_pairs.items()}
    equal = paired_bootstrap([values[0] for values in means.values()],
                             [values[1] for values in means.values()], list(means),
                             seed=20261029, n_resamples=2000)
    equal['estimand'] = 'equal source-group mean paired difference; whole-group resampling'
    return {'image_weighted': image, 'group_equal': equal}


def comparisons(rows):
    regions = sorted({region for row in rows for region in row['regions']})
    pairs = [('q_a', 'c_sam'), ('c_sam', 'mean_margin'), ('variance', 'mean_margin'),
             ('threshold', 'mean_margin'), ('mean_margin', 'base_margin'),
             ('q_a', 'mean_margin')]
    output = {}
    for region in regions:
        output[region] = {}
        for method, baseline in pairs:
            output[region][f'{method}-minus-{baseline}'] = {
                metric: paired_comparison(rows, region, method, baseline, metric)
                for metric in ('auroc', 'auprc', 'aurc', 'risk@0.5')}
    return output


def domain_equal(datasets):
    """Describe equal-domain paired effects without inventing an external CI."""
    values = [summary['paired_differences'] for name, summary in datasets.items() if name != 'ALL']
    output = {}
    for region in ('pos', 'high_pos', 'neg'):
        output[region] = {}
        for metric in ('auroc', 'auprc', 'aurc', 'risk@0.5'):
            differences = [value.get(region, {}).get('q_a-minus-c_sam', {}).get(metric, {})
                           .get('image_weighted', {}).get('difference') for value in values]
            valid = [value for value in differences if value is not None]
            output[region][metric] = {'difference': float(np.mean(valid)) if valid else None,
                                      'n_domains_valid': len(valid), 'n_domains_total': len(values),
                                      'ci95': None, 'reason': 'descriptive equal-domain sensitivity only'}
    return output


def decide(datasets):
    """Apply prespecified development effect floors, never claim confirmation."""
    pairs = datasets['ALL']['paired_differences']
    required = [('high_pos', 'auroc'), ('pos', 'risk@0.5'), ('neg', 'risk@0.5')]
    endpoints = {region: pairs.get(region, {}).get('q_a-minus-c_sam', {}).get(metric, {})
                 for region, metric in required}
    primary, risk, negative = [endpoints[name].get('image_weighted', {})
                               for name in ('high_pos', 'pos', 'neg')]
    deltas = [item.get('difference') for item in (primary, risk, negative)]
    reasons = []
    if any(value is None for value in deltas):
        decision = 'INSUFFICIENT EVIDENCE'
        reasons.append('At least one required endpoint has no valid paired values.')
    elif deltas[0] <= 0 or deltas[1] >= 0 or deltas[2] > .001:
        decision = 'STOP'
        reasons.append('Required high-consensus or matched-coverage direction failed, or negative risk exceeded margin.')
    else:
        image_pass = (deltas[0] >= .02 and deltas[1] <= -.002
                      and primary.get('ci95') is not None and primary['ci95'][0] > 0
                      and risk.get('ci95') is not None and risk['ci95'][1] < 0
                      and negative.get('ci95') is not None and negative['ci95'][1] <= .001)
        group_deltas = [endpoints[name].get('group_equal', {}).get('difference')
                        for name in ('high_pos', 'pos', 'neg')]
        group_pass = (all(value is not None for value in group_deltas)
                      and group_deltas[0] > 0 and group_deltas[1] < 0 and group_deltas[2] <= .001)
        decision = 'DEVELOPMENT PASS' if image_pass and group_pass else 'INSUFFICIENT EVIDENCE'
        if not image_pass:
            reasons.append('Prespecified practical-effect or development interval gate not met.')
        if not group_pass:
            reasons.append('Equal-group sensitivity does not support all required directions.')
    domain_gates = {}
    for dataset, summary in datasets.items():
        if dataset == 'ALL':
            continue
        pair = summary['paired_differences'].get('high_pos', {}).get('q_a-minus-c_sam', {})
        endpoint = pair.get('auroc', {}).get('image_weighted', {})
        valid = endpoint.get('n_images', 0)
        groups = endpoint.get('n_groups', 0)
        errors = summary.get('macro', {}).get('high_pos', {}).get('c_sam', {}).get('n_errors', 0)
        eligible = valid >= 20 and groups >= 20 and errors >= 50
        risk_pair = summary['paired_differences'].get('pos', {}).get('q_a-minus-c_sam', {})
        risk_delta = risk_pair.get('risk@0.5', {}).get('image_weighted', {}).get('difference')
        favorable = (endpoint.get('difference') is not None and endpoint['difference'] > 0
                     and risk_delta is not None and risk_delta < 0)
        domain_gates[dataset] = {'eligible': eligible, 'valid_images': valid,
                                 'valid_groups': groups, 'errors': errors,
                                 'favorable_direction': favorable}
        if decision == 'DEVELOPMENT PASS' and not favorable:
            decision = 'INSUFFICIENT EVIDENCE'
            reasons.append(f'{dataset} does not support both required directions.')
    if decision == 'DEVELOPMENT PASS' and not all(x['eligible'] for x in domain_gates.values()):
        decision = 'INSUFFICIENT EVIDENCE'
        reasons.append('Some declared domains have insufficient independent source groups for the proposed confirmation scope.')
    return {'decision': decision, 'action': 'GO-to-confirm' if decision == 'DEVELOPMENT PASS' else decision,
            'endpoints': endpoints, 'domain_gates': domain_gates, 'reasons': reasons,
            'independent_confirmation': False,
            'new_confirm_gt_opened': False,
            'meaning': 'Development diagnostic only; any confirmation requires separate user authorization.'}
