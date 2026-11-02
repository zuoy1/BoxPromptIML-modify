"""只消费独立开发概率副本，输出完整诊断汇总与固定案例名单。"""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from fecd.diagnostics.stage5a_io import OUTPUT
from fecd.diagnostics.stage5a_metrics import measures, spatial_errors, overlap_errors, summarize
from fecd.tools.stage4f1_v2_io import write_new

MODELS = ('K1', 'K8', 'P0', 'P1', 'P2F', 'P3F')
DOMAINS = ('CASIAv2', 'NIST16', 'Columbia', 'Coverage')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _bin(area, edges):
    if area == 0:
        return 'empty'
    index = max(0, min(len(edges) - 2, np.searchsorted(edges, area, side='left') - 1))
    return f'({edges[index]:.2f},{edges[index + 1]:.2f}]'


def export_source_groups(per_image, threshold_rows):
    for name, rows, fields in (
            ('source_group_errors', per_image, ('model', 'dataset', 'source_group_id')),
            ('source_group_thresholds', threshold_rows, ('model', 'dataset', 'source_group_id', 'threshold'))):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[field] for field in fields)].append(row)
        result = [{**dict(zip(fields, key)), **summarize(values)} for key, values in sorted(groups.items())]
        write_csv(OUTPUT / f'{name}.csv', result)


def _correlation(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['source_group_id'])].append(row)
    result = []
    for unit, values in [('image', rows), ('source_group', [
            {key: float(np.mean([r[key] for r in group])) for key in
             ('prediction_area', 'gt_area', 'area_bias', 'f1')} for group in groups.values()])]:
        for x in ('prediction_area', 'gt_area', 'area_bias'):
            a, b = np.array([r[x] for r in values]), np.array([r['f1'] for r in values])
            valid = len(a) > 1 and a.std() > 0 and b.std() > 0
            result.append({'unit': unit, 'x': x, 'y': 'f1', 'n': len(values),
                           'pearson': float(np.corrcoef(a, b)[0, 1]) if valid else None,
                           'spearman': float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]) if valid else None})
    return result


def _concentration(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['source_group_id'])].append(row)
    result = {}
    for key in ('fp', 'fn'):
        total = sum(r[key] for r in rows)
        image_values = sorted([r[key] for r in rows], reverse=True)
        group_values = sorted([sum(r[key] for r in group) for group in groups.values()], reverse=True)
        result[key + '_top10pct_image_share'] = sum(image_values[:int(np.ceil(.1 * len(rows)))] ) / total if total else 0.
        result[key + '_top10pct_source_group_share'] = sum(group_values[:int(np.ceil(.1 * len(groups)))]) / total if total else 0.
        result[key + '_domain_share'] = {d: sum(r[key] for r in rows if r['dataset'] == d) / total if total else 0. for d in DOMAINS}
    return result


def cases(rows, protocol):
    p3 = [r for r in rows if r['model'] == 'P3F']
    p2 = {r['sample_key']: r for r in rows if r['model'] == 'P2F'}
    selections = []
    def choose(category, candidates, key, count=2):
        for row in sorted(candidates, key=lambda r: (key(r), r['sample_key']))[:count]:
            selections.append({'category': category, 'sample_key': row['sample_key'],
                               'dataset': row['dataset'], 'score': key(row)})
    choose('lowest_f1', p3, lambda r: r['f1'])
    severe = [r for r in rows if r['model'].startswith('P') and r['prediction_area'] >= .2
              and r['prediction_area'] >= 3 * r['gt_area']]
    unique = {}
    for row in severe:
        old = unique.get(row['sample_key'])
        if old is None or row['fp'] > old['fp']:
            unique[row['sample_key']] = row
    choose('severe_overprediction', list(unique.values()), lambda r: -r['fp'])
    choose('severe_miss', [r for r in p3 if r['recall'] <= .2], lambda r: r['recall'])
    choose('small_tamper', p3, lambda r: r['gt_area'])
    choose('P3F_gain', [r for r in p3 if r['f1'] - p2[r['sample_key']]['f1'] >= .1],
           lambda r: -(r['f1'] - p2[r['sample_key']]['f1']))
    choose('P3F_regression', [r for r in p3 if r['f1'] - p2[r['sample_key']]['f1'] <= -.1],
           lambda r: r['f1'] - p2[r['sample_key']]['f1'])
    for domain in DOMAINS:
        choose('domain_' + domain, [r for r in p3 if r['dataset'] == domain], lambda r: r['f1'], 1)
    return selections


def main():
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    members = json.loads((OUTPUT / 'allowlist.json').read_text())['members']
    per_image, threshold_rows, overlap, reliability, calibration = [], [], [], [], []
    for row in members:
        key = row['sample_key']
        identity = {k: row[k] for k in ('sample_key', 'sample_id', 'dataset', 'source_group_id')}
        with np.load(OUTPUT / 'maps' / f'{key}_reference.npz', allow_pickle=False) as ref:
            gt, boxes = ref['gt'].copy(), ref['boxes'].copy()
            maps = {k: ref[k].copy() for k in ('K1', 'K8')}
        for model in MODELS[2:]:
            with np.load(OUTPUT / 'maps' / f'{key}_{model}.npz', allow_pickle=False) as data:
                maps[model] = data['probability'].copy()
        for model, probability in maps.items():
            metrics = measures(probability >= .5, gt)
            record = {**identity, 'model': model, 'area_bin': _bin(metrics['gt_area'], protocol['gt_area_edges']),
                      **metrics, **spatial_errors(probability >= .5, gt, boxes, protocol['boundary_radius_pixels'])}
            record.update(fp_image_fraction=metrics['fp'] / gt.size,
                          fn_image_fraction=metrics['fn'] / gt.size,
                          severe_overprediction_fraction=float(metrics['prediction_area'] >= .2
                              and metrics['prediction_area'] >= 3 * metrics['gt_area']),
                          severe_miss_fraction=float(metrics['recall'] <= .2))
            per_image.append(record)
            for threshold in protocol['thresholds']:
                threshold_rows.append({**identity, 'model': model, 'threshold': threshold,
                                       **measures(probability >= threshold, gt)})
            calibration.append({**identity, 'model': model,
                                'brier': float(np.mean((probability - gt) ** 2)),
                                'foreground_brier': float(np.mean((1 - probability[gt]) ** 2)) if gt.any() else None,
                                'background_brier': float(np.mean(probability[~gt] ** 2)) if (~gt).any() else None})
            indices = np.minimum((probability * 10).astype(int), 9)
            for bucket in range(10):
                selected = indices == bucket
                reliability.append({**identity, 'model': model, 'bucket': bucket,
                    'pixels': int(selected.sum()), 'probability_sum': float(probability[selected].sum()),
                    'gt_sum': int(gt[selected].sum())})
            if model.startswith('P'):
                for teacher in ('K1', 'K8'):
                    overlap.append({**identity, 'model': model, 'teacher': teacher,
                                    **metrics, **overlap_errors(probability >= .5, maps[teacher] >= .5, gt)})
    write_csv(OUTPUT / 'per_image_errors.csv', per_image)
    write_csv(OUTPUT / 'threshold_per_image.csv', threshold_rows)
    write_csv(OUTPUT / 'teacher_student_per_image.csv', overlap)
    write_csv(OUTPUT / 'reliability_per_image.csv', reliability)
    export_source_groups(per_image, threshold_rows)
    summaries, areas, curves, comparisons, correlations, concentration, brier = [], [], [], [], [], {}, []
    for model in MODELS:
        selected = [r for r in per_image if r['model'] == model]
        concentration[model] = _concentration(selected)
        for domain in ('ALL',) + DOMAINS:
            subset = selected if domain == 'ALL' else [r for r in selected if r['dataset'] == domain]
            summaries.append({'model': model, 'dataset': domain, **summarize(subset)})
            correlations.extend({'model': model, 'dataset': domain, **r} for r in _correlation(subset))
            for area_bin in sorted({r['area_bin'] for r in subset}):
                areas.append({'model': model, 'dataset': domain, 'area_bin': area_bin,
                              **summarize([r for r in subset if r['area_bin'] == area_bin])})
        for threshold in protocol['thresholds']:
            rows = [r for r in threshold_rows if r['model'] == model and r['threshold'] == threshold]
            domains = []
            for domain in ('ALL',) + DOMAINS:
                subset = rows if domain == 'ALL' else [r for r in rows if r['dataset'] == domain]
                value = {'model': model, 'threshold': threshold, 'dataset': domain, **summarize(subset)}
                curves.append(value)
                if domain != 'ALL':
                    domains.append(value)
            curves.append({'model': model, 'threshold': threshold, 'dataset': 'DATASET_EQUAL',
                           **{k: float(np.mean([r[k] for r in domains])) for k in domains[0]
                              if k.startswith(('image_', 'group_', 'pooled_'))}})
        for teacher in ('K1', 'K8'):
            for domain in ('ALL',) + DOMAINS:
                rows = [r for r in overlap if r['model'] == model and r['teacher'] == teacher
                        and (domain == 'ALL' or r['dataset'] == domain)]
                if rows:
                    comparisons.append({'model': model, 'teacher': teacher, 'dataset': domain, **summarize(rows)})
        cal = [r for r in calibration if r['model'] == model]
        groups = defaultdict(list)
        for r in cal:
            groups[(r['dataset'], r['source_group_id'])].append(r['brier'])
        brier.append({'model': model, 'image_brier': float(np.mean([r['brier'] for r in cal])),
                      'source_group_brier': float(np.mean([np.mean(g) for g in groups.values()])),
                      **{field: float(np.mean([r[field] for r in cal if r[field] is not None]))
                         for field in ('foreground_brier', 'background_brier')}})
    for name, rows in [('summary', summaries), ('area_bins', areas), ('threshold_sensitivity', curves),
                       ('teacher_student_summary', comparisons), ('correlations', correlations), ('brier', brier)]:
        write_csv(OUTPUT / f'{name}.csv', rows)
        write_new(OUTPUT / f'{name}.json', rows)
    write_new(OUTPUT / 'error_concentration.json', concentration)
    ranked = []
    for model in MODELS:
        values = [r for r in per_image if r['model'] == model]
        for error in ('fp', 'fn'):
            for rank, row in enumerate(sorted(values, key=lambda r: (-r[error], r['sample_key']))[:10], 1):
                ranked.append({'ranked_by': error, 'rank': rank, **row})
    write_csv(OUTPUT / 'high_fp_fn_cases.csv', ranked)
    write_csv(OUTPUT / 'brier_per_image.csv', calibration)
    bins = []
    for model in MODELS:
        for bucket in range(10):
            values = [r for r in reliability if r['model'] == model and r['bucket'] == bucket]
            count = sum(r['pixels'] for r in values)
            bins.append({'model': model, 'bucket': bucket, 'pixel_count': count,
                         'mean_probability': sum(r['probability_sum'] for r in values) / count if count else None,
                         'empirical_foreground_fraction': sum(r['gt_sum'] for r in values) / count if count else None,
                         'meaning': 'descriptive pooled reliability, no independent-pixel inference'})
    write_csv(OUTPUT / 'reliability_summary.csv', bins)
    write_new(OUTPUT / 'case_selection.json', cases(per_image, protocol))
    write_new(OUTPUT / 'analysis_status.json', {'status': 'PASS', 'diagnostic_only': True,
        'images': len(members), 'models': list(MODELS), 'threshold_records': len(threshold_rows),
        'teacher_student_records': len(overlap), 'official_threshold_unchanged': .5})


if __name__ == '__main__':
    main()
