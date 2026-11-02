"""Dataset-stratified paired source-group bootstrap over stored dev records."""
from collections import defaultdict

import numpy as np


def pair_records(left, right):
    a = {r['sample_key']: r for r in left}
    b = {r['sample_key']: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or not a or set(a) != set(b):
        raise ValueError('Paired evaluations must have identical unique members')
    result = []
    for key in sorted(a):
        x, y = a[key], b[key]
        for field in ('dataset', 'source_group_id', 'gt_positive_fraction'):
            if x[field] != y[field]:
                raise ValueError('Paired membership/source/GT identity mismatch')
        row = {k: x[k] for k in ('sample_key', 'dataset', 'source_group_id')}
        for metric in ('f1', 'iou'):
            if not (0 <= x[metric] <= 1 and 0 <= y[metric] <= 1):
                raise ValueError('Invalid paired segmentation metric')
            row[metric] = y[metric] - x[metric]
        result.append(row)
    return result


def _interval(point, samples):
    return {'estimate': float(point), 'ci95': np.quantile(samples, [.025, .975]).tolist()}


def paired_bootstrap(left, right, repeats=2000, seed=20261029):
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError('Bootstrap repeats must be positive')
    rows = pair_records(left, right)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['source_group_id'])].append(row)
    domains = sorted({r['dataset'] for r in rows})
    rng = np.random.default_rng(seed)
    draws = {}
    summaries = {}
    for domain in domains:
        members = [v for (d, _), v in sorted(groups.items()) if d == domain]
        n = np.array([len(v) for v in members], dtype=float)
        sums = np.array([[sum(r[m] for r in v) for m in ('f1', 'iou')] for v in members])
        weights = np.zeros((repeats, len(members)), dtype=np.int64)
        for i in range(repeats):
            weights[i] = np.bincount(rng.integers(len(members), size=len(members)), minlength=len(members))
        draws[domain] = {'n': weights @ n, 'sums': weights @ sums,
                         'means': weights @ (sums / n[:, None]), 'groups': len(members),
                         'point_n': n.sum(), 'point_sums': sums.sum(0),
                         'point_means': (sums / n[:, None]).sum(0)}
        summaries[domain] = {'images': int(n.sum()), 'source_groups': len(members)}
        for j, metric in enumerate(('f1', 'iou')):
            summaries[domain][metric] = _interval(sums[:, j].sum() / n.sum(),
                                                 draws[domain]['sums'][:, j] / draws[domain]['n'])
    total_n = sum(d['n'] for d in draws.values())
    total_sums = sum(d['sums'] for d in draws.values())
    total_means = sum(d['means'] for d in draws.values())
    count_groups = len(groups)
    result = {'images': len(rows), 'source_groups': count_groups, 'seed': seed, 'repeats': repeats,
              'sampling': 'paired source groups, stratified by dataset', 'domains': summaries}
    for j, metric in enumerate(('f1', 'iou')):
        result[metric] = {
            'image_weighted': _interval(np.mean([r[metric] for r in rows]), total_sums[:, j] / total_n),
            'group_equal': _interval(sum(d['point_means'][j] for d in draws.values()) / count_groups,
                                     total_means[:, j] / count_groups),
            'dataset_equal': _interval(np.mean([d['point_sums'][j] / d['point_n'] for d in draws.values()]),
                                       np.mean([d['sums'][:, j] / d['n'] for d in draws.values()], axis=0)),
        }
    return result


def area_diagnostics(rows):
    prediction = np.array([r['prediction_positive_fraction'] for r in rows])
    gt = np.array([r['gt_positive_fraction'] for r in rows])
    return {'images': len(rows), 'prediction_mean': float(prediction.mean()),
            'gt_mean': float(gt.mean()), 'quantiles': [0., .01, .05, .25, .5, .75, .95, .99, 1.],
            'prediction_quantiles': np.quantile(prediction, [0., .01, .05, .25, .5, .75, .95, .99, 1.]).tolist(),
            'gt_quantiles': np.quantile(gt, [0., .01, .05, .25, .5, .75, .95, .99, 1.]).tolist(),
            'severe_overprediction_images': int(((prediction >= .2) & (prediction >= 3 * gt)).sum()),
            'f1_at_most_point1_images': sum(r['f1'] <= .1 for r in rows)}
