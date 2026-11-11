"""固定开发集来源组配对bootstrap；只分析保存的混淆计数。"""
from collections import defaultdict

import numpy as np

METRICS = ('f1', 'iou', 'precision', 'recall', 'prediction_area', 'gt_area', 'area_bias', 'area_mae')
STRATA = ('ALL', 'small', 'large')


def _metrics(row):
    counts = [row[k] for k in ('tp', 'fp', 'fn', 'pixels')]
    if any(type(x) is not int or x < 0 for x in counts) or not counts[3] or sum(counts[:3]) > counts[3]:
        raise ValueError('混淆计数必须为有效非负整数')
    tp, fp, fn, pixels = counts
    gt, pred = (tp+fn)/pixels, (tp+fp)/pixels
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
    iou = tp/(tp+fp+fn) if tp+fp+fn else 1.
    for field, expected in [('f1', f1), ('iou', iou), ('gt_positive_fraction', gt),
                            ('prediction_positive_fraction', pred)]:
        value = row[field]
        if not np.isfinite(value) or not np.isclose(value, expected, rtol=1e-9, atol=1e-12):
            raise ValueError('指标与保存混淆计数不一致: '+field)
    return np.array([f1, iou, tp/(tp+fp) if tp+fp else 0., tp/(tp+fn) if tp+fn else 0.,
                     pred, gt, pred-gt, abs(pred-gt)])


def _paired(left, right):
    maps = [{r['sample_key']: r for r in records} for records in (left, right)]
    if not left or any(len(m) != len(r) for m, r in zip(maps, (left, right))) or set(maps[0]) != set(maps[1]):
        raise ValueError('配对要求相同且唯一的非空成员集合')
    rows, values = [], [[], []]
    for key in sorted(maps[0]):
        a, b = maps[0][key], maps[1][key]
        if any(a[k] != b[k] for k in ('dataset', 'source_group_id', 'gt_positive_fraction', 'pixels')):
            raise ValueError('配对身份或GT分母不一致')
        if any(not isinstance(a[k], str) or not a[k] for k in ('sample_key', 'dataset', 'source_group_id')):
            raise ValueError('配对身份字段必须非空')
        rows.append(a)
        values[0].append(_metrics(a)); values[1].append(_metrics(b))
    return rows, np.asarray(values[0]), np.asarray(values[1])


def _eligible(row, stratum):
    return (stratum == 'ALL' or stratum == 'small' and row['gt_positive_fraction'] <= .01
            or stratum == 'large' and row['gt_positive_fraction'] > .25)


def _summary(rows, values):
    if not rows:
        return {'images': 0, 'source_groups': 0, **{m: None for m in METRICS}}
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
    result = {'images': len(rows), 'source_groups': len({(r['dataset'], r['source_group_id']) for r in rows}),
              **{m: float(values[:, i].mean()) for i, m in enumerate(METRICS)},
              'tp': tp, 'fp': fp, 'fn': fn,
              'micro_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
              'micro_iou': tp/(tp+fp+fn) if tp+fp+fn else 1.,
              'all_background_images': sum(r['tp']+r['fp'] == 0 for r in rows),
              'severe_overprediction_images': sum(r['prediction_positive_fraction'] >= .2 and
                  r['prediction_positive_fraction'] >= 3*r['gt_positive_fraction'] for r in rows)}
    return result


def _interval(point, values):
    valid = values[np.isfinite(values)]
    return {'estimate': float(point) if np.isfinite(point) else None,
            'ci95': np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            'valid_repeats': int(len(valid))}


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator != 0)


def _bootstrap(rows, differences, weights, grouped, domains, stratum):
    tables, effective = {}, {}
    for domain in domains:
        groups = grouped[domain]
        counts = np.array([sum(_eligible(rows[i], stratum) for i in g) for g in groups], float)
        sums = np.array([sum((differences[i] for i in g if _eligible(rows[i], stratum)),
                             start=np.zeros(len(METRICS))) for g in groups])
        present = counts > 0
        means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=present[:, None])
        w = weights[domain]
        tables[domain] = {'n': w@counts, 'sum': w@sums, 'groups': w@present.astype(float),
                          'means': w@means, 'point_n': counts.sum(), 'point_sum': sums.sum(0),
                          'point_groups': int(present.sum()), 'point_means': means.sum(0)}
        effective[domain] = {'images': int(counts.sum()), 'source_groups': int(present.sum()),
                             'valid_repeats': int((w@counts > 0).sum())}
    active = [t for t in tables.values() if t['point_n']]
    repetitions = next(iter(weights.values())).shape[0]
    metrics = {}
    for j, name in enumerate(METRICS):
        if not active:
            metrics[name] = {weight: _interval(np.nan, np.full(repetitions, np.nan))
                             for weight in ('image_equal', 'group_equal', 'dataset_equal')}
            continue
        image_point = sum(t['point_sum'][j] for t in active)/sum(t['point_n'] for t in active)
        group_point = sum(t['point_means'][j] for t in active)/sum(t['point_groups'] for t in active)
        domain_point = np.mean([t['point_sum'][j]/t['point_n'] for t in active])
        image = _ratio(sum(t['sum'][:, j] for t in active), sum(t['n'] for t in active))
        group = _ratio(sum(t['means'][:, j] for t in active), sum(t['groups'] for t in active))
        domain = np.mean([_ratio(t['sum'][:, j], t['n']) for t in active], axis=0)
        metrics[name] = {weight: _interval(point, samples) for weight, point, samples in
                         [('image_equal', image_point, image), ('group_equal', group_point, group),
                          ('dataset_equal', domain_point, domain)]}
    domain_metrics = {d: {m: _interval(t['point_sum'][j]/t['point_n'] if t['point_n'] else np.nan,
                                     _ratio(t['sum'][:, j], t['n'])) for j, m in enumerate(METRICS)}
                      for d, t in tables.items()}
    return {'images': sum(v['images'] for v in effective.values()),
            'source_groups': sum(v['source_groups'] for v in effective.values()),
            'effective_by_domain': effective, 'metrics': metrics, 'domains': domain_metrics}


def classify_signal(state):
    def at_least(key, threshold):
        return state.get(key) is not None and state[key] >= threshold
    domains = state['domain_f1_deltas']
    conditions = {'main_f1_gain': at_least('main_f1_delta', .01),
                  'main_ci_positive': state.get('main_f1_ci_low') is not None and state['main_f1_ci_low'] > 0,
                  'small_recall_point': at_least('small_recall_delta', -.02),
                  'large_recall_point': at_least('large_recall_delta', -.02),
                  'small_recall_ci': at_least('small_recall_ci_low', -.05),
                  'large_recall_ci': at_least('large_recall_ci_low', -.05),
                  'every_domain_f1': bool(domains) and all(v is not None and v >= -.03 for v in domains.values()),
                  'dataset_equal_f1': at_least('dataset_equal_f1_delta', 0),
                  'area_mae_not_increased': state.get('area_mae_delta') is not None and state['area_mae_delta'] <= 0}
    safety = {'small_recall': state.get('small_recall_delta') is not None and state['small_recall_delta'] < -.05,
              'large_recall': state.get('large_recall_delta') is not None and state['large_recall_delta'] < -.05,
              'domain_f1': any(v is not None and v < -.05 for v in domains.values()),
              'fp_fn_tradeoff_no_gain': state['fp_delta'] < 0 and state['fn_delta'] > 0 and state['main_f1_delta'] <= 0}
    label = ('NEGATIVE SIGNAL' if any(safety.values()) else 'POSITIVE SIGNAL' if all(conditions.values())
             else 'NO GAIN' if state['main_f1_delta'] <= 0 else 'UNCERTAIN')
    return {'label': label, 'positive_conditions': conditions, 'safety_negative_conditions': safety,
            'decision_inputs': state, 'meaning': 'single-training-seed exploratory development signal; not independent confirmation or novelty'}


def analyze_records(left, right, repeats=2000, seed=20261029):
    """返回两组描述统计、同源组配对差值区间及冻结门槛判断。"""
    if type(repeats) is not int or repeats < 1 or type(seed) is not int:
        raise ValueError('bootstrap要求正整数次数与整数seed')
    rows, a, b = _paired(left, right)
    right_map = {r['sample_key']: r for r in right}
    domains = sorted({r['dataset'] for r in rows})
    index = defaultdict(list)
    for i, r in enumerate(rows):
        index[(r['dataset'], r['source_group_id'])].append(i)
    grouped = {d: [v for (domain, _), v in sorted(index.items()) if domain == d] for d in domains}
    rng, weights = np.random.default_rng(seed), {}
    for domain in domains:
        size = len(grouped[domain])
        weights[domain] = np.stack([np.bincount(rng.integers(size, size=size), minlength=size)
                                    for _ in range(repeats)])
    arms = {v: {} for v in ('R0', 'R1')}
    paired = {}
    for stratum in STRATA:
        ids = [i for i, row in enumerate(rows) if _eligible(row, stratum)]
        for variant, records, values in [('R0', rows, a), ('R1', [right_map[r['sample_key']] for r in rows], b)]:
            arms[variant][stratum] = {}
            for domain in ['ALL']+domains:
                selected = [i for i in ids if domain == 'ALL' or rows[i]['dataset'] == domain]
                arms[variant][stratum][domain] = _summary([records[i] for i in selected], values[selected])
        paired[stratum] = _bootstrap(rows, b-a, weights, grouped, domains, stratum)
    all_metrics = paired['ALL']['metrics']
    def estimate(stratum, metric):
        return paired[stratum]['metrics'][metric]['image_equal']['estimate']
    def low(stratum, metric):
        ci = paired[stratum]['metrics'][metric]['image_equal']['ci95']
        return ci[0] if ci is not None else None
    state = {'main_f1_delta': estimate('ALL', 'f1'), 'main_f1_ci_low': low('ALL', 'f1'),
             'small_recall_delta': estimate('small', 'recall'), 'large_recall_delta': estimate('large', 'recall'),
             'small_recall_ci_low': low('small', 'recall'), 'large_recall_ci_low': low('large', 'recall'),
             'domain_f1_deltas': {d: paired['ALL']['domains'][d]['f1']['estimate'] for d in domains},
             'dataset_equal_f1_delta': all_metrics['f1']['dataset_equal']['estimate'],
             'area_mae_delta': estimate('ALL', 'area_mae'),
             'fp_delta': arms['R1']['ALL']['ALL']['fp']-arms['R0']['ALL']['ALL']['fp'],
             'fn_delta': arms['R1']['ALL']['ALL']['fn']-arms['R0']['ALL']['ALL']['fn']}
    return {'seed': seed, 'repeats': repeats, 'comparison': 'R1-R0',
            'sampling': 'paired source proxy groups within dataset, retain all group images; same draws for all metrics/strata',
            'stratum_empty_draw_policy': 'exclude undefined draw per weight; dataset-equal requires every point-eligible domain present',
            'arms': arms, 'paired': paired, 'research': classify_signal(state)}
