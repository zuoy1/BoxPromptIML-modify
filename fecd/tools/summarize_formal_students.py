"""GT-free paired inference, domains, failures, and epoch curves from saved records."""
import csv
import json
from pathlib import Path

from fecd.evaluation.f1_statistics import paired_bootstrap, area_diagnostics, pair_records
from fecd.evaluation.strict import write_json
from fecd.training.f1_protocol import NAMES, STATISTICS
from fecd.training.f1_storage import resolve_output_root, parse_output_root


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_csv(path, data):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main(output_root=None):
    output = resolve_output_root(output_root)
    target = output/'analysis'
    target.mkdir(exist_ok=False)
    records = {v:rows(output/'evaluation'/v/'per_image.jsonl') for v in NAMES}
    summaries = {v:read(output/'evaluation'/v/'summary.json') for v in NAMES}
    tables, area, curves, failures = [], {}, [], []
    for variant in NAMES:
        area[variant] = {}
        for domain in ('ALL','CASIAv2','NIST16','Columbia','Coverage'):
            subset = records[variant] if domain == 'ALL' else [r for r in records[variant] if r['dataset'] == domain]
            area[variant][domain] = area_diagnostics(subset)
            tables.append({'variant':variant, 'dataset':domain, **summaries[variant]['domains'][domain],
                           'severe_overprediction_images':area[variant][domain]['severe_overprediction_images'],
                           'low_f1_images':area[variant][domain]['f1_at_most_point1_images']})
        for row in records[variant]:
            over = row['prediction_positive_fraction'] >= .2 and row['prediction_positive_fraction'] >= 3*row['gt_positive_fraction']
            if row['f1'] <= .1 or over or row['all_zero_prediction'] or row['near_full_prediction']:
                failures.append({k:row[k] for k in ('sample_key','dataset','source_group_id','f1','iou',
                                                    'prediction_positive_fraction','gt_positive_fraction')} |
                                {'variant':variant, 'severe_overprediction':over,
                                 'all_zero':row['all_zero_prediction'], 'near_full':row['near_full_prediction']})
        for epoch in range(1,21):
            summary = read(output/variant/f'epoch{epoch:02d}.summary.json')
            for domain, diag in summary['outputs']['domains'].items():
                curves.append({'variant':variant, 'epoch':epoch, 'dataset':domain,
                    'mean_loss':summary['mean_loss'] if domain == 'ALL' else '',
                    'positive_contribution':summary['mean_pos_contribution'] if domain == 'ALL' else '',
                    'negative_contribution':summary['mean_neg_contribution'] if domain == 'ALL' else '',
                    'student_area':diag['student_positive_fraction']['mean'],
                    'teacher_area':diag['teacher_hard_positive_fraction']['mean'],
                    'probability_mean':diag['probability_mean']['mean'],
                    'all_zero':diag['student_all_zero'], 'near_full':diag['student_near_full'],
                    'saturated':diag['probability_saturated'], 'elapsed_seconds':summary['elapsed_seconds'],
                    'updates':summary['updates'], 'gradient_norm_before_max':summary['gradient_norm_before_max'],
                    'gradient_norm_after_max':summary['gradient_norm_after_max']})
    comparisons, delta_csv = {}, []
    for name in STATISTICS['comparisons']:
        right, left = name.split('-')
        value = paired_bootstrap(records[left], records[right], STATISTICS['repeats'], STATISTICS['seed'])
        comparisons[name] = value
        write_json(target/f'{name}_paired_images.json', pair_records(records[left], records[right]))
        for metric in ('f1','iou'):
            for weight in ('image_weighted','group_equal','dataset_equal'):
                result = value[metric][weight]
                delta_csv.append({'comparison':name, 'dataset':'ALL', 'metric':metric, 'weight':weight,
                    'difference':result['estimate'], 'ci95_low':result['ci95'][0], 'ci95_high':result['ci95'][1],
                    'images':value['images'], 'source_groups':value['source_groups']})
            for domain, result in value['domains'].items():
                delta_csv.append({'comparison':name, 'dataset':domain, 'metric':metric, 'weight':'image_weighted',
                    'difference':result[metric]['estimate'], 'ci95_low':result[metric]['ci95'][0],
                    'ci95_high':result[metric]['ci95'][1], 'images':result['images'], 'source_groups':result['source_groups']})
    write_json(target/'paired_bootstrap.json', {'protocol':STATISTICS, 'comparisons':comparisons})
    write_json(target/'area_diagnostics.json', area)
    write_json(target/'domain_metrics.json', tables)
    write_json(target/'training_curves.json', curves)
    write_json(target/'failure_cases.json', failures)
    write_csv(target/'domain_metrics.csv', tables)
    write_csv(target/'paired_differences.csv', delta_csv)
    write_csv(target/'training_curves.csv', curves)
    if failures:
        write_csv(target/'failure_cases.csv', failures)
    print(json.dumps({'status':'PASS', 'comparisons':comparisons}), flush=True)


if __name__ == '__main__':
    main(parse_output_root())
