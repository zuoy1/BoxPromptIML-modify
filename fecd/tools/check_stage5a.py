"""独立开发诊断的最终只读守恒、身份与历史不变性检查。"""
from collections import Counter, defaultdict
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from fecd.diagnostics.stage5a_io import OUTPUT, ROOT, RUN, DiagnosticBoundary
from fecd.diagnostics.stage5a_metrics import measures
from fecd.tools.check_stage3 import analyze_source
from fecd.tools.stage4f1_v2_io import sha, verify_snapshot, write_new


def read_csv(name):
    with (OUTPUT / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='validation_final')
    tag = parser.parse_args().tag
    if not tag.replace('_', '').isalnum():
        raise ValueError('验收标签只能包含字母、数字、下划线')
    allow = json.loads((OUTPUT / 'allowlist.json').read_text())
    members = {r['sample_key']: r for r in allow['members']}
    assert len(members) == len(allow['members']) == 530
    baseline = json.loads((ROOT / 'fecd/assets/stage4f1_v/revalidation_20261030_v2/inputs_before.json').read_text())
    historical = verify_snapshot(baseline, RUN)
    inference = json.loads((OUTPUT / 'inference_summary.json').read_text())
    assert inference['status'] == 'PASS' and not inference['training'] and not inference['denied']
    gt_log = [json.loads(line) for line in (OUTPUT / 'logs/gt_access.jsonl').read_text().splitlines()]
    assert len(gt_log) == 530 and {r['sample_key'] for r in gt_log} == set(members)
    for variant in ('P0', 'P1', 'P2F', 'P3F'):
        old = {r['sample_key']: r for r in (json.loads(line) for line in
            (RUN / f'artifacts/evaluation/{variant}/gt_access.jsonl').read_text().splitlines())}
        assert all(r['role'] == 'new_dev' and r['sha256'] == old[r['sample_key']]['sha256'] for r in gt_log)
    access = [json.loads(line) for line in (OUTPUT / 'logs/file_access.jsonl').read_text().splitlines()]
    for row in access:
        role, key = allow['allowed_paths'][row['path']]
        assert role == row['role'] and key == row['sample_key']
        assert key is None or key in members
    inputs = json.loads((OUTPUT / 'reference_inputs_sha256.json').read_text())
    with (OUTPUT / 'logs' / f'{tag}_input_access.jsonl').open('x') as stream:
        guard = DiagnosticBoundary(allow['allowed_paths'], stream)
        sys.addaudithook(guard)
        assert all(sha(path) == expected for path, expected in inputs.items())
        decoded = {}
        for key in members:
            with np.load(OUTPUT / 'maps' / f'{key}_reference.npz', allow_pickle=False) as ref:
                gt = ref['gt'].copy()
                assert gt.shape == (224, 224) and gt.dtype == bool
                maps = {name: ref[name].copy() for name in ('K1', 'K8')}
            for variant in ('P0', 'P1', 'P2F', 'P3F'):
                with np.load(OUTPUT / 'maps' / f'{key}_{variant}.npz', allow_pickle=False) as data:
                    probability, logits = data['probability'], data['logits']
                    assert logits.dtype == np.float32 and probability.dtype == np.float64
                    assert np.isfinite(logits).all()
                    assert np.array_equal(probability, np.exp(-np.logaddexp(0., -logits.astype(np.float64))))
                    maps[variant] = probability.copy()
            for model, probability in maps.items():
                assert probability.shape == gt.shape and np.isfinite(probability).all()
                assert (probability >= 0).all() and (probability <= 1).all()
                decoded[(key, model)] = measures(probability >= .5, gt)
        per_image = read_csv('per_image_errors.csv')
        assert len(per_image) == len(decoded) == 3180
        assert len({(r['sample_key'], r['model']) for r in per_image}) == 3180
        for row in per_image:
            actual = decoded[(row['sample_key'], row['model'])]
            for field, value in actual.items():
                assert abs(float(row[field]) - value) < 1e-14
            for total, first, second in (
                    ('fp', 'fp_gt_boundary', 'fp_gt_far'), ('fp', 'fp_pred_boundary', 'fp_pred_interior'),
                    ('fn', 'fn_gt_boundary', 'fn_gt_interior'), ('fn', 'fn_pred_boundary', 'fn_pred_far')):
                assert int(row[total]) == int(row[first]) + int(row[second])
        thresholds = read_csv('threshold_per_image.csv')
        assert len(thresholds) == 28620
        grouped = defaultdict(list)
        for row in thresholds:
            grouped[(row['sample_key'], row['model'])].append(row)
        for rows in grouped.values():
            rows.sort(key=lambda r: float(r['threshold']))
            assert [float(r['threshold']) for r in rows] == [.1, .2, .3, .4, .5, .6, .7, .8, .9]
            for first, second in zip(rows, rows[1:]):
                for field in ('prediction_area', 'recall', 'tp', 'fp'):
                    assert float(first[field]) >= float(second[field])
        comparisons = read_csv('teacher_student_per_image.csv')
        assert len(comparisons) == 4240
        assert len({(r['sample_key'], r['model'], r['teacher']) for r in comparisons}) == 4240
        for row in comparisons:
            assert sum(int(row[k]) for k in ('both_wrong', 'both_correct',
                'teacher_wrong_student_correct', 'teacher_correct_student_wrong')) == int(row['pixels'])
            for field in ('fp', 'fn'):
                assert int(row[field]) == int(row[f'student_{field}_shared']) + int(row[f'student_{field}_extra'])
        assert len(read_csv('source_group_errors.csv')) == 1398
        assert len(read_csv('source_group_thresholds.csv')) == 12582
        paths = sorted((ROOT / 'fecd/diagnostics').glob('stage5a*.py'))
        paths += [ROOT / 'fecd/tools/run_stage5a_inference.py', Path(__file__), ROOT / 'fecd/tests/test_stage5a.py']
        paths.append(ROOT / 'fecd/tools/finalize_stage5a_inference.py')
        issues = [issue for path in paths for issue in analyze_source(path)]
        assert not issues, issues
        assert not guard.denied
        cases = json.loads((OUTPUT / 'case_manifest.json').read_text())
        assert all(r['sample_key'] in members and (OUTPUT / 'cases' / r['file']).is_file() for r in cases)
        result = {'status': 'PASS', 'scope': 'stage5a diagnostic integrity; no training or scientific gain claim',
            'historical_snapshot': historical, 'inference': inference,
            'new_dev_unique_identities': 530, 'gt_decodes': 530,
            'inference_open_counts': dict(Counter(r['role'] for r in access)),
            'final_input_hash_open_counts': guard.counts.copy(),
            'locked_gt_accesses': 0, 'audit_scope': 'CPython open audit, not an OS sandbox',
            'teacher_cache_verified': {'K1': 530, 'K8': 530}, 'input_hashes_unchanged': len(inputs),
            'per_image_records': 3180, 'threshold_records': 28620, 'teacher_student_records': 4240,
            'source_group_error_records': 1398, 'source_group_threshold_records': 12582,
            'probability_maps_verified': 2120, 'reference_maps_verified': 530,
            'error_partitions_and_threshold_monotonicity': 'PASS', 'static_files': len(paths),
            'static_issues': issues, 'visual_cases': len(cases),
            'official_threshold': .5, 'scientific_conclusion': 'NO CLEAR GAIN',
            'unresolved': ['single seed and development-only associations cannot establish causal mechanism']}
        write_new(OUTPUT / f'{tag}.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
