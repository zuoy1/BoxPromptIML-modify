"""Read-only final audit of training/evaluation artifacts, never GT image reads."""
from collections import Counter
import json
import math
from pathlib import Path

import torch

from fecd.diagnostics.stage3ra_split import read_split
from fecd.evaluation.metrics import summarize_records
from fecd.evaluation.f1_statistics import paired_bootstrap
from fecd.pseudo.cache_schema import canonical_hash
from fecd.tools.check_stage3 import analyze_source
from fecd.tools.preflight_formal_student import sha
from fecd.training.f1_protocol import ROOT, NAMES, DOMAINS, MEMBER_HASH, EVAL_HASH, STATISTICS, configuration
from fecd.training.f1_runner import write_json
from fecd.training.f1_storage import resolve_output_root, parse_output_root


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def check_training(variant, schedule, selected, members, output_root=None):
    output = resolve_output_root(output_root)
    directory = output/variant
    config = configuration(variant)
    result = read(directory/'result.json')
    assert read(directory/'config.json') == config
    assert result['epochs'] == 20 and result['logical_updates'] == 17660 and result['gt_reads'] == 0
    assert result['membership_hash'] == MEMBER_HASH
    allow = read(output/'preflight_access.json')['opened_counts']
    allowed_rgb = {str(Path(r['image_path']).resolve()) for r in members.values()}
    cache_fragment = '/k1/' if variant == 'P0' else '/k8/'
    allowed = allowed_rgb | {p for p in allow if cache_fragment in p}
    assert len(allowed) == 7060
    for epoch in range(1,21):
        summary = read(directory/f'epoch{epoch:02d}.summary.json')
        steps = rows(directory/f'epoch{epoch:02d}.steps.jsonl')
        images = rows(directory/f'epoch{epoch:02d}.outputs.jsonl')
        access = read(directory/f'epoch{epoch:02d}.access.json')
        assert summary['status'] == 'PASS' and finite(summary) and finite(steps) and finite(images)
        assert summary['identity'] == schedule[epoch-1]
        assert summary['initial_hash'] == result['initial_hash']
        assert summary['dataset_forward_backward_counts'] == DOMAINS
        assert len(steps) == 883 and len(images) == 3530 and steps[-1]['batch_size'] == 2
        assert Counter(r['dataset'] for r in images) == Counter(DOMAINS)
        assert not access['denied'] and access['gt_reads'] == 0
        assert set(access['opened_counts']) == allowed
        keys = [key for step in steps for key in step['sample_keys']]
        assert len(set(keys)) == 3530 and set(keys) == set(members)
        assert keys == [r['sample_key'] for r in images]
        assert canonical_hash(keys) == schedule[epoch-1]['sample_order_hash']
        assert canonical_hash([r['operation'] for r in images]) == schedule[epoch-1]['geometry_hash']
        for i, step in enumerate(steps):
            assert step['update'] == (epoch-1)*883+i+1
            assert step['epoch_update'] == i+1 and step['epoch'] == epoch
            assert step['batch_size'] == (2 if i == 882 else 4)
            assert step['gradient_norm_before'] > 0 and 0 <= step['gradient_norm_after'] <= .50001
            assert abs(step['loss']-step['pos_contribution']-step['neg_contribution']) < 1e-7
            assert abs(step['pos_contribution']-.5*step['pos_mean']) < 1e-7
            assert abs(step['neg_contribution']-.5*step['neg_mean']) < 1e-7
        for row in images:
            key = row['sample_key']
            assert row['dataset'] == members[key]['dataset']
            assert row['candidate_positive_count']+row['candidate_negative_count'] == 224*224
            for short, sign in (('pos','positive'),('neg','negative')):
                n, k = row[f'candidate_{sign}_count'], row[f'selected_{sign}_count']
                assert k == (n if variant in ('P0','P1') else n//2)
                if variant != 'P0':
                    assert n == selected[key][f'candidate_{short}']
            if variant in ('P2F','P3F'):
                assert row['canonical_selection_hash'] == selected[key][f'{variant}_canonical_hash']
        assert abs(summary['mean_loss']-sum(r['loss'] for r in steps)/883) < 1e-12
    final = torch.load(directory/'final.pt', map_location='cpu', weights_only=False)
    assert final['config'] == config and final['update'] == 17660
    assert final['position'] == {'epoch':20,'next_batch':0}
    assert final['model_hash'] == result['final_hash'] == summary['model_memory_hash']
    assert sha(directory/'final.pt') == sha(directory/'last.pt')
    assert set(final['rng']) == {'python','numpy','torch','cuda'} and final['optimizer']['state']
    return {'status':'PASS', 'epochs':20, 'updates':17660, 'images':70600,
            'final_checkpoint_sha256':sha(directory/'final.pt'), 'gt_reads':0}


def check_evaluation(variant, dev, output_root=None):
    output = resolve_output_root(output_root)
    directory = output/'evaluation'/variant
    summary = read(directory/'summary.json')
    records = rows(directory/'per_image.jsonl')
    accesses = rows(directory/'gt_access.jsonl')
    allow = read(directory/'gt_allowlist.json')
    assert len(records) == len(accesses) == summary['gt_access_count'] == 530
    keys = sorted(dev)
    assert [r['sample_key'] for r in records] == [r['sample_key'] for r in accesses] == keys
    assert allow['members'] == [dev[k] for k in keys]
    for entry in accesses:
        key = entry['sample_key']
        expected = allow['sources'][key]['prepared_mask'].replace('/hy-tmp/datasets/','/home/datasets/')
        assert entry['role'] == 'new_dev' and entry['path'] == expected
        assert entry['operation'] == 'dedicated_strict_evaluator_gt_read'
    assert summary['protocol_hash'] == EVAL_HASH and summary['members_hash'] == canonical_hash(keys)
    assert summary['new_confirm_gt_access_count'] == 0 and finite(summary) and finite(records)
    assert summary['checkpoint_sha256'] == summary['checkpoint_sha256_after'] == sha(output/variant/'final.pt')
    assert summary['per_image_sha256'] == sha(directory/'per_image.jsonl')
    assert summary['gt_access_log_sha256'] == sha(directory/'gt_access.jsonl')
    assert summary['result_hash'] == canonical_hash({k:v for k,v in summary.items() if k != 'result_hash'})
    aggregate = summarize_records(records)
    assert aggregate['domains'] == summary['domains'] and aggregate['dataset_equal'] == summary['dataset_equal']
    memory = read(output/'evaluation'/f'{variant}_memory_gate.json')
    assert memory['status'] == 'PASS' and all(memory['checks'].values())
    return {'images':530, 'gt_reads':530, 'new_confirm_gt_reads':0,
            'memory_order_gate':'PASS', 'stop_formal_request':summary['stop_formal_request'],
            'metrics':summary['domains'], 'dataset_equal':summary['dataset_equal']}


def main(output_root=None):
    output = resolve_output_root(output_root)
    preflight = read(output/'preflight.json')
    assert preflight['status'] == 'PASS'
    protected = read(output/'protected_before.json')
    for path, digest in {**protected, **preflight['runtime_hashes']}.items():
        assert sha(ROOT/path) == digest, path
    split = read_split()
    members = {r['sample_key']:r for r in split['records'] if r['subset'] == 'student_train'}
    dev = {r['sample_key']:r for r in split['records'] if r['subset'] == 'new_dev'}
    assert canonical_hash(sorted(members)) == MEMBER_HASH
    selected = {r['sample_key']:r for r in read(ROOT/'fecd/assets/stage4f0/selection_audit.json')['per_image']}
    schedule = read(output/'epoch_schedule.json')
    results = {}
    for variant in NAMES:
        results[variant] = check_training(variant, schedule, selected, members, output)
        results[variant]['evaluation'] = check_evaluation(variant, dev, output)
        print(json.dumps({'variant':variant, 'audit':'PASS'}), flush=True)
    initial = [read(output/v/'result.json')['initial_hash'] for v in NAMES]
    assert len(set(initial)) == 1 and initial[0] == preflight['initial_hashes']['P0']
    paired = read(output/'analysis/paired_bootstrap.json')
    assert paired['protocol'] == STATISTICS
    for name in STATISTICS['comparisons']:
        right, left = name.split('-')
        expected = paired_bootstrap(rows(output/'evaluation'/left/'per_image.jsonl'),
                                    rows(output/'evaluation'/right/'per_image.jsonl'),
                                    STATISTICS['repeats'], STATISTICS['seed'])
        assert paired['comparisons'][name] == expected
    assert len(read(output/'analysis/training_curves.json')) == 400
    paths = set(preflight['runtime_hashes']) | {str(p.relative_to(ROOT)) for p in ROOT.glob('fecd/tools/*formal_students.py')}
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    paths.update(str(p.relative_to(ROOT)) for p in ROOT.glob('fecd/tools/*formal_curves.py'))
    issues = {p:analyze_source(ROOT/p) for p in paths if p.endswith('.py') and
              ('f1_' in p or 'formal_student' in p or 'test_stage4f1' in p or 'check_stage4f1' in p or 'formal_curves' in p)}
    assert not any(issues.values()), issues
    result = {'status':'PASS', 'engineering':'PASS', 'variants':results,
              'protected_files_unchanged':len(protected), 'runtime_hashes_unchanged':len(preflight['runtime_hashes']),
              'training_gt_reads':0, 'new_dev_gt_reads':2120, 'new_confirm_gt_reads':0,
              'teacher_diag_or_test_gt_reads':0, 'auditor_gt_image_reads':0, 'static_issues':issues,
              'formal_training_seeds':1, 'additional_seed_training':False}
    result['result_hash'] = canonical_hash(result)
    write_json(output/'validation.json', result)
    print(json.dumps({'status':'PASS','result_hash':result['result_hash']}), flush=True)


if __name__ == '__main__':
    main(parse_output_root())
