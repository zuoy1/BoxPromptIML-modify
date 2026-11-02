"""4F1-V 完整 CPU 工程复验；与原运行隔离写出 PASS/FAIL。"""
import argparse
import ast
from datetime import datetime, timezone
import inspect
from pathlib import Path
import sys
import traceback

import torch

from fecd.diagnostics.stage3ra_split import read_split
from fecd.evaluation.f1_statistics import paired_bootstrap
from fecd.pseudo.cache_schema import canonical_hash
from fecd.tools import check_stage4f1 as original
from fecd.tools.check_stage3 import analyze_source
from fecd.tools.check_stage4f1_v2 import check_training
from fecd.tools.stage4f1_v2_checks import (read, rows, check_metrics, checkpoint_hash,
                                         check_research, preserved_training_source)
from fecd.tools.stage4f1_v2_io import (sha, write_new, install_boundary, verify_snapshot,
                                     source_hashes, boundary_evidence)
from fecd.training.f1_protocol import (ROOT, NAMES, MEMBER_HASH, FORMAL_HASH, EVAL_HASH,
                                       STATISTICS, configuration, epoch_identity, verify_frozen)

VERSION = 'stage4f1-v2-float32-exact-1'
RUN_ID = 'formal20_seed20261029_noshutdown_20261029a'
CHECKPOINT_HASHES = {
    'P0': '9ea55e01e5e354d252fb6470c7a3aa04712840eb657917e694c19f56650bd1ea',
    'P1': 'b0e56a4eb4ffc361f80476af580b30c2351ad885bea10f251b08ae119cdf4d89',
    'P2F': '1d1997cd71b8ff82b5e42debdf5e1e9e8ea659b8c1194c231481c3e0088a9404',
    'P3F': '8c6dc3512bdebaf0885b85dbe9b65af737b804f90e21238405073cbe4fa11ea6',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _frozen(output):
    verify_frozen()
    pre = read(output / 'preflight.json')
    _require(pre['status'] == 'PASS', '预检未通过')
    _require(pre['formal_proposal_hash'] == FORMAL_HASH and pre['evaluation_protocol_hash'] == EVAL_HASH
             and pre['member_hash'] == MEMBER_HASH, '冻结协议身份不一致')
    protected = read(output / 'protected_before.json')
    for path, digest in {**protected, **pre['runtime_hashes']}.items():
        _require(sha(ROOT / path) == digest, f'冻结文件变化: {path}')
    revised = ast.dump(ast.parse(inspect.getsource(check_training)))
    _require(revised == preserved_training_source(original.check_training), '非数值逻辑变化')
    return {'protected_unchanged': len(protected), 'runtime_unchanged': len(pre['runtime_hashes']),
            'original_non_numeric_training_logic': 'unchanged', 'preflight': pre}


def _identities(output, pre):
    split = read_split()
    members = {r['sample_key']: r for r in split['records'] if r['subset'] == 'student_train'}
    dev = {r['sample_key']: r for r in split['records'] if r['subset'] == 'new_dev'}
    _require(len(members) == 3530 and len(dev) == 530 and canonical_hash(sorted(members)) == MEMBER_HASH,
             '成员身份错误')
    schedule = read(output / 'epoch_schedule.json')
    _require(schedule == [epoch_identity(sorted(members), 20261029, e) for e in range(20)],
             '冻结顺序或几何计划错误')
    selected = {r['sample_key']: r for r in read(ROOT / 'fecd/assets/stage4f0/selection_audit.json')['per_image']}
    _require(set(selected) == set(members), '选择审计成员错误')
    initial = [read(output / v / 'result.json')['initial_hash'] for v in NAMES]
    _require(len(set(initial)) == 1 and initial[0] == pre['initial_hashes']['P0'], '共同初始化错误')
    for v in NAMES:
        _require(pre['initial_hashes'][v] == initial[0], f'{v} 预检初始化错误')
        initialization = read(output / v / 'initialization.json')
        _require(initialization['initial_hash'] == initial[0] and initialization['fresh_initialization'],
                 f'{v} 初始化记录错误')
        _require(pre['resume_probes'][v]['loss_exact']
                 and pre['resume_probes'][v]['all_model_adam_rng_memory_exact'], '历史恢复探针失败')
    return members, dev, selected, schedule


def _variant(output, variant, identities):
    members, dev, selected, schedule = identities
    result = check_training(variant, schedule, selected, members, output)
    result['evaluation'] = original.check_evaluation(variant, dev, output)
    records = rows(output / 'evaluation' / variant / 'per_image.jsonl')
    check_metrics(records)
    payload = torch.load(output / variant / 'final.pt', map_location='cpu', weights_only=False)
    _require(payload['schema'] == 'boxprompt-stage4p-checkpoint-v1', 'Checkpoint schema 错误')
    actual = checkpoint_hash(payload)
    _require(actual == payload['model_hash'] == read(output / variant / 'result.json')['final_hash'],
             f'{variant} 模型 tensor/memory 内容哈希错误')
    _require(result['final_checkpoint_sha256'] == CHECKPOINT_HASHES[variant], '与原报告 checkpoint SHA 不一致')
    _require(payload['config'] == configuration(variant), 'Checkpoint 配置改变')
    old_false_alarms = []
    for epoch in range(1, 21):
        for step in rows(output / variant / f'epoch{epoch:02d}.steps.jsonl'):
            residual = abs(step['loss'] - step['pos_contribution'] - step['neg_contribution'])
            if residual >= 1e-7:
                old_false_alarms.append({**{k: step[k] for k in ('epoch', 'epoch_update', 'update',
                                          'loss', 'pos_contribution', 'neg_contribution')},
                                        'old_residual': residual})
    result.update(verified_loss_records=17660, model_content_hash=actual,
                  old_false_alarms=old_false_alarms, per_image_confusion_metrics='PASS')
    return result


def _statistics(output):
    paired = read(output / 'analysis/paired_bootstrap.json')
    _require(paired['protocol'] == STATISTICS, '统计协议错误')
    for name in STATISTICS['comparisons']:
        right, left = name.split('-')
        expected = paired_bootstrap(rows(output / 'evaluation' / left / 'per_image.jsonl'),
                                    rows(output / 'evaluation' / right / 'per_image.jsonl'),
                                    STATISTICS['repeats'], STATISTICS['seed'])
        _require(paired['comparisons'][name] == expected, f'配对统计复算不一致: {name}')
    _require(len(read(output / 'analysis/training_curves.json')) == 400, '训练曲线记录不完整')
    return {'paired_comparisons': list(STATISTICS['comparisons']), 'bootstrap': 'exact match',
            'curve_records': 400, 'research': check_research(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run, delivery = args.input_run.resolve(strict=True), args.output_dir.resolve(strict=True)
    _require(__debug__, '禁止以 python -O 运行包含原始 assert 的验收器')
    _require(run.name == RUN_ID and not delivery.is_relative_to(run), '输入运行或独立输出目录无效')
    _require(not (delivery / 'validation_v2.json').exists(), '禁止覆盖已有复验结果')
    torch.set_num_threads(4)
    boundary = install_boundary(delivery)
    snapshot = read(delivery / 'inputs_before.json')
    result = {'validator_version': VERSION, 'input_run_id': run.name,
              'started_at': datetime.now(timezone.utc).isoformat(), 'checks': {},
              'correction_reason': 'float32 tensor 加法转换为 JSON 后，Python float 固定绝对容差产生误报',
              'historical_status_relationship': '补充工程复验；不改写原执行器 FAILED',
              'device': 'cpu', 'model_constructed': False, 'training_performed': False,
              'new_gt_evaluation': False, 'new_seeds': False, 'source_sha256': source_hashes(),
              'input_manifest_sha256': sha(delivery / 'inputs_before.json')}

    def stage(name, function):
        try:
            value = function()
            result['checks'][name] = {'status': 'PASS', 'details': value}
            print(f'{name}: PASS', flush=True)
            return value
        except Exception as exc:
            result['checks'][name] = {'status': 'FAIL', 'error_type': type(exc).__name__,
                                     'error': str(exc), 'traceback': traceback.format_exc()}
            print(f'{name}: FAIL\n{traceback.format_exc()}', flush=True)
            return None

    output = run / 'artifacts'
    stage('input_snapshot_before', lambda: verify_snapshot(snapshot, run))
    stage('original_failed_preserved', lambda: _require(read(run / 'status.json')['status'] == 'FAILED',
                                                      '原始终态改变'))
    stage('frozen_and_original_logic', lambda: _frozen(output))
    identities = stage('members_initialization_schedule', lambda: _identities(output, read(output / 'preflight.json')))
    if identities is not None:
        result['checks']['members_initialization_schedule']['details'] = {
            'training_members': len(identities[0]), 'dev_members': len(identities[1]),
            'frozen_selection_members': len(identities[2]), 'epoch_identities': len(identities[3]),
            'common_initialization': 'PASS'}
        for variant in NAMES:
            stage(variant, lambda v=variant: _variant(output, v, identities))
    else:
        for variant in NAMES:
            result['checks'][variant] = {'status': 'FAIL', 'error': '成员/协议检查失败，禁止签发该组PASS'}
    stage('statistics_curves_research', lambda: _statistics(output))

    def static():
        paths = set(read(output / 'preflight.json')['runtime_hashes']) | set(source_hashes())
        paths.update(str(p.relative_to(ROOT)) for p in ROOT.glob('fecd/tools/*formal_students.py'))
        paths.update(str(p.relative_to(ROOT)) for p in ROOT.glob('fecd/tools/*formal_curves.py'))
        paths.add('fecd/tools/check_stage4f1.py')
        issues = {p: analyze_source(ROOT / p) for p in paths if p.endswith('.py')}
        _require(not any(issues.values()), f'静态检查错误: {issues}')
        return {'files': len(issues), 'issues': issues}

    stage('static_analysis', static)
    stage('strict_existing_json', lambda: _strict_files(run))
    stage('input_snapshot_after', lambda: verify_snapshot(snapshot, run))
    stage('read_boundary', lambda: _require(not boundary.denied, '只读边界违规'))
    evidence = stage('consumed_inputs_and_boundary', lambda: boundary_evidence(boundary, snapshot, run))
    if evidence is None:
        evidence = {'status': 'FAIL', 'denied': boundary.denied,
                    'reason': result['checks']['consumed_inputs_and_boundary']['error']}
    else:
        result['checks']['consumed_inputs_and_boundary']['details'] = {
            'read_files': len(evidence['read_files_sha256']), 'dataset_images_opened': 0,
            'input_snapshot_check': evidence['input_snapshot_check']}
    write_new(delivery / 'access_and_inputs.json', evidence)
    result['access_evidence_sha256'] = sha(delivery / 'access_and_inputs.json')
    result['ended_at'] = datetime.now(timezone.utc).isoformat()
    result['status'] = 'PASS' if all(c['status'] == 'PASS' for c in result['checks'].values()) else 'FAIL'
    result['engineering'] = result['status']
    result['research_conclusion'] = 'NO CLEAR GAIN'
    result['result_hash'] = canonical_hash(result)
    write_new(delivery / 'validation_v2.json', result)
    print(f"FINAL {result['status']}; result_hash={result['result_hash']}", flush=True)
    if result['status'] != 'PASS':
        raise SystemExit(1)


def _strict_files(run):
    count = 0
    for path in sorted(run.rglob('*')):
        if path.suffix == '.json':
            read(path)
        elif path.suffix == '.jsonl':
            rows(path)
        else:
            continue
        count += 1
    return {'strict_json_and_jsonl_files': count, 'nonfinite_or_duplicate_fields': 0}


if __name__ == '__main__':
    main()
