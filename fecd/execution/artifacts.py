"""Artifact gates supplement subprocess status; the existing full auditor remains mandatory."""
import json
import math
from pathlib import Path
import re
import sys

import torch

from fecd.execution.shutdown import file_sha
from fecd.pseudo.cache_schema import canonical_hash
from fecd.training.f1_protocol import NAMES, ROOT, MEMBER_HASH, FORMAL_HASH, EVAL_HASH, configuration

STAGES = ('preflight', *NAMES, 'evaluation', 'statistics', 'curves', 'tests', 'validation')
MODULES = {'preflight':'preflight_formal_student', 'evaluation':'evaluate_formal_students',
           'statistics':'summarize_formal_students', 'curves':'plot_formal_curves',
           'validation':'check_stage4f1'}


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(stage, artifacts):
    prefix = [sys.executable, '-u']
    if stage == 'tests':
        return prefix+['-m','unittest','discover','-s','fecd/tests']
    module = 'train_formal_student' if stage in NAMES else MODULES[stage]
    result = prefix+['-m',f'fecd.tools.{module}','--output-root',str(artifacts)]
    return result+['--variant',stage] if stage in NAMES else result


def verify_training(artifacts, variant, dry_run):
    directory = artifacts/variant
    result = read(directory/'result.json')
    config = configuration(variant)
    require(result['status'] == 'PASS' and result['epochs'] == 20 and
            result['logical_updates'] == 17660 and result['gt_reads'] == 0, 'Incomplete formal training')
    require(bool(result.get('simulation',False)) == dry_run, 'Simulation cannot substitute formal training')
    require(read(directory/'config.json') == config and result['config_hash'] == config['config_hash'], 'Config mismatch')
    require(result['membership_hash'] == MEMBER_HASH, 'Training membership mismatch')
    for epoch in range(1,21):
        record = read(directory/f'epoch{epoch:02d}.summary.json')
        require(record['status'] == 'PASS' and record['updates'] == 883 and record['images'] == 3530,
                'Missing/incomplete epoch summary')
    checkpoint = directory/'final.pt'
    digest = file_sha(checkpoint)
    if 'checkpoint_sha256' in result:
        require(result['checkpoint_sha256'] == digest, 'Final checkpoint SHA256 mismatch')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(payload['config'] == config and payload['update'] == 17660 and
            payload['position'] == {'epoch':20,'next_batch':0}, 'Wrong final checkpoint identity')
    require(payload['model_hash'] == result['final_hash'], 'Model identity mismatch')
    if not dry_run:
        from fecd.evaluation.strict import load_independent_model
        model, snapshot, metadata = load_independent_model(checkpoint)
        require(metadata['checkpoint_model_hash'] == result['final_hash'], 'Full model/memory load failed')
        del model, snapshot
        torch.cuda.empty_cache()
    return {'checkpoint':str(checkpoint), 'sha256':digest, 'model_hash':result['final_hash']}


def verify_stage(stage, artifacts, dry_run):
    artifacts = Path(artifacts)
    if stage == 'preflight':
        result = read(artifacts/'preflight.json')
        require(result['status'] == 'PASS' and result['member_hash'] == MEMBER_HASH, 'Preflight failed')
        require(result['formal_proposal_hash'] == FORMAL_HASH and result['evaluation_protocol_hash'] == EVAL_HASH,
                'Frozen protocol mismatch')
        require(len(result['initial_hashes']) == 4 and len(set(result['initial_hashes'].values())) == 1,
                'Unequal initial weights')
        require(result['gt_reads'] == 0 and result['new_confirm_gt_reads'] == 0, 'Preflight GT access')
    elif stage in NAMES:
        return verify_training(artifacts, stage, dry_run)
    elif stage == 'evaluation':
        for variant in NAMES:
            record = read(artifacts/'evaluation'/variant/'summary.json')
            require(record['protocol_hash'] == EVAL_HASH and record['gt_access_count'] == 530 and
                    record['new_confirm_gt_access_count'] == 0, 'Incomplete/unsafe dev evaluation')
            require(record['checkpoint_sha256'] == file_sha(artifacts/variant/'final.pt'), 'Evaluated checkpoint changed')
            require(record['result_hash'] == canonical_hash({k:v for k,v in record.items() if k != 'result_hash'}),
                    'Evaluation result hash mismatch')
            for domain in ('ALL','CASIAv2','NIST16','Columbia','Coverage'):
                for metric in ('image_macro_f1','image_macro_iou','pooled_micro_f1','pooled_micro_iou'):
                    value = record['domains'][domain][metric]
                    require(math.isfinite(value) and 0 <= value <= 1, 'Nonfinite/missing domain metric')
    elif stage == 'statistics':
        from fecd.training.f1_protocol import STATISTICS
        result = read(artifacts/'analysis/paired_bootstrap.json')
        require(result['protocol'] == STATISTICS, 'Bootstrap protocol mismatch')
        for name in STATISTICS['comparisons']:
            value = result['comparisons'][name]
            for metric in ('f1','iou'):
                for weight in ('image_weighted','group_equal','dataset_equal'):
                    item = value[metric][weight]
                    require(len(item['ci95']) == 2 and all(math.isfinite(x) for x in
                            [item['estimate'], *item['ci95']]), 'Incomplete paired inference')
        require((artifacts/'analysis/domain_metrics.csv').is_file() and
                (artifacts/'analysis/area_diagnostics.json').is_file(), 'Missing domain/area analysis')
    elif stage == 'curves':
        require((artifacts/'analysis/training_curves.svg').is_file() and
                len(read(artifacts/'analysis/training_curves.json')) == 400, 'Incomplete epoch curves')
    elif stage == 'tests':
        log = (artifacts.parent/'logs/tests.log').read_text()
        require(bool(re.search(r'Ran [1-9][0-9]* tests?',log)) and log.rstrip().endswith('OK'), 'Tests did not pass')
    elif stage == 'validation':
        result = read(artifacts/'validation.json')
        require(result['status'] == result['engineering'] == 'PASS', 'Final engineering audit failed')
        require(result['training_gt_reads'] == result['new_confirm_gt_reads'] ==
                result['teacher_diag_or_test_gt_reads'] == 0, 'Forbidden GT access')
        require(result['new_dev_gt_reads'] == 2120, 'Incomplete dev GT audit')
        require(result['result_hash'] == canonical_hash({k:v for k,v in result.items() if k != 'result_hash'}),
                'Final validation hash mismatch')
    else:
        raise ValueError('Unknown execution stage')
    return {'stage':stage, 'status':'PASS'}


def frozen_files():
    path = ROOT/'fecd/assets/stage4f1_storage_audit/delivery_manifest.json'
    result = {str(ROOT/p):item['sha256'] for p,item in read(path)['files'].items()}
    result[str(path)] = file_sha(path)
    prior = read(ROOT/'fecd/assets/stage4f1_storage_audit/before.json')
    result.update({str(ROOT/p):digest for p,digest in prior['existing_4f1_artifacts_and_configs'].items()})
    for folder in ('fecd/execution','fecd/tests'):
        for source in (ROOT/folder).glob('*.py'):
            result[str(source)] = file_sha(source)
    for name in ('run_stage4f1_autoshutdown','check_stage4f1_autoshutdown'):
        source = ROOT/f'fecd/tools/{name}.py'
        result[str(source)] = file_sha(source)
    evidence = ROOT/'fecd/configs/stage4f1_autoshutdown/shutdown_evidence.json'
    result[str(evidence)] = file_sha(evidence)
    return result
