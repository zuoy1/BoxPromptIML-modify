"""Synthetic artifact producer for full orchestration dry runs; no real dataset IO."""
import json
from pathlib import Path

import torch

from fecd.execution.shutdown import file_sha
from fecd.pseudo.cache_schema import canonical_hash
from fecd.training.f1_protocol import NAMES, MEMBER_HASH, FORMAL_HASH, EVAL_HASH, STATISTICS, configuration


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream)


def simulate_stage(stage, artifacts, progress):
    artifacts = Path(artifacts)
    if stage == 'preflight':
        artifacts.mkdir(exist_ok=False)
        write(artifacts/'preflight.json', {'status':'PASS', 'member_hash':MEMBER_HASH,
            'formal_proposal_hash':FORMAL_HASH, 'evaluation_protocol_hash':EVAL_HASH,
            'initial_hashes':{name:'mock-common-init' for name in NAMES}, 'gt_reads':0,
            'new_confirm_gt_reads':0, 'simulation':True})
    elif stage in NAMES:
        directory = artifacts/stage
        directory.mkdir()
        config = configuration(stage)
        write(directory/'config.json', config)
        for epoch in range(1,21):
            write(directory/f'epoch{epoch:02d}.summary.json',
                  {'status':'PASS', 'epoch':epoch, 'updates':883, 'images':3530, 'simulation':True})
        with (directory/'final.pt').open('xb') as stream:
            torch.save({'config':config, 'update':17660, 'position':{'epoch':20,'next_batch':0},
                        'model_hash':'mock-model-hash', 'simulation':True}, stream)
        write(directory/'result.json', {'status':'PASS', 'simulation':True, 'epochs':20,
            'logical_updates':17660, 'gt_reads':0, 'membership_hash':MEMBER_HASH,
            'config_hash':config['config_hash'], 'final_hash':'mock-model-hash',
            'checkpoint_sha256':file_sha(directory/'final.pt')})
        progress(epoch=20, latest_checkpoint=str(directory/'final.pt'))
    elif stage == 'evaluation':
        for name in NAMES:
            metrics = {d:{m:.1 for m in ('image_macro_f1','image_macro_iou','pooled_micro_f1','pooled_micro_iou')}
                       for d in ('ALL','CASIAv2','NIST16','Columbia','Coverage')}
            result = {'protocol_hash':EVAL_HASH, 'gt_access_count':530, 'simulation':True,
                      'new_confirm_gt_access_count':0, 'checkpoint_sha256':file_sha(artifacts/name/'final.pt'),
                      'domains':metrics}
            result['result_hash'] = canonical_hash(result)
            write(artifacts/'evaluation'/name/'summary.json', result)
    elif stage == 'statistics':
        value = {m:{w:{'estimate':0., 'ci95':[-.1,.1]} for w in
                    ('image_weighted','group_equal','dataset_equal')} for m in ('f1','iou')}
        write(artifacts/'analysis/paired_bootstrap.json', {'protocol':STATISTICS,
              'comparisons':{name:value for name in STATISTICS['comparisons']}, 'simulation':True})
        write(artifacts/'analysis/area_diagnostics.json', {'simulation':True})
        with (artifacts/'analysis/domain_metrics.csv').open('x') as stream:
            stream.write('simulation\ntrue\n')
    elif stage == 'curves':
        write(artifacts/'analysis/training_curves.json', [{'simulation':True} for _ in range(400)])
        with (artifacts/'analysis/training_curves.svg').open('x') as stream:
            stream.write('<svg xmlns="http://www.w3.org/2000/svg"><title>SIMULATION ONLY</title></svg>')
    elif stage == 'tests':
        with (artifacts.parent/'logs/tests.log').open('x') as stream:
            stream.write('Ran 1 test in 0.001s\n\nOK\n')
    elif stage == 'validation':
        value = {'status':'PASS', 'engineering':'PASS', 'simulation':True,
                 'training_gt_reads':0, 'new_confirm_gt_reads':0, 'teacher_diag_or_test_gt_reads':0,
                 'new_dev_gt_reads':2120, 'note':'all counts are synthetic; no GT opened'}
        value['result_hash'] = canonical_hash(value)
        write(artifacts/'validation.json', value)
    else:
        raise ValueError('Unknown simulated stage')
