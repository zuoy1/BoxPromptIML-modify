"""Validate completed 4F0 artifacts without opening any ground-truth image."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from fecd.diagnostics.stage3ra_split import read_split
from fecd.evaluation.metrics import summarize_records
from fecd.pseudo.cache_schema import canonical_hash
from fecd.tools.check_stage3 import analyze_source
from fecd.training.data import K1_ROOT, K8_ROOT, _build_samples
from fecd.training.f0_diagnostics import summarize_outputs
from fecd.training.f0_protocol import configuration, EVAL_HASH, FORMAL_HASH

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'fecd/assets/stage4f0'
NAMES=('P0','P1','P2F','P3F')
DOMAINS={'CASIAv2':3278,'NIST16':172,'Columbia':52,'Coverage':28}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    if isinstance(value,list):
        return all(finite(v) for v in value)
    return not isinstance(value,(int,float)) or math.isfinite(value)


def verify_training(name,members,ordered_keys,selection_rows):
    directory=OUTPUT/name
    result=read(directory/'result.json')
    config=configuration(name)
    steps=lines(directory/'steps.jsonl')
    outputs=lines(directory/'train_outputs.jsonl')
    resume=read(directory/'resume.json')
    assert read(directory/'config.json')==config
    assert result['status']=='PASS' and result['config_hash']==config['config_hash']
    assert result['images']==3530 and result['logical_updates']==883
    assert result['physical_updates']==884 and result['last_batch']==2
    assert result['dataset_forward_backward_counts']==DOMAINS
    assert result['membership_hash']==canonical_hash(sorted(members))
    assert result['gt_reads']==0 and not result['denied'] and finite(result)
    assert len(steps)==883 and len(outputs)==3530 and finite(steps) and finite(outputs)
    assert resume=={'status':'PASS','update':5,'loss_exact':True,'full_state_exact':True}
    used=[key for step in steps for key in step['sample_keys']]
    assert used==ordered_keys and [r['sample_key'] for r in outputs]==used
    assert Counter(r['dataset'] for r in outputs)==Counter(DOMAINS)
    per_image={r['sample_key']:r for r in outputs}
    for key,row in per_image.items():
        assert row['dataset']==members[key]['dataset']
        assert row['candidate_positive_count']+row['candidate_negative_count']==224*224
        for sign in ('positive','negative'):
            n=row[f'candidate_{sign}_count'];k=row[f'selected_{sign}_count']
            assert k==(n if name in ('P0','P1') else n//2)
            assert abs(row[f'selected_{sign}_full_fraction']-k/(224*224))<1e-7
        if name in ('P2F','P3F'):
            assert row['canonical_selection_hash']==selection_rows[key][f'{name}_canonical_hash']
    for i,step in enumerate(steps):
        assert step['update']==i+1 and step['batch_size']==(2 if i==882 else 4)
        group=[per_image[key] for key in step['sample_keys']]
        assert all(r['update']==i+1 for r in group)
        assert step['operations']==[r['operation'] for r in group]
        assert step['gradient_norm_before']>0 and 0<=step['gradient_norm_after']<=.50001
        assert step['step_seconds']>0 and step['peak_cuda_allocated_bytes']>0
        assert abs(step['loss']-step['pos_contribution']-step['neg_contribution'])<1e-7
        assert abs(step['pos_contribution']-.5*step['pos_mean'])<1e-7
        assert abs(step['neg_contribution']-.5*step['neg_mean'])<1e-7
        for short,long in (('pos','positive'),('neg','negative')):
            assert step[f'selected_{short}']==sum(r[f'selected_{long}_count'] for r in group)
            assert step[f'valid_{short}_images']==sum(r[f'selected_{long}_count']>0 for r in group)
    cache_root=K1_ROOT if name=='P0' else K8_ROOT
    samples=_build_samples(list(members.values()),read(cache_root/'dataset_index.json'),cache_root)
    allowed={str(Path(row[field]).resolve()) for row in samples for field in ('image_path','cache_path')}
    access=read(directory/'access.json')
    assert not access['denied'] and set(access['opened'])==allowed
    assert result['access_unique_files']==len(allowed)==7060
    summary=read(directory/'training_summary.json')
    assert summary==summarize_outputs(outputs) and summary['domains']['ALL']['images']==3530
    assert abs(result['mean_loss']-statistics.mean(r['loss'] for r in steps))<1e-12
    table={'loss_first':steps[0]['loss'],'loss_last':steps[-1]['loss'],
           'loss_mean':statistics.mean(r['loss'] for r in steps),
           'positive_contribution_mean':statistics.mean(r['pos_contribution'] for r in steps),
           'negative_contribution_mean':statistics.mean(r['neg_contribution'] for r in steps),
           'warm_step_median_seconds':statistics.median(r['step_seconds'] for r in steps[1:]),
           'peak_allocated_bytes':max(r['peak_cuda_allocated_bytes'] for r in steps),
           'peak_reserved_bytes':max(r['peak_cuda_reserved_bytes'] for r in steps),
           'train_outputs':summary,'elapsed_seconds':result['elapsed_seconds']}
    return result,steps,per_image,table


def verify_evaluation(name,dev,training_result):
    directory=OUTPUT/'evaluation'/name
    summary=read(directory/'summary.json')
    rows=lines(directory/'per_image.jsonl')
    accesses=lines(directory/'gt_access.jsonl')
    metadata=read(directory/'evaluation_config.json')
    allow=read(directory/'gt_allowlist.json')
    keys=sorted(dev)
    assert summary['protocol_hash']==EVAL_HASH and metadata['protocol_hash']==EVAL_HASH
    assert summary['split_hash']==metadata['split_hash']==configuration(name)['split_hash']
    assert summary['construction']['checkpoint_config']==configuration(name)
    assert summary['construction']['checkpoint_model_hash']==training_result['final_hash']
    assert summary['members_hash']==metadata['members_hash']==canonical_hash(keys)
    assert len(rows)==len(accesses)==summary['gt_access_count']==530
    assert [r['sample_key'] for r in rows]==[r['sample_key'] for r in accesses]==keys
    assert [r['sample_key'] for r in allow['members']]==keys
    assert all(row==dev[row['sample_key']] for row in allow['members'])
    assert summary['new_confirm_gt_access_count']==metadata['new_confirm_gt_access_count']==0
    for entry in accesses:
        key=entry['sample_key'];source=allow['sources'][key]
        expected=str(source['prepared_mask']).replace('/hy-tmp/datasets/','/home/datasets/')
        assert entry['path']==expected and entry['role']=='new_dev'
        assert entry['operation']=='dedicated_strict_evaluator_gt_read'
        assert entry['native_hw']==[source['height'],source['width']]
        assert len(entry['sha256'])==64 and source['image_sha256']==dev[key]['image_sha256']
    assert finite(summary) and finite(rows)
    assert summary['gt_access_log_sha256']==sha(directory/'gt_access.jsonl')
    assert summary['per_image_sha256']==sha(directory/'per_image.jsonl')
    checkpoint=(OUTPUT/name/'epoch1.pt').resolve()
    assert Path(summary['checkpoint']).resolve()==checkpoint
    assert summary['checkpoint_sha256']==summary['checkpoint_sha256_after']==sha(checkpoint)
    assert summary['result_hash']==canonical_hash({k:v for k,v in summary.items() if k!='result_hash'})
    aggregate=summarize_records(rows)
    assert aggregate['domains']==summary['domains'] and aggregate['dataset_equal']==summary['dataset_equal']
    assert summary['domains']['ALL']['images']==530
    assert Counter(r['dataset'] for r in rows)==Counter(r['dataset'] for r in dev.values())
    area=sum(r['all_zero_prediction'] or r['near_full_prediction'] for r in rows)/530
    saturated=sum(r['saturated_probability_mean'] for r in rows)/530
    expected_stop=(area>=.5 and summary['domains']['ALL']['image_macro_f1']<=.05) or saturated>=.9
    assert summary['stop_formal_request']==expected_stop
    return {k:summary[k] for k in ('domains','dataset_equal','stop_formal_request',
                                  'zero_or_near_full_fraction','saturated_probability_fraction',
                                  'seconds','checkpoint_sha256','result_hash')}


def main():
    target=OUTPUT/'validation.json'
    if target.exists():
        raise FileExistsError('Refusing to replace 4F0 validation')
    split=read_split()
    members={r['sample_key']:r for r in split['records'] if r['subset']=='student_train'}
    dev={r['sample_key']:r for r in split['records'] if r['subset']=='new_dev'}
    assert len(members)==3530 and len(dev)==530 and not set(members)&set(dev)
    gates=read(OUTPUT/'gates.json')
    assert gates['status']=='PASS' and gates['eval_protocol_hash']==EVAL_HASH
    assert gates['formal_proposal_hash']==FORMAL_HASH and gates['gt_reads']==0
    changed_code=[p for p,h in gates['code_hashes'].items() if sha(ROOT/p)!=h]
    assert not changed_code,changed_code
    protected=read(OUTPUT/'protected_before.json')
    assert len(protected)==gates['protected_files']==17610
    changed=[p for p,h in protected.items() if sha(ROOT/p)!=h]
    assert not changed,changed
    selection=read(OUTPUT/'selection_audit.json')
    assert selection['status']=='PASS' and selection['result_hash']==gates['selection_result_hash']
    assert selection['result_hash']==canonical_hash({k:v for k,v in selection.items() if k!='result_hash'})
    selected={r['sample_key']:r for r in selection['per_image']}
    assert set(selected)==set(members)
    memory=read(OUTPUT/gates['memory_gate'])
    assert memory['eval_protocol_hash']==EVAL_HASH and all(memory['checks'].values())
    keys=sorted(members)
    order=torch.randperm(3530,generator=torch.Generator().manual_seed(20261029)).tolist()
    ordered_keys=[keys[i] for i in order]
    results={};steps={};images={};tables={}
    for name in NAMES:
        results[name],steps[name],images[name],tables[name]=verify_training(name,members,ordered_keys,selected)
        assert results[name]['order_hash']==canonical_hash(order)
        tables[name]['evaluation']=verify_evaluation(name,dev,results[name])
    for name in NAMES[1:]:
        for field in ('initial_hash','order_hash','membership_hash'):
            assert results[name][field]==results['P0'][field]
        for first,other in zip(steps['P0'],steps[name]):
            assert first['sample_keys']==other['sample_keys'] and first['operations']==other['operations']
    for key in members:
        for field in ('candidate_positive_count','candidate_negative_count',
                      'selected_positive_count','selected_negative_count'):
            assert images['P2F'][key][field]==images['P3F'][key][field]
        for field in ('candidate_positive_count','candidate_negative_count'):
            assert images['P1'][key][field]==images['P2F'][key][field]
    issues=analyze_source(Path(__file__))
    assert not issues,issues
    stopped=any(t['evaluation']['stop_formal_request'] for t in tables.values())
    result={'status':'PASS','engineering_checks':'PASS','formal_request_ready':not stopped,
            'action':'BLOCKED' if stopped else 'PASS','variants':tables,
            'protected_files_unchanged':len(protected),'gate_code_files_unchanged':len(gates['code_hashes']),
            'static_issues':issues,'new_confirm_gt_access_count':0,'auditor_gt_image_reads':0,
            'resume_evidence':'exact next-step loss and model+buffer+memory digest; no direct Adam/RNG tensor equality claim',
            'interpretation':'one-epoch engineering preflight, no model ranking or formal20 authorization'}
    result['result_hash']=canonical_hash(result)
    with target.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':result['status'],'action':result['action'],
                      'formal_request_ready':result['formal_request_ready'],'result_hash':result['result_hash']}))
    for name,table in tables.items():
        print(json.dumps({'variant':name,'loss':table['loss_mean'],
                          **table['evaluation']['domains']['ALL'],
                          'stop':table['evaluation']['stop_formal_request']}))


if __name__=='__main__':
    main()
