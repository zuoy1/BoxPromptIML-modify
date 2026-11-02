"""Verify actual controlled smoke artifacts and immutable data boundaries."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from fecd.diagnostics.stage3ra_split import read_split
from fecd.tools.check_stage3 import analyze_source
from fecd.training.protocol import configuration
from fecd.pseudo.cache_schema import canonical_hash

PROJECT=Path(__file__).resolve().parents[2]
ASSETS=PROJECT/'fecd/assets/stage4p'


def main():
    if (ASSETS/'validation.json').exists():
        raise FileExistsError('Refusing to replace stage4p verification')
    split=read_split()
    members={r['sample_key']:r for r in split['records'] if r['subset']=='student_train'}
    keys=sorted(members)
    base=None
    results={}
    all_steps={}
    for name in ('P0','P1','P2','P3'):
        directory=ASSETS/name
        result=json.loads((directory/'result.json').read_text())
        steps=[json.loads(r) for r in (directory/'steps.jsonl').read_text().splitlines()]
        access=json.loads((directory/'access.json').read_text())
        assert result['status']=='PASS' and result['gt_files_read']==0 and not access['denied']
        assert result['config_hash']==configuration(name)['config_hash']
        assert len(steps)==8 and result['consumed_images']==32 and result['training_images']==3530
        assert result['train_member_keys_hash']==canonical_hash(keys)
        if base is None:base=result
        for field in ('initial_model_hash','order_hash','train_member_keys_hash','tinyvit_sha256'):
            assert base[field]==result[field],field
        used=[k for step in steps for k in step['sample_keys']]
        assert len(set(used))==32 and set(used)<=set(members)
        for step in steps:
            assert step['finite'] and np.isfinite(step['loss'])
            assert step['gradient_norm_before']>0 and step['gradient_norm_after']<=.50001
            assert abs(step['loss']-step['pos_contribution']-step['neg_contribution'])<1e-7
            assert abs(step['pos_contribution']-.5*step['pos_mean'])<1e-7
            assert abs(step['neg_contribution']-.5*step['neg_mean'])<1e-7
            for sign in ('pos','neg'):
                c=step[f'candidate_{sign}_per_image']
                selected=step[f'selected_{sign}_per_image']
                assert selected==[n if name in ('P0','P1') else n//2 for n in c]
        summary={'status':'PASS','updates':8,'consumed_images':32,'physical_updates_with_probe':9,
                 'datasets':dict(Counter(members[k]['dataset'] for k in used)),
                 'loss_first':steps[0]['loss'],'loss_last':steps[-1]['loss'],
                 'loss_mean':float(np.mean([r['loss'] for r in steps])),
                 'positive_contribution_mean':float(np.mean([r['pos_contribution'] for r in steps])),
                 'negative_contribution_mean':float(np.mean([r['neg_contribution'] for r in steps])),
                 'selected_positive_total':sum(r['selected_pos'] for r in steps),
                 'selected_negative_total':sum(r['selected_neg'] for r in steps),
                 'prediction_positive_mean':float(np.mean([r['prediction_positive_fraction'] for r in steps])),
                 'prediction_negative_mean':float(np.mean([1-r['prediction_positive_fraction'] for r in steps])),
                 'peak_allocated_bytes':max(r['peak_cuda_allocated_bytes'] for r in steps),
                 'peak_reserved_bytes':max(r['peak_cuda_reserved_bytes'] for r in steps),
                 'gradient_norm_before_range':[min(r['gradient_norm_before'] for r in steps),
                                               max(r['gradient_norm_before'] for r in steps)],
                 'step_first_seconds':steps[0]['step_seconds'],
                 'warm_step_median_seconds':float(np.median([r['step_seconds'] for r in steps[1:]])),
                 'full_elapsed_including_probe':result['elapsed_seconds'],
                 'resume':json.loads((directory/'resume_check.json').read_text()),
                 'tinyvit':result['tinyvit'],'initial_hash':result['initial_model_hash'],
                 'last_hash':result['final_model_hash'],
                 'effective_class_coefficients':{
                     sign:[.5*r[f'valid_{sign}_images']/4 for r in steps] for sign in ('pos','neg')}}
        results[name]=summary
        all_steps[name]=steps
    for i in range(8):
        for n in ('P1','P2','P3'):
            assert all_steps[n][i]['sample_keys']==all_steps['P0'][i]['sample_keys']
            assert all_steps[n][i]['operations']==all_steps['P0'][i]['operations']
        for sign in ('pos','neg'):
            assert all_steps['P2'][i][f'selected_{sign}_per_image']==all_steps['P3'][i][f'selected_{sign}_per_image']
            assert all_steps['P1'][i][f'candidate_{sign}_per_image']==all_steps['P3'][i][f'candidate_{sign}_per_image']
    sources=list((PROJECT/'fecd/training').glob('*.py'))
    sources.extend(PROJECT/'fecd/tools'/name for name in ('train_controlled_baseline.py','check_stage4p.py'))
    sources.extend((PROJECT/'fecd/tests').glob('test_stage4p*.py'))
    issues=[issue for p in sources for issue in analyze_source(p)]
    assert not issues,issues
    old=json.loads(Path('/tmp/stage4p_protected.json').read_text())
    changed=[name for name,h in old.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    output={'status':'PASS','variants':results,'shared_initialization':True,
            'shared_order_augmentation_optimizer_budget':True,'p2_p3_exact_class_budgets':True,
            'training_members':3530,'gt_accesses':0,'new_confirm_accesses':0,
            'validation_performed':False,'protected_files_checked':len(old),'protected_changes':changed,
            'static_files':len(sources),'static_issues':issues,
            'code_sha256':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'interpretation':'training-smoke-only; no validation metrics or performance comparison',
            'access_boundary':'CPython audit hook covers /home/datasets and frozen SAM cache opens; not whole-OS monitor'}
    (ASSETS/'validation.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','protected':len(old),'static_files':len(sources),
                      'variants':{k:{t:v[t] for t in ('loss_first','loss_last','selected_positive_total',
                                        'selected_negative_total','warm_step_median_seconds')}
                                  for k,v in results.items()}},indent=2))


if __name__=='__main__':
    main()
