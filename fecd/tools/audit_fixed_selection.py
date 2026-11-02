"""Read-only K8 budget audit: canonical masks, never RGB, GT or student training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fecd.confidence.teacher_scores import sam_confidence
from fecd.diagnostics.stage3ra_split import read_split
from fecd.pseudo.cache_schema import canonical_hash, load_cache
from fecd.training.data import CONFIGS, K8_ROOT, _build_samples
from fecd.training.f0_selection import expected_unique_coverage, select_for_variant

OUTPUT=Path(__file__).resolve().parents[1]/'assets/stage4f0/selection_audit.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError('Refusing to replace a 4F0 selection audit')
    started=time.monotonic()
    split=read_split()
    records=sorted((r for r in split['records'] if r['subset']=='student_train'),
                   key=lambda r:r['sample_key'])
    if len(records)!=3530:
        raise ValueError('Unexpected student_train membership')
    index=json.loads((K8_ROOT/'dataset_index.json').read_text())
    samples=_build_samples(records,index,K8_ROOT)
    results=[]
    dynamic=[]
    dynamic_counts=Counter()
    for row,source in zip(records,samples):
        record=load_cache(source['cache_path'],CONFIGS[8],expected={
            **row,'split':'train','k':8,'image_sha1':source['image_sha1'],
            'box_hash':source['box_hash']})
        target=(record['mu']>=.5).astype(np.float32)
        confidence=sam_confidence(record['mu'],record['var'],record['agreement'])
        item={'sample_key':row['sample_key'],'dataset':row['dataset']}
        reference_counts=None
        for variant in ('P2F','P3F'):
            baseline,counts=select_for_variant(target,confidence,variant,20261029,
                                               row['sample_key'],0)
            for epoch in (1,19):
                later,next_counts=select_for_variant(target,confidence,variant,20261029,
                                                     row['sample_key'],epoch)
                if not np.array_equal(baseline,later) or counts!=next_counts:
                    raise AssertionError('Fixed canonical supervision changed across epochs')
            if reference_counts is not None and counts!=reference_counts:
                raise AssertionError('P2F/P3F budgets differ')
            reference_counts=counts
            item[f'{variant}_canonical_hash']=hashlib.sha256(baseline.tobytes()).hexdigest()
        item.update(reference_counts)
        item['P2E_expected_unique20']={sign:expected_unique_coverage(
            item[f'candidate_{sign}'],'P2E',20) for sign in ('pos','neg')}
        results.append(item)
        if dynamic_counts[row['dataset']]<8:
            union=np.zeros_like(target,dtype=bool)
            masks=[]
            for epoch in range(20):
                mask,_=select_for_variant(target,confidence,'P2E',20261029,
                                         row['sample_key'],epoch)
                union|=mask
                masks.append(hashlib.sha256(mask.tobytes()).hexdigest())
            union_counts={sign:int((union&(target==label)).sum())
                          for sign,label in (('pos',1),('neg',0))}
            dynamic.append({'sample_key':row['sample_key'],'dataset':row['dataset'],
                            'distinct_epoch_mask_hashes':len(set(masks)),
                            'unique_counts':union_counts,
                            'unique_fraction':{sign:union_counts[sign]/item[f'candidate_{sign}']
                                if item[f'candidate_{sign}'] else None for sign in ('pos','neg')}})
            dynamic_counts[row['dataset']]+=1
        if len(results)%500==0:
            print(json.dumps({'audited':len(results),'seconds':time.monotonic()-started}),flush=True)
    total={field:sum(r[field] for r in results) for field in
           ('candidate_pos','candidate_neg','selected_pos','selected_neg')}
    weighted={sign:sum(r[f'candidate_{sign}']*(r['P2E_expected_unique20'][sign] or 0)
                       for r in results)/total[f'candidate_{sign}'] for sign in ('pos','neg')}
    result={'status':'PASS','protocol':'stage4f0-fixed-selection-v1','seed':20261029,
            'images':len(records),'datasets':dict(Counter(r['dataset'] for r in records)),
            'membership_hash':canonical_hash([r['sample_key'] for r in records]),
            'split_hash':split['manifest_sha256'],'k8_config_hash':CONFIGS[8],
            'checked_epochs':[0,1,19],'fixed_all20':'epoch-independent by construction; no training',
            'selection':'per class floor(n/2); priority hash seed:0:sample_key:class',
            'tie_rule':'fixed random priority, same priority as P2F',
            'P0_P1_unique_fraction':1.,'P2F_P3F_totals':total,
            'P2F_P3F_unique20_fraction':{s:total[f'selected_{s}']/total[f'candidate_{s}']
                                       for s in ('pos','neg')},
            'P2E_theoretical_unique20_fraction':weighted,
            'theory_assumption':'independent uniform epoch priorities; actual SHA256 deterministic draws audited only in subset',
            'empty_class':'N/A; singleton floor(1/2)=0 forever',
            'dynamic_subset_rule':'sample_key sorted, first up to 8 per dataset',
            'dynamic_subset_counts':dict(dynamic_counts),'dynamic_subset_actual20':dynamic,
            'per_image':results,'gt_reads':0,'rgb_reads':0,'sam_runs':0,'training_updates':0,
            'elapsed_seconds':time.monotonic()-started}
    result['result_hash']=canonical_hash(result)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','images','elapsed_seconds','result_hash')}),flush=True)


if __name__=='__main__':
    main()
