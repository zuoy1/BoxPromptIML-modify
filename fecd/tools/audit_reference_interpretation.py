"""Read existing B0 JSON only; no image/cache/GT or feature imports."""
from collections import Counter, defaultdict
from itertools import combinations, product
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'fecd/assets/stage3rb0'
OUTPUT = ROOT/'fecd/assets/stage3rb0_interpretation_audit.json'
INPUTS = [BASE/name for name in (
    'per_image.jsonl', 'summary.json', 'protocol.json', 'geometry_audit.json')]
INPUTS += [ROOT/'fecd/forensic_reference'/name for name in
           ('patches.py','matching.py','statistics.py','summary.py')]
INPUTS += [ROOT/'fecd/tools/audit_reference_feasibility.py']


def average(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def describe(values):
    values = [v for v in values if v is not None]
    return {'n':len(values), 'min':min(values) if values else None,
            'median':float(np.median(values)) if values else None,
            'mean':average(values), 'max':max(values) if values else None}


def bounds(identifier):
    size,y,x=map(int,identifier.split(':'))
    return (x,y,x+size,y+size)


def overlap(first, second):
    a,b=bounds(first),bounds(second)
    return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])


def valid(records):
    return [r for r in records if r['statistics']['status']=='applicable']


def primary(record):
    return record['statistics']['standardized_quantile_distance']


def target(record):
    return bool(set(record['categories']) & {'pos','box_inside','box_edge'})


def reason(record):
    c=record['matching']['counts']
    return ('reference_pool_lt3' if c['reference_eligible']<3 else
            'nonoverlap_lt3' if c['nonoverlap']<3 else
            'joint_content_lt3' if c['joint_content']<3 else
            'mutually_nonoverlap_selection_lt3')


def anchor_pairs(records):
    ids=[r['patch']['id'] for r in records]
    pairs=list(combinations(ids,2))
    return {'anchors':len(ids),'pairs':len(pairs),
            'spatial_overlap_pairs':sum(overlap(a,b) for a,b in pairs)}


def category_summary(records, q95):
    vs=valid(records)
    return {'audited':len(records),'matched':len(vs),
            'unmatched':dict(Counter(reason(r) for r in records
                                    if r['statistics']['status']!='applicable')),
            'above_q95':sum(primary(r)>q95 for r in vs) if q95 is not None else None,
            'exceed_fraction':average([primary(r)>q95 for r in vs]) if q95 is not None else None}


def image_audit(row,size,area):
    scale=row['scales'][size]
    controls=scale['controls']
    cv=valid(controls)
    cs=scale['candidates']
    ts=[r for r in cs if target(r)]
    tv=valid(ts)
    values=[primary(c) for c in cv]
    q95=float(np.quantile(values,.95,method='linear')) if values else None
    out={k:row[k] for k in ('sample_key','sample_id','dataset','source_group_id')}
    out.update(size=int(size),box_area=area,
               controls_audited=len(controls),controls_valid=len(cv),targets_valid=len(tv),
               control_q95=q95,control_unique_values=len(set(values)),
               controls_equal_q95=sum(v==q95 for v in values) if values else None,
               controls_exceed=average([v>q95 for v in values]) if values else None,
               targets_exceed=average([primary(t)>q95 for t in tv]) if values and tv else None)
    out['paired_difference']=(out['targets_exceed']-out['controls_exceed']
                              if out['targets_exceed'] is not None else None)
    out['categories']={name:category_summary([r for r in cs if name in r['categories']],q95)
                       for name in ('pos','neg','box_inside','box_edge')}
    out['categories']['stable_negative_controls']=category_summary(controls,q95)
    out['target_overlap_all']=anchor_pairs(ts)
    out['target_overlap_valid']=anchor_pairs(tv)
    out['control_overlap_all']=anchor_pairs(controls)
    out['control_overlap_valid']=anchor_pairs(cv)
    self_refs=[r['patch']['id'] for r in cs+controls
               if r['patch']['id'] in r['matching']['reference_ids']]
    assert not self_refs
    out['self_reference_count']=len(self_refs)
    pairs=list(product(tv,cv))
    shares=[len(set(a['matching']['reference_ids']) & set(b['matching']['reference_ids']))
            for a,b in pairs]
    out['target_control_reference_reuse']={
        'anchor_pairs':len(pairs),'pairs_sharing_exact_reference':sum(n>0 for n in shares),
        'pairs_sharing_or_overlapping_reference':sum(
            any(overlap(x,y) for x in a['matching']['reference_ids']
                for y in b['matching']['reference_ids']) for a,b in pairs),
        'shared_reference_instances':sum(shares)}
    out['cross_control_anchor_as_reference_instances']=sum(
        ref in {r['patch']['id'] for r in cv}
        for c in cv for ref in c['matching']['reference_ids'])
    for r in cs+controls:
        refs=r['matching']['reference_ids']
        assert not any(overlap(r['patch']['id'],ref) for ref in refs)
        assert not any(overlap(a,b) for a,b in combinations(refs,2))
        assert (r['statistics']['status']=='applicable')==(len(refs)>=3)
        assert r['statistics']['n_references']==len(refs)
    return out


def aggregate(rows):
    paired=[r for r in rows if r['paired_difference'] is not None]
    output={'images':len(rows),'groups':len({r['source_group_id'] for r in rows}),
            'paired_images':len(paired),'paired_groups':len({r['source_group_id'] for r in paired}),
            'controls_valid_distribution':describe([r['controls_valid'] for r in rows]),
            'controls_exceed_macro':average([r['controls_exceed'] for r in paired]),
            'targets_exceed_macro':average([r['targets_exceed'] for r in paired]),
            'difference_macro':average([r['paired_difference'] for r in paired]),
            'difference_distribution':describe([r['paired_difference'] for r in paired]),
            'categories':{}}
    for name in ('pos','neg','box_inside','box_edge','stable_negative_controls'):
        records=[r['categories'][name] for r in rows]
        relevant=[r for r in rows if r['categories'][name]['audited']]
        matched=[r for r in relevant if r['categories'][name]['matched']]
        comparable=[r for r in matched if r['categories'][name]['exceed_fraction'] is not None]
        fail=Counter()
        for r in records:fail.update(r['unmatched'])
        output['categories'][name]={
            'images_with_anchors':len(relevant),'groups_with_anchors':len({r['source_group_id'] for r in relevant}),
            'images_matched':len(matched),'groups_matched':len({r['source_group_id'] for r in matched}),
            'comparable_images':len(comparable),'comparable_groups':len({r['source_group_id'] for r in comparable}),
            'audited':sum(r['audited'] for r in records),'matched':sum(r['matched'] for r in records),
            'unmatched':dict(fail),'exceed_macro':average([r['categories'][name]['exceed_fraction'] for r in comparable]),
            'exceed_pooled_descriptive':(sum(r['categories'][name]['above_q95'] for r in comparable)/
                sum(r['categories'][name]['matched'] for r in comparable)) if comparable else None}
    for name in ('target_overlap_all','target_overlap_valid','control_overlap_all','control_overlap_valid',
                 'target_control_reference_reuse'):
        output[name]={k:sum(r[name][k] for r in rows) for k in rows[0][name]}
    output['self_reference_count']=sum(r['self_reference_count'] for r in rows)
    output['cross_control_anchor_as_reference_instances']=sum(
        r['cross_control_anchor_as_reference_instances'] for r in rows)
    return output


def main():
    if OUTPUT.exists():
        raise FileExistsError('Refusing to overwrite interpretation audit')
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}
    source=[json.loads(s) for s in INPUTS[0].read_text().splitlines()]
    original=json.loads((BASE/'summary.json').read_text())
    areas={r['sample_key']:r['cached_box_union_area_224']
           for r in json.loads((BASE/'geometry_audit.json').read_text())['per_image']}
    output={'type':'posthoc_readonly_interpretation_audit','action':'STOP',
            'new_gt_accesses':0,'feature_reextraction':False,'original_inputs_sha256':before,
            'quantile_method':'NumPy linear, index=(n-1)*p; strict >; ties equal go to <=',
            'scales':{}}
    for size in ('64','32'):
        rows=[image_audit(r,size,areas[r['sample_key']]) for r in source]
        domains={d:aggregate(rows if d=='ALL' else [r for r in rows if r['dataset']==d])
                 for d in ['ALL']+sorted({r['dataset'] for r in rows})}
        groups=defaultdict(list)
        for r in rows:groups[(r['dataset'],r['source_group_id'])].append(r)
        group_records=[{'dataset':d,'source_group_id':g,**aggregate(rs)}
                       for (d,g),rs in sorted(groups.items())]
        paired_groups=[r for r in group_records if r['paired_images']]
        assert abs(1-domains['ALL']['targets_exceed_macro']-original[
            'domains']['ALL'][size]['natural_proxy_controls']['image_macro_target_below_control_q95'])<1e-12
        box_strata={}
        for label,subset in [('small_lt1pct',[r for r in rows if r['box_area']<.01]),
                             ('large_gt50pct',[r for r in rows if r['box_area']>.5]),
                             ('middle',[r for r in rows if .01<=r['box_area']<=.5])]:
            if subset:box_strata[label]=aggregate(subset)
        output['scales'][size]={'per_image':rows,'domains':domains,'groups':group_records,
            'group_equal_paired':{k:average([r[k] for r in paired_groups])
                for k in ('controls_exceed_macro','targets_exceed_macro','difference_macro')},
            'box_strata':box_strata}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before.items())
    # Runnable mathematical and overlap checks, with no image IO.
    assert np.quantile([0,1],.95,method='linear')==.95
    assert sum(x>np.quantile([1,1,1],.95) for x in [1,1,1])==0
    assert overlap('64:0:0','64:0:32') and not overlap('64:0:0','64:0:64')
    OUTPUT.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'action':'STOP','images':len(source),'output':str(OUTPUT),
                      'primary':output['scales']['64']['domains']['ALL'],
                      'group_equal':output['scales']['64']['group_equal_paired']},indent=2))


if __name__=='__main__':
    main()
