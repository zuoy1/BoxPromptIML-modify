"""GT-free applicability, matching quality and confounding summaries."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def distribution(values):
    a = np.asarray([v for v in values if v is not None],float)
    if not len(a):
        return {'n':0,'min':None,'q25':None,'median':None,'q75':None,'q95':None,'max':None}
    return dict(zip(('min','q25','median','q75','q95','max'),
                    map(float,np.quantile(a,[0,.25,.5,.75,.95,1]))),n=len(a))


def correlation(x, y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return None
    def rank(values):
        order=np.argsort(values,kind='stable')
        result=np.empty(len(values),float)
        values=np.asarray(values)
        for value in np.unique(values):
            indices=order[values[order]==value]
            result[indices]=np.mean(np.flatnonzero(values[order]==value))
        return result
    return float(np.corrcoef(rank(x),rank(y))[0,1])


def target(candidate):
    return bool(set(candidate['categories']) & {'pos','box_inside','box_edge'})


def collect(rows, size):
    candidates = [c for r in rows for c in r['scales'][size]['candidates']]
    valid = [c for c in candidates if c['statistics']['status']=='applicable']
    matched_images = sum(any(c['statistics']['status']=='applicable'
                             for c in r['scales'][size]['candidates']) for r in rows)
    target_images = sum(any(target(c) and c['statistics']['status']=='applicable'
                           for c in r['scales'][size]['candidates']) for r in rows)
    controls = [c for r in rows for c in r['scales'][size]['controls']
                if c['statistics']['status']=='applicable']
    control_pairs=[]
    for row in rows:
        scale=row['scales'][size]
        cv=[c['statistics']['standardized_quantile_distance'] for c in scale['controls']
            if c['statistics']['status']=='applicable']
        tv=[c['statistics']['standardized_quantile_distance'] for c in scale['candidates']
            if target(c) and c['statistics']['status']=='applicable']
        if cv and tv:
            q95=float(np.quantile(cv,.95))
            control_pairs.append({'sample_key':row['sample_key'],'control_q95':q95,
                                  'target_below_control_q95_fraction':float(np.mean(np.asarray(tv)<=q95)),
                                  'target_n':len(tv),'control_n':len(cv)})
    reasons=Counter()
    for c in candidates:
        if c['statistics']['status']!='applicable':
            counts=c['matching']['counts']
            reason=('reference_pool_lt3' if counts['reference_eligible']<3 else
                    'nonoverlap_lt3' if counts['nonoverlap']<3 else
                    'joint_content_lt3' if counts['joint_content']<3 else
                    'mutually_nonoverlap_selection_lt3')
            reasons[reason]+=1
    quality={}
    for feature in ('brightness','gradient','texture'):
        before=[c['matching']['matching_quality']['before_content_gate'][feature]['mean'] for c in valid]
        after=[c['matching']['matching_quality']['selected'][feature]['mean'] for c in valid]
        stats=[c['statistics']['standardized_quantile_distance'] for c in valid]
        quality[feature]={'before':distribution(before),'after':distribution(after),
                          'statistic_spearman':correlation(stats,after)}
    groups=Counter(r['source_group_id'] for r in rows)
    category={}
    for name in ('pos','neg','box_inside','box_edge'):
        cs=[c for c in candidates if name in c['categories']]
        vs=[c for c in cs if c['statistics']['status']=='applicable']
        category[name]={'candidates':len(cs),'matched':len(vs),
                        'applicable_fraction':len(vs)/len(cs) if cs else None,
                        'distance':distribution([c['statistics']['standardized_quantile_distance'] for c in vs])}
    return {'images':len(rows),'source_groups':len(groups),
            'grid_patches':sum(r['scales'][size]['grid_patch_count'] for r in rows),
            'candidates':len(candidates),'matched_candidates':len(valid),
            'matched_fraction':len(valid)/len(candidates) if candidates else None,
            'matched_images':matched_images,'matched_image_fraction':matched_images/len(rows),
            'target_matched_images':target_images,'target_matched_image_fraction':target_images/len(rows),
            'reference_pool':distribution([r['scales'][size]['reference_pool_count'] for r in rows]),
            'outside_pool':distribution([r['scales'][size]['outside_patch_count'] for r in rows]),
            'reference_counts':dict(Counter(c['matching']['counts']['selected'] for c in candidates)),
            'unmatched_reasons':dict(reasons),'matching_quality':quality,
            'primary_distance':distribution([c['statistics']['standardized_quantile_distance'] for c in valid]),
            'raw_distance':distribution([c['statistics']['raw_quantile_distance'] for c in valid]),
            'mad_log_ratio':distribution([c['statistics']['log_mad_ratio'] for c in valid]),
            'leave_one_out_span':distribution([np.ptp(c['statistics']['leave_one_out_range']) for c in valid]),
            'categories':category,'natural_proxy_controls':{
                'matched_controls':len(controls),
                'distance':distribution([c['statistics']['standardized_quantile_distance'] for c in controls]),
                'paired_images':len(control_pairs), 'per_image':control_pairs,
                'image_macro_target_below_control_q95':float(np.mean([
                    r['target_below_control_q95_fraction'] for r in control_pairs])) if control_pairs else None}}


def summarize(output):
    output=Path(output)
    rows=[json.loads(s) for s in (output/'per_image.jsonl').read_text().splitlines()]
    domains={}
    for domain in ['ALL']+sorted({r['dataset'] for r in rows}):
        subset=rows if domain=='ALL' else [r for r in rows if r['dataset']==domain]
        domains[domain]={size:collect(subset,size) for size in ('64','32')}
    image_stats=[]
    for r in rows:
        cs=[c for c in r['scales']['64']['candidates'] if target(c)
            and c['statistics']['status']=='applicable']
        image_stats.append({'sample_key':r['sample_key'],'dataset':r['dataset'],
            'format':r['input_metadata']['format'],'native_longest':max(r['input_metadata']['native_hw']),
            'work_hw':r['scales']['64']['work_hw'],'num_boxes':r['num_boxes'],
            'mean_candidate_box_fraction':float(np.mean([c['patch']['box_fraction']
                for c in r['scales']['64']['candidates']])) if r['scales']['64']['candidates'] else None,
            'median_distance':float(np.median([c['statistics']['standardized_quantile_distance'] for c in cs])) if cs else None,
            'target_match_count':len(cs),'high_texture':float(np.mean([c['patch']['texture']
                for c in r['scales']['64']['candidates']])) if r['scales']['64']['candidates'] else None})
    valid=[r for r in image_stats if r['median_distance'] is not None]
    by_format={fmt:distribution([r['median_distance'] for r in valid if r['format']==fmt])
               for fmt in sorted({r['format'] for r in image_stats})}
    correlations={feature:correlation([r[feature] for r in valid],[r['median_distance'] for r in valid])
                  for feature in ('native_longest','high_texture','mean_candidate_box_fraction','num_boxes')}
    all64=domains['ALL']['64']
    reasons=[]
    if all64['matched_image_fraction']<.5:
        reasons.append('more_than_half_images_have_no_valid_reference')
    if all64['target_matched_image_fraction']<.5:
        reasons.append('more_than_half_images_have_no_matched_positive_or_box_candidate')
    proxy=all64['natural_proxy_controls']
    if proxy['paired_images']>=10 and proxy['image_macro_target_below_control_q95']>=.5:
        reasons.append('candidate_difference_not_separated_from_ordinary_reference_proxy_variation')
    for feature in ('gradient','texture'):
        corr=all64['matching_quality'][feature]['statistic_spearman']
        if all64['matched_candidates']>=30 and corr is not None and abs(corr)>=.7:
            reasons.append(feature+'_difference_dominates_statistic')
    formats=[d['median'] for d in by_format.values() if d['n']>=10]
    if len(formats)>=2 and min(formats)>0 and max(formats)/min(formats)>=4:
        reasons.append('large_format_association_requires_resolution')
    result={'version':'fecd-reference-feasibility-summary-v1','gt_files_opened':0,
            'decision':'STOP' if reasons else 'READY FOR B-R DEVELOPMENT DIAGNOSTIC',
            'decision_reasons':reasons,'domains':domains,'format_distributions':by_format,
            'image_correlations':correlations,'image_metadata':image_stats,
            'limitations':['reference controls are not GT-confirmed natural',
                           'patches overlap across candidate anchors; no independent-pixel inference',
                           'format/dataset/resolution confounded; correlations not causal']}
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
