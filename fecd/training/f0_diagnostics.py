"""Output diagnostics against pseudo-labels only; no ground-truth interface."""
import numpy as np
import torch

LEVELS=[0.,.01,.05,.5,.95,.99,1.]


def describe_output(logits,target,selection):
    if logits.shape!=target.shape or target.shape!=selection.shape:
        raise ValueError('Output/teacher/selection grid mismatch')
    rows=[]
    for logit,teacher,mask in zip(logits.detach(),target,selection):
        p=logit.sigmoid()
        pred=p>=.5
        t=teacher>=.5
        area=float(pred.float().mean().item())
        selected=mask.bool()
        lq=torch.quantile(logit.float(),torch.tensor(LEVELS,device=logit.device)).cpu().tolist()
        pq=torch.quantile(p.float(),torch.tensor(LEVELS,device=p.device)).cpu().tolist()
        meanp=float(p.mean().item())
        rows.append({
            'logit_quantiles':lq,'probability_quantiles':pq,'probability_mean':meanp,
            'student_positive_fraction':area,'teacher_hard_positive_fraction':float(t.float().mean().item()),
            'selected_positive_full_fraction':float((selected&t).float().mean().item()),
            'selected_negative_full_fraction':float((selected&~t).float().mean().item()),
            'candidate_positive_count':int(t.sum().item()),
            'candidate_negative_count':int((~t).sum().item()),
            'selected_positive_count':int((selected&t).sum().item()),
            'selected_negative_count':int((selected&~t).sum().item()),
            'student_all_zero':area==0.,'student_all_foreground':area==1.,
            'student_near_full':area>=.99,'student_abnormal_area':area<=.001 or area>=.95,
            'teacher_all_zero':not bool(t.any()),'teacher_near_full':float(t.float().mean().item())>=.99,
            'predicted_positive_confidence':float(p[pred].mean().item()) if pred.any() else None,
            'predicted_negative_confidence':float((1-p[~pred]).mean().item()) if (~pred).any() else None,
            'probability_saturated':meanp<=.01 or meanp>=.99,
        })
    return rows


def summarize_outputs(rows):
    domains={}
    for domain in ['ALL']+sorted({r['dataset'] for r in rows}):
        records=rows if domain=='ALL' else [r for r in rows if r['dataset']==domain]
        summary={'images':len(records)}
        for key in ('probability_mean','student_positive_fraction','teacher_hard_positive_fraction',
                    'selected_positive_full_fraction','selected_negative_full_fraction',
                    'predicted_positive_confidence','predicted_negative_confidence'):
            values=[r[key] for r in records if r[key] is not None]
            summary[key]={'mean':float(np.mean(values)) if values else None,'valid':len(values)}
        for key in ('student_all_zero','student_all_foreground','student_near_full',
                    'student_abnormal_area','teacher_all_zero','teacher_near_full','probability_saturated'):
            summary[key]=sum(r[key] for r in records)
        for key in ('logit_quantiles','probability_quantiles'):
            summary['image_mean_'+key]=np.mean([r[key] for r in records],axis=0).tolist()
        domains[domain]=summary
    return {'quantile_levels':LEVELS,'domains':domains,
            'interpretation':'pre-update training outputs from changing parameters, not held-out evaluation'}
