"""Raw-logit segmentation metrics under the frozen development protocol."""
import numpy as np

from fecd.training.f0_protocol import EVAL_PROTOCOL


def segmentation_metrics(pred, gt):
    pred=np.asarray(pred,dtype=bool);gt=np.asarray(gt,dtype=bool)
    if pred.shape!=gt.shape or pred.ndim!=2:
        raise ValueError('Metric grids must be equal two-dimensional arrays')
    tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'pixels':pred.size,
            'gt_positive_fraction':float(gt.mean()),
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.}


def output_diagnostics(logits):
    x=np.asarray(logits,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError('Expected finite two-dimensional raw logits')
    probability=np.exp(-np.logaddexp(0.,-x));pred=probability>=EVAL_PROTOCOL['threshold']
    area=float(pred.mean());mean=float(probability.mean())
    quantiles=[0.,.01,.05,.5,.95,.99,1.]
    return {'prediction_positive_fraction':area,'probability_mean':mean,
            'logit_quantiles':np.quantile(x,quantiles).tolist(),
            'probability_quantiles':np.quantile(probability,quantiles).tolist(),
            'quantile_levels':quantiles,'all_zero_prediction':bool(not pred.any()),
            'all_full_prediction':bool(pred.all()),
            'near_full_prediction':area>=EVAL_PROTOCOL['near_full_area'],
            'near_zero_prediction':area<=EVAL_PROTOCOL['near_zero_area'],
            'abnormal_high_area':area>=EVAL_PROTOCOL['abnormal_high_area'],
            'abnormal_low_area':area<=EVAL_PROTOCOL['abnormal_low_area'],
            'saturated_probability_mean':mean<=EVAL_PROTOCOL['probability_saturation_low'] or mean>=EVAL_PROTOCOL['probability_saturation_high'],
            'positive_mean_probability':float(probability[pred].mean()) if pred.any() else None,
            'negative_mean_confidence':float((1-probability[~pred]).mean()) if (~pred).any() else None}


def _aggregate(rows):
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    result={'images':len(rows),'pixels':sum(r['pixels'] for r in rows),'tp':tp,'fp':fp,'fn':fn,
            'source_groups':len({r['source_group_id'] for r in rows if 'source_group_id' in r}),
            'image_macro_f1':float(np.mean([r['f1'] for r in rows])),
            'image_macro_iou':float(np.mean([r['iou'] for r in rows])),
            'pooled_micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'pooled_micro_iou':tp/(tp+fp+fn) if tp+fp+fn else 1.}
    for field in ('all_zero_prediction','all_full_prediction','near_full_prediction','near_zero_prediction',
                  'abnormal_high_area','abnormal_low_area','saturated_probability_mean'):
        result[field+'_images']=sum(bool(r.get(field,False)) for r in rows)
    for field in ('prediction_positive_fraction','probability_mean','gt_positive_fraction'):
        available=[r[field] for r in rows if field in r]
        result['mean_'+field]=float(np.mean(available)) if available else None
    return result


def summarize_records(rows):
    if not rows:
        raise ValueError('Cannot summarize zero evaluation images')
    domains={d:_aggregate([r for r in rows if r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})}
    equal={'f1':float(np.mean([r['image_macro_f1'] for r in domains.values()])),
           'iou':float(np.mean([r['image_macro_iou'] for r in domains.values()]))}
    domains['ALL']=_aggregate(rows)
    return {'domains':domains,'dataset_equal':equal}
