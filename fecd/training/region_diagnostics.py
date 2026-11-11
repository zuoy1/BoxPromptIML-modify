"""不参与选择决策的困难负类分布与反事实Epoch选择统计。"""
import hashlib

import numpy as np


def digest(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {'n': 0, 'mean': None, 'quantiles': None}
    return {'n': int(values.size), 'mean': float(values.mean()),
            'quantiles': np.quantile(values, [0, .1, .5, .9, 1]).tolist()}


def describe(logits, target, boxes, mask, reference, next_mask, metadata, mu, var, confidence,
             student_probability):
    logits, target, boxes, mask = (np.asarray(x).reshape(224, 224) if np.asarray(x).size == 50176
                                   else np.asarray(x).squeeze() for x in (logits, target, boxes, mask))
    target, boxes = target.astype(bool), boxes.astype(bool)
    probability = np.asarray(student_probability).squeeze()
    bce = np.logaddexp(0, logits.astype(np.float64))
    row = {**metadata, 'selected_positive': int((mask & target).sum()),
           'candidate_positive': int(target.sum()), 'box_area': float(boxes.mean()),
           'selection_hash': digest(mask), 'positive_hash': digest(mask & target),
           'prediction_area': float((logits >= 0).mean()),
           'probability_mean': float(probability.mean()), 'logit_distribution': distribution(logits),
           'all_zero': not bool((logits >= 0).any()), 'near_full': float((logits >= 0).mean()) >= .99,
           'counterfactual_next_epoch_hash': digest(next_mask),
           'counterfactual_changed': int((mask != next_mask).sum()), 'regions': {}}
    for side, region in (('in', boxes), ('out', ~boxes)):
        candidate = ~target & region
        selected, ignored = candidate & mask, candidate & ~mask
        count, total = int(selected.sum()), int(candidate.sum())
        if count != total//2:
            raise AssertionError('区域floor预算不一致')
        row['selected_negative_'+side] = count
        row['candidate_negative_'+side] = total
        row['regions'][side] = {
            'selected': {name: distribution(np.asarray(array).squeeze()[selected])
                         for name, array in [('mu', mu), ('var', var), ('c_sam', confidence),
                                             ('student_probability', probability), ('bce', bce)]},
            'unselected': {name: distribution(np.asarray(array).squeeze()[ignored])
                           for name, array in [('mu', mu), ('var', var), ('c_sam', confidence),
                                               ('student_probability', probability), ('bce', bce)]},
            'random_reference_probability_mean': float(probability[candidate & reference].mean())
            if count else None,
            'candidate_hash': digest(candidate), 'selected_hash': digest(selected),
            'next_epoch_jaccard': float((selected & next_mask).sum()/(selected | (candidate & next_mask)).sum())
            if count else None,
            'selected_mu_ge_04': int((selected & (np.asarray(mu).squeeze() >= .4)).sum()),
            'selected_var_ge_01': int((selected & (np.asarray(var).squeeze() >= .1)).sum()),
        }
    return row
