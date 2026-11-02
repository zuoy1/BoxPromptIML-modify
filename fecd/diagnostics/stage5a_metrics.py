"""开发错误的图像级分解；像素只作计数，不作独立统计样本。"""
from collections import defaultdict

import cv2
import numpy as np


def measures(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.ndim != 2 or pred.shape != gt.shape or not pred.size:
        raise ValueError('预测和GT必须共享非空二维网格')
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return {'tp': tp, 'fp': fp, 'fn': fn, 'pixels': pred.size,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'prediction_area': float(pred.mean()), 'gt_area': float(gt.mean()),
            'area_bias': float(pred.mean() - gt.mean())}


def _band(mask, radius):
    kernel = np.ones((3, 3), np.uint8)
    value = np.asarray(mask, np.uint8)
    expanded = cv2.dilate(value, kernel, iterations=radius, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    eroded = cv2.erode(value, kernel, iterations=radius, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return expanded.astype(bool) ^ eroded.astype(bool)


def spatial_errors(pred, gt, boxes, radius=3):
    pred, gt, boxes = (np.asarray(x, bool) for x in (pred, gt, boxes))
    if pred.ndim != 2 or not pred.size or pred.shape != gt.shape or pred.shape != boxes.shape:
        raise ValueError('区域诊断输入必须共享非空二维网格')
    if type(radius) is not int or radius < 1:
        raise ValueError('边界半径必须为正整数')
    fp, fn = pred & ~gt, ~pred & gt
    gb, pb = _band(gt, radius), _band(pred, radius)
    count, _, stats, _ = cv2.connectedComponentsWithStats(fp.astype(np.uint8), connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    result = {'fp_gt_boundary': int((fp & gb).sum()), 'fp_gt_far': int((fp & ~gb).sum()),
              'fn_gt_boundary': int((fn & gb).sum()), 'fn_gt_interior': int((fn & ~gb).sum()),
              'fp_pred_boundary': int((fp & pb).sum()), 'fp_pred_interior': int((fp & ~pb).sum()),
              'fn_pred_boundary': int((fn & pb).sum()), 'fn_pred_far': int((fn & ~pb).sum()),
              'fp_outside_weak_boxes': int((fp & ~boxes).sum()),
              'fp_components': count - 1,
              'fp_largest_component_fraction': largest / int(fp.sum()) if fp.any() else 0.}
    for name in list(result):
        if name.startswith(('fp_', 'fn_')) and name not in ('fp_components', 'fp_largest_component_fraction'):
            denominator = int(fp.sum()) if name.startswith('fp_') else int(fn.sum())
            result[name + '_fraction'] = result[name] / denominator if denominator else None
    return result


def overlap_errors(student, teacher, gt):
    student, teacher, gt = (np.asarray(x, bool) for x in (student, teacher, gt))
    if student.ndim != 2 or not student.size or student.shape != teacher.shape or student.shape != gt.shape:
        raise ValueError('教师学生错误对照必须共享非空二维网格')
    sw, tw = student != gt, teacher != gt
    return {'both_wrong': int((sw & tw).sum()), 'both_correct': int((~sw & ~tw).sum()),
            'teacher_wrong_student_correct': int((tw & ~sw).sum()),
            'teacher_correct_student_wrong': int((~tw & sw).sum()),
            'student_fp_shared': int((student & teacher & ~gt).sum()),
            'student_fp_extra': int((student & ~teacher & ~gt).sum()),
            'student_fn_shared': int((~student & ~teacher & gt).sum()),
            'student_fn_extra': int((~student & teacher & gt).sum()),
            'student_error_overlap_fraction': int((sw & tw).sum()) / int(sw.sum()) if sw.any() else None,
            'teacher_error_corrected_fraction': int((tw & ~sw).sum()) / int(tw.sum()) if tw.any() else None,
            'student_fp_extra_fraction': int((student & ~teacher & ~gt).sum()) / int((student & ~gt).sum()) if (student & ~gt).any() else None}


def summarize(rows):
    if not rows:
        raise ValueError('不能汇总空集合')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['source_group_id'])].append(row)
    result = {'images': len(rows), 'source_groups': len(groups)}
    fields = ['f1', 'iou', 'precision', 'recall', 'prediction_area', 'gt_area']
    fields.extend(k for k in rows[0] if k.endswith('_fraction') or k == 'area_bias')
    for field in fields:
        values = [r[field] for r in rows if r.get(field) is not None]
        per_group = [np.mean([r[field] for r in group if r.get(field) is not None])
                     for group in groups.values() if any(r.get(field) is not None for r in group)]
        result['image_' + field] = float(np.mean(values)) if values else None
        result['group_' + field] = float(np.mean(per_group)) if per_group else None
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
    result.update(tp=tp, fp=fp, fn=fn,
                  pooled_precision=tp / (tp + fp) if tp + fp else 0.,
                  pooled_recall=tp / (tp + fn) if tp + fn else 0.)
    for field in rows[0]:
        if field.startswith(('fp_', 'fn_', 'student_', 'teacher_', 'both_')):
            if all(type(r.get(field)) is int for r in rows):
                result['count_' + field] = sum(r[field] for r in rows)
    return result
