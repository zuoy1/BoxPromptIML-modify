"""在原始网格按区域匹配预算；困难选择只使用已脱离梯度的学生概率。"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _priority(length, seed, sample_key, epoch, region):
    digest = hashlib.sha256(
        f'stage5c0:{seed}:{epoch}:{sample_key}:{region}'.encode('utf-8')).digest()
    permutation = np.random.default_rng(int.from_bytes(digest[:8], 'big')).permutation(length)
    priority = np.empty(length, dtype=np.int64)
    priority[permutation] = np.arange(length)
    return permutation, priority


def select_region_pixels(target, box_union, variant, seed, sample_key, epoch,
                         probabilities=None):
    """返回同形状的numpy布尔mask及每类候选/已选数量。

    target与box_union必须对应尚未增强的同一原始网格，形状为[H,W]或
    [1,H,W]。R1的probabilities须是当前forward概率逆几何变换后的同网格
    数组；torch输入会先detach。输出只表示选择，不覆盖教师标签。

    正类使用epoch=0的独立随机顺序；负类按epoch和区域分别生成顺序。
    R1按概率降序，同概率时复用R0的随机顺序，禁止坐标优先。
    """
    labels, inside = _array(target), _array(box_union)
    shape = labels.shape
    if (labels.ndim not in (2, 3) or labels.size == 0
            or (labels.ndim == 3 and labels.shape[0] != 1)
            or labels.shape != inside.shape):
        raise ValueError('标签和Box并集必须是同形状的非空二维或单通道网格')
    if (not np.issubdtype(labels.dtype, np.number) and labels.dtype != np.bool_
            or not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all()):
        raise ValueError('教师hard标签必须是有限的二值数组')
    if inside.dtype != np.bool_:
        raise ValueError('Box并集必须是布尔数组')
    if variant not in ('R0', 'R1'):
        raise ValueError('仅支持独立编号R0和R1')
    if (type(seed) is not int or type(epoch) is not int or epoch < 0
            or not isinstance(sample_key, str) or not sample_key.strip()):
        raise ValueError('选择需要整数seed、非负整数epoch和非空样本身份')
    scores = None
    if probabilities is not None:
        scores = _array(probabilities)
        if (scores.shape != labels.shape or not np.issubdtype(scores.dtype, np.number)
                or not np.isrealobj(scores) or not np.isfinite(scores).all()
                or (scores < 0).any() or (scores > 1).any()):
            raise ValueError('学生概率必须是同网格、有限且处于[0,1]')
        scores = scores.astype(np.float64, copy=False).reshape(-1)
    if variant == 'R1' and scores is None:
        raise ValueError('R1必须使用当前学生预测概率')

    labels, inside = labels.reshape(-1), inside.reshape(-1)
    negative = labels == 0
    candidates = (('pos', labels == 1),
                  ('inside_neg', negative & inside),
                  ('outside_neg', negative & ~inside))
    selected = np.zeros(labels.size, dtype=bool)
    counts = {}
    for region, candidate in candidates:
        indices = np.flatnonzero(candidate)
        count = len(indices)
        budget = count // 2
        selection_epoch = 0 if region == 'pos' else epoch
        permutation, priority = _priority(count, seed, sample_key, selection_epoch, region)
        if variant == 'R1' and region != 'pos':
            order = np.lexsort((priority, -scores[indices]))
        else:
            order = permutation
        selected[indices[order[:budget]]] = True
        counts[f'candidate_{region}'] = count
        counts[f'selected_{region}'] = budget
    counts['candidate_neg'] = counts['candidate_inside_neg'] + counts['candidate_outside_neg']
    counts['selected_neg'] = counts['selected_inside_neg'] + counts['selected_outside_neg']
    counts['candidate_total'] = int(labels.size)
    counts['selected_total'] = int(selected.sum())
    return selected.reshape(shape), counts
