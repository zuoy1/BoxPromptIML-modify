"""从实际canonical选择mask统计相邻Epoch重叠及负类累计唯一覆盖。"""
from __future__ import annotations

import numpy as np

_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def _counts(packed):
    return _POPCOUNT[packed].sum(axis=1, dtype=np.int64)


def _validate(value, shape, name):
    if (not isinstance(value, np.ndarray) or value.dtype != np.uint8
            or value.ndim != 2 or not value.size
            or shape is not None and value.shape != shape):
        raise ValueError(f'{name}必须是同形状非空uint8二维packed网格')


def summarize_coverage(current_packed, previous_packed, cumulative_packed,
                       candidate_inside_packed, candidate_outside_packed):
    """返回按输入行顺序的每图统计与更新后的负类累计packed并集。

    所有输入为[N,bytes]的np.packbits输出，位序须一致。current和previous
    保存完整选择（包括固定正类），cumulative只保存此前已选负类并集。
    首Epoch previous/cumulative均为None，其余Epoch必须同时提供。
    inside/outside须为互斥的hard负类候选，主runner固定成员行序并持久化。
    空候选的coverage和空并集的Jaccard为None；不将0/0记成完美重叠。
    """
    _validate(current_packed, None, '当前选择')
    shape = current_packed.shape
    inside, outside = candidate_inside_packed, candidate_outside_packed
    _validate(inside, shape, '框内候选')
    _validate(outside, shape, '框外候选')
    if np.any(inside & outside):
        raise ValueError('框内与框外负类候选必须互斥')
    if (previous_packed is None) != (cumulative_packed is None):
        raise ValueError('上一轮选择与累计并集必须同时提供或同时为None')
    candidates = inside | outside
    current_negative = current_packed & candidates
    if previous_packed is None:
        updated = current_negative.copy()
    else:
        _validate(previous_packed, shape, '上一轮选择')
        _validate(cumulative_packed, shape, '累计选择')
        if np.any((current_packed ^ previous_packed) & ~candidates):
            raise ValueError('候选负类之外的固定正类选择发生变化')
        if np.any(cumulative_packed & ~candidates):
            raise ValueError('累计并集包含非候选负类像素')
        if np.any((previous_packed & candidates) & ~cumulative_packed):
            raise ValueError('累计并集未包含上一轮已选负类')
        updated = cumulative_packed | current_negative
    rows = [{'inside': {}, 'outside': {}, 'negative': {}} for _ in range(shape[0])]
    for name, candidate in (('inside', inside), ('outside', outside), ('negative', candidates)):
        candidate_counts = _counts(candidate)
        selected_counts = _counts(current_packed & candidate)
        unique_counts = _counts(updated & candidate)
        if name != 'negative' and np.any(selected_counts != candidate_counts//2):
            raise ValueError('当前每图每区域选择违反floor半预算')
        intersection = union = None
        if previous_packed is not None:
            if name != 'negative' and np.any(_counts(previous_packed & candidate) != candidate_counts//2):
                raise ValueError('上一轮每图每区域选择违反floor半预算')
            intersection = _counts(current_packed & previous_packed & candidate)
            union = _counts((current_packed | previous_packed) & candidate)
        for i, row in enumerate(rows):
            n, selected, unique = map(int, (candidate_counts[i], selected_counts[i], unique_counts[i]))
            row[name] = {'candidate_count': n, 'selected_count': selected,
                         'cumulative_unique_count': unique,
                         'cumulative_coverage': unique/n if n else None,
                         'previous_intersection_count': int(intersection[i]) if intersection is not None else None,
                         'previous_union_count': int(union[i]) if union is not None else None,
                         'previous_jaccard': float(intersection[i]/union[i])
                         if union is not None and union[i] else None}
    return rows, updated
