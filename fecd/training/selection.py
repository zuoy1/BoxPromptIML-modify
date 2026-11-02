"""Deterministic classwise pixel budgets for fixed hard pseudo-labels."""
from __future__ import annotations

import hashlib
import numpy as np


def select_pixels(target, c_sam, mode, seed, sample_key, epoch):
    """Return a bool mask and exact class counts for one image.

    Half-budget modes retain floor(n_class/2), including zero for singletons.
    Ties share the seeded random permutation priority with random selection.
    """
    target = np.asarray(target)
    score = np.asarray(c_sam)
    if target.shape != score.shape or target.ndim == 0:
        raise ValueError('Target and reliability must share a non-scalar image shape')
    if not np.isfinite(target).all() or not np.isin(target, (0,1)).all():
        raise ValueError('Hard target must be finite and binary')
    if not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
        raise ValueError('Reliability must be finite in [0,1]')
    if mode not in ('all','random','confidence'):
        raise ValueError('Unsupported pixel selection mode')
    if not isinstance(seed, int) or not isinstance(epoch, int) or epoch < 0 or not sample_key:
        raise ValueError('Selection requires integer seed/nonnegative epoch and sample identity')
    result = np.zeros(target.size, dtype=bool)
    labels = target.reshape(-1)
    scores = score.reshape(-1)
    counts = {}
    for sign, value in (('pos',1),('neg',0)):
        indices = np.flatnonzero(labels == value)
        n = len(indices)
        k = n if mode == 'all' else n // 2
        if mode == 'all':
            chosen = indices
        else:
            digest = hashlib.sha256(f'{seed}:{epoch}:{sample_key}:{sign}'.encode()).digest()
            class_seed = int.from_bytes(digest[:8], 'big')
            permutation = np.random.default_rng(class_seed).permutation(n)
            if mode == 'random':
                order = permutation
            else:
                priority = np.empty(n, dtype=np.int64)
                priority[permutation] = np.arange(n)
                order = np.lexsort((priority, -scores[indices]))
            chosen = indices[order[:k]]
        result[chosen] = True
        counts[f'candidate_{sign}'] = n
        counts[f'selected_{sign}'] = k
    counts['selected_total'] = int(result.sum())
    counts['candidate_total'] = int(target.size)
    return result.reshape(target.shape), counts
