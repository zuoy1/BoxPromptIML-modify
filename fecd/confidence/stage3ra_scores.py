"""Six prespecified teacher reliability scores; no fitting or label access."""
from __future__ import annotations

import numpy as np

from fecd.confidence.teacher_scores import _bounded_array, sam_confidence


def teacher_scores(mu, var, agreement, base_mu):
    """Return scores on one common grid; a high score always means reliable."""
    mean = _bounded_array(mu, 'mu')
    base = _bounded_array(base_mu, 'base_mu')
    variance = _bounded_array(var, 'var', .2501)
    vote = _bounded_array(agreement, 'agreement')
    if any(value.shape != mean.shape for value in (base, variance, vote)):
        raise ValueError('All teacher arrays must have identical shapes')
    confidence = sam_confidence(mean, variance, vote)
    return {
        'base_margin': np.asarray(2 * abs(base - .5), np.float32),
        'mean_margin': np.asarray(2 * abs(mean - .5), np.float32),
        'variance': np.asarray(1 - np.clip(4 * variance, 0, 1), np.float32),
        'threshold': np.asarray(1 - 4 * vote * (1 - vote), np.float32),
        'c_sam': confidence,
        'q_a': np.asarray(confidence / (1 + abs(mean - base)), np.float32),
    }
