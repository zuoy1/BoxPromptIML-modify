"""GT-free heuristic stability scores; these are not calibrated probabilities."""

from __future__ import annotations

import numpy as np


def _bounded_array(value, name: str, maximum: float = 1.) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f'{name} must be nonempty and finite')
    if (array < 0).any() or (array > maximum).any():
        raise ValueError(f'{name} must be in [0, {maximum}]')
    return array


def sam_confidence(mu, var, agreement) -> np.ndarray:
    """Return the frozen SAM stability formula on equally shaped arrays.

    Variance up to .2501 is accepted for the existing float16 cache tolerance;
    the formula clamps 4*variance to [0,1]. All arithmetic uses float32.
    """
    mean = _bounded_array(mu, 'mu')
    variance = _bounded_array(var, 'var', .2501)
    vote = _bounded_array(agreement, 'agreement')
    if mean.shape != variance.shape or mean.shape != vote.shape:
        raise ValueError('mu, var and agreement must have identical shapes')
    uncertainty = 4. * vote * (1. - vote)
    score = (
        (1. - np.clip(4. * variance, 0., 1.))
        * (1. - uncertainty)
        * np.sqrt(2. * np.abs(mean - .5))
    )
    return np.asarray(np.clip(score, 0., 1.), dtype=np.float32)


def rule_confidence(c_sam, r) -> tuple[np.ndarray, np.ndarray]:
    """Return c_pos=c_sam*(.5+.5*r), c_neg=c_sam without reading GT."""
    confidence = _bounded_array(c_sam, 'c_sam')
    evidence = _bounded_array(r, 'r')
    if confidence.shape != evidence.shape:
        raise ValueError('c_sam and r must have identical shapes')
    positive = np.asarray(confidence * (.5 + .5 * evidence), dtype=np.float32)
    return positive, confidence.copy()
