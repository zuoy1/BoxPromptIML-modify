"""Error detection with reliability-high scores and explicit single-class cases.

Exact metrics group identical scores. Histogram metrics use uniform reliability
bins: within-bin ordering is deliberately treated as a tie. Summing histograms
therefore computes pooled pixel (micro) metrics without retaining every pixel.
"""
from __future__ import annotations

import numpy as np

HISTOGRAM_BINS = 4096


def _validate_inputs(error, reliability):
    errors = np.asarray(error)
    scores = np.asarray(reliability, dtype=np.float64)
    if errors.shape != scores.shape:
        raise ValueError('Error and reliability must have identical shapes')
    if (not np.isfinite(scores).all() or np.any(scores < 0)
            or np.any(scores > 1) or not np.isin(errors, (0, 1)).all()):
        raise ValueError('Expected binary errors and finite reliability in [0, 1]')
    return errors.ravel().astype(np.int64), scores.ravel()


def _exact_groups(error, reliability):
    errors, scores = _validate_inputs(error, reliability)
    if not len(scores):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    _, indices = np.unique(scores, return_inverse=True)
    counts = np.bincount(indices).astype(np.int64)
    error_counts = np.bincount(indices, weights=errors).astype(np.int64)
    return counts, error_counts


def reliability_histogram(error, reliability, bins=HISTOGRAM_BINS):
    """Return [candidate counts, error counts] in ascending reliability bins.

    Score 1 is in the final bin. This protocol has bin width 1/bins, with no
    claim of a uniform bound on metric error when scores are highly clustered.
    """
    if not isinstance(bins, int) or bins < 2:
        raise ValueError('bins must be an integer >= 2')
    errors, scores = _validate_inputs(error, reliability)
    indices = np.minimum((scores * bins).astype(np.int64), bins - 1)
    return np.stack((np.bincount(indices, minlength=bins),
                     np.bincount(indices, weights=errors, minlength=bins)))\
        .astype(np.int64)


def _histogram_groups(hist):
    hist = np.asarray(hist)
    if (hist.ndim != 2 or hist.shape[0] != 2
            or not np.isfinite(hist).all() or np.any(hist < 0)
            or np.any(hist[1] > hist[0]) or np.any(hist != np.floor(hist))):
        raise ValueError('Expected nonnegative integer [counts, errors] histogram')
    occupied = hist[0] > 0
    return hist[0, occupied].astype(np.int64), hist[1, occupied].astype(np.int64)


def _metrics_from_groups(counts, errors):
    n = int(counts.sum())
    positive = int(errors.sum())
    negative = n - positive
    output = {'n': n, 'error_count': positive,
              'error_prevalence': positive / n if n else None,
              'auroc': None, 'auprc': None,
              'auroc_reason': None, 'auprc_reason': None}
    if not n:
        output.update(auroc_reason='empty candidate set',
                      auprc_reason='empty candidate set')
        return output
    if not positive or not negative:
        output['auroc_reason'] = 'candidate errors contain only one class'
    else:
        # Low confidence is a positive error detection; high confidence negatives
        # following each error group count as correctly ordered pairs.
        correct = counts - errors
        correct_after = negative - np.cumsum(correct)
        output['auroc'] = float(np.sum(errors * (correct_after + .5 * correct))
                                / (positive * negative))
    if not positive:
        output['auprc_reason'] = 'no error positives; average precision undefined'
    else:
        cumulative_errors = np.cumsum(errors, dtype=np.float64)
        cumulative_candidates = np.cumsum(counts, dtype=np.float64)
        # Non-interpolated average precision, including all tied detections at
        # once, rather than an optimistic sort order inside a tie group.
        output['auprc'] = float(np.sum(errors / positive * cumulative_errors
                                      / cumulative_candidates))
    return output


def error_detection_metrics(error, reliability):
    """Exact AUROC and non-interpolated AP, with error score = 1 - reliability."""
    return _metrics_from_groups(*_exact_groups(error, reliability))


def metrics_from_hist(hist):
    """Compute pooled detection metrics from additive reliability histograms."""
    return _metrics_from_groups(*_histogram_groups(hist))
