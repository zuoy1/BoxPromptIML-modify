"""Canonical-pixel budgets for the separately versioned 4F0 baselines."""
from __future__ import annotations

from fecd.training.selection import select_pixels

MODES={'P0':'all','P1':'all','P2F':'random','P3F':'confidence','P2E':'random'}


def select_for_variant(target,confidence,variant,seed,sample_key,epoch):
    """Choose before geometry; only P2E varies its priority with the epoch."""
    if variant not in MODES:
        raise ValueError('Unknown 4F0 baseline')
    if not isinstance(epoch,int) or epoch<0:
        raise ValueError('Epoch must be a nonnegative integer')
    priority_epoch=epoch if variant=='P2E' else 0
    return select_pixels(target,confidence,MODES[variant],seed,sample_key,priority_epoch)


def expected_unique_coverage(candidate_count,variant,epochs):
    """P2E expectation assumes independent uniform epoch priorities, not an audit result."""
    if variant not in MODES:
        raise ValueError('Unknown 4F0 baseline')
    if not isinstance(candidate_count,int) or candidate_count<0:
        raise ValueError('Candidate count must be a nonnegative integer')
    if not isinstance(epochs,int) or epochs<1:
        raise ValueError('Epoch count must be a positive integer')
    if candidate_count==0:
        return None
    if MODES[variant]=='all':
        return 1.0
    fraction=(candidate_count//2)/candidate_count
    return 1-(1-fraction)**epochs if variant=='P2E' else fraction
