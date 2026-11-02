"""Integer offline selections with shared label-independent tie priorities."""
from __future__ import annotations

import hashlib

import numpy as np

from fecd.diagnostics.stage3ra_protocol import PROTOCOL


def realized_budgets(error, scores, masks, sample_key):
    """Count actual retained pixels; primary inference keeps fractional ties."""
    result = {}
    for sign in ('pos', 'neg'):
        candidate = masks[sign]
        errors = error[candidate]
        n = len(errors)
        identity = f"{PROTOCOL['seed']}:{sample_key}:{sign}"
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big')
        permutation = np.random.default_rng(seed).permutation(n)
        priority = np.empty(n, np.int64)
        priority[permutation] = np.arange(n)
        orders = {'random': permutation}
        for method in ('c_sam', 'q_a'):
            orders[method] = np.lexsort((priority, -scores[method][candidate]))
        result[sign] = {}
        for method, order in orders.items():
            result[sign][method] = {}
            for coverage in PROTOCOL['coverages']:
                k = round(coverage * n)
                count_errors = int(errors[order[:k]].sum())
                sufficient = k >= PROTOCOL['minimum_coverage_pixels']
                risk = count_errors / k if sufficient else None
                result[sign][method][f'{coverage:g}'] = {
                    'n_candidates': n, 'selected_count': k, 'error_count': count_errors,
                    'risk': risk, 'precision': 1 - risk if risk is not None else None,
                    'conditional_coverage': k / n if n else None,
                    'full_coverage': k / candidate.size,
                    'reason': None if sufficient else 'fewer than 10 selected pixels',
                    'count_policy': 'actual integer selection; common deterministic random tie order',
                }
    return result
