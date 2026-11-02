"""Verify diagnostic run completeness and agreement of additive artifact totals."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np

from fecd.diagnostics.metrics import metrics_from_hist
from fecd.diagnostics.protocol import PROTOCOL, PROTOCOL_HASH


def validate_run(input_dir, rows, index, histograms):
    """Reject incomplete or mixed runs before any summary or curve is written."""
    manifest = json.loads((input_dir / 'run_manifest.json').read_text())
    membership = hashlib.sha256('\n'.join(sorted(row['sample_key'] for row in rows)).encode()).hexdigest()
    if (manifest.get('status') != 'passed' or manifest.get('evaluated') != len(rows)
            or manifest.get('requested') != len(rows)
            or manifest.get('protocol_hash') != PROTOCOL_HASH
            or manifest.get('evaluated_sample_keys_sha256') != membership):
        raise ValueError('Diagnostic run is incomplete or protocol/membership metadata differs')
    expected = defaultdict(lambda: np.zeros(2, np.int64))
    for row in rows:
        scopes = ((row['dataset'], 'ALL'), ('ALL', 'ALL'),
                  (row['dataset'], row['area_stratum']), ('ALL', row['area_stratum']))
        for region, methods in row['regions'].items():
            for method, result in methods.items():
                counts = np.asarray([result['n'], result['error_count']], dtype=np.int64)
                if (counts < 0).any() or counts[1] > counts[0]:
                    raise ValueError('Invalid per-image diagnostic counts')
                for dataset, area in scopes:
                    expected[(dataset, area, region, method)] += counts
    keys = [record['key'] for record in index]
    if len(set(keys)) != len(keys) or set(keys) != set(histograms.files):
        raise ValueError('Histogram NPZ/index keys are duplicated or inconsistent')
    scopes = [(record['dataset'], record['area_stratum'], record['region'], record['method'])
              for record in index]
    if len(set(scopes)) != len(scopes) or set(scopes) != set(expected):
        raise ValueError('Histogram scopes are duplicated, missing, or unexpected')
    for record, scope in zip(index, scopes):
        histogram = histograms[record['key']]
        if histogram.shape != (2, PROTOCOL['micro_histogram_bins']):
            raise ValueError('Histogram does not match frozen 4096-bin protocol')
        metrics_from_hist(histogram)
        if not np.array_equal(histogram.sum(axis=1), expected[scope]):
            raise ValueError(f'Histogram count totals disagree with per-image records: {scope}')
    return {'status': 'passed', 'images': len(rows), 'histogram_scopes': len(scopes),
            'membership_sha256': membership, 'protocol_hash': PROTOCOL_HASH}
