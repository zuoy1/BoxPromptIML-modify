"""Global robust normalization fitted exclusively from authorized train_fit maps."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from fecd.evidence.extractors import EXTRACTOR_CONFIG


FEATURES = ('hp', 'noise', 'jpeg', 'fft')


def _hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def _validate_maps(maps):
    for name in FEATURES:
        value = np.asarray(maps[name])
        if value.shape != (224, 224) or not np.isfinite(value).all() or value.min() < 0:
            raise ValueError(f'Invalid {name}: expected finite nonnegative [224,224]')


def fit_stats(maps_iterable, provenance, pixels_per_image=1024):
    """Fit global q01/q99 using equal, deterministic spatial samples per image.

    Caller must provide only train_fit maps and a frozen split hash in provenance.
    The partition guard is defense in depth; identity membership is checked by the
    fitting CLI. Nonfinite/missing features fail instead of being silently imputed.
    """
    if provenance.get('partition') != 'train_fit':
        raise ValueError('Statistics may only be fitted on train_fit')
    if not isinstance(pixels_per_image, int) or not 1 <= pixels_per_image <= 224 * 224:
        raise ValueError('pixels_per_image must be an integer in [1,50176]')
    indices = np.linspace(0, 224 * 224 - 1, pixels_per_image, dtype=np.int64)
    samples = {name: [] for name in FEATURES}
    count = 0
    for maps in maps_iterable:
        _validate_maps(maps)
        for name in FEATURES:
            samples[name].append(np.asarray(maps[name], np.float32).ravel()[indices])
        count += 1
    if not count:
        raise ValueError('No train_fit evidence supplied')
    stats = {
        'schema_version': 'fecd-evidence-normalization-v1',
        'provenance': dict(provenance), 'extractor_config': EXTRACTOR_CONFIG,
        'image_count': count, 'pixels_per_image': pixels_per_image,
        'sampling': 'equal-image linspace(flattened 224x224) indices including endpoints',
        'quantiles': [0.01, 0.50, 0.99], 'quantile_method': 'linear',
        'transform': 'clip((raw-low)/(high-low),0,1); degenerate high<=low maps to zero',
        'missing_values': 'fail; no imputation', 'features': {},
    }
    for name in FEATURES:
        values = np.concatenate(samples[name])
        low, median, high = np.quantile(values, [.01, .50, .99], method='linear')
        stats['features'][name] = {'low': float(low), 'median': float(median),
                                   'high': float(high), 'sample_count': int(values.size)}
    stats['stats_hash'] = _hash(stats)
    return stats


def normalize_evidence(maps, stats):
    """Apply identical frozen train_fit scaling; HP never enters default fusion."""
    payload = {key: value for key, value in stats.items() if key != 'stats_hash'}
    if stats.get('stats_hash') != _hash(payload):
        raise ValueError('Evidence statistics hash mismatch')
    if stats.get('extractor_config') != EXTRACTOR_CONFIG:
        raise ValueError('Evidence extractor configuration mismatch')
    _validate_maps(maps)
    normalized = {}
    for name in FEATURES:
        low = stats['features'][name]['low']
        high = stats['features'][name]['high']
        if not np.isfinite([low, high]).all() or high < low:
            raise ValueError(f'Invalid normalization bounds for {name}')
        value = np.asarray(maps[name], dtype=np.float32)
        normalized[name] = (np.clip((value - low) / (high - low), 0, 1).astype(np.float32)
                            if high > low else np.zeros_like(value))
    normalized['r'] = np.median(np.stack([normalized[name] for name in ('noise', 'jpeg', 'fft')]),
                                axis=0).astype(np.float32)
    return normalized
