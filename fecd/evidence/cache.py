"""Image-only evidence records, independently keyed from frozen SAM caches."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image, features

from fecd.evidence.extractors import EXTRACTOR_CONFIG, extract_evidence
from fecd.evidence.normalize import FEATURES
from fecd.pseudo.cache_schema import canonical_hash, load_cache, make_sample_key

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DEFAULT_SPLIT = ROOT / 'assets/stage3_split_manifest.json'
DEFAULT_CACHE = ROOT / 'assets/stage3_evidence/raw'
SAM_ROOT = ROOT / 'assets/stage2c_cache/k8/2bff688546a7'
SCHEMA = 'fecd-evidence-cache-v1'


def evidence_config_hash():
    payload = {'extractor': EXTRACTOR_CONFIG,
               'implementation_sha256': hashlib.sha256((Path(__file__).parent / 'extractors.py').read_bytes()).hexdigest(),
               'numpy': np.__version__, 'opencv': cv2.__version__, 'pillow': PIL.__version__,
               'jpeg_codec': features.version_codec('jpg')}
    return canonical_hash(payload)


def evidence_path(record, cache_root=DEFAULT_CACHE):
    return Path(cache_root) / evidence_config_hash()[:12] / f"{record['sample_key']}.npz"


def _identity(record):
    expected = make_sample_key(record['dataset'], 'train', record['sample_id'], record['image_sha256'])
    if record['sample_key'] != expected:
        raise ValueError('Evidence sample_key does not match source identity')
    return {key: record[key] for key in ('dataset', 'sample_id', 'sample_key', 'image_sha256')}


def read_split(path=DEFAULT_SPLIT):
    payload = json.loads(Path(path).read_text())
    actual = canonical_hash({k: v for k, v in payload.items() if k != 'manifest_sha256'})
    if payload.get('manifest_sha256') != actual or payload.get('records_sha256') != canonical_hash(payload['records']):
        raise ValueError('Frozen stage3 split hash mismatch')
    keys = []
    for row in payload['records']:
        _identity(row)
        if row['subset'] not in ('train_fit', 'teacher_diag'):
            raise ValueError('Unknown stage3 subset')
        keys.append(row['sample_key'])
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate stage3 sample_key')
    return payload


def aligned_sam_index(split_manifest):
    rows = json.loads((SAM_ROOT / 'dataset_index.json').read_text())
    index = {row['sample_key']: row for row in rows}
    if len(index) != len(rows) or set(index) != {r['sample_key'] for r in split_manifest['records']}:
        raise ValueError('SAM and stage3 identity sets differ')
    for row in split_manifest['records']:
        sam = index[row['sample_key']]
        if any(sam[k] != row[k] for k in ('dataset', 'sample_id', 'image_path')) or sam['split'] != 'train':
            raise ValueError('SAM/source identity mismatch')
    return index


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_record(maps, source, image_metadata):
    metadata = {**_identity(source), **image_metadata, 'schema_version': SCHEMA,
                'extractor_config_hash': evidence_config_hash(), 'extractor_config': EXTRACTOR_CONFIG,
                'map_sha256': {name: _array_hash(maps[name]) for name in FEATURES}}
    record = {name: np.asarray(maps[name], np.float32) for name in FEATURES}
    record['metadata'] = metadata
    _validate_record(record)
    return record


def _validate_record(record, expected=None):
    meta = record['metadata']
    _identity(meta)
    if meta.get('schema_version') != SCHEMA or meta.get('extractor_config_hash') != evidence_config_hash():
        raise ValueError('Evidence schema/config mismatch')
    if meta.get('extractor_config') != EXTRACTOR_CONFIG:
        raise ValueError('Evidence extractor parameters mismatch')
    if len(meta['native_hw']) != 2 or min(meta['native_hw']) <= 0 or not meta['original_format']:
        raise ValueError('Invalid native evidence metadata')
    if expected and any(meta[k] != v for k, v in _identity(expected).items()):
        raise ValueError('Evidence/source identity mismatch')
    for name in FEATURES:
        value = record[name]
        if value.shape != (224, 224) or value.dtype != np.float32 or not np.isfinite(value).all() or value.min() < 0:
            raise ValueError(f'Invalid evidence map: {name}')
        if meta['map_sha256'][name] != _array_hash(value):
            raise ValueError(f'Evidence map checksum mismatch: {name}')


def load_evidence(path, expected_record=None):
    try:
        with np.load(path, allow_pickle=False) as data:
            record = {name: data[name].copy() for name in FEATURES}
            text = str(data['metadata_json'])
            if str(data['metadata_sha256']) != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError('Evidence metadata checksum mismatch')
            record['metadata'] = json.loads(text)
        _validate_record(record, expected_record)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(f'Invalid evidence cache {path}: {exc}') from exc
    return record


def save_evidence(path, record):
    _validate_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record['metadata'], sort_keys=True, separators=(',', ':'))
    descriptor, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **{name: record[name] for name in FEATURES},
                                metadata_json=np.asarray(text),
                                metadata_sha256=np.asarray(hashlib.sha256(text.encode()).hexdigest()))
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _image_metadata(image):
    tables = getattr(image, 'quantization', None)
    quantization = None
    if tables:
        quantization = {'tables': {str(k): {'min': min(v), 'median': float(np.median(v)), 'max': max(v)}
                                   for k, v in tables.items()},
                        'tables_sha256': canonical_hash(tables), 'exact_quality_factor': 'not inferred'}
    return {'native_hw': [image.height, image.width], 'original_format': image.format or 'unknown',
            'original_jpeg_quantization': quantization,
            'jpeg_recompression': EXTRACTOR_CONFIG['jpeg']}


def generate_record(source, sam_index_record, cache_root=DEFAULT_CACHE, validate_only=False):
    cv2.setNumThreads(1)
    identity = _identity(source)
    path = Path(source['image_path'])
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != source['image_sha256']:
        raise ValueError('Image SHA256 differs from frozen source')
    sha1 = hashlib.sha1(contents).hexdigest()
    if sha1 != sam_index_record['image_sha1']:
        raise ValueError('Image SHA1 differs from frozen SAM index')
    expected = {**identity, 'split': 'train', 'image_sha1': sha1,
                'box_hash': sam_index_record['box_hash'], 'k': 8}
    sam_path = Path(sam_index_record['cache_path'])
    if not sam_path.is_absolute():
        sam_path = PROJECT / sam_path
    sam = load_cache(sam_path, expected=expected)
    output = evidence_path(source, cache_root)
    reused = output.exists()
    if reused:
        record = load_evidence(output, source)
    elif validate_only:
        raise ValueError(f'Missing evidence cache: {output}')
    else:
        with Image.open(path) as image:
            metadata = _image_metadata(image)
            rgb = np.asarray(image.convert('RGB'))
        metadata.update({'sam_cache_key': str(sam['cache_key']), 'sam_config_hash': str(sam['config_hash']),
                         'image_sha1': sha1, 'box_hash': str(sam['box_hash'])})
        record = build_record(extract_evidence(rgb), source, metadata)
        save_evidence(output, record)
        record = load_evidence(output, source)
    meta = record['metadata']
    if meta['native_hw'] != [int(sam['image_h']), int(sam['image_w'])]:
        raise ValueError('Evidence/SAM native shape mismatch')
    for key in ('sam_cache_key', 'sam_config_hash', 'box_hash', 'image_sha1'):
        sam_key = {'sam_cache_key': 'cache_key', 'sam_config_hash': 'config_hash'}.get(key, key)
        if meta[key] != str(sam[sam_key]):
            raise ValueError(f'Evidence/SAM metadata mismatch: {key}')
    return {'sample_key': source['sample_key'], 'dataset': source['dataset'], 'subset': source['subset'],
            'path': str(output), 'reused': reused, 'bytes': output.stat().st_size,
            'original_format': meta['original_format'], 'native_hw': meta['native_hw']}
