"""Dedicated new_dev-only label access and fixed six-score development audit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from fecd.diagnostics.stage3ra_split import read_split
from fecd.diagnostics.stage3ra_protocol import PROTOCOL, PROTOCOL_HASH
from fecd.diagnostics.stage3ra_sample import audit_sample
from fecd.pseudo.cache_schema import canonical_hash, load_cache

PROJECT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT / 'fecd/assets'
SOURCE = Path('/home/datasets/processed/boxprompt_paper_scale_20261016/manifest.json')
K8 = ASSETS / 'stage2c_cache/k8/2bff688546a7'
K1 = ASSETS / 'stage2c_cache/k1/d7ccb769414f'


def load_dev_gt(row, source, native_hw, allowed_rows, log_path):
    """Check trusted full-record membership before touching a label file."""
    trusted = allowed_rows.get(row.get('sample_key'))
    if (row.get('subset') != 'new_dev' or trusted != row
            or trusted.get('subset') != 'new_dev' or source.get('split') != 'train'):
        raise ValueError('GT access requires exact frozen new_dev membership')
    if (row['dataset'], row['sample_id'], row['image_sha256']) != (
            source['dataset'], source['id'], source['image_sha256']):
        raise ValueError('Development label/source identity mismatch')
    path = Path(source['prepared_mask'].replace('/hy-tmp/datasets/', '/home/datasets/'))
    data = path.read_bytes()
    access = {'sample_key': row['sample_key'], 'partition': 'new_dev',
              'prepared_mask_sha256': hashlib.sha256(data).hexdigest(),
              'mapping': 'native binary L; PIL NEAREST to224x224; foreground >127'}
    with Path(log_path).open('a') as stream:
        stream.write(json.dumps(access, separators=(',', ':')) + '\n')
    with Image.open(io.BytesIO(data)) as image:
        if [image.height, image.width] != list(native_hw):
            raise ValueError('Development label/native cache dimensions differ')
        gray = np.asarray(image.convert('L'))
        if not np.isin(gray, (0, 255)).all():
            raise ValueError('Prepared development label must contain only 0 and 255')
        resized = np.asarray(image.convert('L').resize((224, 224), Image.Resampling.NEAREST))
    return resized > PROTOCOL['gt_threshold']


def _cache(index_row, row, k):
    expected = {key: row[key] for key in ('dataset', 'sample_id', 'sample_key', 'image_sha256')}
    expected.update(split='train', k=k, image_sha1=index_row['image_sha1'],
                    box_hash=index_row['box_hash'])
    path = Path(index_row['cache_path'])
    return load_cache(path if path.is_absolute() else PROJECT / path, expected=expected)


def _freeze_protocol():
    path = ASSETS / 'stage3ra_protocol.json'
    payload = {'protocol': PROTOCOL, 'protocol_hash': PROTOCOL_HASH}
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError('Frozen development protocol differs from source')
    else:
        with path.open('x') as stream:
            stream.write(json.dumps(payload, indent=2) + '\n')


def _save_histograms(storage, assets):
    arrays, index = {}, []
    for i, (scope, value) in enumerate(sorted(storage.items())):
        key = f'h{i:06d}'
        arrays[key] = value
        dataset, area, region, method = scope
        index.append(dict(key=key, dataset=dataset, area_stratum=area,
                          region=region, method=method))
    np.savez_compressed(assets / 'stage3ra_histograms.npz', **arrays)
    (assets / 'stage3ra_hist_index.json').write_text(json.dumps(index, indent=2) + '\n')


def run(assets=ASSETS, limit=None):
    """Run once, failing closed on A0, protocol, membership or source mismatch."""
    split = read_split()
    _freeze_protocol()
    assets = Path(assets).resolve()
    if assets != ASSETS.resolve() and not assets.is_relative_to((ASSETS / 'stage3ra_pilot').resolve()):
        raise ValueError('Audit outputs must use dedicated stage3ra diagnostic paths')
    assets.mkdir(parents=True, exist_ok=True)
    output = assets / 'stage3ra_per_image.jsonl'
    run_path = assets / 'stage3ra_run_manifest.json'
    access_path = assets / 'stage3ra_gt_access.jsonl'
    if any(p.exists() for p in (output, run_path, access_path)):
        raise FileExistsError('Refusing to overwrite development diagnostic artifacts')
    allowed = {row['sample_key']: row for row in split['records'] if row['subset'] == 'new_dev'}
    rows = list(allowed.values())
    if limit is not None:
        rows = rows[:limit]
    k8_rows = json.loads((K8 / 'dataset_index.json').read_text())
    k1_rows = json.loads((K1 / 'dataset_index.json').read_text())
    k8_index = {r['sample_key']: r for r in k8_rows}
    k1_index = {r['sample_key']: r for r in k1_rows}
    if len(k8_index) != len(k8_rows) or len(k1_index) != len(k1_rows) or set(k8_index) != set(k1_index):
        raise ValueError('K1/K8 index membership differs or is duplicated')
    source_rows = json.loads(SOURCE.read_text())
    source_index = {(r['dataset'], r['id']): r for r in source_rows if r['split'] == 'train'}
    source_index = {key: source_index[key] for key in ((r['dataset'], r['sample_id']) for r in rows)}
    membership = hashlib.sha256('\n'.join(sorted(r['sample_key'] for r in rows)).encode()).hexdigest()
    code_files = [p for folder in ('confidence', 'diagnostics', 'tools')
                  for p in (PROJECT / 'fecd' / folder).glob('*stage3ra*.py')]
    code_files.extend(PROJECT / name for name in (
        'fecd/diagnostics/metrics.py', 'fecd/diagnostics/risk_coverage.py',
        'fecd/diagnostics/sample_audit.py', 'fecd/confidence/teacher_scores.py'))
    manifest = dict(schema_version='fecd-stage3ra-run-v1', status='running',
                    protocol_hash=PROTOCOL_HASH, split_hash=split['manifest_sha256'],
                    requested=len(rows), evaluated=0, gt_partition='new_dev',
                    confirmation_gt_accesses=0, subset_limit=limit,
                    evaluated_sample_keys_sha256=membership,
                    source_manifest_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    implementation_sha256={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted(code_files)})
    run_path.write_text(json.dumps(manifest, indent=2) + '\n')
    start, storage = time.monotonic(), {}
    cv2.setNumThreads(1)
    with output.open('x') as stream:
        for i, row in enumerate(rows, 1):
            k8, k1 = _cache(k8_index[row['sample_key']], row, 8), _cache(k1_index[row['sample_key']], row, 1)
            if not np.array_equal(k8['base_mu'], k1['mu']):
                raise ValueError('K8 Base and K1 cached mean differ')
            gt = load_dev_gt(row, source_index[(row['dataset'], row['sample_id'])],
                             (int(k8['image_h']), int(k8['image_w'])), allowed, access_path)
            result, histograms = audit_sample(k8['mu'].astype(np.float32), k8['var'].astype(np.float32),
                                              k8['agreement'].astype(np.float32), k8['base_mu'].astype(np.float32),
                                              k8['box_masks'], gt, sample_key=row['sample_key'])
            result.update({key: row[key] for key in ('sample_id', 'dataset', 'source_group_id')})
            stream.write(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
            stream.flush()
            for (region, method), histogram in histograms.items():
                for dataset, area in ((row['dataset'], 'ALL'), ('ALL', 'ALL'),
                                      (row['dataset'], result['area_stratum']), ('ALL', result['area_stratum'])):
                    scope = (dataset, area, region, method)
                    if scope not in storage:
                        storage[scope] = np.zeros_like(histogram)
                    storage[scope] += histogram
            if i % 25 == 0 or i == len(rows):
                print(json.dumps({'evaluated': i, 'requested': len(rows),
                                  'elapsed_seconds': time.monotonic() - start}), flush=True)
    _save_histograms(storage, assets)
    accesses = [json.loads(x) for x in access_path.read_text().splitlines()]
    if len(accesses) != len(rows) or {a['sample_key'] for a in accesses} != {r['sample_key'] for r in rows}:
        raise ValueError('GT access log differs from evaluated development identities')
    manifest.update(status='passed', evaluated=len(rows), dataset_counts=dict(Counter(r['dataset'] for r in rows)),
                    elapsed_seconds=time.monotonic() - start, gt_accesses=len(accesses),
                    gt_access_log_sha256=canonical_hash(accesses))
    run_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, default=ASSETS)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and (args.limit < 1 or args.assets_dir.resolve() == ASSETS.resolve()):
        parser.error('A positive pilot limit requires --assets-dir under fecd/assets/stage3ra_pilot')
    run(args.assets_dir, args.limit)


if __name__ == '__main__':
    main()
