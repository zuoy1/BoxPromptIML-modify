"""Exclusive teacher_diag GT loader and reproducible frozen-method audit entry."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fecd.diagnostics.protocol import PROTOCOL, PROTOCOL_HASH
from fecd.diagnostics.sample_audit import audit_sample
from fecd.evidence.cache import (DEFAULT_SPLIT, PROJECT, ROOT, aligned_sam_index,
                                 evidence_config_hash, evidence_path, load_evidence, read_split)
from fecd.evidence.normalize import normalize_evidence
from fecd.pseudo.cache_schema import canonical_hash, load_cache
from fecd.tools.check_stage3 import check_gate

DIAGNOSTIC_ROOT = ROOT / 'assets/stage3_diagnostics'
K1_ROOT = ROOT / 'assets/stage2c_cache/k1/d7ccb769414f'
SOURCE_MANIFEST = Path('/home/datasets/processed/boxprompt_paper_scale_20261016/manifest.json')
_GATE_STATUS = None


def _ensure_gate():
    global _GATE_STATUS
    if _GATE_STATUS is None:
        _GATE_STATUS = check_gate(require_3a=True)
    return _GATE_STATUS


def _project_path(path):
    value = Path(path)
    return value if value.is_absolute() else PROJECT / value


def load_diagnostic_gt(row, source, native_hw):
    """This is the only stage3 function that reads a ground-truth image file."""
    _ensure_gate()
    if row['subset'] != 'teacher_diag' or source['split'] != 'train':
        raise ValueError('GT access is restricted to frozen teacher_diag training records')
    if (row['dataset'], row['sample_id'], row['image_sha256']) != (
            source['dataset'], source['id'], source['image_sha256']):
        raise ValueError('Diagnostic GT/source identity mismatch')
    declared = source['prepared_mask']
    path = Path(declared.replace('/hy-tmp/datasets/', '/home/datasets/'))
    with Image.open(path) as image:
        if [image.height, image.width] != list(native_hw):
            raise ValueError('Diagnostic GT/native SAM dimensions differ')
        gray = np.asarray(image.convert('L'))
        if not np.isin(gray, (0, 255)).all():
            raise ValueError('Prepared diagnostic GT must contain only 0 and 255')
        resized = np.asarray(image.convert('L').resize((224, 224), Image.Resampling.NEAREST))
    return resized > PROTOCOL['gt_threshold'], {
        'sample_key': row['sample_key'], 'prepared_mask_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'mapping': 'PIL nearest-neighbor native binary L to 224x224, foreground >127'}


def _load_stats(path, split):
    stats = json.loads(Path(path).read_text())
    fit_rows = [row for row in split['records'] if row['subset'] == 'train_fit']
    key_hash = hashlib.sha256('\n'.join(sorted(row['sample_key'] for row in fit_rows)).encode()).hexdigest()
    provenance = stats['provenance']
    if (provenance.get('partition') != 'train_fit'
            or provenance.get('split_manifest_sha256') != split['manifest_sha256']
            or provenance.get('sample_keys_sha256') != key_hash
            or provenance.get('extractor_config_hash') != evidence_config_hash()
            or stats['image_count'] != len(fit_rows)):
        raise ValueError('Normalization provenance must match the complete frozen train_fit')
    return stats


def _sam_record(index_row, source, k):
    expected = {'dataset': source['dataset'], 'sample_id': source['sample_id'],
                'sample_key': source['sample_key'], 'image_sha256': source['image_sha256'],
                'image_sha1': index_row['image_sha1'], 'box_hash': index_row['box_hash'],
                'split': 'train', 'k': k}
    return load_cache(_project_path(index_row['cache_path']), expected=expected)


def _add_histograms(storage, row, histograms):
    scopes = ((row['dataset'], 'ALL'), ('ALL', 'ALL'),
              (row['dataset'], row['area_stratum']), ('ALL', row['area_stratum']))
    for (region, method), hist in histograms.items():
        for dataset, area in scopes:
            key = (dataset, area, region, method)
            if key not in storage:
                storage[key] = np.zeros_like(hist)
            storage[key] += hist


def run(split_path=DEFAULT_SPLIT, stats_path=ROOT / 'assets/stage3_evidence_stats.json',
        output_dir=DIAGNOSTIC_ROOT, limit=None):
    _ensure_gate()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(DIAGNOSTIC_ROOT.resolve()):
        raise ValueError('GT-bearing audit outputs must remain under stage3_diagnostics')
    split = read_split(split_path)
    stats = _load_stats(stats_path, split)
    k8_index = aligned_sam_index(split)
    k1_rows = json.loads((K1_ROOT / 'dataset_index.json').read_text())
    k1_index = {row['sample_key']: row for row in k1_rows}
    if len(k1_index) != len(k1_rows) or set(k1_index) != set(k8_index):
        raise ValueError('K1/K8 frozen identity sets differ')
    source_rows = json.loads(SOURCE_MANIFEST.read_text())
    source_index = {(row['dataset'], row['id']): row for row in source_rows}
    if len(source_index) != len(source_rows):
        raise ValueError('Duplicate original manifest identity')
    rows = [row for row in split['records'] if row['subset'] == 'teacher_diag']
    if limit is not None:
        rows = rows[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    histogram_storage = {}
    accesses = []
    start = time.monotonic()
    cv2.setNumThreads(1)
    membership = hashlib.sha256('\n'.join(sorted(row['sample_key'] for row in rows)).encode()).hexdigest()
    manifest = {'schema_version': 'fecd-stage3-diagnostic-run-v1', 'status': 'running',
                'protocol': PROTOCOL, 'protocol_hash': PROTOCOL_HASH,
                'split_manifest_sha256': split['manifest_sha256'], 'stats_hash': stats['stats_hash'],
                'extractor_config_hash': evidence_config_hash(), 'requested': len(rows),
                'evaluated_sample_keys_sha256': membership, 'gt_partition': 'teacher_diag',
                'gt_threshold': 127, 'gt_resize': 'PIL NEAREST to224x224',
                'source_manifest_sha256': hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
                'subset_limit': limit, 'contains_dense_gt': False}
    run_path = output_dir / 'run_manifest.json'
    run_path.write_text(json.dumps(manifest, indent=2) + '\n')
    with (output_dir / 'per_image.jsonl').open('w') as stream:
        for count, row in enumerate(rows, 1):
            k8 = _sam_record(k8_index[row['sample_key']], row, 8)
            k1 = _sam_record(k1_index[row['sample_key']], row, 1)
            if (str(k8['image_sha1']), str(k8['box_hash'])) != (str(k1['image_sha1']), str(k1['box_hash'])):
                raise ValueError('K1/K8 source or Box mismatch')
            raw = load_evidence(evidence_path(row), row)
            normalized = normalize_evidence(raw, stats)
            gt, access = load_diagnostic_gt(row, source_index[(row['dataset'], row['sample_id'])],
                                            [int(k8['image_h']), int(k8['image_w'])])
            result, histograms = audit_sample(k8['mu'].astype(np.float32), k8['var'].astype(np.float32),
                                              k8['agreement'].astype(np.float32), k1['mu'].astype(np.float32),
                                              k8['box_masks'], normalized, gt)
            result.update({key: row[key] for key in ('sample_key', 'sample_id', 'dataset', 'source_group_id')})
            stream.write(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
            stream.flush()
            _add_histograms(histogram_storage, result, histograms)
            accesses.append(access)
            if count % 25 == 0 or count == len(rows):
                print(json.dumps({'evaluated': count, 'requested': len(rows),
                                  'elapsed_seconds': time.monotonic() - start}), flush=True)
    histogram_arrays = {}
    histogram_index = []
    for count, (scope, value) in enumerate(sorted(histogram_storage.items())):
        key = f'h{count:06d}'
        histogram_arrays[key] = value
        dataset, area, region, method = scope
        histogram_index.append({'key': key, 'dataset': dataset, 'area_stratum': area,
                                'region': region, 'method': method})
    np.savez_compressed(output_dir / 'histograms.npz', **histogram_arrays)
    (output_dir / 'hist_index.json').write_text(json.dumps(histogram_index, indent=2) + '\n')
    (output_dir / 'gt_access_log.json').write_text(json.dumps(accesses, indent=2) + '\n')
    manifest.update(status='passed', evaluated=len(rows),
                    dataset_counts=dict(Counter(row['dataset'] for row in rows)),
                    elapsed_seconds=time.monotonic() - start,
                    gt_access_log_sha256=canonical_hash(accesses))
    run_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    return manifest


def run_auxiliary(split_path, stats_path, output_dir, visualize=False, sensitivity=False):
    _ensure_gate()
    from fecd.diagnostics.cases import micro_sensitivity, visualize_cases
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(DIAGNOSTIC_ROOT.resolve()):
        raise ValueError('Diagnostic auxiliary outputs must remain under stage3_diagnostics')
    split = read_split(split_path)
    stats = _load_stats(stats_path, split)
    k8_index = aligned_sam_index(split)
    k1_index = {r['sample_key']: r for r in json.loads((K1_ROOT / 'dataset_index.json').read_text())}
    source_index = {(r['dataset'], r['id']): r for r in json.loads(SOURCE_MANIFEST.read_text())}
    accesses = []

    def load_sample(row, include_rgb=False):
        k8 = _sam_record(k8_index[row['sample_key']], row, 8)
        k1 = _sam_record(k1_index[row['sample_key']], row, 1)
        if (str(k8['image_sha1']), str(k8['box_hash'])) != (str(k1['image_sha1']), str(k1['box_hash'])):
            raise ValueError('Auxiliary K1/K8 identity mismatch')
        raw = load_evidence(evidence_path(row), row)
        normalized = normalize_evidence(raw, stats)
        gt = None
        if row['subset'] == 'teacher_diag':
            gt, access = load_diagnostic_gt(row, source_index[(row['dataset'], row['sample_id'])],
                                            [int(k8['image_h']), int(k8['image_w'])])
            accesses.append(access)
        result = {'k8': k8, 'k1': k1, 'normalized': normalized,
                  'gt': gt, 'evidence_metadata': raw['metadata']}
        if include_rgb:
            with Image.open(row['image_path']) as image:
                rgb = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR))
            boxes = np.asarray(k8_index[row['sample_key']]['boxes_xyxy'], np.float32).reshape(-1, 4)
            boxes[:, [0, 2]] *= 224 / int(k8['image_w'])
            boxes[:, [1, 3]] *= 224 / int(k8['image_h'])
            result.update(rgb224=rgb, boxes224=boxes)
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {'protocol_hash': PROTOCOL_HASH, 'split_hash': split['manifest_sha256'],
               'stats_hash': stats['stats_hash']}
    if visualize:
        result = visualize_cases(output_dir / 'per_image.jsonl', split['records'], load_sample, output_dir)
        summary['case_count'] = sum(c['sample_key'] is not None for c in result['cases'])
    if sensitivity:
        result = micro_sensitivity(split['records'], load_sample, output_dir)
        summary['sensitivity_sample_count'] = len(result['sample_keys'])
    (output_dir / 'auxiliary_gt_access_log.json').write_text(json.dumps(accesses, indent=2) + '\n')
    summary.update(gt_access_count=len(accesses), gt_access_log_sha256=canonical_hash(accesses))
    (output_dir / 'auxiliary_run_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--stats', type=Path, default=ROOT / 'assets/stage3_evidence_stats.json')
    parser.add_argument('--output-dir', type=Path, default=DIAGNOSTIC_ROOT)
    parser.add_argument('--limit', type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--visualize-only', action='store_true')
    modes.add_argument('--micro-sensitivity-only', action='store_true')
    modes.add_argument('--cases-only', action='store_true', help='visualizations and exact/histogram sensitivity')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.visualize_only or args.micro_sensitivity_only or args.cases_only:
        run_auxiliary(args.split, args.stats, args.output_dir,
                      visualize=args.visualize_only or args.cases_only,
                      sensitivity=args.micro_sensitivity_only or args.cases_only)
    else:
        run(args.split, args.stats, args.output_dir, args.limit)


if __name__ == '__main__':
    main()
