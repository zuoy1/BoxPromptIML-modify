"""GT-free descriptive domain/texture bias and fixed JPEG-quality sensitivity audit."""
from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from fecd.evidence.cache import (DEFAULT_CACHE, DEFAULT_SPLIT, evidence_path,
                                 load_evidence, read_split)
from fecd.evidence.extractors import _log_anomaly
from fecd.evidence.normalize import FEATURES, normalize_evidence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'assets/stage3_diagnostics/bias_checks.json'


def jpeg_quality_map(rgb, quality):
    """Reproduce main native JPEG path at a diagnostic quality without config mutation."""
    value = np.asarray(rgb)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError('Sensitivity input must be RGB uint8 [H,W,3]')
    if not isinstance(quality, int) or not 1 <= quality <= 100 or min(value.shape[:2]) < 1:
        raise ValueError('JPEG quality must be an integer [1,100], image must be nonempty')
    buffer = io.BytesIO()
    Image.fromarray(value).save(buffer, format='JPEG', quality=quality, subsampling=0,
                                optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        reconstruction = np.asarray(compressed.convert('RGB'), dtype=np.float32)
    residual = np.abs(value.astype(np.float32) - reconstruction).mean(axis=2) / 255.0
    height, width = value.shape[:2]
    scale = min(1.0, 512.0 / max(height, width))
    work_wh = (max(1, round(width * scale)), max(1, round(height * scale)))
    anomaly = _log_anomaly(cv2.resize(residual, work_wh, interpolation=cv2.INTER_AREA))
    result = cv2.resize(anomaly, (224, 224), interpolation=cv2.INTER_LINEAR)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite JPEG sensitivity map')
    return np.ascontiguousarray(np.maximum(result, 0), dtype=np.float32)


def safe_spearman(x, y):
    """Return N/A explicitly for insufficient or constant image-level observations."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Spearman inputs must match and be finite')
    if x.size < 3:
        return {'rho': None, 'n': int(x.size), 'reason': 'fewer than 3 images'}
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {'rho': None, 'n': int(x.size), 'reason': 'constant image-level variable'}
    rho, p = spearmanr(x, y)
    return {'rho': float(rho), 'n': int(x.size), 'descriptive_p_value': float(p)}


def texture_summary(rgb224, r):
    """Describe fixed-grid gradient quartiles; edge is >=Q75 and smooth is <=Q25."""
    if rgb224.shape != (224, 224, 3) or np.shape(r) != (224, 224):
        raise ValueError('Texture and evidence must share the 224 grid')
    gray = cv2.cvtColor(rgb224.astype(np.float32) / 255, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8,
                   borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8,
                   borderType=cv2.BORDER_REFLECT_101)
    gradient = np.hypot(dx, dy)
    lower, upper = np.quantile(gradient, [.25, .75])
    edge, smooth = gradient >= upper, gradient <= lower
    return {'texture_sobel_mean': float(gradient.mean()),
            'texture_sobel_p75': float(upper), 'texture_sobel_p25': float(lower),
            'r_edge_mean': float(r[edge].mean()), 'r_smooth_mean': float(r[smooth].mean()),
            'edge_pixel_count': int(edge.sum()), 'smooth_pixel_count': int(smooth.sum())}


def _distribution(values):
    values = np.asarray(values)
    return {'mean': float(values.mean()), 'p10': float(np.quantile(values, .1)),
            'p50': float(np.median(values)), 'p90': float(np.quantile(values, .9))}


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped['all'].append(row)
        for field in ('dataset', 'partition', 'original_format'):
            grouped[f'{field}={row[field]}'].append(row)
        grouped[f"dataset={row['dataset']}|partition={row['partition']}"].append(row)
        grouped[f"dataset={row['dataset']}|format={row['original_format']}"].append(row)
    result = {}
    for name, group in grouped.items():
        texture = [row['texture_sobel_mean'] for row in group]
        result[name] = {'image_count': len(group), 'features': {},
                        'r_edge_mean': _distribution([row['r_edge_mean'] for row in group]),
                        'r_smooth_mean': _distribution([row['r_smooth_mean'] for row in group])}
        for feature in (*FEATURES, 'r'):
            means = [row['features'][feature]['mean'] for row in group]
            result[name]['features'][feature] = {
                'image_mean_distribution': _distribution(means),
                'texture_spearman': safe_spearman(texture, means)}
    return result


def _sensitivity(row, record, normalized, stats):
    with Image.open(row['image_path']) as image:
        rgb = np.asarray(image.convert('RGB'))
    result = []
    for quality in (75, 95):
        raw = jpeg_quality_map(rgb, quality)
        alternate = normalize_evidence({**record, 'jpeg': raw}, stats)
        result.append({'dataset': row['dataset'], 'sample_id': row['sample_id'],
                       'sample_key': row['sample_key'], 'quality': quality,
                       'jpeg_normalized_mean': float(alternate['jpeg'].mean()),
                       'q90_normalized_mean': float(normalized['jpeg'].mean()),
                       'jpeg_mae_vs_q90': float(np.abs(alternate['jpeg'] - normalized['jpeg']).mean()),
                       'jpeg_spearman_vs_q90': safe_spearman(alternate['jpeg'].ravel(), normalized['jpeg'].ravel()),
                       'r_mean_delta_if_jpeg_replaced': float((alternate['r'] - normalized['r']).mean()),
                       'original_format': record['metadata']['original_format']})
    return result


def run(split_path=DEFAULT_SPLIT, cache_root=DEFAULT_CACHE,
        stats_path=ROOT / 'assets/stage3_evidence_stats.json', output=DEFAULT_OUTPUT, limit=None):
    cv2.setNumThreads(1)
    split = read_split(split_path)
    stats = json.loads(Path(stats_path).read_text())
    selected = defaultdict(list)
    for row in split['records']:
        if row['subset'] == 'teacher_diag':
            selected[row['dataset']].append(row['sample_key'])
    sensitivity_keys = {key for keys in selected.values() for key in sorted(keys)[:8]}
    rows = sorted(split['records'], key=lambda row: row['sample_key'])
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        rows = rows[:limit]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    per_image_path = output.with_name('per_image_bias.jsonl')
    per_image, sensitivity = [], []
    with per_image_path.open('w') as stream:
        for row in rows:
            record = load_evidence(evidence_path(row, cache_root), row)
            normalized = normalize_evidence(record, stats)
            with Image.open(row['image_path']) as image:
                rgb224 = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR))
            meta = record['metadata']
            entry = {key: row[key] for key in ('dataset', 'sample_id', 'sample_key')}
            entry.update({'partition': row['subset'], 'original_format': meta['original_format'],
                          'original_jpeg_quantization': meta['original_jpeg_quantization'],
                          'non_jpeg_semantic_limit': meta['original_format'] != 'JPEG',
                          'features': {name: {'mean': float(value.mean()),
                                              'p10': float(np.quantile(value, .1)),
                                              'p50': float(np.median(value)),
                                              'p90': float(np.quantile(value, .9))}
                                       for name, value in normalized.items()},
                          **texture_summary(rgb224, normalized['r'])})
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
            per_image.append(entry)
            if row['sample_key'] in sensitivity_keys:
                sensitivity.extend(_sensitivity(row, record, normalized, stats))
    sensitivity_summary = {}
    for dataset in sorted(selected):
        sensitivity_summary[dataset] = {}
        for quality in (75, 95):
            group = [row for row in sensitivity if row['dataset'] == dataset and row['quality'] == quality]
            sensitivity_summary[dataset][str(quality)] = {
                'image_count': len(group),
                **({key: _distribution([row[key] for row in group])
                    for key in ('jpeg_mae_vs_q90', 'r_mean_delta_if_jpeg_replaced')}
                   if group else {'reason': 'no selected images processed'})}
    report = {'schema_version': 'fecd-stage3-bias-v1', 'ground_truth_read': False,
              'split_manifest_sha256': split['manifest_sha256'], 'stats_hash': stats['stats_hash'],
              'processed_image_count': len(per_image), 'limit': limit,
              'per_image_path': str(per_image_path), 'aggregate': _aggregate(per_image),
              'sensitivity_selection': 'first 8 teacher_diag sample_key-sorted per dataset, fixed without GT',
              'sensitivity': sensitivity, 'sensitivity_summary': sensitivity_summary,
              'interpretation': ['Associations are descriptive; dataset/camera/codec confounding remains.',
                                 'Edge/smooth quartile sets can overlap on constant-gradient images.',
                                 'Non-JPEG residual describes counterfactual encoding distortion, not original JPEG tampering.',
                                 'Q75/Q95 are diagnostic only; Q90 stays frozen and no feature is selected here.']}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'processed': len(per_image),
                      'sensitivity_images': len(sensitivity) // 2, 'ground_truth_read': False}))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--stats', type=Path, default=ROOT / 'assets/stage3_evidence_stats.json')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.split, args.cache_root, args.stats, args.output, args.limit)


if __name__ == '__main__':
    main()
