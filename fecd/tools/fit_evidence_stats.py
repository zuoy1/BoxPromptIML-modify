"""Fit frozen robust evidence scales from train_fit raw maps only."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fecd.evidence.cache import (DEFAULT_CACHE, DEFAULT_SPLIT, evidence_path,
                                 evidence_config_hash, load_evidence, read_split)
from fecd.evidence.normalize import FEATURES, fit_stats, normalize_evidence

ROOT = Path(__file__).resolve().parents[1]


def run(split_path=DEFAULT_SPLIT, cache_root=DEFAULT_CACHE, output=ROOT / 'assets/stage3_evidence_stats.json'):
    split = read_split(split_path)
    rows = [row for row in split['records'] if row['subset'] == 'train_fit']
    key_hash = hashlib.sha256('\n'.join(sorted(row['sample_key'] for row in rows)).encode()).hexdigest()
    def maps_iter():
        for row in rows:
            record = load_evidence(evidence_path(row, cache_root), row)
            yield {name: record[name] for name in FEATURES}

    provenance = {'partition': 'train_fit', 'split_manifest_sha256': split['manifest_sha256'],
                  'sample_keys_sha256': key_hash,
                  'extractor_config_hash': evidence_config_hash(),
                  'source_split_manifest': str(Path(split_path)),
                  'ground_truth_read': False}
    stats = fit_stats(maps_iter(), provenance)
    summaries = defaultdict(lambda: defaultdict(list))
    for row in rows:
        record = load_evidence(evidence_path(row, cache_root), row)
        normalized = normalize_evidence(record, stats)
        for name in (*FEATURES, 'r'):
            summaries[row['dataset']][name].append(float(normalized[name].mean()))
    distributions = {}
    for dataset, feature_summaries in summaries.items():
        distributions[dataset] = {}
        for feature, image_means in feature_summaries.items():
            means = np.asarray(image_means)
            distributions[dataset][feature + '_per_image_mean'] = {
                'image_count': int(len(means)), 'mean': float(means.mean()),
                'p50': float(np.quantile(means, .5)), 'p95': float(np.quantile(means, .95))}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + '\n')
    distribution_path = output.with_name(output.stem + '_train_fit_distributions.json')
    distribution_path.write_text(json.dumps({'stats_hash': stats['stats_hash'],
                                            'partition': 'train_fit',
                                            'definition': 'per-image means of frozen normalized maps',
                                            'datasets': distributions}, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'stats_hash': stats['stats_hash'],
                      'train_fit_count': len(rows), 'sample_keys_sha256': key_hash,
                      'feature_stats': stats['features'],
                      'distribution_path': str(distribution_path),
                      'dataset_counts': {k: v['hp_per_image_mean']['image_count'] for k, v in distributions.items()}},
                     indent=2, ensure_ascii=False))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/stage3_evidence_stats.json')
    args = parser.parse_args()
    run(args.split, args.cache_root, args.output)


if __name__ == '__main__':
    main()
