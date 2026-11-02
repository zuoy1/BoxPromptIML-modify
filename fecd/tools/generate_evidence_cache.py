"""Generate GT-free evidence cache and verify alignment with frozen SAM v2."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fecd.evidence.cache import (DEFAULT_CACHE, DEFAULT_SPLIT, aligned_sam_index,
                                 evidence_config_hash, generate_record, read_split)


def run(split_path=DEFAULT_SPLIT, cache_root=DEFAULT_CACHE, subset='all', limit=None,
        workers=2, validate_only=False):
    split = read_split(split_path)
    sam_index = aligned_sam_index(split)
    rows = [row for row in split['records'] if subset == 'all' or row['subset'] == subset]
    if limit is not None:
        rows = rows[:limit]
    outcomes = []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_record, row, sam_index[row['sample_key']], cache_root,
                               validate_only): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                outcomes.append(future.result())
            except Exception as exc:
                failures.append({'sample_key': row['sample_key'], 'identity': f"{row['dataset']}/{row['sample_id']}",
                                 'error': f'{type(exc).__name__}: {exc}'})
    report = {'schema_version': 'fecd-evidence-cache-audit-v1', 'subset': subset,
              'requested': len(rows), 'succeeded': len(outcomes), 'failed': failures,
              'config_hash': evidence_config_hash(), 'split_manifest_sha256': split['manifest_sha256'],
              'sample_keys_sha256': hashlib.sha256('\n'.join(sorted(r['sample_key'] for r in rows)).encode()).hexdigest(),
              'cache_root': str(cache_root), 'validate_only': validate_only,
              'reused': sum(x['reused'] for x in outcomes), 'written': sum(not x['reused'] for x in outcomes),
              'dataset_counts': dict(Counter(x['dataset'] for x in outcomes)),
              'original_format_counts': dict(Counter(x['original_format'] for x in outcomes))}
    report_path = Path(cache_root).parent / 'stage3_evidence_cache_audit.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({**report, 'audit_path': str(report_path)}, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--subset', choices=('train_fit', 'teacher_diag', 'all'), default='all')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.workers < 1:
        parser.error('--workers must be positive')
    run(args.split, args.cache_root, args.subset, args.limit, args.workers, args.validate_only)


if __name__ == '__main__':
    main()
