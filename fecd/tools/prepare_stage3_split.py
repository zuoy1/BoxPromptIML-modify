"""Freeze a source-proxy-disjoint internal train_fit/teacher_diag split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = Path('/home/datasets/processed/boxprompt_paper_scale_20261016')
SEED = 20261029
TARGET_FRACTION = 0.20


def source_tokens(row: dict) -> tuple[list[str], str]:
    dataset, sample_id = row['dataset'], row['sample_id']
    if dataset == 'CASIAv2':
        tokens = re.findall(r'[a-z]{3}\d{5}', sample_id.lower())
        if len(tokens) == 2:
            return tokens, 'filename donor-token proxy; connected components'
    elif dataset == 'NIST16':
        group = str(row.get('group') or '')
        if group:
            return [group], 'manifest base-source group'
    elif dataset == 'Columbia':
        match = re.search(r'_sub_(\d+)$', sample_id)
        if match:
            return [f'scene-{int(match.group(1)):02d}'], 'scene-number proxy; camera pair excluded'
    elif dataset == 'Coverage':
        group = str(row.get('group') or '')
        if group:
            return [group], 'manifest source-image group'
    return [f'unresolved:{dataset}/{sample_id}'], 'unresolved; singleton identity only, not source isolation'


def component_groups(rows: list[dict]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    row_tokens: dict[str, list[str]] = {}
    for row in rows:
        identity = row['sample_key']
        tokens, _ = source_tokens(row)
        nodes = [f"{row['dataset']}::{token}" for token in tokens]
        row_tokens[identity] = nodes
        for node in nodes:
            find(node)
        for node in nodes[1:]:
            parent[find(node)] = find(nodes[0])
    members: dict[str, list[str]] = defaultdict(list)
    for identity, nodes in row_tokens.items():
        members[find(nodes[0])].append(identity)
    return {identity: hashlib.sha256('|'.join(sorted(members[find(nodes[0])])).encode()).hexdigest()
            for identity, nodes in row_tokens.items()}


def assign_groups(rows: list[dict]) -> dict[str, str]:
    group_ids = component_groups(rows)
    members = defaultdict(list)
    lookup = {r['sample_key']: r for r in rows}
    for identity, group in group_ids.items():
        members[group].append(identity)
    target = {ds: round(sum(r['dataset'] == ds for r in rows) * TARGET_FRACTION)
              for ds in sorted({r['dataset'] for r in rows})}
    selected: set[str] = set()
    for dataset, goal in target.items():
        candidates = []
        for root, identities in members.items():
            ds_ids = [i for i in identities if lookup[i]['dataset'] == dataset]
            if ds_ids and len(ds_ids) <= goal:
                tie = hashlib.sha256(f'{SEED}:{dataset}:{root}'.encode()).hexdigest()
                candidates.append((tie, root, len(ds_ids)))
        # Deterministic subset-sum, preferring a count nearest the per-dataset target.
        candidates.sort()
        states: dict[int, tuple[str, ...]] = {0: ()}
        for tie, root, size in candidates:
            for count, chosen in list(states.items())[::-1]:
                new_count = count + size
                if new_count <= goal and new_count not in states:
                    states[new_count] = chosen + (root,)
        best = max(states, key=lambda n: (n, tuple(reversed(states[n]))))
        selected.update(states[best])
    return {identity: ('teacher_diag' if group in selected else 'train_fit')
            for identity, group in group_ids.items()}


def prepare(manifest_root: Path = MANIFEST_ROOT, output: Path = ROOT / 'assets/stage3_split_manifest.json') -> dict:
    rows = list(csv.DictReader((manifest_root / 'train_manifest.csv').open(newline='', encoding='utf-8')))
    records = json.loads((manifest_root / 'manifest.json').read_text())
    metadata = {(r['dataset'], r['id']): r for r in records}
    joined = []
    for row in rows:
        source = metadata[(row['dataset'], row['id'])]
        if row['split'] != 'train' or source.get('split') != 'train':
            raise ValueError('stage3 split accepts locked train records only')
        item = {'dataset': row['dataset'], 'sample_id': row['id'],
                'sample_key': hashlib.sha256('\0'.join((row['dataset'], 'train', row['id'], source['image_sha256'])).encode()).hexdigest(),
                'image_sha256': source['image_sha256'], 'group': str(source.get('group') or ''),
                'image_path': source['image'].replace('/hy-tmp/datasets/', '/home/datasets/')}
        tokens, method = source_tokens(item)
        item['source_group_proxy'] = '|'.join(tokens)
        item['source_group_method'] = method
        joined.append(item)
    assignment = assign_groups(joined)
    group_ids = component_groups(joined)
    for item in joined:
        item['subset'] = assignment[item['sample_key']]
        item['source_group_id'] = group_ids[item['sample_key']]
    payload = {'schema_version': 'fecd-stage3-split-v1', 'seed': SEED,
               'target_teacher_diag_fraction': TARGET_FRACTION,
               'source_group_policy': 'dataset-scoped source proxies; transitive union of CASIAv2 donor tokens; no identity invented for unresolved records',
               'parent_manifest': str(manifest_root / 'train_manifest.csv'),
               'records': sorted(joined, key=lambda x: (x['dataset'], x['sample_id']))}
    payload['records_sha256'] = hashlib.sha256(json.dumps(payload['records'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    payload['manifest_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-root', type=Path, default=MANIFEST_ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/stage3_split_manifest.json')
    args = parser.parse_args()
    payload = prepare(args.manifest_root, args.output)
    counts = defaultdict(lambda: defaultdict(int))
    for row in payload['records']:
        counts[row['dataset']][row['subset']] += 1
    print(json.dumps({'path': str(args.output), 'manifest_sha256': payload['manifest_sha256'],
                      'records_sha256': payload['records_sha256'], 'counts': counts}, indent=2))


if __name__ == '__main__':
    main()
