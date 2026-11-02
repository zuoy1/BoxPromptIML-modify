"""Deterministic image-only derivation and integrity gate for route-A roles."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fecd.evidence.cache import DEFAULT_SPLIT as PARENT_PATH, PROJECT, SAM_ROOT
from fecd.evidence.cache import read_split as read_parent
from fecd.pseudo.cache_schema import canonical_hash, load_cache, make_sample_key

VERSION = 'fecd-stage3r-proposed-v1'
SCHEMA = 'fecd-stage3ra-split-v1'
SEED = 20261029
PARENT_HASH = '7f18b459cded7cc2a8577167c5e2958cfac715691217a32bf457c8ae611cd66e'
DEFAULT_SPLIT = PROJECT / 'fecd/assets/stage3ra_split_manifest.json'
DEFAULT_AUDIT = PROJECT / 'fecd/assets/stage3ra_split_audit.json'
ROLES = ('student_train', 'new_dev', 'new_confirm')
GIANT = '07cc57f1a6ed7e53fbd482c40f5a726515fcb519715a85bc8ed710cc8027e2de'
K8_CONFIG = '2bff688546a79079dcbac4ae29bb2c5169823704025c69b746a8cb57df99aafd'


def _assign(parent):
    groups = defaultdict(list)
    for row in parent['records']:
        if row['subset'] == 'train_fit':
            groups[(row['dataset'], row['source_group_id'])].append(row)
    assignments = {group: 'student_train' for group in groups}
    selected = set()
    for dataset in sorted({group[0] for group in groups}):
        total = sum(len(rows) for group, rows in groups.items() if group[0] == dataset)
        goal = round(total * (.1 if dataset == 'CASIAv2' else .25))
        for role in ('new_confirm', 'new_dev'):
            candidates = [(hashlib.sha256(f'{VERSION}:{SEED}:{role}:{dataset}:{group[1]}'.encode()).hexdigest(),
                           group, len(rows)) for group, rows in groups.items()
                          if group[0] == dataset and group not in selected and len(rows) <= goal]
            states = {0: ()}
            for _, group, size in sorted(candidates):
                for count, chosen in list(states.items())[::-1]:
                    new_count = count + size
                    if new_count <= goal and new_count not in states:
                        states[new_count] = chosen + (group,)
            for group in states[max(states)]:
                assignments[group] = role
                selected.add(group)
    return assignments


def build_split(parent):
    assignments = _assign(parent)
    records = [{**row, 'parent_subset': row['subset'],
                'subset': assignments[(row['dataset'], row['source_group_id'])]}
               for row in parent['records'] if row['subset'] == 'train_fit']
    payload = {'schema_version': SCHEMA, 'assignment_version': VERSION, 'seed': SEED,
               'parent_manifest_sha256': parent['manifest_sha256'],
               'parent_records_sha256': parent['records_sha256'],
               'new_confirm_gt_access': 'closed; no authorization in stage3ra',
               'role_membership_sha256': {role: canonical_hash(sorted(
                   row['sample_key'] for row in records if row['subset'] == role)) for role in ROLES},
               'records': records, 'records_sha256': canonical_hash(records)}
    payload['manifest_sha256'] = canonical_hash(payload)
    validate_structure(payload, parent)
    return payload


def validate_structure(payload, parent):
    actual = canonical_hash({k: v for k, v in payload.items() if k != 'manifest_sha256'})
    if payload.get('manifest_sha256') != actual or payload.get('records_sha256') != canonical_hash(payload['records']):
        raise ValueError('Derived manifest hash mismatch')
    if (payload.get('schema_version') != SCHEMA or payload.get('assignment_version') != VERSION
            or payload.get('seed') != SEED or payload.get('parent_manifest_sha256') != parent['manifest_sha256']
            or payload.get('parent_records_sha256') != parent['records_sha256']):
        raise ValueError('Derived split parent/version mismatch')
    expected = {row['sample_key']: row for row in parent['records'] if row['subset'] == 'train_fit'}
    records = payload['records']
    keys = [row['sample_key'] for row in records]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError('Derived roles do not exactly cover unique parent train_fit members')
    assignments = _assign(parent)
    grouped_roles, image_roles = defaultdict(set), defaultdict(set)
    for row in records:
        key = row['sample_key']
        original = {k: v for k, v in row.items() if k not in ('parent_subset', 'subset')}
        if {**original, 'subset': row['parent_subset']} != expected[key]:
            raise ValueError('Derived split changed parent identity or metadata')
        group = (row['dataset'], row['source_group_id'])
        if row['subset'] != assignments[group]:
            raise ValueError('Role does not match frozen deterministic algorithm')
        if key != make_sample_key(row['dataset'], 'train', row['sample_id'], row['image_sha256']):
            raise ValueError('Invalid source sample identity')
        grouped_roles[group].add(row['subset'])
        image_roles[row['image_sha256']].add(row['subset'])
    for row in parent['records']:
        if row['subset'] == 'teacher_diag':
            grouped_roles[(row['dataset'], row['source_group_id'])].add('teacher_diag')
            image_roles[row['image_sha256']].add('teacher_diag')
    if any(len(roles) != 1 for roles in grouped_roles.values()) or any(len(roles) != 1 for roles in image_roles.values()):
        raise ValueError('Source group or identical image crosses roles')
    for role in ROLES:
        membership = canonical_hash(sorted(row['sample_key'] for row in records if row['subset'] == role))
        if payload['role_membership_sha256'].get(role) != membership:
            raise ValueError('Role membership hash mismatch')
    giant = [row for row in records if row['source_group_id'] == GIANT]
    if giant and (len(giant) != 3020 or any(row['subset'] != 'student_train' for row in giant)):
        raise ValueError('CASIAv2 giant component was changed or split')
    return True


def freeze_json(path, payload):
    path = Path(path)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Frozen output differs; refusing overwrite: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def read_split(path=DEFAULT_SPLIT, audit_path=DEFAULT_AUDIT):
    parent = read_parent(PARENT_PATH)
    if parent['manifest_sha256'] != PARENT_HASH:
        raise ValueError('Historical parent is not the accepted frozen version')
    payload = json.loads(Path(path).read_text())
    validate_structure(payload, parent)
    audit = json.loads(Path(audit_path).read_text())
    audit_hash = canonical_hash({k: v for k, v in audit.items() if k != 'audit_sha256'})
    if (audit.get('status') != 'passed' or audit.get('manifest_sha256') != payload['manifest_sha256']
            or audit.get('audit_sha256') != audit_hash or audit.get('new_confirm_gt_access_count') != 0):
        raise ValueError('A0 identity gate is missing, invalid or not passed')
    return payload


def audit_images(payload, parent):
    index_rows = json.loads((SAM_ROOT / 'dataset_index.json').read_text())
    index = {row['sample_key']: row for row in index_rows}
    if len(index) != len(index_rows) or set(index) != {r['sample_key'] for r in parent['records']}:
        raise ValueError('Frozen SAM index membership mismatch')
    summary = defaultdict(lambda: {'images': 0, 'groups': set(), 'formats': Counter(),
                                   'height': [], 'width': [], 'aspect_ratio': []})
    for row in payload['records']:
        sam_row = index[row['sample_key']]
        if any(sam_row[key] != row[key] for key in ('dataset', 'sample_id', 'image_path')) or sam_row['split'] != 'train':
            raise ValueError('SAM index identity/path mismatch')
        image_path = Path(row['image_path'])
        contents = image_path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != row['image_sha256']:
            raise ValueError(f'Image SHA256 mismatch: {row["sample_key"]}')
        sha1 = hashlib.sha1(contents).hexdigest()
        if sha1 != sam_row['image_sha1']:
            raise ValueError('Image/SAM SHA1 mismatch')
        cache_path = Path(sam_row['cache_path'])
        if not cache_path.is_absolute():
            cache_path = PROJECT / cache_path
        if not cache_path.resolve().is_relative_to(SAM_ROOT.resolve()):
            raise ValueError('SAM index cache path leaves frozen K8 directory')
        sam = load_cache(cache_path, K8_CONFIG, expected={**row, 'split': 'train', 'image_sha1': sha1,
                                             'box_hash': sam_row['box_hash'], 'k': 8})
        with Image.open(image_path) as image:
            native_hw = [image.height, image.width]
            original_format = image.format or 'unknown'
            image.verify()
        if native_hw != [int(sam['image_h']), int(sam['image_w'])]:
            raise ValueError('Native image/SAM dimensions differ')
        stats = summary[(row['dataset'], row['subset'])]
        stats['images'] += 1
        stats['groups'].add(row['source_group_id'])
        stats['formats'][original_format] += 1
        stats['height'].append(native_hw[0])
        stats['width'].append(native_hw[1])
        stats['aspect_ratio'].append(native_hw[1] / native_hw[0])
    output = {}
    for (dataset, role), stats in sorted(summary.items()):
        output.setdefault(dataset, {})[role] = {
            'images': stats['images'], 'source_groups': len(stats['groups']),
            'original_formats': dict(sorted(stats['formats'].items())),
            'native_distribution': {key: dict(zip(('min', 'median', 'max'),
                map(float, np.quantile(stats[key], [0, .5, 1])))) for key in ('height', 'width', 'aspect_ratio')}}
    return output


def prepare(output=DEFAULT_SPLIT, audit_output=DEFAULT_AUDIT):
    parent = read_parent(PARENT_PATH)
    if parent['manifest_sha256'] != PARENT_HASH:
        raise ValueError('Historical parent hash mismatch')
    payload = build_split(parent)
    counts = Counter(row['subset'] for row in payload['records'])
    if dict(counts) != {'student_train': 3530, 'new_dev': 530, 'new_confirm': 530}:
        raise ValueError(f'Derived role counts differ from accepted protocol: {counts}')
    dataset_roles = audit_images(payload, parent)
    audit = {'status': 'passed', 'manifest_sha256': payload['manifest_sha256'],
             'parent_manifest_sha256': parent['manifest_sha256'], 'counts': dict(counts),
             'dataset_roles': dataset_roles, 'verified_images': len(payload['records']),
             'verified_k8_cache_records': len(payload['records']), 'source_group_crossings': 0,
             'identical_image_crossings': 0, 'old_teacher_diag_unchanged': True,
             'gt_files_opened': 0, 'new_confirm_gt_access_count': 0,
             'limitations': ['dataset-scoped source proxies are not verified real provenance',
                            '3020-image CASIAv2 component excluded from development and confirmation',
                            'new confirmation previously observed without diagnostic pixel GT',
                            'small NIST16/Columbia/Coverage source-group counts limit inference']}
    audit['audit_sha256'] = canonical_hash(audit)
    freeze_json(output, payload)
    freeze_json(audit_output, audit)
    read_split(output, audit_output)
    return audit
