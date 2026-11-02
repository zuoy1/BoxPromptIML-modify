"""Derived roles preserve source groups, parent identities and frozen bytes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fecd.diagnostics.stage3ra_split import build_split, validate_structure, freeze_json
from fecd.pseudo.cache_schema import canonical_hash, make_sample_key


def parent_fixture():
    rows = []
    for index in range(20):
        digest = f'{index:064x}'
        rows.append({'dataset': 'CASIAv2', 'sample_id': str(index),
                     'sample_key': make_sample_key('CASIAv2', 'train', str(index), digest),
                     'image_sha256': digest, 'image_path': f'/image/{index}.png',
                     'source_group_id': 'giant' if index < 8 else str(index),
                     'source_group_proxy': str(index), 'source_group_method': 'fixture',
                     'subset': 'train_fit' if index < 18 else 'teacher_diag'})
    parent = {'records': rows, 'records_sha256': canonical_hash(rows)}
    parent['manifest_sha256'] = canonical_hash(parent)
    return parent


def resign(payload):
    payload['records_sha256'] = canonical_hash(payload['records'])
    for role in payload['role_membership_sha256']:
        payload['role_membership_sha256'][role] = canonical_hash(sorted(
            row['sample_key'] for row in payload['records'] if row['subset'] == role))
    payload['manifest_sha256'] = canonical_hash({k: v for k, v in payload.items() if k != 'manifest_sha256'})


class Stage3raSplitTests(unittest.TestCase):
    def test_repeatable_complete_disjoint_roles_preserve_parent(self):
        parent = parent_fixture()
        before = json.dumps(parent, sort_keys=True)
        result = build_split(parent)
        self.assertEqual(result, build_split(parent))
        self.assertEqual(before, json.dumps(parent, sort_keys=True))
        self.assertEqual(len(result['records']), 18)
        self.assertEqual(set(r['subset'] for r in result['records']),
                         {'student_train', 'new_dev', 'new_confirm'})
        self.assertEqual({r['subset'] for r in result['records']
                          if r['source_group_id'] == 'giant'}, {'student_train'})
        self.assertTrue(validate_structure(result, parent))

    def test_rehashed_modified_parent_identity_is_rejected(self):
        parent = parent_fixture()
        result = build_split(parent)
        result['records'][0]['image_path'] = '/changed.png'
        resign(result)
        with self.assertRaises(ValueError):
            validate_structure(result, parent)

    def test_crossing_group_and_wrong_assignment_are_rejected(self):
        parent = parent_fixture()
        result = build_split(parent)
        result['records'][0]['subset'] = 'new_confirm'
        resign(result)
        with self.assertRaises(ValueError):
            validate_structure(result, parent)

    def test_duplicate_image_across_roles_is_rejected(self):
        parent = parent_fixture()
        result = build_split(parent)
        dev = next(r for r in result['records'] if r['subset'] == 'new_dev')
        train = next(r for r in result['records'] if r['subset'] == 'student_train')
        dev['image_sha256'] = train['image_sha256']
        resign(result)
        with self.assertRaises(ValueError):
            validate_structure(result, parent)

    def test_frozen_output_reuses_identical_and_rejects_different(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'frozen.json'
            freeze_json(path, {'value': 1})
            original = path.read_bytes()
            freeze_json(path, {'value': 1})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                freeze_json(path, {'value': 2})
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
