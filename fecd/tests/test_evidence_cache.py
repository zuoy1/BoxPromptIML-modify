"""Evidence cache identity, integrity and GT-free fitting checks."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fecd.evidence.cache import build_record, generate_record, load_evidence, save_evidence, read_split
from fecd.evidence.normalize import fit_stats
from fecd.pseudo.cache_schema import canonical_hash, make_sample_key


class EvidenceCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'image.png'
        Image.fromarray(np.full((24, 32, 3), 100, np.uint8)).save(self.image)
        digest = hashlib.sha256(self.image.read_bytes()).hexdigest()
        self.row = {'dataset': 'test', 'sample_id': '1', 'image_sha256': digest,
                    'image_path': str(self.image), 'subset': 'train_fit'}
        self.row['sample_key'] = make_sample_key('test', 'train', '1', digest)
        self.maps = {name: np.ones((224, 224), np.float32) for name in ('hp', 'noise', 'jpeg', 'fft')}
        self.record = build_record(self.maps, self.row, {'native_hw': [24, 32], 'original_format': 'PNG'})

    def test_cache_valid_identity_roundtrip_preserves_maps(self):
        path = self.root / 'cache.npz'
        save_evidence(path, self.record)
        loaded = load_evidence(path, self.row)
        for name in self.maps:
            np.testing.assert_array_equal(loaded[name], self.maps[name])
        self.assertEqual(loaded['metadata']['sample_key'], self.row['sample_key'])

    def test_cache_wrong_identity_or_config_fails(self):
        path = self.root / 'cache.npz'
        save_evidence(path, self.record)
        wrong = {**self.row, 'sample_id': '2'}
        with self.assertRaises(ValueError):
            load_evidence(path, wrong)
        changed = copy.deepcopy(self.record)
        changed['metadata']['extractor_config_hash'] = 'bad'
        with self.assertRaises(ValueError):
            save_evidence(path, changed)

    def test_cache_corrupt_nonfinite_map_fails(self):
        changed = copy.deepcopy(self.record)
        changed['noise'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            save_evidence(self.root / 'cache.npz', changed)

    def test_cache_finite_tampered_map_hash_fails(self):
        changed = copy.deepcopy(self.record)
        changed['noise'][0, 0] = 2.
        with self.assertRaises(ValueError):
            save_evidence(self.root / 'cache.npz', changed)

    def test_source_file_changed_since_freeze_is_rejected(self):
        self.image.write_bytes(b'changed after manifest freeze')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            generate_record(self.row, {}, self.root)

    def test_fitting_teacher_diag_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_stats([self.maps], {'partition': 'teacher_diag'})

    def test_split_manifest_mutation_is_rejected(self):
        payload = {'records': [self.row]}
        payload['records_sha256'] = canonical_hash(payload['records'])
        payload['manifest_sha256'] = canonical_hash(payload)
        path = self.root / 'split.json'
        path.write_text(json.dumps(payload))
        self.assertEqual(read_split(path)['records'], [self.row])
        payload['records'][0]['subset'] = 'teacher_diag'
        path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            read_split(path)


if __name__ == '__main__':
    unittest.main()
