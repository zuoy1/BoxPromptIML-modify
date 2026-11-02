"""GT access must require exact membership in the frozen development allowlist."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from fecd.tools.audit_stage3ra import load_dev_gt


class Stage3RAGTTests(unittest.TestCase):
    def test_confirmation_and_forged_members_are_rejected_before_file_access(self):
        row = dict(sample_key='confirm', subset='new_confirm', dataset='Toy',
                   sample_id='one', image_sha256='sha')
        source = dict(dataset='Toy', id='one', split='train', image_sha256='sha',
                      prepared_mask='/missing/never-open-this.png')
        with TemporaryDirectory() as directory:
            log = Path(directory) / 'access.jsonl'
            for candidate in (row, {**row, 'subset': 'new_dev'}):
                with self.assertRaises(ValueError):
                    load_dev_gt(candidate, source, (2, 2), {}, log)
            self.assertFalse(log.exists())

    def test_development_mask_uses_binary_nearest_mapping_and_access_log(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mask = root / 'mask.png'
            Image.fromarray(np.array([[0, 255], [0, 255]], np.uint8)).save(mask)
            row = dict(sample_key='dev', subset='new_dev', dataset='Toy',
                       sample_id='one', image_sha256='sha')
            source = dict(dataset='Toy', id='one', split='train', image_sha256='sha',
                          prepared_mask=str(mask))
            gt = load_dev_gt(row, source, (2, 2), {'dev': row}, root / 'access.jsonl')
            self.assertEqual(gt.shape, (224, 224))
            self.assertFalse(gt[:, :112].any())
            self.assertTrue(gt[:, 112:].all())
            accesses = [json.loads(x) for x in (root / 'access.jsonl').read_text().splitlines()]
            self.assertEqual(len(accesses), 1)
            self.assertEqual(accesses[0]['sample_key'], 'dev')
            self.assertEqual(accesses[0]['partition'], 'new_dev')
            self.assertEqual(accesses[0]['prepared_mask_sha256'],
                             hashlib.sha256(mask.read_bytes()).hexdigest())

    def test_wrong_dimensions_nonbinary_or_source_identity_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mask = root / 'mask.png'
            Image.fromarray(np.full((2, 2), 128, np.uint8)).save(mask)
            row = dict(sample_key='dev', subset='new_dev', dataset='Toy',
                       sample_id='one', image_sha256='sha')
            source = dict(dataset='Toy', id='one', split='train', image_sha256='sha',
                          prepared_mask=str(mask))
            for native in ((3, 2), (2, 2)):
                with self.assertRaises(ValueError):
                    load_dev_gt(row, source, native, {'dev': row}, root / 'log.jsonl')
            with self.assertRaises(ValueError):
                load_dev_gt(row, {**source, 'id': 'other'}, (2, 2), {'dev': row},
                            root / 'log.jsonl')


if __name__ == '__main__':
    unittest.main()
