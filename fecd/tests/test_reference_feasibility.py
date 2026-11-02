"""GT-free runner gates and deterministic selection contracts."""
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from tempfile import TemporaryDirectory

from fecd.tools.audit_reference_feasibility import choose_rows, ReadGuard


class ReferenceFeasibilityTests(unittest.TestCase):
    def test_selection_uses_only_new_dev_sorted_keys_per_domain(self):
        rows = [{'subset':role,'dataset':d,'sample_key':f'{i:03}'}
                for d in ('A','B') for role in ('new_dev','new_confirm','student_train')
                for i in range(20,-1,-1)]
        chosen = choose_rows(rows)
        self.assertEqual(len(chosen),32)
        self.assertTrue(all(x['subset']=='new_dev' for x in chosen))
        self.assertEqual([x['sample_key'] for x in chosen[:16]], [f'{i:03}' for i in range(16)])

    def test_guard_allows_initialized_rgb_and_numpy_cache_readers(self):
        with TemporaryDirectory() as directory:
            image=Path(directory)/'rgb.png'; cache=Path(directory)/'cache.npz'
            Image.fromarray(np.zeros((8,12,3),np.uint8)).save(image)
            np.savez_compressed(cache, value=np.zeros(2))
            with ReadGuard([image,cache]) as guard:
                with Image.open(image) as rgb:
                    self.assertEqual(rgb.convert('RGB').size,(12,8))
                with np.load(cache,allow_pickle=False) as data:
                    self.assertEqual(data['value'].tolist(),[0,0])
            self.assertEqual(guard.denied,[])

    def test_read_guard_rejects_unlisted_file_before_open(self):
        with TemporaryDirectory() as directory:
            allowed=Path(directory)/'rgb.bin'; forbidden=Path(directory)/'other.bin'
            allowed.write_bytes(b'rgb'); forbidden.write_bytes(b'not allowed')
            guard=ReadGuard([allowed])
            with guard:
                self.assertEqual(allowed.read_bytes(),b'rgb')
                with self.assertRaises(PermissionError): forbidden.read_bytes()
            self.assertEqual(len(guard.opens),1)
            self.assertEqual(len(guard.denied),1)


if __name__=='__main__':
    unittest.main()
