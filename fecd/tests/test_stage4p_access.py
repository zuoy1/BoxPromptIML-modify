"""Training read boundary rejects nonstudent image/cache paths before IO."""
import unittest

from fecd.training.access import DataReadAudit


class AccessTests(unittest.TestCase):
    def test_allowlisted_rgb_and_cache_are_recorded(self):
        rgb='/home/datasets/raw/only-student.png'
        cache='/tmp/fecd/assets/stage2c_cache/own.npz'
        with DataReadAudit([rgb],[cache]) as guard:
            guard._event('open',(rgb,'r',0))
            guard._event('open',(cache,'r',0))
        self.assertEqual(guard.opened,[rgb,cache])
        self.assertEqual(guard.denied,[])

    def test_unlisted_image_cache_and_writes_are_denied(self):
        rgb='/home/datasets/raw/only-student.png'
        with DataReadAudit([rgb],[]) as guard:
            for path,mode in [('/home/datasets/unapproved.png','r'),
                              ('/tmp/fecd/assets/stage2c_cache/other.npz','r'),(rgb,'w')]:
                with self.assertRaises(PermissionError):guard._event('open',(path,mode,0))
        self.assertEqual(len(guard.denied),3)


if __name__=='__main__':
    unittest.main()
