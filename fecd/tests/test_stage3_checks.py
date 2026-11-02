"""The scoped static analyzer must expose missing names and GT-path fields."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fecd.tools.check_stage3 import analyze_source


class Stage3ChecksTest(unittest.TestCase):
    def test_missing_global_is_reported_and_builtins_are_allowed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.py'
            path.write_text('def calculate(values):\n    return sum(values) + missing_name\n')
            issues = analyze_source(path)
            self.assertEqual(len(issues), 1)
            self.assertIn('missing_name', issues[0])

    def test_image_evidence_rejects_ground_truth_path_field(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence' / 'sample.py'
            path.parent.mkdir()
            path.write_text("FIELD = 'prepared_mask'\n")
            self.assertTrue(any('GT-path' in issue for issue in analyze_source(path)))


if __name__ == '__main__':
    unittest.main()
