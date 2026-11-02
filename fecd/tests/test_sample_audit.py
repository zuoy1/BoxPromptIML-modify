"""Behavior of array-only teacher diagnosis and candidate selection."""
import inspect
import unittest

import numpy as np

from fecd.diagnostics import sample_audit
from fecd.diagnostics.sample_audit import audit_sample, confidence_methods, region_masks


class SampleAuditTest(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((224, 224), bool)
        self.gt[50:100, 50:100] = True
        self.mu = np.zeros((224, 224), np.float32)
        self.mu[50:100, 50:110] = 1
        self.maps = {name: np.full((224, 224), .4, np.float32)
                     for name in ('noise', 'jpeg', 'fft', 'hp', 'r')}
        self.boxes = np.zeros((3, 224, 224), np.uint8)
        self.boxes[0, 40:120, 40:120] = 1

    def test_rule_methods_leave_negative_confidence_identical(self):
        pred = self.mu >= .5
        methods = confidence_methods(np.ones_like(self.mu), self.maps, pred)
        for method in ('S1', 'S2', 'S3', 'S4', 'S_HP'):
            np.testing.assert_array_equal(methods[method][~pred], methods['S0'][~pred])
        np.testing.assert_allclose(methods['S4'][pred], .7)
        np.testing.assert_allclose(methods['S5'][~pred], .6)

    def test_regions_box_partition_and_high_consensus(self):
        pred = self.mu >= .5
        regions = region_masks(pred, np.ones_like(self.mu), np.zeros_like(self.mu), self.boxes)
        partition = sum(regions[k].astype(int) for k in ('box_inside', 'box_edge', 'box_ring', 'box_far'))
        np.testing.assert_array_equal(partition, np.ones_like(self.mu))
        np.testing.assert_array_equal(regions['high_pos'], pred)

    def test_sample_audit_detects_false_positives_and_matching_negative_methods(self):
        row, hist = audit_sample(self.mu, np.zeros_like(self.mu), self.mu,
                                 self.mu, self.boxes, self.maps, self.gt)
        self.assertEqual(row['segmentation']['K8']['fp'], 500)
        self.assertEqual(row['segmentation']['K8']['fn'], 0)
        self.assertEqual(row['regions']['pos']['S0']['n'], 3000)
        self.assertEqual(row['regions']['pos']['S0']['error_count'], 500)
        for method in ('S1', 'S2', 'S3', 'S4'):
            self.assertEqual(row['regions']['neg'][method], row['regions']['neg']['S0'])
        np.testing.assert_array_equal(hist[('pos', 'S0')].sum(axis=1), [3000, 500])

    def test_small_coverage_is_na_with_reason(self):
        self.mu[:] = 0
        self.mu[0, :5] = 1
        row, _ = audit_sample(self.mu, np.zeros_like(self.mu), self.mu,
                              self.mu, self.boxes, self.maps, self.gt)
        coverage = row['regions']['pos']['S0']['coverage']['0.5']
        self.assertIsNone(coverage['risk'])
        self.assertIn('10', coverage['reason'])

    def test_array_only_module_has_no_gt_file_access(self):
        source = inspect.getsource(sample_audit)
        for token in ('Image.open', 'read_text(', 'np.load(', 'prepared_mask', 'gt_root'):
            self.assertNotIn(token, source)


if __name__ == '__main__':
    unittest.main()
