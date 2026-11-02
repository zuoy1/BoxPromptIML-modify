"""GT-free codec-sensitivity checks; these do not test real forgery detection."""
import unittest

import numpy as np

from fecd.evidence.extractors import extract_evidence
from fecd.tools.stage3_bias_checks import jpeg_quality_map, safe_spearman, texture_summary


class Stage3BiasTests(unittest.TestCase):
    def test_q90_reproduces_frozen_extractor(self):
        rgb = np.random.default_rng(17).integers(0, 256, (93, 151, 3), np.uint8)
        expected = extract_evidence(rgb)['jpeg']
        np.testing.assert_array_equal(jpeg_quality_map(rgb, 90), expected)
        np.testing.assert_array_equal(jpeg_quality_map(rgb, 90), expected)

    def test_jpeg_quality_changes_residual_on_textured_image(self):
        rgb = np.random.default_rng(13).integers(0, 256, (99, 149, 3), np.uint8)
        q75, q95 = jpeg_quality_map(rgb, 75), jpeg_quality_map(rgb, 95)
        self.assertGreater(float(np.abs(q75 - q95).mean()), 1e-4)

    def test_singleton_codec_maps_and_texture_remain_finite(self):
        for shape in ((1, 1), (1, 9), (9, 1)):
            rgb = np.full((*shape, 3), 128, np.uint8)
            for quality in (75, 90, 95):
                value = jpeg_quality_map(rgb, quality)
                self.assertEqual(value.shape, (224, 224))
                self.assertTrue(np.isfinite(value).all())
                self.assertLess(float(value.max()), 1e-4)
        stats = texture_summary(np.full((224, 224, 3), 128, np.uint8), np.zeros((224, 224)))
        self.assertEqual(stats['texture_sobel_mean'], 0)
        self.assertEqual(stats['r_edge_mean'], 0)

    def test_spearman_marks_constant_or_small_samples_na(self):
        self.assertIsNone(safe_spearman([1], [2])['rho'])
        self.assertIsNone(safe_spearman([1, 1, 1], [2, 3, 4])['rho'])
        self.assertAlmostEqual(safe_spearman([1, 2, 3], [3, 2, 1])['rho'], -1)


if __name__ == '__main__':
    unittest.main()
