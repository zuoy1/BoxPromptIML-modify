"""Behavioral checks for GT-free forensic descriptors and frozen normalization."""
import copy
import unittest

import numpy as np

from fecd.evidence.extractors import EXTRACTOR_CONFIG, extract_evidence
from fecd.evidence.normalize import fit_stats, normalize_evidence


class EvidenceTests(unittest.TestCase):
    def test_constant_images_have_no_local_anomaly(self):
        for value in (0, 127, 255):
            maps = extract_evidence(np.full((67, 113, 3), value, np.uint8))
            self.assertEqual(set(maps), {'hp', 'noise', 'jpeg', 'fft'})
            for feature in maps.values():
                self.assertEqual(feature.shape, (224, 224))
                self.assertEqual(feature.dtype, np.float32)
                self.assertTrue(np.isfinite(feature).all())
                self.assertLess(float(feature.max()), 1e-4)

    def test_smooth_gradient_noise_and_edge_remain_finite(self):
        ramp = np.linspace(0, 255, 129, dtype=np.uint8)
        gradient = np.broadcast_to(ramp[None, :, None], (79, 129, 3)).copy()
        edge = np.zeros((100, 180, 3), np.uint8)
        edge[:, 90:] = 255
        noise = np.random.default_rng(12).integers(0, 256, (79, 129, 3), np.uint8)
        for image in (gradient, edge, noise):
            for feature in extract_evidence(image).values():
                self.assertTrue(np.isfinite(feature).all())
                self.assertTrue((feature >= 0).all())

    def test_aspect_ratio_and_tiny_inputs_preserve_contract(self):
        for shape in ((1, 1), (2, 7), (120, 3), (3, 120), (700, 129)):
            image = np.random.default_rng(7).integers(0, 256, (*shape, 3), np.uint8)
            for feature in extract_evidence(image).values():
                self.assertEqual(feature.shape, (224, 224))
                self.assertTrue(np.isfinite(feature).all())
        self.assertEqual(EXTRACTOR_CONFIG['work_long_edge'], 512)

    def test_float_rgb_and_explicit_bgr_agree(self):
        rgb = np.random.default_rng(18).integers(0, 256, (40, 90, 3), np.uint8)
        reference = extract_evidence(rgb)
        float_maps = extract_evidence(rgb.astype(np.float32) / 255)
        bgr_maps = extract_evidence(rgb[:, :, ::-1], color_order='BGR')
        for name in reference:
            np.testing.assert_allclose(reference[name], float_maps[name], atol=1e-6)
            np.testing.assert_allclose(reference[name], bgr_maps[name], atol=1e-6)

    def test_invalid_inputs_fail_instead_of_becoming_missing_values(self):
        for image in (np.zeros((10, 10)), np.zeros((10, 10, 4)),
                      np.zeros((10, 10, 3), np.int16),
                      np.full((10, 10, 3), np.nan, np.float32),
                      np.full((10, 10, 3), 2.0, np.float32)):
            with self.assertRaises(ValueError):
                extract_evidence(image)
        with self.assertRaises(ValueError):
            extract_evidence(np.zeros((10, 10, 3), np.uint8), color_order='GRAY')

    def test_synthetic_local_noise_has_local_high_pass_response(self):
        image = np.full((224, 224, 3), 128, np.uint8)
        image[64:160, 64:160] = np.random.default_rng(4).integers(64, 192, (96, 96, 3), np.uint8)
        maps = extract_evidence(image)
        for name in ('hp', 'noise', 'fft'):
            self.assertGreater(float(maps[name][70:154, 70:154].mean()),
                               float(maps[name][8:40, 8:40].mean()) + 1e-4)

    def test_normalization_is_global_frozen_and_excludes_hp_from_median(self):
        maps = [{name: np.full((224, 224), value, np.float32)
                 for name in ('hp', 'noise', 'jpeg', 'fft')} for value in (0, 1, 2)]
        stats = fit_stats(iter(maps), {'partition': 'train_fit', 'split_hash': 'test'})
        self.assertEqual(stats['image_count'], 3)
        again = fit_stats(iter(maps), {'partition': 'train_fit', 'split_hash': 'test'})
        self.assertEqual(stats, again)
        probe = {name: np.full((224, 224), 1, np.float32) for name in maps[0]}
        probe['hp'][:] = 999
        normalized = normalize_evidence(probe, stats)
        for feature in normalized.values():
            self.assertTrue(np.isfinite(feature).all())
            self.assertTrue(((feature >= 0) & (feature <= 1)).all())
        np.testing.assert_allclose(normalized['r'], .5)
        self.assertEqual(float(normalized['hp'].min()), 1)

    def test_fit_and_normalization_reject_nonfit_or_corrupt_statistics(self):
        zero = {name: np.zeros((224, 224), np.float32) for name in ('hp', 'noise', 'jpeg', 'fft')}
        with self.assertRaises(ValueError):
            fit_stats([zero], {'partition': 'teacher_diag'})
        with self.assertRaises(ValueError):
            fit_stats([], {'partition': 'train_fit'})
        stats = fit_stats([zero], {'partition': 'train_fit'})
        self.assertTrue(all(np.count_nonzero(v) == 0 for v in normalize_evidence(zero, stats).values()))
        corrupt = copy.deepcopy(stats)
        corrupt['features']['noise']['high'] += 1
        with self.assertRaises(ValueError):
            normalize_evidence(zero, corrupt)


if __name__ == '__main__':
    unittest.main()
