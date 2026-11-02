"""GT-free patch extraction contracts for the frozen reference precheck."""
import unittest

import numpy as np

from fecd.forensic_reference.patches import extract_patches


def cache():
    return (np.zeros((224, 224), np.float32), np.zeros((224, 224), np.float32),
            np.zeros((224, 224), np.float32), np.zeros((0, 224, 224), np.uint8))


class ReferencePatchTests(unittest.TestCase):
    def test_constant_rgb_has_zero_noise_and_stable_negative_patches(self):
        result = extract_patches(np.full((96, 128, 3), 128, np.uint8), *cache())
        self.assertEqual(result['work_hw'], [96, 128])
        self.assertEqual(len(result['patches']), 6)
        for patch in result['patches']:
            self.assertEqual(patch['mad'], 0)
            self.assertTrue(np.allclose(patch['quantiles'], 0))
            self.assertEqual(patch['stable_negative_fraction'], 1)
            self.assertEqual(patch['outside_dilated_fraction'], 1)

    def test_non_square_grid_keeps_aspect_and_half_open_coordinates(self):
        result = extract_patches(np.zeros((256, 1024, 3), np.uint8), *cache())
        self.assertEqual(result['work_hw'], [128, 512])
        patch = result['patches'][-1]
        self.assertEqual([patch[k] for k in ('x0', 'y0', 'x1', 'y1')], [448, 64, 512, 128])
        self.assertEqual(patch['bounds_native'], [896., 128., 1024., 256.])
        self.assertEqual(patch['bounds_sam224'], [196., 112., 224., 224.])

    def test_tiny_input_has_no_padded_or_upsampled_patches(self):
        result = extract_patches(np.zeros((1, 7, 3), np.uint8), *cache())
        self.assertEqual(result['work_hw'], [1, 7])
        self.assertEqual(result['patches'], [])

    def test_rgb_bgr_and_float_inputs_preserve_semantics(self):
        rgb = np.zeros((64, 64, 3), np.uint8)
        rgb[:, :, 0] = 220
        first = extract_patches(rgb, *cache())['patches'][0]
        second = extract_patches(rgb[:, :, ::-1], *cache(), color_order='BGR')['patches'][0]
        third = extract_patches(rgb.astype(np.float32) / 255., *cache())['patches'][0]
        self.assertEqual(first, second)
        self.assertAlmostEqual(first['brightness'], third['brightness'], places=6)
        self.assertGreater(first['brightness'], .25)

    def test_sampler_projects_known_sam_boundary_without_axis_swap(self):
        mu, var, agreement, box = cache()
        mu[:, 112:] = 1
        agreement[:, 112:] = 1
        box = np.zeros((1, 224, 224), np.uint8)
        box[:, :, 112:] = 1
        result = extract_patches(np.zeros((64, 128, 3), np.uint8), mu, var,
                                 agreement, box, box_dilation=0)
        first, middle, last = result['patches']
        self.assertEqual(first['pred_fraction'], 0)
        self.assertEqual(last['pred_fraction'], 1)
        self.assertEqual(middle['pred_fraction'], .5)
        self.assertEqual(last['box_fraction'], 1)
        self.assertEqual(first['outside_dilated_fraction'], 1)

    def test_noise_response_and_last_edge_lattice_are_finite(self):
        rgb = np.random.default_rng(7).integers(0, 256, (99, 101, 3), dtype=np.uint8)
        result = extract_patches(rgb, *cache(), patch_size=32)
        last = result['patches'][-1]
        self.assertEqual([last[k] for k in ('x0', 'y0', 'x1', 'y1')], [69, 67, 101, 99])
        self.assertGreater(last['mad'], 0)
        self.assertGreater(last['gradient'], 0)
        self.assertGreater(last['texture'], 0)
        self.assertTrue(np.isfinite(last['quantiles']).all())
        self.assertEqual(len(last['quantiles']), 19)
        self.assertTrue(np.all(np.diff(last['quantiles']) >= 0))

    def test_invalid_image_cache_and_parameters_fail_closed(self):
        rgb = np.zeros((64, 64, 3), np.uint8)
        bad_inputs = [np.zeros((64, 64), np.uint8), rgb.astype(np.int16),
                      np.full((64, 64, 3), np.nan), np.ones((64, 64, 3)) * 2]
        for image in bad_inputs:
            with self.subTest(dtype=image.dtype, shape=image.shape):
                with self.assertRaises(ValueError):
                    extract_patches(image, *cache())
        for bad in [np.zeros((32, 32)), np.full((224, 224), np.nan)]:
            with self.assertRaises(ValueError):
                extract_patches(rgb, bad, *cache()[1:])
        with self.assertRaises(ValueError):
            extract_patches(rgb, *cache(), patch_size=48)
        with self.assertRaises(ValueError):
            extract_patches(rgb, *cache(), color_order='YUV')

    def test_residual_uses_full_image_halo_before_patch_crop(self):
        rgb = np.zeros((64, 96, 3), np.uint8)
        rgb[:, 63:] = 255
        result = extract_patches(rgb, *cache())
        for patch in result['patches']:
            self.assertEqual(patch['mad'], 0)
            self.assertTrue(np.allclose(patch['quantiles'], 0))

    def test_smooth_gradient_does_not_become_positive_noise_evidence(self):
        ramp = np.linspace(0, 1, 128, dtype=np.float32)
        rgb = np.broadcast_to(ramp[None, :, None], (64, 128, 3)).copy()
        result = extract_patches(rgb, *cache())
        for patch in result['patches']:
            self.assertGreater(patch['gradient'], 0)
            self.assertLess(patch['mad'], 1e-6)
            self.assertLess(np.max(np.abs(patch['quantiles'])), 1e-6)

    def test_box_dilation_happens_on_sam_grid_before_work_projection(self):
        mu, var, agreement, box = cache()
        box = np.zeros((1, 224, 224), np.uint8)
        box[:, :, 112:] = 1
        rgb = np.zeros((64, 128, 3), np.uint8)
        first = extract_patches(rgb, mu, var, agreement, box, box_dilation=0)['patches'][0]
        expanded = extract_patches(rgb, mu, var, agreement, box, box_dilation=5)['patches'][0]
        self.assertEqual(first['outside_dilated_fraction'], 1)
        self.assertEqual(expanded['box_fraction'], 0)
        self.assertLess(expanded['outside_dilated_fraction'], 1)


if __name__ == '__main__':
    unittest.main()
