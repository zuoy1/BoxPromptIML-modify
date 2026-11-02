"""Runnable numerical contracts for teacher-error diagnosis."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from fecd.diagnostics.bootstrap import paired_bootstrap
from fecd.diagnostics.metrics import (
    error_detection_metrics, metrics_from_hist, reliability_histogram,
)
from fecd.diagnostics.risk_coverage import risk_coverage, risk_coverage_from_hist
from fecd.diagnostics.visualizations import save_diagnostic, save_risk_curves


class DetectionTests(unittest.TestCase):
    def test_low_reliability_errors_produce_perfect_detection(self):
        result = error_detection_metrics([1, 1, 0, 0], [.1, .2, .8, .9])
        self.assertEqual(result['auroc'], 1)
        self.assertEqual(result['auprc'], 1)
        self.assertEqual(result['error_prevalence'], .5)

    def test_reversed_order_produces_zero_auroc(self):
        result = error_detection_metrics([1, 1, 0, 0], [.9, .8, .2, .1])
        self.assertEqual(result['auroc'], 0)
        self.assertAlmostEqual(result['auprc'], (1 / 3 + 2 / 4) / 2)

    def test_equal_scores_use_tie_group_average_precision(self):
        result = error_detection_metrics([1, 0, 1, 0], [.5] * 4)
        self.assertEqual(result['auroc'], .5)
        self.assertEqual(result['auprc'], .5)

    def test_single_class_and_empty_candidates_report_na(self):
        no_error = error_detection_metrics([0, 0], [.1, .9])
        all_error = error_detection_metrics([1, 1], [.1, .9])
        empty = error_detection_metrics([], [])
        self.assertIsNone(no_error['auroc'])
        self.assertIsNone(no_error['auprc'])
        self.assertIsNone(all_error['auroc'])
        self.assertEqual(all_error['auprc'], 1)
        self.assertIsNone(empty['error_prevalence'])
        self.assertTrue(no_error['auroc_reason'])

    def test_histograms_are_additive_and_match_exact_distinct_bins(self):
        errors = np.array([1, 0, 1, 0, 0, 1])
        scores = np.array([0, .2, .4, .6, .8, 1])
        hist = reliability_histogram(errors, scores, bins=16)
        halves = (reliability_histogram(errors[:3], scores[:3], bins=16)
                  + reliability_histogram(errors[3:], scores[3:], bins=16))
        np.testing.assert_array_equal(hist, halves)
        exact = error_detection_metrics(errors, scores)
        binned = metrics_from_hist(hist)
        self.assertAlmostEqual(exact['auroc'], binned['auroc'])
        self.assertAlmostEqual(exact['auprc'], binned['auprc'])

    def test_invalid_values_are_rejected(self):
        for errors, scores in (([0], [np.nan]), ([0], [1.1]), ([2], [.5]),
                               ([0, 1], [.5])):
            with self.assertRaises(ValueError):
                reliability_histogram(errors, scores)


class CoverageTests(unittest.TestCase):
    def test_equal_coverage_selects_reliable_pixels_first(self):
        result = risk_coverage([1, 1, 0, 0], [.1, .2, .8, .9], [.5, 1])
        self.assertEqual(result['coverage']['0.5']['risk'], 0)
        self.assertEqual(result['coverage']['1']['risk'], .5)
        self.assertAlmostEqual(result['aurc'], .5 - .5 * np.log(2))

    def test_ties_use_fractional_selection_and_unbiased_risk(self):
        result = risk_coverage([1, 0, 0], [.5, .5, .5], [.1, .5, .9])
        for item in result['coverage'].values():
            self.assertAlmostEqual(item['risk'], 1 / 3)
            self.assertAlmostEqual(item['precision'], 2 / 3)
        self.assertAlmostEqual(result['aurc'], 1 / 3)
        self.assertAlmostEqual(result['coverage']['0.1']['selected_count'], .3)

    def test_small_and_empty_sets_keep_exact_coverage_and_na(self):
        result = risk_coverage([1], [.1], [.1])
        self.assertEqual(result['coverage']['0.1']['risk'], 1)
        empty = risk_coverage([], [], [.1])
        self.assertIsNone(empty['aurc'])
        self.assertIsNone(empty['coverage']['0.1']['risk'])

    def test_hist_and_exact_coverage_match_without_bin_collisions(self):
        errors, scores = [0, 1, 0, 1], [1, .7, .3, 0]
        exact = risk_coverage(errors, scores)
        binned = risk_coverage_from_hist(reliability_histogram(errors, scores))
        self.assertAlmostEqual(exact['aurc'], binned['aurc'])
        self.assertEqual(exact['coverage'], binned['coverage'])


class BootstrapTests(unittest.TestCase):
    def test_constant_paired_improvement_has_exact_interval(self):
        result = paired_bootstrap([.5, .8, .9], [.4, .7, .8], ['a', 'a', 'b'])
        self.assertAlmostEqual(result['difference'], .1)
        np.testing.assert_allclose(result['ci95'], [.1, .1])
        self.assertEqual(result['n_groups'], 2)
        self.assertEqual(result['n_images'], 3)

    def test_na_pairs_are_removed_and_one_group_has_no_interval(self):
        result = paired_bootstrap([.5, None, .9], [.4, .7, None], ['a', 'b', 'c'])
        self.assertEqual(result['n_images'], 1)
        self.assertIsNone(result['ci95'])
        self.assertTrue(result['reason'])

    def test_bootstrap_is_deterministic_and_image_weighted(self):
        args = ([1, 1, 0], [0, 0, 0], ['a', 'a', 'b'])
        first = paired_bootstrap(*args, n_resamples=400)
        second = paired_bootstrap(*args, n_resamples=400)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first['difference'], 2 / 3)


class VisualizationTests(unittest.TestCase):
    def test_gt_panel_is_guarded_by_diagnostic_directory(self):
        rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_diagnostic(Path(directory) / 'unsafe.png', rgb, [], {'GT': rgb[:, :, 0]}, 'test')
            with self.assertRaises(ValueError):
                save_diagnostic(Path(directory) / 'unsafe.png', rgb, [], {'diagnostic GT': rgb[:, :, 0]},
                                'test', contains_gt=False)

    def test_non_gt_panels_and_risk_curve_are_saved_as_rasters(self):
        rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'panels.png'
            maps = {'confidence': np.ones((224, 224)),
                    'selection delta': np.eye(224) * 2 - 1}
            save_diagnostic(path, rgb, [[20, 30, 90, 100]], maps, 'test', contains_gt=False)
            with Image.open(path) as image:
                self.assertGreaterEqual(image.width, 3 * 224)
                self.assertGreater(image.height, 224)
            curve_path = Path(directory) / 'curve.png'
            save_risk_curves(curve_path, {'S0': {'coverage': [.1, .5, 1], 'risk': [0, .1, .4]}})
            with Image.open(curve_path) as image:
                self.assertEqual(image.size, (800, 600))

    def test_gt_panels_can_be_saved_inside_dedicated_directory(self):
        root = Path(__file__).resolve().parents[1] / 'assets/stage3_diagnostics'
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / 'safe.png'
            save_diagnostic(path, np.zeros((224, 224, 3), dtype=np.uint8), [],
                            {'GT': np.zeros((224, 224))}, 'diagnostic only')
            self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
