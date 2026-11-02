"""开发诊断的区域分解、阈值、配对错误和汇总契约。"""
import unittest
from io import StringIO
import os

import numpy as np

from fecd.diagnostics.stage5a_metrics import measures, spatial_errors, overlap_errors, summarize
from fecd.diagnostics.stage5a_io import DiagnosticBoundary, OUTPUT


class FailureDiagnosisTests(unittest.TestCase):
    def test_diagnostic_boundary_rejects_external_write_and_records_denial(self):
        log = StringIO()
        boundary = DiagnosticBoundary({}, log)
        with self.assertRaises(PermissionError):
            boundary('open', ('/tmp/stage5a_forbidden', 'w', os.O_WRONLY | os.O_CREAT))
        self.assertEqual(boundary.denied, ['/tmp/stage5a_forbidden'])
        self.assertIn('denied_write', log.getvalue())
        boundary('open', (str(OUTPUT / 'tmp/allowed'), 'w', os.O_WRONLY | os.O_CREAT))
        self.assertEqual(len(boundary.denied), 1)

    def test_diagnostic_boundary_rejects_unlisted_gt_before_reading(self):
        log = StringIO()
        boundary = DiagnosticBoundary({}, log)
        with self.assertRaises(PermissionError):
            boundary('open', ('/home/datasets/locked_mask.png', 'r', os.O_RDONLY))
        self.assertIn('denied_read', log.getvalue())
        self.assertEqual(boundary.counts, {})

    def test_measures_known_counts_match_precision_recall_and_area(self):
        gt = np.array([[1, 1], [0, 0]], dtype=bool)
        probability = np.array([[.9, .2], [.8, .1]])
        result = measures(probability >= .5, gt)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 1))
        for key in ('f1', 'precision', 'recall', 'prediction_area', 'gt_area'):
            self.assertEqual(result[key], .5)
        self.assertAlmostEqual(result['iou'], 1 / 3)

    def test_increasing_threshold_never_increases_area_or_recall(self):
        probability = np.linspace(0, 1, 64).reshape(8, 8)
        gt = np.eye(8, dtype=bool)
        rows = [measures(probability >= t, gt) for t in np.arange(.1, 1., .1)]
        for first, second in zip(rows, rows[1:]):
            self.assertGreaterEqual(first['prediction_area'], second['prediction_area'])
            self.assertGreaterEqual(first['recall'], second['recall'])

    def test_spatial_error_partitions_recover_all_false_positives_and_negatives(self):
        gt = np.zeros((40, 40), bool)
        gt[12:28, 12:28] = True
        pred = gt.copy()
        pred[0:3, 0:3] = True
        pred[16:20, 16:20] = False
        stats = spatial_errors(pred, gt, np.ones_like(gt), radius=3)
        self.assertEqual(stats['fp_gt_boundary'] + stats['fp_gt_far'], 9)
        self.assertEqual(stats['fn_gt_boundary'] + stats['fn_gt_interior'], 16)
        self.assertEqual(stats['fp_pred_boundary'] + stats['fp_pred_interior'], 9)
        self.assertEqual(stats['fn_pred_boundary'] + stats['fn_pred_far'], 16)
        self.assertEqual(stats['fp_largest_component_fraction'], 1.)

    def test_teacher_student_error_cross_table_captures_correction_and_new_error(self):
        gt = np.array([[1, 1], [0, 0]], bool)
        teacher = np.array([[0, 1], [1, 0]], bool)
        student = np.array([[1, 0], [0, 1]], bool)
        result = overlap_errors(student, teacher, gt)
        self.assertEqual(result['teacher_wrong_student_correct'], 2)
        self.assertEqual(result['teacher_correct_student_wrong'], 2)
        self.assertEqual(result['both_wrong'], 0)
        self.assertEqual(result['student_fp_extra'], 1)
        self.assertEqual(result['student_fn_extra'], 1)

    def test_source_groups_are_equal_weighted_and_scoped_by_dataset(self):
        rows = [{'dataset': 'A', 'source_group_id': 'same', 'f1': .9, 'iou': .8,
                 'precision': .9, 'recall': .9, 'prediction_area': .2, 'gt_area': .1,
                 'tp': 2, 'fp': 1, 'fn': 1, 'pixels': 16} for _ in range(3)]
        rows.append({**rows[0], 'dataset': 'B', 'f1': .1, 'iou': .05})
        result = summarize(rows)
        self.assertEqual(result['images'], 4)
        self.assertEqual(result['source_groups'], 2)
        self.assertAlmostEqual(result['image_f1'], .7)
        self.assertAlmostEqual(result['group_f1'], .5)

    def test_diagnostic_spatial_inputs_with_wrong_shape_or_radius_are_rejected(self):
        value = np.zeros((4, 4), bool)
        for invalid in (np.zeros((1, 4)), np.zeros((0, 0))):
            with self.assertRaises(ValueError):
                spatial_errors(value, invalid, value)
            with self.assertRaises(ValueError):
                overlap_errors(value, invalid, value)
        with self.assertRaises(ValueError):
            spatial_errors(value, value, value, radius=0)


if __name__ == '__main__':
    unittest.main()
