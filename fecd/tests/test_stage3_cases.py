"""Fixed-case ranking and fractional coverage selection behavior."""
import unittest
from unittest.mock import patch

import numpy as np

from fecd.diagnostics.cases import fractional_selection, select_cases
from fecd.tools import audit_pseudo


class Stage3CasesTest(unittest.TestCase):
    def test_fractional_selection_ties_match_target_without_position_bias(self):
        score = np.array([.9, .8, .8, .1], np.float32)
        weight = fractional_selection(score, np.ones(4, bool), .5)
        np.testing.assert_allclose(weight, [1., .5, .5, 0.])
        self.assertAlmostEqual(float(weight.sum()), 2.)

    def test_fractional_selection_boundary_and_empty_candidates(self):
        score = np.array([0., 1., 1.])
        candidate = np.array([False, True, True])
        np.testing.assert_array_equal(fractional_selection(score, candidate, 1.), [0., 1., 1.])
        np.testing.assert_array_equal(fractional_selection(score, np.zeros(3, bool), .5), np.zeros(3))
        with self.assertRaises(ValueError):
            fractional_selection(score, candidate, 0.)

    def test_case_selection_absent_success_explicitly_records_unavailable(self):
        per_image = [{'sample_key': 'diag', 'sample_id': 'x', 'dataset': 'D',
                      'stable_counts': {'high_pos': {'fp': 60}, 'high_neg': {'fn': 3}},
                      'regions': {'pos': {'S0': {'n': 100, 'error_count': 60, 'coverage': {'0.5': {'risk': .2}}},
                                          'S4': {'n': 100, 'error_count': 60, 'coverage': {'0.5': {'risk': .4}}}}},
                      'evidence_distributions': {'TN': {'r': {'mean': .7, 'n': 5}, 'jpeg': {'mean': .8, 'n': 5}}}}]
        selected = select_cases(per_image, [])
        success = next(row for row in selected if row['case'] == 'evidence_success')
        self.assertIsNone(success['sample_key'])
        self.assertIn('no eligible', success['reason'])
        failure = next(row for row in selected if row['case'] == 'evidence_failure')
        self.assertEqual(failure['sample_key'], 'diag')

    def test_gt_access_requires_stage3a_gate_before_loading(self):
        with patch.object(audit_pseudo, '_GATE_STATUS', None), \
                patch.object(audit_pseudo, 'check_gate', side_effect=ValueError('gate not passed')), \
                patch.object(audit_pseudo.Image, 'open') as image_open:
            with self.assertRaisesRegex(ValueError, 'gate not passed'):
                audit_pseudo.load_diagnostic_gt({'subset': 'teacher_diag'}, {}, [224, 224])
            image_open.assert_not_called()

    def test_train_fit_gt_access_rejected_after_gate(self):
        with patch.object(audit_pseudo, '_GATE_STATUS', {'status': 'passed'}), \
                patch.object(audit_pseudo.Image, 'open') as image_open:
            with self.assertRaisesRegex(ValueError, 'teacher_diag'):
                audit_pseudo.load_diagnostic_gt({'subset': 'train_fit'}, {'split': 'train'}, [224, 224])
            image_open.assert_not_called()


if __name__ == '__main__':
    unittest.main()
