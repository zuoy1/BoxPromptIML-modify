"""Behavior checks for frozen SAM and Rule-FECD confidence definitions."""

import unittest

import numpy as np

from fecd.confidence.teacher_scores import rule_confidence, sam_confidence


class TeacherScoresTest(unittest.TestCase):
    def test_sam_confidence_extreme_inputs_bounded_and_finite(self):
        mu = np.array([0., .5, 1., 0., 1.], dtype=np.float16)
        variance = np.array([0., 0., 0., .25, .25], dtype=np.float16)
        agreement = np.array([0., .5, 1., 0., 1.], dtype=np.float16)
        result = sam_confidence(mu, variance, agreement)
        np.testing.assert_array_equal(result, [1., 0., 1., 0., 0.])
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_sam_confidence_variance_increases_score_does_not_increase(self):
        variance = np.linspace(0., .2501, 128, dtype=np.float32)
        result = sam_confidence(np.full(128, .9), variance, np.ones(128))
        self.assertTrue((np.diff(result) <= 0).all())
        self.assertEqual(float(result[-1]), 0.)

    def test_sam_confidence_threshold_uncertainty_increases_score_does_not_increase(self):
        agreement = np.linspace(0., .5, 128, dtype=np.float32)
        result = sam_confidence(np.full(128, .9), np.zeros(128), agreement)
        self.assertTrue((np.diff(result) <= 0).all())
        np.testing.assert_allclose(
            result,
            sam_confidence(np.full(128, .9), np.zeros(128), 1. - agreement),
            atol=2e-7,
        )

    def test_sam_confidence_foreground_background_symmetry(self):
        mu = np.array([.1, .2, .4], dtype=np.float32)
        variance = np.full(3, .03)
        agreement = np.full(3, .1)
        np.testing.assert_allclose(
            sam_confidence(mu, variance, agreement),
            sam_confidence(1. - mu, variance, 1. - agreement),
            atol=2e-7,
        )

    def test_rule_confidence_evidence_changes_only_positive_confidence(self):
        confidence = np.full((2, 3), .8)
        evidence = np.array([[0., .25, .5], [.75, 1., 0.]])
        positive, negative = rule_confidence(confidence, evidence)
        np.testing.assert_allclose(positive, [[.4, .5, .6], [.7, .8, .4]])
        np.testing.assert_allclose(negative, confidence)
        self.assertEqual(positive.dtype, np.float32)
        self.assertEqual(negative.dtype, np.float32)
        self.assertTrue((positive <= negative).all())

    def test_sam_confidence_all_zero_inputs_return_one(self):
        zero = np.zeros((224, 224), dtype=np.float32)
        np.testing.assert_array_equal(sam_confidence(zero, zero, zero), np.ones_like(zero))

    def test_confidence_invalid_shapes_nonfinite_and_ranges_fail_fast(self):
        valid = np.array([.8, .9])
        cases = [
            (np.array([np.nan, .5]), np.zeros(2), np.ones(2)),
            (valid, np.array([np.inf, 0.]), np.ones(2)),
            (valid, np.zeros(2), np.array([-.1, 1.])),
            (np.array([1.01, .5]), np.zeros(2), np.ones(2)),
            (valid, np.array([-.01, 0.]), np.ones(2)),
            (valid, np.array([.251, 0.]), np.ones(2)),
            (valid, np.zeros(1), np.ones(2)),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sam_confidence(*args)
        for evidence in (np.array([1.1, 0.]), np.array([np.nan, 0.]), np.zeros(1)):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                rule_confidence(valid, evidence)
        with self.assertRaises(ValueError):
            rule_confidence(np.array([-1., .5]), np.zeros(2))


if __name__ == '__main__':
    unittest.main()
