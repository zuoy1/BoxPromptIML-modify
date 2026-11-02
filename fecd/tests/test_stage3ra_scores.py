"""Behavior contract for the six fixed stage-3R-A teacher scores."""
import unittest

import numpy as np

from fecd.confidence.stage3ra_scores import teacher_scores


class Stage3RAScoreTests(unittest.TestCase):
    def test_fixed_scores_use_base_margin_and_q_has_no_fitted_parameter(self):
        mu = np.array([.9, .1, .5], np.float32)
        base = np.array([.7, .3, .5], np.float32)
        variance = np.array([.01, .01, .25], np.float32)
        agreement = np.array([1., 0., .5], np.float32)
        scores = teacher_scores(mu, variance, agreement, base)
        self.assertEqual(set(scores), {'base_margin', 'mean_margin', 'variance',
                                       'threshold', 'c_sam', 'q_a'})
        np.testing.assert_allclose(scores['base_margin'], [.4, .4, 0], atol=1e-6)
        np.testing.assert_allclose(scores['mean_margin'], [.8, .8, 0], atol=1e-6)
        np.testing.assert_allclose(scores['variance'], [.96, .96, 0], atol=1e-6)
        np.testing.assert_allclose(scores['threshold'], [1, 1, 0], atol=1e-6)
        np.testing.assert_allclose(scores['q_a'], scores['c_sam'] / (1 + abs(mu-base)))
        for score in scores.values():
            self.assertTrue(np.isfinite(score).all())
            self.assertTrue(((score >= 0) & (score <= 1)).all())

    def test_same_base_and_mean_leaves_q_equal_to_consensus(self):
        mu = np.array([0., .5, 1.], np.float32)
        scores = teacher_scores(mu, np.zeros(3), np.array([0, .5, 1]), mu)
        np.testing.assert_array_equal(scores['q_a'], scores['c_sam'])

    def test_invalid_base_range_shape_or_nonfinite_fails(self):
        for base in (np.array([1.1]), np.array([np.nan]), np.zeros(2)):
            with self.assertRaises(ValueError):
                teacher_scores(np.array([.9]), np.array([0]), np.array([1]), base)


if __name__ == '__main__':
    unittest.main()
