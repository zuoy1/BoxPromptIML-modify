"""Paired source-cluster inference, not independent-pixel or training-seed inference."""
import unittest

from fecd.evaluation.f1_statistics import paired_bootstrap, pair_records, area_diagnostics


def record(key, domain, group, score):
    return {'sample_key': key, 'dataset': domain, 'source_group_id': group,
            'f1': score, 'iou': score, 'gt_positive_fraction': .1,
            'prediction_positive_fraction': .4}


class PairedStatisticsTests(unittest.TestCase):
    def test_reordered_records_are_paired_and_missing_or_changed_members_fail(self):
        left = [record('a', 'A', 'g', .1), record('b', 'B', 'h', .2)]
        right = [record('b', 'B', 'h', .4), record('a', 'A', 'g', .3)]
        paired = pair_records(left, right)
        self.assertEqual([r['sample_key'] for r in paired], ['a', 'b'])
        self.assertAlmostEqual(paired[0]['f1'], .2)
        with self.assertRaises(ValueError):
            pair_records(left, right[:1])
        with self.assertRaises(ValueError):
            pair_records(left, [record('a', 'B', 'g', .3), right[0]])

    def test_cluster_sampling_preserves_constant_effect_and_is_reproducible(self):
        left = [record(str(i), 'A' if i < 4 else 'B', str(i // 2), .1) for i in range(6)]
        right = [{**r, 'f1': .3, 'iou': .3} for r in left]
        result = paired_bootstrap(left, right, repeats=2000, seed=20261029)
        self.assertEqual(result, paired_bootstrap(left, right, repeats=2000, seed=20261029))
        self.assertEqual(result['domains']['A']['source_groups'], 2)
        self.assertEqual(result['domains']['B']['source_groups'], 1)
        for method in ('image_weighted', 'group_equal', 'dataset_equal'):
            self.assertAlmostEqual(result['f1'][method]['estimate'], .2)
            for bound in result['f1'][method]['ci95']:
                self.assertAlmostEqual(bound, .2)

    def test_image_weighted_group_equal_and_dataset_equal_are_distinct(self):
        left = [record(str(i), 'A' if i < 3 else 'B', 'big' if i < 3 else 'small', 0) for i in range(4)]
        right = [{**r, 'f1': 1. if i < 3 else 0., 'iou': 1. if i < 3 else 0.} for i, r in enumerate(left)]
        result = paired_bootstrap(left, right, repeats=10, seed=4)
        self.assertEqual(result['f1']['image_weighted']['estimate'], .75)
        self.assertEqual(result['f1']['group_equal']['estimate'], .5)
        self.assertEqual(result['f1']['dataset_equal']['estimate'], .5)

    def test_severe_overprediction_uses_explicit_area_and_ratio_conjunction(self):
        rows = [record('a', 'A', 'g', .2), {**record('b', 'A', 'g', .2), 'prediction_positive_fraction': .2}]
        result = area_diagnostics(rows)
        self.assertEqual(result['severe_overprediction_images'], 1)
        self.assertEqual(result['images'], 2)

