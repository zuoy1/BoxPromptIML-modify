"""真实逐轮区域选择重叠和累计唯一覆盖的行为契约。"""
import unittest

import numpy as np

from fecd.training.region_coverage import summarize_coverage


def packed(rows):
    return np.packbits(np.array(rows, dtype=np.uint8), axis=1)


class RegionCoverageTests(unittest.TestCase):
    def setUp(self):
        self.inside = packed([[1, 1, 1, 1, 0, 0, 0, 0]])
        self.outside = packed([[0, 0, 0, 0, 1, 1, 0, 0]])
        self.first = packed([[1, 1, 0, 0, 1, 0, 1, 0]])
        self.second = packed([[0, 1, 1, 0, 0, 1, 1, 0]])

    def run_epoch(self, current, previous=None, cumulative=None):
        return summarize_coverage(current, previous, cumulative, self.inside, self.outside)

    def test_first_epoch_reports_half_coverage_without_previous_overlap(self):
        rows, cumulative = self.run_epoch(self.first)
        self.assertEqual(rows[0]['inside']['candidate_count'], 4)
        self.assertEqual(rows[0]['inside']['selected_count'], 2)
        self.assertEqual(rows[0]['inside']['cumulative_unique_count'], 2)
        self.assertEqual(rows[0]['inside']['cumulative_coverage'], .5)
        self.assertIsNone(rows[0]['inside']['previous_jaccard'])
        self.assertEqual(rows[0]['outside']['cumulative_coverage'], .5)
        self.assertEqual(rows[0]['negative']['selected_count'], 3)
        np.testing.assert_array_equal(cumulative, packed([[1, 1, 0, 0, 1, 0, 0, 0]]))

    def test_actual_adjacent_overlap_and_cumulative_unique_coverage_are_distinct(self):
        _, first_union = self.run_epoch(self.first)
        rows, union = self.run_epoch(self.second, self.first, first_union)
        self.assertAlmostEqual(rows[0]['inside']['previous_jaccard'], 1/3)
        self.assertEqual(rows[0]['outside']['previous_jaccard'], 0)
        self.assertEqual(rows[0]['inside']['cumulative_unique_count'], 3)
        self.assertEqual(rows[0]['inside']['cumulative_coverage'], .75)
        self.assertEqual(rows[0]['outside']['cumulative_coverage'], 1)
        self.assertEqual(rows[0]['negative']['cumulative_unique_count'], 5)
        np.testing.assert_array_equal(union, packed([[1, 1, 1, 0, 1, 1, 0, 0]]))
        self.assertEqual(rows[0]['negative']['previous_intersection_count'], 1)
        self.assertEqual(rows[0]['negative']['previous_union_count'], 5)

    def test_unchanged_hard_set_does_not_increase_unique_coverage(self):
        _, union = self.run_epoch(self.first)
        rows, after = self.run_epoch(self.first, self.first, union)
        self.assertEqual(rows[0]['inside']['previous_jaccard'], 1)
        self.assertEqual(rows[0]['outside']['previous_jaccard'], 1)
        self.assertEqual(rows[0]['negative']['cumulative_coverage'], .5)
        np.testing.assert_array_equal(after, union)

    def test_empty_and_singleton_regions_keep_undefined_overlap_and_floor_zero(self):
        inside = packed([[0]*8, [1, 0, 0, 0, 0, 0, 0, 0]])
        outside = packed([[0]*8, [0]*8])
        current = packed([[0, 1, 0, 0, 0, 0, 0, 0], [0]*8])
        rows, cumulative = summarize_coverage(current, None, None, inside, outside)
        self.assertIsNone(rows[0]['inside']['cumulative_coverage'])
        self.assertEqual(rows[1]['inside']['cumulative_coverage'], 0)
        after, _ = summarize_coverage(current, current, cumulative, inside, outside)
        self.assertIsNone(after[1]['inside']['previous_jaccard'])
        self.assertIsNone(after[0]['negative']['previous_jaccard'])

    def test_changed_positive_selection_is_rejected(self):
        _, union = self.run_epoch(self.first)
        changed = self.second.copy()
        changed[0, 0] ^= np.uint8(2)
        with self.assertRaisesRegex(ValueError, '正类'):
            self.run_epoch(changed, self.first, union)

    def test_invalid_partition_budget_and_cumulative_state_fail_fast(self):
        with self.assertRaises(ValueError):
            summarize_coverage(self.first.astype(float), None, None, self.inside, self.outside)
        with self.assertRaises(ValueError):
            summarize_coverage(self.first, None, None, self.inside[:, :0], self.outside)
        with self.assertRaisesRegex(ValueError, '互斥'):
            summarize_coverage(self.first, None, None, self.inside, self.inside)
        invalid = packed([[1, 0, 0, 0, 1, 0, 1, 0]])
        with self.assertRaisesRegex(ValueError, 'floor'):
            self.run_epoch(invalid)
        with self.assertRaisesRegex(ValueError, '累计'):
            self.run_epoch(self.second, self.first, np.zeros_like(self.first))
        with self.assertRaisesRegex(ValueError, '累计'):
            self.run_epoch(self.second, self.first, self.first)
        with self.assertRaisesRegex(ValueError, '同时'):
            self.run_epoch(self.second, self.first, None)

    def test_multiple_rows_and_bytes_do_not_mix_members_or_mutate_inputs(self):
        inside = np.repeat(self.inside, 2, axis=0)
        outside = np.repeat(self.outside, 2, axis=0)
        current = np.concatenate([self.first, self.second], axis=0)
        original = current.copy()
        rows, union = summarize_coverage(current, None, None, inside, outside)
        np.testing.assert_array_equal(current, original)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(int(union[0, 0]), int(union[1, 0]))
        wide = np.concatenate([current, current], axis=1)
        doubled, _ = summarize_coverage(wide, None, None,
                                       np.concatenate([inside, inside], axis=1),
                                       np.concatenate([outside, outside], axis=1))
        self.assertEqual(doubled[0]['inside']['candidate_count'], 8)
        self.assertEqual(doubled[0]['negative']['selected_count'], 6)


if __name__ == '__main__':
    unittest.main()
