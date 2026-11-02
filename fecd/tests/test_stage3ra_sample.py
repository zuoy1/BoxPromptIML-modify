"""Common candidate sets, tie budgets and condition strata must be preserved."""
import unittest

import numpy as np

from fecd.diagnostics.stage3ra_sample import audit_sample


def sample():
    mu = np.full((224, 224), .95, np.float32)
    mu[:, 112:] = .05
    base = mu.copy()
    base[20:40, 20:40] = .2
    gt = mu >= .5
    gt[20:30, 20:30] = False
    gt[60:70, 140:150] = True
    box = np.zeros((1, 224, 224), np.uint8)
    box[:, 10:180, 10:180] = 1
    return mu, np.zeros_like(mu), (mu >= .5).astype(np.float32), base, box, gt


class Stage3RASampleTests(unittest.TestCase):
    def test_all_six_scores_share_k8_candidates_and_exact_error_counts(self):
        row, histograms = audit_sample(*sample())
        self.assertEqual(row['regions']['pos']['base_margin']['n'], 224 * 112)
        for region, methods in row['regions'].items():
            self.assertEqual(len(methods), 6)
            self.assertEqual(len({(m['n'], m['error_count']) for m in methods.values()}), 1)
            for method, metric in methods.items():
                self.assertEqual(int(histograms[(region, method)][0].sum()), metric['n'])
        self.assertEqual(row['region_counts']['pos']['fp'], 100)
        self.assertEqual(row['region_counts']['neg']['fn'], 100)

    def test_random_budget_uses_expected_same_coverage_not_new_random_formula(self):
        row, _ = audit_sample(*sample())
        budget = row['budgets']['pos']
        for coverage in ('0.1', '0.25', '0.5', '0.75', '0.9'):
            random = budget['random'][coverage]
            self.assertAlmostEqual(random['risk'], 100 / (224 * 112))
            for method in ('c_sam', 'q_a'):
                chosen = budget[method][coverage]
                self.assertEqual(chosen['selected_count'], random['selected_count'])
                self.assertEqual(chosen['full_coverage'], random['full_coverage'])

    def test_region_composition_and_condition_bins_partition_candidates(self):
        row, _ = audit_sample(*sample())
        for method in ('c_sam', 'q_a'):
            parts = row['selection_composition']['pos'][method]
            self.assertAlmostEqual(sum(v['selected_count'] for v in parts.values()),
                                   row['budgets']['pos'][method]['0.5']['selected_count'])
        for dimension in ('mean_margin', 'variance'):
            values = row['conditioned_metrics']['pos'][dimension]
            self.assertEqual(sum(x['c_sam']['n'] for x in values.values()), 224 * 112)

    def test_realized_masks_have_equal_integer_budget_and_deterministic_ties(self):
        first, _ = audit_sample(*sample(), sample_key='toy-id')
        second, _ = audit_sample(*sample(), sample_key='toy-id')
        self.assertEqual(first['realized_budgets'], second['realized_budgets'])
        for sign in ('pos', 'neg'):
            budgets = first['realized_budgets'][sign]
            self.assertEqual(budgets['random'], budgets['c_sam'])
            for coverage in ('0.1', '0.25', '0.5', '0.75', '0.9'):
                counts = {budgets[method][coverage]['selected_count'] for method in budgets}
                self.assertEqual(counts, {round(float(coverage) * 224 * 112)})

    def test_empty_positive_set_preserves_na_and_rejects_nonbinary_gt(self):
        values = list(sample())
        values[0] = np.zeros((224, 224), np.float32)
        values[3] = values[0].copy()
        row, _ = audit_sample(*values)
        self.assertIsNone(row['regions']['pos']['q_a']['auroc'])
        self.assertIsNone(row['budgets']['pos']['random']['0.5']['precision'])
        values[-1] = np.full((224, 224), .1)
        with self.assertRaises(ValueError):
            audit_sample(*values)


if __name__ == '__main__':
    unittest.main()
