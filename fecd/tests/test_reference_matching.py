"""GT-free reference contracts use only synthetic patch descriptors."""
import math
import unittest

import numpy as np

from fecd.forensic_reference.matching import (
    candidate_categories, match_references, reference_eligible,
)
from fecd.forensic_reference.statistics import compare_regions


def patch(name, x=0, y=0, **changes):
    result = dict(id=name, x0=x, y0=y, x1=x+64, y1=y+64,
                  brightness=.5, gradient=.05, texture=.08, mad=.01,
                  quantiles=np.linspace(-.02, .02, 19).tolist(),
                  box_fraction=0., pred_fraction=0.,
                  stable_negative_fraction=1., outside_dilated_fraction=1.)
    result.update(changes)
    return result


class ReferenceMatchingTests(unittest.TestCase):
    def test_reference_gates_do_not_call_stability_confirmed_background(self):
        self.assertTrue(reference_eligible(patch('r')))
        for changes in ({'outside_dilated_fraction': .99},
                        {'pred_fraction': .051}, {'stable_negative_fraction': .89}):
            self.assertFalse(reference_eligible(patch('r', **changes)))

    def test_candidate_categories_allow_fixed_box_sam_overlap(self):
        self.assertEqual(candidate_categories(patch('x', pred_fraction=.8, box_fraction=.8)),
                         ['pos', 'box_inside', 'box_edge'])
        self.assertEqual(candidate_categories(patch('x', pred_fraction=.4)), [])
        self.assertEqual(candidate_categories(patch('x')), ['neg'])

    def test_nonoverlap_allows_touching_edges_and_excludes_shared_pixels(self):
        c = patch('c')
        refs = [patch('overlap', 32), patch('a', 64), patch('b', 128), patch('d', 192)]
        result = match_references(c, refs)
        self.assertEqual(result['reference_ids'], ['a', 'b', 'd'])
        self.assertEqual(result['status'], 'applicable')
        self.assertEqual(result['rejection_reasons']['candidate_overlap'], 1)

    def test_matching_order_is_content_distance_then_id_not_spatial_distance(self):
        c = patch('c')
        refs = [patch('near', 64, brightness=.55), patch('far', 256),
                patch('middle', 128), patch('final', 384)]
        a = match_references(c, refs)
        b = match_references(c, refs[::-1])
        self.assertEqual(a['reference_ids'], ['far', 'final', 'middle', 'near'])
        self.assertEqual(a, b)

    def test_reference_overlap_greedy_exclusion_never_relaxes_minimum_count(self):
        c = patch('c')
        result = match_references(c, [patch('a', 64), patch('b', 96), patch('d', 128)])
        self.assertEqual(result['reference_ids'], ['a', 'd'])
        self.assertEqual(result['status'], 'not_applicable')
        self.assertEqual(result['rejection_reasons']['reference_overlap'], 1)

    def test_content_gates_record_why_unmatched_without_zero_fallback(self):
        c = patch('c')
        refs = [patch('bright', 64, brightness=.7),
                patch('gradient', 128, gradient=.2),
                patch('texture', 192, texture=.4)]
        result = match_references(c, refs)
        self.assertEqual(result['reference_ids'], [])
        self.assertEqual(result['counts']['joint_content'], 0)
        self.assertEqual(result['rejection_reasons']['brightness'], 1)
        self.assertEqual(result['rejection_reasons']['gradient'], 1)
        self.assertEqual(result['rejection_reasons']['texture'], 1)
        self.assertIsNone(result['matching_quality']['selected']['brightness']['mean'])

    def test_maximum_five_references_and_nonfinite_identity_errors_fail_fast(self):
        c = patch('c')
        self.assertEqual(len(match_references(c, [patch(str(i), 64*(i+1))
                                                for i in range(8)])['references']), 5)
        for refs in ([patch('bad', 64, brightness=float('nan'))],
                     [patch('bad', 64, x1=float('nan'))],
                     [patch('dup', 64), patch('dup', 128)]):
            with self.assertRaises(ValueError):
                match_references(c, refs)

    def test_identical_constant_statistics_are_finite_zero(self):
        c = patch('c', mad=0., quantiles=[0.]*19)
        refs = [patch(str(i), 64*(i+1), mad=0., quantiles=[0.]*19) for i in range(3)]
        result = compare_regions(c, refs)
        self.assertEqual(result['status'], 'applicable')
        self.assertEqual(result['standardized_quantile_distance'], 0.)
        self.assertEqual(result['raw_quantile_distance'], 0.)
        self.assertEqual(result['log_mad_ratio'], 0.)

    def test_quantile_distance_definition_and_missing_references_are_explicit(self):
        c = patch('c')
        q = (np.asarray(c['quantiles'])+.01).tolist()
        refs = [patch(str(i), 64*(i+1), quantiles=q) for i in range(3)]
        result = compare_regions(c, refs)
        self.assertAlmostEqual(result['raw_quantile_distance'], .01)
        self.assertAlmostEqual(result['standardized_quantile_distance'], .01/.015)
        result = compare_regions(c, refs[:2])
        self.assertEqual(result['status'], 'not_applicable')
        self.assertIsNone(result['standardized_quantile_distance'])
        self.assertEqual(result['n_references'], 2)

    def test_invalid_statistics_fail_instead_of_nan_output(self):
        refs = [patch(str(i), 64*(i+1)) for i in range(3)]
        for c in (patch('c', mad=-1.), patch('c', quantiles=[math.inf]*19),
                  patch('c', quantiles=[0.]*18),
                  patch('c', quantiles=list(range(19))[::-1]),
                  patch('c', quantiles=[1e308]*19)):
            with self.assertRaises(ValueError):
                compare_regions(c, refs)


if __name__ == '__main__':
    unittest.main()
