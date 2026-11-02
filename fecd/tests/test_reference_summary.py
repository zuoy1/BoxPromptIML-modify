"""Aggregation tests use synthetic records, never image masks."""
import unittest
from fecd.forensic_reference.summary import collect, correlation, distribution


class ReferenceSummaryTests(unittest.TestCase):
    def test_constant_correlation_is_not_available(self):
        self.assertIsNone(correlation([1,1,1],[1,2,3]))
        self.assertAlmostEqual(correlation([1,1,2,3],[4,4,2,1]),-1)

    def test_empty_distribution_preserves_not_available(self):
        self.assertEqual(distribution([])['n'],0)
        self.assertIsNone(distribution([])['median'])

    def test_unmatched_image_not_reported_as_zero_distance(self):
        row={'source_group_id':'g','sample_key':'s','scales':{'64':{
            'candidates':[],'controls':[],'grid_patch_count':0,
            'reference_pool_count':0,'outside_patch_count':0}}}
        result=collect([row],'64')
        self.assertEqual(result['matched_images'],0)
        self.assertEqual(result['target_matched_images'],0)
        self.assertIsNone(result['primary_distance']['median'])
        self.assertIsNone(result['matched_fraction'])


if __name__=='__main__':
    unittest.main()
