"""Fixed versus dynamic canonical-pixel supervision contracts; no real image IO."""
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from fecd.datasets.transforms import apply_spatial_transform
from fecd.training.data import ControlledDataset
from fecd.training.f0_data import EpochDataset
from fecd.training.f0_selection import select_for_variant, expected_unique_coverage
import fecd.tests.test_stage4p_data as synthetic


class FixedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.target=np.r_[np.zeros(65),np.ones(63)].reshape(1,8,16)
        self.score=np.linspace(0,1,128).reshape(self.target.shape)

    def test_fixed_masks_and_counts_are_identical_across_twenty_epochs(self):
        for variant in ('P2F','P3F'):
            first,counts=select_for_variant(self.target,self.score,variant,13,'image',0)
            for epoch in range(1,20):
                mask,next_counts=select_for_variant(self.target,self.score,variant,13,'image',epoch)
                np.testing.assert_array_equal(mask,first)
                self.assertEqual(next_counts,counts)
            self.assertEqual(counts['selected_pos'],31)
            self.assertEqual(counts['selected_neg'],32)

    def test_fixed_random_and_confidence_share_budget_not_ranking(self):
        random,rc=select_for_variant(self.target,self.score,'P2F',13,'image',1)
        confidence,cc=select_for_variant(self.target,self.score,'P3F',13,'image',19)
        self.assertEqual(rc,cc)
        self.assertFalse(np.array_equal(random,confidence))
        for sign in (0,1):
            selected=self.score[(self.target==sign)&confidence]
            rejected=self.score[(self.target==sign)&~confidence]
            self.assertGreaterEqual(selected.min(),rejected.max())

    def test_confidence_ties_use_same_fixed_random_priority(self):
        tied=np.ones_like(self.score)
        random,_=select_for_variant(self.target,tied,'P2F',13,'image',0)
        confidence,_=select_for_variant(self.target,tied,'P3F',13,'image',19)
        np.testing.assert_array_equal(random,confidence)
        self.assertFalse(random.reshape(-1)[:32].all())

    def test_tied_priority_is_not_systematically_raster_ordered(self):
        target=np.ones((1,8,16))
        frequency=np.zeros_like(target)
        for image in range(256):
            mask,_=select_for_variant(target,target,'P3F',13,f'image-{image}',19)
            frequency+=mask
        frequency/=256
        self.assertTrue(((frequency>.35)&(frequency<.65)).all())

    def test_dynamic_random_changes_canonical_set_and_is_repeatable(self):
        first,_=select_for_variant(self.target,self.score,'P2E',13,'image',0)
        second,_=select_for_variant(self.target,self.score,'P2E',13,'image',1)
        again,_=select_for_variant(self.target,self.score,'P2E',13,'image',1)
        self.assertFalse(np.array_equal(first,second))
        np.testing.assert_array_equal(second,again)
        fixed,_=select_for_variant(self.target,self.score,'P2F',13,'image',1)
        np.testing.assert_array_equal(first,fixed)

    def test_full_supervision_and_singleton_or_empty_classes(self):
        for variant in ('P0','P1'):
            mask,counts=select_for_variant(self.target,self.score,variant,13,'image',0)
            self.assertTrue(mask.all())
            self.assertEqual(counts['selected_total'],128)
        tiny=np.array([0,0,1])
        for variant in ('P2F','P3F','P2E'):
            mask,counts=select_for_variant(tiny,np.ones(3),variant,13,'image',0)
            self.assertEqual(counts['selected_pos'],0)
            self.assertEqual(counts['selected_neg'],1)
            self.assertEqual(mask.sum(),1)
            empty=np.zeros(8)
            _,counts=select_for_variant(empty,np.ones(8),variant,13,'image',0)
            self.assertEqual(counts['candidate_pos'],0)

    def test_unique_coverage_formula_and_invalid_variant(self):
        self.assertIsNone(expected_unique_coverage(0,'P2F',20))
        self.assertEqual(expected_unique_coverage(1,'P2E',20),0)
        self.assertEqual(expected_unique_coverage(100,'P2F',20),.5)
        self.assertEqual(expected_unique_coverage(63,'P3F',20),31/63)
        self.assertAlmostEqual(expected_unique_coverage(100,'P2E',20),1-.5**20)
        self.assertEqual(expected_unique_coverage(7,'P1',20),1)
        with self.assertRaises(ValueError):
            select_for_variant(self.target,self.score,'P3',13,'image',0)
        with self.assertRaises(ValueError):
            expected_unique_coverage(2,'P2F',0)


class EpochDatasetAlignmentTests(unittest.TestCase):
    def dataset(self,row,index,variant):
        base={'P2F':'P2','P3F':'P3','P2E':'P2'}[variant]
        samples=synthetic._build_samples([row],[index],synthetic.Path(index['cache_path']).parent)
        obj=EpochDataset.__new__(EpochDataset)
        ControlledDataset._configure(obj,base,20261029,0,[row],samples,'config')
        obj.formal_variant=variant
        return obj

    def test_inverse_geometry_restores_fixed_original_pixels_and_hash(self):
        with TemporaryDirectory() as directory:
            helper=synthetic.Stage4PDataTests()
            row,index=helper.fixture(directory)
            for variant in ('P2F','P3F'):
                dataset=self.dataset(row,index,variant)
                originals=[]
                hashes=[]
                seen=set()
                for epoch in range(100):
                    dataset.set_epoch(epoch)
                    item=dataset[0]
                    if item['operation'] in seen:
                        continue
                    seen.add(item['operation'])
                    inverse={'identity':'identity','hflip':'hflip','vflip':'vflip',
                             'rot90:1':'rot90:3','rot90:2':'rot90:2','rot90:3':'rot90:1'}[item['operation']]
                    mask=apply_spatial_transform(item['selection'][0].numpy(),operation=inverse)[0]
                    originals.append(mask)
                    hashes.append(item['canonical_selection_hash'])
                    unnormalized=item['image'][0]*.229+.485
                    self.assertTrue(torch.equal(unnormalized>.5,item['target'][0].bool()))
                    self.assertEqual(item['selection_epoch'],0)
                    if len(seen)==6:
                        break
                self.assertEqual(len(seen),6)
                np.testing.assert_array_equal(originals[0],originals[1])
                np.testing.assert_array_equal(originals[0],originals[2])
                self.assertEqual(len(set(hashes)),1)

    def test_dynamic_dataset_records_real_selection_epoch(self):
        with TemporaryDirectory() as directory:
            helper=synthetic.Stage4PDataTests()
            row,index=helper.fixture(directory)
            dataset=self.dataset(row,index,'P2E')
            first=dataset[0]
            dataset.set_epoch(1)
            second=dataset[0]
            self.assertNotEqual(first['canonical_selection_hash'],second['canonical_selection_hash'])
            self.assertEqual(second['selection_epoch'],1)


if __name__=='__main__':
    unittest.main()
