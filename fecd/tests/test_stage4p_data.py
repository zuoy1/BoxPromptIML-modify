"""Student-only identity and synchronized geometry contracts with synthetic IO."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from fecd.training.data import ControlledDataset, _build_samples
from fecd.pseudo.cache_schema import build_cache_record, make_sample_key


class Stage4PDataTests(unittest.TestCase):
    def fixture(self, directory, k=8):
        root=Path(directory)
        rgb=np.zeros((224,224,3),np.uint8)
        rgb[:,112:]=255
        image=root/'image.png'
        Image.fromarray(rgb).save(image)
        contents=image.read_bytes()
        sha256=hashlib.sha256(contents).hexdigest()
        sha1=hashlib.sha1(contents).hexdigest()
        key=make_sample_key('synthetic','train','sample',sha256)
        row={'subset':'student_train','dataset':'synthetic','sample_id':'sample',
             'sample_key':key,'image_sha256':sha256,'image_path':str(image)}
        mu=np.zeros((224,224),np.float32);mu[:,112:]=1
        rec=build_cache_record(maps={'mu':mu,'var':np.zeros_like(mu),'agreement':mu},
             box_masks=np.zeros((3,224,224),np.uint8),
             quality_mean=0.,quality_std=0.,image_hw=(224,224),schema_config_hash='config',
             image_sha1=sha1,box_hash='box',sample_id='sample',dataset='synthetic',
             split='train',sample_key=key,num_boxes=0,k=k,empty_boxes=True,
             base_mu=mu if k==8 else None)
        cache=root/'cache.npz';np.savez_compressed(cache,**rec)
        index={'image_path':str(image),'cache_path':str(cache),'dataset':'synthetic',
               'split':'train','sample_id':'sample','sample_key':key,'boxes_xyxy':[],
               'image_sha1':sha1,'box_hash':'box'}
        return row,index

    def dataset(self,row,index,variant):
        samples=_build_samples([row],[index],Path(index['cache_path']).parent)
        obj=ControlledDataset.__new__(ControlledDataset)
        obj._configure(variant,20261029,0,[row],samples,'config')
        return obj

    def test_index_filters_roles_and_rejects_duplicate_or_label_fields(self):
        with TemporaryDirectory() as directory:
            row,index=self.fixture(directory)
            other={**row,'subset':'new_dev','sample_key':'dev'}
            samples=_build_samples([other,row],[index],Path(directory))
            self.assertEqual(len(samples),1)
            with self.assertRaises(ValueError):
                _build_samples([row],[index,{**index}],Path(directory))
            with self.assertRaises(ValueError):
                _build_samples([row],[{**index,'prepared_mask':'never_read'}],Path(directory))

    def test_index_rejects_role_identity_and_cache_escape(self):
        with TemporaryDirectory() as directory:
            row,index=self.fixture(directory)
            self.assertEqual(_build_samples([{**row,'subset':'new_confirm'}],[index],Path(directory)),[])
            with self.assertRaises(ValueError):
                _build_samples([row],[{**index,'sample_id':'other'}],Path(directory))
            with self.assertRaises(ValueError):
                _build_samples([row],[{**index,'cache_path':'/outside/cache.npz'}],Path(directory))

    def test_transformed_image_target_and_selection_are_aligned(self):
        with TemporaryDirectory() as directory:
            row,index=self.fixture(directory)
            data=self.dataset(row,index,'P3')
            result=data[0]
            unnormalized=result['image'][0]*.229+.485
            self.assertTrue(torch.equal(unnormalized>.5,result['target'][0].bool()))
            self.assertTrue(result['selection'].dtype==torch.bool)
            self.assertEqual(int((result['selection']&result['target'].bool()).sum()),224*112//2)
            self.assertEqual(int((result['selection']&~result['target'].bool()).sum()),224*112//2)
            result2=data[0]
            self.assertTrue(torch.equal(result['selection'],result2['selection']))
            self.assertEqual(data.input_paths(0),[Path(index['image_path']),Path(index['cache_path'])])

    def test_p2_p3_equal_budget_same_augmentation(self):
        with TemporaryDirectory() as directory:
            row,index=self.fixture(directory)
            random=self.dataset(row,index,'P2')[0]
            confidence=self.dataset(row,index,'P3')[0]
            self.assertEqual(random['operation'],confidence['operation'])
            self.assertTrue(torch.equal(random['target'],confidence['target']))
            for key in ('candidate_pos','candidate_neg','selected_pos','selected_neg'):
                self.assertEqual(random[key],confidence[key])

    def test_p0_and_p1_binarize_corresponding_teacher_and_select_all(self):
        for variant,k in [('P0',1),('P1',8)]:
            with TemporaryDirectory() as directory:
                row,index=self.fixture(directory,k)
                sample=self.dataset(row,index,variant)[0]
                self.assertTrue(sample['selection'].all())
                self.assertTrue(torch.isin(sample['target'],torch.tensor([0.,1.])).all())
                self.assertEqual(sample['k'],k)

    def test_read_rejects_image_byte_hash_drift_before_data_load(self):
        with TemporaryDirectory() as directory:
            row,index=self.fixture(directory)
            data=self.dataset(row,index,'P1')
            Image.fromarray(np.ones((224,224,3),np.uint8)).save(index['image_path'])
            with self.assertRaisesRegex(ValueError,'SHA256'):
                data[0]

    def test_read_rejects_native_cache_dimension_mismatch(self):
        with TemporaryDirectory() as directory:
            row,index=self.fixture(directory)
            rec=dict(np.load(index['cache_path'],allow_pickle=False))
            rec['image_h']=np.asarray(225,dtype=np.int32)
            np.savez_compressed(index['cache_path'],**rec)
            data=self.dataset(row,index,'P1')
            with self.assertRaisesRegex(ValueError,'image_h'):
                data[0]

    def test_epoch_is_deterministic_and_invalid_variant_is_rejected(self):
        with TemporaryDirectory() as directory:
            row,index=self.fixture(directory)
            data=self.dataset(row,index,'P2')
            data.set_epoch(3)
            first=data[0];second=data[0]
            self.assertEqual(first['operation'],second['operation'])
            self.assertTrue(torch.equal(first['selection'],second['selection']))
            with self.assertRaises(ValueError):
                data.set_epoch(-1)
            with self.assertRaises(ValueError):
                self.dataset(row,index,'unknown')


if __name__=='__main__':
    unittest.main()
