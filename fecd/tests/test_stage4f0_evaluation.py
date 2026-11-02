"""Strict evaluator contracts; synthetic masks only."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

from fecd.evaluation.metrics import segmentation_metrics, output_diagnostics, summarize_records
from fecd.evaluation.memory import initial_memory, reset_memory, infer_one
from fecd.evaluation.data import load_dev_gt


class Stateful(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(1.))
        self.memory_bank=torch.tensor([0.])
    def forward(self,x):
        value=x[:,:1]+self.memory_bank.sum()
        self.memory_bank=self.memory_bank+1
        return value


class StrictEvaluationTests(unittest.TestCase):
    def test_empty_predictions_and_empty_gt_use_frozen_convention(self):
        z=np.zeros((2,2),bool)
        self.assertEqual(segmentation_metrics(z,z)['f1'],1.)
        self.assertEqual(segmentation_metrics(z,z)['iou'],1.)
        o=z.copy();o[0,0]=True
        self.assertEqual(segmentation_metrics(o,z)['f1'],0.)
        self.assertEqual(segmentation_metrics(z,o)['iou'],0.)

    def test_no_minmax_and_fixed_threshold(self):
        logits=np.full((2,2),-0.1)
        diag=output_diagnostics(logits)
        self.assertEqual(diag['prediction_positive_fraction'],0.)
        self.assertAlmostEqual(diag['probability_mean'],1/(1+np.exp(.1)))
        self.assertTrue(diag['all_zero_prediction'])
        self.assertIsNone(diag['positive_mean_probability'])

    def test_macro_and_micro_are_distinct(self):
        rows=[{'dataset':'A',**segmentation_metrics(np.ones((2,2),bool),np.ones((2,2),bool))},
              {'dataset':'B',**segmentation_metrics(np.zeros((2,2),bool),np.ones((2,2),bool))}]
        result=summarize_records(rows)
        self.assertEqual(result['domains']['ALL']['image_macro_f1'],.5)
        self.assertAlmostEqual(result['domains']['ALL']['pooled_micro_f1'],2/3)
        self.assertEqual(result['dataset_equal']['f1'],.5)

    def test_memory_reset_makes_repeat_and_reverse_independent(self):
        model=Stateful().eval();snapshot=initial_memory(model)
        a=torch.ones(1,3,2,2);b=torch.ones(1,3,2,2)*2
        first=infer_one(model,a,snapshot)
        infer_one(model,b,snapshot)
        second=infer_one(model,a,snapshot)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertEqual(snapshot['']['memory_bank'].item(),0.)
        reset_memory(model,snapshot)
        torch.testing.assert_close(model.memory_bank,torch.tensor([0.]))

    def test_without_reset_stateful_model_is_order_dependent(self):
        model=Stateful().eval();image=torch.ones(1,3,2,2)
        self.assertFalse(torch.equal(model(image),model(image)))

    def test_forged_same_key_membership_rejected_before_gt_open(self):
        frozen={'sample_key':'ok','subset':'new_dev','dataset':'A'}
        forged={**frozen,'dataset':'B'}
        with self.assertRaises(ValueError):
            load_dev_gt(forged,{},(2,2),{'ok':frozen},Path('/does-not-exist/log.jsonl'))

    def test_extreme_logits_are_finite_and_report_saturation(self):
        for value in (-1000.,1000.):
            result=output_diagnostics(np.full((2,2),value))
            self.assertTrue(result['saturated_probability_mean'])
            self.assertTrue(np.isfinite(result['probability_quantiles']).all())
        with self.assertRaises(ValueError):
            output_diagnostics(np.array([[np.nan]]))

    def test_invalid_role_rejected_before_gt_open(self):
        row={'sample_key':'bad','subset':'new_confirm'}
        with self.assertRaises(ValueError):
            load_dev_gt(row,{},(2,2),{},Path('/does-not-exist/log.jsonl'))

    def test_gt_binary_nearest_pair_and_access_log(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'mask.png';log=Path(directory)/'access.jsonl'
            Image.fromarray(np.array([[0,255],[255,0]],np.uint8)).save(path)
            row={'sample_key':'ok','subset':'new_dev','dataset':'A','sample_id':'id',
                 'image_sha256':'rgb'}
            source={'split':'train','dataset':'A','id':'id','image_sha256':'rgb',
                    'prepared_mask':str(path),'width':2,'height':2}
            out=load_dev_gt(row,source,(2,2),{'ok':row},log)
            self.assertEqual(out.shape,(224,224))
            self.assertTrue(out[0,-1])
            self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(),log.read_text())
            source['width']=3
            with self.assertRaises(ValueError):
                load_dev_gt(row,source,(2,2),{'ok':row},log)


if __name__=='__main__':
    unittest.main()
