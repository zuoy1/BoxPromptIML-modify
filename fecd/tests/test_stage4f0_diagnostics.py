"""Different probability/area quantities are not interchangeable."""
import unittest
import torch

from fecd.training.f0_diagnostics import describe_output


class OutputTests(unittest.TestCase):
    def test_probability_mean_differs_from_threshold_area(self):
        z=torch.tensor([[[[-.01,.01,1.,3.]]]])
        t=torch.tensor([[[[0.,0.,0.,1.]]]])
        s=torch.tensor([[[[False,True,False,True]]]])
        d=describe_output(z,t,s)[0]
        self.assertEqual(d['student_positive_fraction'],.75)
        self.assertEqual(d['teacher_hard_positive_fraction'],.25)
        self.assertNotEqual(d['probability_mean'],.75)
        self.assertEqual(d['selected_positive_full_fraction'],.25)
        self.assertEqual(d['selected_negative_full_fraction'],.25)
        self.assertEqual(d['candidate_positive_count'],1)
        self.assertEqual(d['candidate_negative_count'],3)
        self.assertEqual(d['selected_positive_count'],1)
        self.assertEqual(d['selected_negative_count'],1)

    def test_empty_predicted_class_confidence_is_not_available(self):
        z=torch.full((1,1,4,4),-20.)
        d=describe_output(z,torch.zeros_like(z),torch.ones_like(z,dtype=torch.bool))[0]
        self.assertTrue(d['student_all_zero'])
        self.assertIsNone(d['predicted_positive_confidence'])
        self.assertTrue(d['probability_saturated'])
        self.assertAlmostEqual(d['predicted_negative_confidence'],1.)


if __name__=='__main__':
    unittest.main()
