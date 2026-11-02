"""Behavior checks for frozen selection budgets and class-balanced masked BCE."""
import math
import unittest

import numpy as np
import torch

from fecd.training.losses import class_balanced_masked_bce
from fecd.training.selection import select_pixels


class SelectionTests(unittest.TestCase):
    def test_random_and_confidence_keep_same_floor_half_per_class(self):
        target = np.asarray([1, 1, 1, 0, 0, 0, 0, 0], np.float32)
        score = np.linspace(0, 1, target.size)
        for mode in ('random', 'confidence'):
            mask, counts = select_pixels(target, score, mode, 42, 'sample', 0)
            self.assertEqual(mask.dtype, np.bool_)
            self.assertEqual(int(mask[target == 1].sum()), 1)
            self.assertEqual(int(mask[target == 0].sum()), 2)
            self.assertEqual(counts['selected_pos'], 1)
            self.assertEqual(counts['selected_neg'], 2)

    def test_selection_repeats_and_confidence_selects_highest_scores(self):
        target = np.asarray([1] * 8 + [0] * 8)
        score = np.arange(16, dtype=float) / 16
        for mode in ('random', 'confidence'):
            first, _ = select_pixels(target, score, mode, 21, 'key', 2)
            second, _ = select_pixels(target, score, mode, 21, 'key', 2)
            np.testing.assert_array_equal(first, second)
        chosen, _ = select_pixels(target, score, 'confidence', 21, 'key', 2)
        np.testing.assert_array_equal(np.flatnonzero(chosen), [4,5,6,7,12,13,14,15])

    def test_tied_confidence_uses_shared_random_priority(self):
        target = np.asarray([1] * 5 + [0] * 7)
        score = np.ones(12)
        random_mask, _ = select_pixels(target, score, 'random', 42, 'key', 3)
        confidence_mask, _ = select_pixels(target, score, 'confidence', 42, 'key', 3)
        np.testing.assert_array_equal(random_mask, confidence_mask)

    def test_one_pixel_class_and_empty_class_are_not_forced_selected(self):
        target = np.ones((1,1,1), dtype=np.float32)
        for mode in ('random', 'confidence'):
            chosen, counts = select_pixels(target, np.zeros_like(target), mode, 42, 's', 0)
            self.assertFalse(chosen.any())
            self.assertEqual(counts['candidate_neg'], 0)
        chosen, _ = select_pixels(target, np.zeros_like(target), 'all', 42, 's', 0)
        self.assertTrue(chosen.all())

    def test_selection_rejects_invalid_shapes_values_and_mode(self):
        with self.assertRaises(ValueError):
            select_pixels(np.asarray([.5]), np.asarray([.5]), 'random', 1, 's', 0)
        with self.assertRaises(ValueError):
            select_pixels(np.asarray([1]), np.asarray([float('nan')]), 'all', 1, 's', 0)
        with self.assertRaises(ValueError):
            select_pixels(np.asarray([1]), np.asarray([[.5]]), 'all', 1, 's', 0)
        with self.assertRaises(ValueError):
            select_pixels(np.asarray([1]), np.asarray([.5]), 'other', 1, 's', 0)


class MaskedBCETests(unittest.TestCase):
    def test_positive_and_negative_contributions_have_fixed_equal_weight(self):
        logits = torch.tensor([[[[0.,0.,0.,0.]]]], requires_grad=True)
        target = torch.tensor([[[[1.,0.,0.,0.]]]])
        mask = torch.ones_like(target, dtype=torch.bool)
        loss, details = class_balanced_masked_bce(logits, target, mask)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertAlmostEqual(details['pos_contribution'].item(), math.log(2)/2, places=6)
        self.assertAlmostEqual(details['neg_contribution'].item(), math.log(2)/2, places=6)
        loss.backward()
        self.assertAlmostEqual(logits.grad[0,0,0,0].item(), -.25, places=6)
        self.assertAlmostEqual(logits.grad[0,0,0,1].item(), 1/12, places=6)

    def test_ignored_pixels_have_zero_gradient(self):
        logits = torch.tensor([[[[1.,-2.,3.,-4.]]]], requires_grad=True)
        target = torch.tensor([[[[1.,0.,1.,0.]]]])
        selection = torch.tensor([[[[True,True,False,False]]]])
        loss, _ = class_balanced_masked_bce(logits, target, selection)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[...,2:], torch.zeros_like(logits.grad[...,2:])))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_single_class_weight_stays_half_and_empty_selection_is_differentiable_zero(self):
        for value, field in ((0.,'neg_contribution'),(1.,'pos_contribution')):
            logits = torch.zeros((1,1,2,2), requires_grad=True)
            target = torch.full_like(logits, value)
            loss, details = class_balanced_masked_bce(logits, target, torch.ones_like(target, dtype=torch.bool))
            self.assertAlmostEqual(loss.item(), math.log(2)/2, places=6)
            self.assertAlmostEqual(details[field].item(), math.log(2)/2, places=6)
            self.assertTrue(torch.isfinite(loss))
        logits = torch.randn((2,1,2,2), requires_grad=True)
        target = torch.zeros_like(logits)
        loss, details = class_balanced_masked_bce(logits, target, torch.zeros_like(target, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        self.assertEqual(details['selected_neg'].item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_loss_is_equal_image_weight_not_pixel_weight(self):
        logits = torch.tensor([[[[0.,0.]]], [[[2.,2.]]]], requires_grad=True)
        target = torch.tensor([[[[1.,0.]]], [[[1.,0.]]]])
        selection = torch.tensor([[[[True,True]]], [[[True,False]]]])
        loss, details = class_balanced_masked_bce(logits, target, selection)
        expected = (math.log(2) + .5 * math.log1p(math.exp(-2))) / 2
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertEqual(details['valid_pos_images'].item(),2)
        self.assertEqual(details['valid_neg_images'].item(),1)

    def test_loss_rejects_invalid_nonfinite_hard_target_and_mask(self):
        logits = torch.zeros((1,1,2,2), requires_grad=True)
        target = torch.zeros_like(logits)
        mask = torch.ones_like(logits, dtype=torch.bool)
        bad = target.clone(); bad[0,0,0,0] = .5
        with self.assertRaises(ValueError): class_balanced_masked_bce(logits,bad,mask)
        bad = logits.detach().clone(); bad[0,0,0,0] = float('inf')
        with self.assertRaises(ValueError): class_balanced_masked_bce(bad,target,mask)
        with self.assertRaises(ValueError): class_balanced_masked_bce(logits,target,mask.float())
        with self.assertRaises(ValueError): class_balanced_masked_bce(logits.squeeze(1),target.squeeze(1),mask.squeeze(1))


if __name__ == '__main__':
    unittest.main()
