"""区域分层负类选择的行为契约。"""
import unittest

import numpy as np
import torch

from fecd.training.region_negative_selection import select_region_pixels
from fecd.datasets.transforms import apply_spatial_transform
from fecd.training.losses import class_balanced_masked_bce


class RegionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.target = np.zeros((1, 8, 8), dtype=np.float32)
        self.target[0, :2, :3] = 1
        self.box = np.zeros_like(self.target, dtype=bool)
        self.box[0, :4, :5] = True
        self.score = np.linspace(.01, .99, 64).reshape(self.target.shape)

    def choose(self, variant, epoch=0, score=None, target=None, box=None):
        return select_region_pixels(
            self.target if target is None else target,
            self.box if box is None else box,
            variant, 20261029, 'domain:sample', epoch,
            probabilities=self.score if score is None else score)

    def test_positive_set_and_each_region_budget_match_between_arms(self):
        for epoch in (0, 1, 19):
            random_mask, random_counts = self.choose('R0', epoch)
            hard_mask, hard_counts = self.choose('R1', epoch)
            self.assertEqual(random_counts, hard_counts)
            self.assertEqual(random_counts['selected_pos'], 3)
            np.testing.assert_array_equal(
                random_mask & (self.target == 1), hard_mask & (self.target == 1))
            for region in (self.box, ~self.box):
                candidate = region & (self.target == 0)
                self.assertEqual(int((random_mask & candidate).sum()), int(candidate.sum()) // 2)
                self.assertEqual(int((hard_mask & candidate).sum()), int(candidate.sum()) // 2)

    def test_hard_selection_takes_highest_probability_in_each_region(self):
        selection, _ = self.choose('R1')
        for region in (self.box, ~self.box):
            candidate = region & (self.target == 0)
            selected = self.score[candidate & selection]
            omitted = self.score[candidate & ~selection]
            self.assertGreaterEqual(float(selected.min()), float(omitted.max()))

    def test_positive_selection_is_fixed_but_random_negatives_change_each_epoch(self):
        masks = [self.choose('R0', epoch)[0] for epoch in range(3)]
        for mask in masks[1:]:
            np.testing.assert_array_equal(mask & (self.target == 1), masks[0] & (self.target == 1))
            self.assertFalse(np.array_equal(mask & (self.target == 0), masks[0] & (self.target == 0)))

    def test_equal_probabilities_use_seeded_random_order_and_are_repeatable(self):
        tied = np.full_like(self.score, .5)
        hard, counts = self.choose('R1', score=tied)
        random_mask, random_counts = self.choose('R0')
        np.testing.assert_array_equal(hard, random_mask)
        np.testing.assert_array_equal(hard, self.choose('R1', score=tied)[0])
        self.assertEqual(counts, random_counts)
        indices = np.flatnonzero((self.box & (self.target == 0)).ravel())
        chosen = np.flatnonzero((hard & self.box & (self.target == 0)).ravel())
        self.assertFalse(np.array_equal(chosen, indices[:len(indices) // 2]))

    def test_empty_singleton_and_all_positive_classes_follow_floor_without_refill(self):
        cases = [
            (np.array([[1, 0]], dtype=np.float32), np.array([[True, True]]), 0),
            (np.array([[0, 0]], dtype=np.float32), np.array([[True, False]]), 0),
            (np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=bool), 2),
            (np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=bool), 2),
        ]
        for target, box, selected in cases:
            for arm in ('R0', 'R1'):
                with self.subTest(target=target.tolist(), arm=arm):
                    mask, counts = self.choose(arm, target=target, box=box,
                                               score=np.full_like(target, .5))
                    self.assertEqual(int(mask.sum()), selected)
                    self.assertEqual(counts['selected_total'], selected)

    def test_multiple_box_union_never_overwrites_teacher_positive_class(self):
        union = np.zeros_like(self.box)
        union[0, :2, :2] = True
        union[0, 5:, 5:] = True
        for arm in ('R0', 'R1'):
            mask, counts = self.choose(arm, box=union)
            self.assertEqual(counts['candidate_pos'], int(self.target.sum()))
            self.assertEqual(counts['selected_pos'], int(self.target.sum()) // 2)
            self.assertEqual(counts['candidate_neg'], int((self.target == 0).sum()))
            self.assertEqual(int((mask & (self.target == 1)).sum()), 3)

    def test_detached_selection_and_ignored_bce_gradients_are_exactly_zero(self):
        logits = torch.tensor(self.score * 6 - 3, dtype=torch.float32).unsqueeze(0)
        logits.requires_grad_()
        mask, counts = self.choose('R1', score=logits.sigmoid()[0])
        selection = torch.from_numpy(mask).unsqueeze(0)
        target = torch.from_numpy(self.target).unsqueeze(0)
        self.assertFalse(selection.requires_grad)
        self.assertIsNone(selection.grad_fn)
        loss, detail = class_balanced_masked_bce(logits, target, selection)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[~selection] == 0))
        self.assertTrue(torch.all(logits.grad[selection & (target == 0)] > 0))
        self.assertEqual(int(detail['selected_neg']), counts['selected_neg'])
        direct = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
        expected = .5 * direct[selection & (target == 1)].mean()
        expected += .5 * direct[selection & (target == 0)].mean()
        torch.testing.assert_close(loss, expected)

    def test_synchronous_geometry_keeps_mask_target_box_and_probabilities_aligned(self):
        canonical, _ = self.choose('R1')
        image = np.repeat(self.score[0, :, :, None], 3, axis=2)
        for operation in ('identity', 'hflip', 'vflip', 'rot90:1', 'rot90:2', 'rot90:3'):
            _, target, box, probability, mask = apply_spatial_transform(
                image, self.target[0], self.box[0], self.score[0], canonical[0],
                operation=operation)
            for region in (box, ~box):
                candidate = region & (target == 0)
                self.assertGreaterEqual(float(probability[candidate & mask].min()),
                                        float(probability[candidate & ~mask].max()))
            self.assertEqual(int((mask & (target == 1)).sum()), 3)

    def test_two_dimensional_and_channel_inputs_have_identical_decisions(self):
        mask, counts = self.choose('R1')
        planar, planar_counts = self.choose('R1', target=self.target[0], box=self.box[0],
                                           score=self.score[0])
        np.testing.assert_array_equal(mask[0], planar)
        self.assertEqual(counts, planar_counts)
        self.assertEqual(mask.dtype, np.bool_)

    def test_integer_probabilities_keep_descending_numeric_order(self):
        target = np.zeros((2, 4), dtype=np.float32)
        box = np.zeros_like(target, dtype=bool)
        score = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.uint8)
        mask, _ = self.choose('R1', target=target, box=box, score=score)
        np.testing.assert_array_equal(mask, score.astype(bool))

    def test_random_arm_needs_no_student_output_and_empty_loss_stays_differentiable(self):
        random_mask, counts = select_region_pixels(
            self.target, self.box, 'R0', 20261029, 'domain:sample', 0)
        np.testing.assert_array_equal(random_mask, self.choose('R0')[0])
        self.assertGreater(counts['selected_total'], 0)
        target = np.array([[1, 0]], dtype=np.float32)
        mask, _ = self.choose('R1', target=target, box=np.ones_like(target, dtype=bool),
                              score=np.full_like(target, .5))
        logits = torch.zeros((1, 1, 1, 2), requires_grad=True)
        loss, _ = class_balanced_masked_bce(
            logits, torch.from_numpy(target)[None, None], torch.from_numpy(mask)[None, None])
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.all(logits.grad == 0))

    def test_invalid_shapes_labels_probabilities_and_identity_fail_fast(self):
        invalid = [
            {'target': np.zeros((0, 0)), 'box_union': np.zeros((0, 0), dtype=bool)},
            {'target': np.zeros((2, 8, 8))},
            {'target': np.full_like(self.target, .5)},
            {'box_union': np.zeros((8, 7), dtype=bool)},
            {'box_union': np.zeros_like(self.target, dtype=float)},
            {'probabilities': None},
            {'probabilities': np.full_like(self.score, np.nan)},
            {'probabilities': np.full_like(self.score, 1.01)},
            {'probabilities': np.full_like(self.score, -.01)},
            {'variant': 'P3F'}, {'seed': True}, {'epoch': -1}, {'sample_key': ''},
        ]
        for replacement in invalid:
            kwargs = dict(target=self.target, box_union=self.box, variant='R1', seed=1,
                          sample_key='sample', epoch=0, probabilities=self.score)
            kwargs.update(replacement)
            with self.subTest(fields=list(replacement)):
                with self.assertRaises(ValueError):
                    select_region_pixels(**kwargs)


if __name__ == '__main__':
    unittest.main()
