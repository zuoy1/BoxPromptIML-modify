"""区域训练入口的行为契约，先于实现。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fecd.training.region_data import spatial
from fecd.training.region_engine import train_region_step
from fecd.training.region_access import RegionReadAudit


class RuntimeTests(unittest.TestCase):
    def _batch(self, image, target, boxes, op):
        return {'image': spatial(image, op), 'target': spatial(target, op),
                'canonical_target': target, 'box_union': boxes,
                'mu': target*.8+.05, 'var': torch.zeros_like(image),
                'confidence': torch.ones_like(image), 'operation': [op],
                'sample_key': ['same'], 'dataset': ['synthetic'], 'num_boxes': [2]}

    def test_augmented_forward_selection_loss_and_update_match_canonical_for_all_ops(self):
        grid = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)/10
        target, boxes = (grid > 1).float(), grid > .5
        for image in (grid, torch.full_like(grid, 40)):
            references = []
            for op in ('identity', 'hflip', 'vflip', 'rot90:1', 'rot90:2', 'rot90:3'):
                model = torch.nn.Conv2d(1, 1, 1)
                with torch.no_grad():
                    model.weight.fill_(.7)
                    model.bias.fill_(-.2)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                batch = self._batch(image, target, boxes, op)
                config = {'device': 'cpu', 'clip_norm': .5, 'seed': 20261029, 'variant': 'R1'}
                record, _, masks = train_region_step(model, optimizer, batch, config, 0)
                references.append((record, masks, model.weight.detach().clone()))
            for record, masks, weight in references[1:]:
                self.assertTrue(np.array_equal(masks, references[0][1]))
                # 旋转改变FP32归约顺序；容差限定为相对约八个machine epsilon。
                self.assertLessEqual(abs(record['loss']-references[0][0]['loss']),
                                     1e-6*max(1, abs(references[0][0]['loss'])))
                self.assertTrue(torch.allclose(weight, references[0][2], atol=1e-7, rtol=0))

    def test_diagnostics_report_actual_float32_saturated_selection_probability(self):
        image = torch.arange(17, 25, dtype=torch.float32).reshape(1, 1, 2, 4)
        target = torch.zeros_like(image)
        boxes = torch.ones_like(image, dtype=torch.bool)
        model = torch.nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        batch = self._batch(image, target, boxes, 'identity')
        config = {'device': 'cpu', 'clip_norm': .5, 'seed': 20261029, 'variant': 'R1'}
        _, rows, _ = train_region_step(model, optimizer, batch, config, 0)
        self.assertEqual(rows[0]['regions']['in']['selected']['student_probability']['quantiles'], [1]*5)

    def test_six_geometries_round_trip_keeps_all_maps_aligned(self):
        x = torch.arange(25).reshape(1, 5, 5)
        for op in ('identity', 'hflip', 'vflip', 'rot90:1', 'rot90:2', 'rot90:3'):
            moved = spatial(x, op)
            self.assertTrue(torch.equal(spatial(moved, op, inverse=True), x))
            self.assertTrue(torch.equal(spatial(x > 11, op), moved > 11))

    def test_real_update_obeys_region_budget_and_clipping_on_cpu(self):
        image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)/10
        target = (image > 1).float()
        boxes = image > .5
        batch = {'image': image, 'target': target, 'canonical_target': target,
                 'box_union': boxes, 'mu': target*.8+.05, 'var': torch.zeros_like(image),
                 'confidence': torch.ones_like(image), 'operation': ['identity'],
                 'sample_key': ['tiny'], 'dataset': ['synthetic'], 'num_boxes': [2]}
        for variant in ('R0', 'R1'):
            model = torch.nn.Conv2d(1, 1, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            before = model.weight.detach().clone()
            config = {'device': 'cpu', 'clip_norm': .5, 'seed': 20261029, 'variant': variant}
            record, rows, masks = train_region_step(model, optimizer, batch, config, 0)
            self.assertTrue(record['finite'])
            self.assertLessEqual(record['gradient_norm_after'], .50001)
            self.assertFalse(torch.equal(before, model.weight))
            self.assertEqual(len(rows), 1)
            self.assertEqual(masks.shape, (1, 4, 4))
            self.assertEqual(rows[0]['selected_positive'], int(target.sum())//2)
            self.assertEqual(rows[0]['selected_negative_in'], int(((target==0)&boxes).sum())//2)
            self.assertEqual(rows[0]['selected_negative_out'], int(((target==0)&~boxes).sum())//2)

    def test_boundary_rejects_non_allowlisted_data_and_historical_writes(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)/'output'
            output.mkdir()
            rgb = Path(root)/'rgb.png'
            rgb.write_bytes(b'rgb')
            with RegionReadAudit([rgb], [], output) as audit:
                self.assertEqual(rgb.read_bytes(), b'rgb')
                with self.assertRaises(PermissionError):
                    Path('/home/datasets/new_confirm/never-open.png').read_bytes()
                with self.assertRaises(PermissionError):
                    rgb.write_bytes(b'changed')
            self.assertEqual(rgb.read_bytes(), b'rgb')
            self.assertEqual(len(audit.denied), 2)


if __name__ == '__main__':
    unittest.main()
