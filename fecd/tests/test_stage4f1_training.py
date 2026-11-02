"""Formal epoch boundaries, frozen inputs, and complete resume contracts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from fecd.training.f1_protocol import configuration, epoch_batches, epoch_identity
from fecd.training.f1_state import save_last, training_state_hash
from fecd.training.checkpoint import restore_checkpoint


class ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.memory_bank = torch.ones(2)

    def update(self, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = self.linear(torch.ones(2)).square().sum()
        loss.backward()
        optimizer.step()
        self.memory_bank = self.memory_bank + 1
        return float(loss.detach())


class FormalTrainingTests(unittest.TestCase):
    def test_formal_config_has_twenty_epochs_and_rejects_dynamic_variant(self):
        for variant in ('P0', 'P1', 'P2F', 'P3F'):
            config = configuration(variant)
            self.assertEqual(config['epochs'], 20)
            self.assertEqual(config['steps_per_epoch'], 883)
            self.assertEqual(config['updates'], 17660)
        with self.assertRaises(ValueError):
            configuration('P2E')

    def test_epoch_order_consumes_every_image_and_preserves_partial_batch(self):
        keys = [f'image{i}' for i in range(3530)]
        for epoch in range(20):
            batches = list(epoch_batches(len(keys), 20261029, epoch, 4))
            self.assertEqual(len(batches), 883)
            self.assertEqual(len(batches[-1]), 2)
            self.assertEqual(sorted(i for batch in batches for i in batch), list(range(3530)))
            self.assertEqual(batches, list(epoch_batches(len(keys), 20261029, epoch, 4)))
            identity = epoch_identity(keys, 20261029, epoch)
            self.assertEqual(identity, epoch_identity(keys, 20261029, epoch))
        self.assertNotEqual(epoch_identity(keys, 20261029, 0), epoch_identity(keys, 20261029, 1))

    def test_epoch_boundary_resume_restores_adam_rng_memory_and_next_update(self):
        config = configuration('P0')
        model = ToyStudent()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        model.update(optimizer)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            save_last(path, model, optimizer, config, 883, {'epoch': 1, 'next_batch': 0})
            saved_hash = training_state_hash(model, optimizer)
            expected_loss = model.update(optimizer)
            expected_hash = training_state_hash(model, optimizer)
            resumed = ToyStudent()
            new_optimizer = torch.optim.Adam(resumed.parameters(), lr=.01)
            payload = restore_checkpoint(path, resumed, new_optimizer, config['config_hash'])
            self.assertEqual(payload['position'], {'epoch': 1, 'next_batch': 0})
            self.assertEqual(training_state_hash(resumed, new_optimizer), saved_hash)
            self.assertEqual(resumed.update(new_optimizer), expected_loss)
            self.assertEqual(training_state_hash(resumed, new_optimizer), expected_hash)
            save_last(path, resumed, new_optimizer, config, 1766, {'epoch': 2, 'next_batch': 0})
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload['position']['epoch'], 2)
            self.assertFalse(path.with_suffix('.pending.pt').exists())

