"""Independent output paths preserve training identity and complete checkpoints."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fecd.training.f1_storage import resolve_output_root
from fecd.training.f1_protocol import configuration, verify_frozen
from fecd.training.f1_state import save_last, training_state_hash
from fecd.training.checkpoint import restore_checkpoint
from fecd.tests.test_stage4f1_training import ToyStudent


class StorageTests(unittest.TestCase):
    def test_custom_output_root_does_not_change_training_configuration(self):
        before = configuration('P0')
        with TemporaryDirectory() as directory:
            target = resolve_output_root(Path(directory)/'custom-output')
            self.assertEqual(target, Path(directory)/'custom-output')
        self.assertEqual(before, configuration('P0'))
        self.assertNotIn('output_root', before)
        verify_frozen()

    def test_relative_output_root_fails_before_creating_files(self):
        with self.assertRaises(ValueError):
            resolve_output_root('relative/output')

    def test_data_path_on_root_filesystem_is_rejected_before_writes(self):
        with patch('fecd.training.f1_storage.Path.stat', return_value=SimpleNamespace(st_dev=67)):
            with self.assertRaises(ValueError):
                resolve_output_root('/hy-tmp/fecd-storage-test')

    def test_atomic_rolling_checkpoint_and_continuation_use_custom_directory(self):
        config = configuration('P1')
        model = ToyStudent()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        model.update(optimizer)
        with TemporaryDirectory() as directory:
            root = resolve_output_root(directory)
            path = root/'last.pt'
            save_last(path, model, optimizer, config, 883, {'epoch':1,'next_batch':0})
            digest = training_state_hash(model, optimizer)
            expected = model.update(optimizer)
            expected_digest = training_state_hash(model, optimizer)
            resumed = ToyStudent()
            new_optimizer = torch.optim.Adam(resumed.parameters(), lr=.01)
            payload = restore_checkpoint(path, resumed, new_optimizer, config['config_hash'])
            self.assertEqual(training_state_hash(resumed, new_optimizer), digest)
            self.assertEqual(resumed.update(new_optimizer), expected)
            self.assertEqual(training_state_hash(resumed, new_optimizer), expected_digest)
            self.assertEqual(payload['position'], {'epoch':1,'next_batch':0})
            self.assertFalse(path.with_suffix('.pending.pt').exists())
            self.assertEqual(payload['config'], config)

