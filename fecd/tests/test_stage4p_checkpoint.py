"""Full training-state restoration behavior without the production model."""
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from fecd.training.checkpoint import save_checkpoint, restore_checkpoint, state_hash


class Stateful(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(2,1)
        self.memory_bank=torch.ones(2)


class CheckpointTests(unittest.TestCase):
    def test_resume_restores_parameters_memory_optimizer_rng_and_position(self):
        model=Stateful();optim=torch.optim.Adam(model.parameters(),lr=.01)
        model.linear(torch.ones(2)).sum().backward();optim.step()
        model.memory_bank=torch.arange(3,dtype=torch.float32)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt'
            save_checkpoint(path,model,optim,{'variant':'P2','config_hash':'x'},4,{'epoch':0,'next_batch':4})
            expected=(random.random(),np.random.random(),torch.rand(2))
            resumed=Stateful();newoptim=torch.optim.Adam(resumed.parameters(),lr=.01)
            payload=restore_checkpoint(path,resumed,newoptim,'x')
            self.assertEqual(payload['update'],4)
            self.assertEqual(payload['position'],{'epoch':0,'next_batch':4})
            self.assertEqual(state_hash(model),state_hash(resumed))
            torch.testing.assert_close(resumed.memory_bank,model.memory_bank)
            self.assertEqual(len(newoptim.state),len(optim.state))
            self.assertEqual(random.random(),expected[0])
            self.assertEqual(np.random.random(),expected[1])
            torch.testing.assert_close(torch.rand(2),expected[2])

    def test_wrong_configuration_and_existing_output_fail_closed(self):
        model=Stateful();optim=torch.optim.Adam(model.parameters())
        with TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt'
            save_checkpoint(path,model,optim,{'config_hash':'x'},0,{'epoch':0,'next_batch':0})
            with self.assertRaises(ValueError):restore_checkpoint(path,model,optim,'y')
            with self.assertRaises(FileExistsError):
                save_checkpoint(path,model,optim,{'config_hash':'x'},0,{})


if __name__=='__main__':
    unittest.main()
