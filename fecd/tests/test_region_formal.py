"""正式运行位置、冻结配置和塌缩门禁的行为契约。"""
import unittest

from fecd.training.region_formal_protocol import configuration, validate_position, collapse_gate


class FormalTests(unittest.TestCase):
    def test_formal_configuration_differs_only_in_variant(self):
        left, right = configuration('R0'), configuration('R1')
        self.assertEqual({k:v for k,v in left.items() if k not in ('config_hash','variant')},
                         {k:v for k,v in right.items() if k not in ('config_hash','variant')})
        self.assertEqual(left['epochs'],20)
        self.assertEqual(left['updates'],17660)
        self.assertFalse(left['automatic_shutdown'])
        with self.assertRaises(ValueError):configuration('P1')

    def test_resume_only_complete_epoch_positions_accepted(self):
        for epoch in (1,10,20):
            self.assertEqual(validate_position({'epoch':epoch,'next_batch':0},epoch*883),epoch)
        for position,update in [({'epoch':0,'next_batch':0},0),({'epoch':1,'next_batch':1},883),
                                ({'epoch':21,'next_batch':0},18543),({'epoch':1,'next_batch':0},882)]:
            with self.assertRaises(ValueError):validate_position(position,update)

    def test_global_collapse_requires_two_severe_epochs_not_thirteen_cases(self):
        safe={'images':3530,'all_zero_outputs':13,'near_full_outputs':0}
        severe={'images':3530,'all_zero_outputs':3200,'near_full_outputs':0}
        self.assertFalse(collapse_gate([safe,severe]))
        self.assertFalse(collapse_gate([severe,safe]))
        self.assertTrue(collapse_gate([safe,severe,severe]))
        self.assertFalse(collapse_gate([severe]))


if __name__=='__main__':unittest.main()

class ResumeTests(unittest.TestCase):
    def test_real_disk_restore_and_incomplete_epoch_isolation(self):
        import tempfile
        from pathlib import Path
        import torch
        import numpy as np
        from fecd.training.checkpoint import save_checkpoint
        from fecd.training.f1_state import training_state_hash
        from fecd.training.region_formal_protocol import sha,write
        from fecd.training.region_formal_runner import resume_boundary
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);done=root/'epoch01';done.mkdir()
            model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters())
            model(torch.ones(1,2)).sum().backward();optimizer.step()
            config={'config_hash':'synthetic'}
            save_checkpoint(done/'checkpoint.pt',model,optimizer,config,883,{'epoch':1,'next_batch':0})
            expected=training_state_hash(model,optimizer)
            np.savez(done/'selection.npz',masks=np.zeros((1,1),np.uint8))
            summary={'status':'PASS','epoch':1,'images':3530,'total_updates':883,
                     'checkpoint_sha256':sha(done/'checkpoint.pt'),'selection_sha256':sha(done/'selection.npz'),
                     'complete_state_hash':expected,'all_zero_outputs':0,'near_full_outputs':0,
                     'parameter_update_verified':True}
            write(done/'summary.json',summary)
            (root/'epoch02').mkdir();(root/'epoch02/partial').write_text('incomplete')
            fresh=torch.nn.Linear(2,1);adam=torch.optim.Adam(fresh.parameters())
            result=resume_boundary(root,fresh,adam,config)
            self.assertEqual(len(result),1)
            self.assertEqual(training_state_hash(fresh,adam),expected)
            self.assertFalse((root/'epoch02').exists())
            self.assertEqual(len(list(root.glob('interrupted_epoch02_*'))),1)

    def test_resume_rejects_persisted_failed_parameter_update_gate(self):
        import tempfile
        from pathlib import Path
        import json
        import torch
        from fecd.training.region_formal_runner import resume_boundary
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);done=root/'epoch01';done.mkdir()
            (done/'summary.json').write_text(json.dumps({'status':'PASS','epoch':1,'images':3530,
                'total_updates':883,'all_zero_outputs':0,'near_full_outputs':0,
                'parameter_update_verified':False}))
            model=torch.nn.Linear(1,1);optimizer=torch.optim.Adam(model.parameters())
            with self.assertRaises(ValueError):resume_boundary(root,model,optimizer,{'config_hash':'synthetic'})

class ParameterGateTests(unittest.TestCase):
    def test_memory_change_without_parameter_update_is_rejected(self):
        import torch
        from fecd.training.region_formal_epoch import require_parameter_change
        model=torch.nn.Linear(2,1)
        before=[p.detach().clone() for p in model.parameters()]
        model.memory_bank=torch.ones(1)
        with self.assertRaises(ValueError):require_parameter_change(model,before)
        with torch.no_grad():model.weight.add_(.01)
        require_parameter_change(model,before)

    def test_atomic_summary_rejects_existing_and_keeps_old_contents(self):
        import tempfile,json
        from pathlib import Path
        from fecd.training.region_formal_protocol import commit_summary
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'summary.json'
            commit_summary(path,{'status':'PASS'})
            with self.assertRaises(FileExistsError):commit_summary(path,{'status':'FAIL'})
            self.assertEqual(json.loads(path.read_text()),{'status':'PASS'})

    def test_failed_epoch_without_commit_marker_is_never_replayed(self):
        import tempfile
        from pathlib import Path
        import torch
        from fecd.training.region_formal_runner import resume_boundary
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'epoch01').mkdir();(root/'epoch01/BLOCKED.json').write_text('{}')
            model=torch.nn.Linear(1,1);opt=torch.optim.Adam(model.parameters())
            with self.assertRaises(ValueError):resume_boundary(root,model,opt,{'config_hash':'synthetic'})

class RecoveryFailureTests(unittest.TestCase):
    def test_entrypoint_failure_record_prevents_resume_replay(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from fecd.tools.train_region_formal import main
        from fecd.training.region_formal_runner import resume_boundary
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);arm=root/'R0';arm.mkdir()
            with patch('fecd.tools.train_region_formal.OUTPUT',root), \
                 patch('fecd.tools.train_region_formal.run',side_effect=ValueError('NaN fixture')), \
                 patch('sys.argv',['formal','--variant','R0']):
                with self.assertRaises(ValueError):main()
            self.assertEqual(len(list(root.glob('BLOCKED_R0_*.json'))),1)
            model=torch.nn.Linear(1,1);opt=torch.optim.Adam(model.parameters())
            with self.assertRaisesRegex(ValueError,'BLOCKED'):
                resume_boundary(arm,model,opt,{'config_hash':'synthetic'})

    def _checkpoint(self,root,number,payload_epoch,collapsed=False):
        import torch,numpy as np
        from fecd.training.checkpoint import save_checkpoint
        from fecd.training.f1_state import training_state_hash
        from fecd.training.region_formal_protocol import sha,write
        target=root/f'epoch{number:02d}';target.mkdir()
        model=torch.nn.Linear(1,1);opt=torch.optim.Adam(model.parameters())
        save_checkpoint(target/'checkpoint.pt',model,opt,{'config_hash':'synthetic'},
                        payload_epoch*883,{'epoch':payload_epoch,'next_batch':0})
        np.savez(target/'selection.npz',masks=np.zeros((1,1),np.uint8))
        write(target/'summary.json',{'status':'PASS','epoch':number,'images':3530,
            'total_updates':number*883,'parameter_update_verified':True,
            'all_zero_outputs':3200 if collapsed else 0,'near_full_outputs':0,
            'checkpoint_sha256':sha(target/'checkpoint.pt'),
            'selection_sha256':sha(target/'selection.npz'),
            'complete_state_hash':training_state_hash(model,opt)})
        return model,opt

    def test_legal_checkpoint_position_disagrees_with_epoch_directory_rejected(self):
        import tempfile
        from pathlib import Path
        from fecd.training.region_formal_runner import resume_boundary
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model,opt=self._checkpoint(root,1,2)
            with self.assertRaisesRegex(ValueError,'轮次'):
                resume_boundary(root,model,opt,{'config_hash':'synthetic'})

    def test_two_collapsed_old_pass_summaries_rejected_on_resume(self):
        import tempfile
        from pathlib import Path
        from fecd.training.region_formal_runner import resume_boundary
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self._checkpoint(root,1,1,True)
            model,opt=self._checkpoint(root,2,2,True)
            with self.assertRaisesRegex(ValueError,'门禁'):
                resume_boundary(root,model,opt,{'config_hash':'synthetic'})
