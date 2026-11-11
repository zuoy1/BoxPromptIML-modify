"""正式评估门禁及真实文件访问边界的契约。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path('/hy-tmp/BoxPromtIML/fecd_runs/stage5c1/code/evaluate.py')
spec = importlib.util.spec_from_file_location('stage5c1_evaluate', SCRIPT)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def complete(stage):
    for variant in ('R0', 'R1'):
        directory = stage/variant
        directory.mkdir()
        checkpoint = directory/'final.pt'
        checkpoint.write_bytes(variant.encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        save(directory/'epoch20/summary.json', {'status':'PASS','epoch':20,
             'total_updates':17660,'checkpoint_sha256':digest})
        save(directory/'result.json', {'status':'PASS','epochs':20,'logical_updates':17660,
             'variant':variant,'gt_reads':0,'automatic_shutdown':False,
             'initial_hash':'shared','final_checkpoint_sha256':digest})


class EvaluationGateTests(unittest.TestCase):
    def test_both_complete_and_same_initial_state_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            complete(stage)
            self.assertEqual(set(wrapper.complete_results(stage)), {'R0','R1'})
            for field, value in [('epochs',19),('logical_updates',17659),('status','BLOCKED'),
                                 ('gt_reads',1),('automatic_shutdown',True),('initial_hash','other')]:
                result = stage/'R1/result.json'
                original = json.loads(result.read_text())
                save(result, {**original,field:value})
                with self.assertRaises(ValueError):
                    wrapper.complete_results(stage)
                save(result, original)

    def test_missing_member_changed_checkpoint_or_existing_evaluation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                wrapper.complete_results(stage)
            complete(stage)
            (stage/'R1/final.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                wrapper.complete_results(stage)
            (stage/'evaluation').mkdir()
            with self.assertRaises(FileExistsError):
                wrapper.reserve_evaluation(stage)

    def test_checkpoint_uses_actual_nested_config_and_epoch_boundary(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'final.pt'
            config = wrapper.configuration('R0')
            payload = {'config':config,'position':{'epoch':20,'next_batch':0},'update':17660}
            torch.save(payload,checkpoint)
            wrapper.check_checkpoint('R0',checkpoint,{'config_hash':config['config_hash']})
            payload['position']['next_batch'] = 1
            torch.save(payload,checkpoint)
            with self.assertRaises(ValueError):
                wrapper.check_checkpoint('R0',checkpoint,{'config_hash':config['config_hash']})

    def test_true_open_audit_allows_only_stage_rgb_and_explicit_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root/'fecd/assets'
            assets.mkdir(parents=True)
            rgb, gt, metadata = [assets/name for name in ('rgb.png','mask.png','split.json')]
            for path in (rgb,gt,metadata):
                path.write_bytes(b'bytes')
            output = root/'output'
            output.mkdir()
            with wrapper.RegionReadAudit([rgb],[metadata],output) as audit:
                self.assertEqual(rgb.read_bytes(),b'bytes')
                self.assertEqual(metadata.read_bytes(),b'bytes')
                with self.assertRaises(PermissionError):
                    gt.read_bytes()
                with self.assertRaises(PermissionError):
                    rgb.write_bytes(b'changed')
                (output/'result.json').write_bytes(b'allowed')
            self.assertEqual(set(audit.opened),{str(rgb),str(metadata)})
            self.assertEqual(len(audit.denied),2)
            with wrapper.RegionReadAudit([gt],[metadata],output):
                self.assertEqual(gt.read_bytes(),b'bytes')
                with self.assertRaises(PermissionError):
                    rgb.read_bytes()


if __name__ == '__main__':
    unittest.main()
