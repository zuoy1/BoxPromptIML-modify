"""4F1-V 的数值边界、损坏记录和原验收逻辑保留契约。"""
import ast
import inspect
import math
import unittest

import numpy as np

from fecd.tools.stage4f1_v2_checks import check_loss, check_metrics, read, rows


REGRESSIONS = (
    {'loss': 2.0073587894439697, 'pos_mean': 3.667693614959717,
     'neg_mean': 0.34702417254447937, 'pos_contribution': 1.8338468074798584,
     'neg_contribution': 0.17351208627223969},
    {'loss': 2.1430554389953613, 'pos_mean': 3.1435434818267822,
     'neg_mean': 1.1425676345825195, 'pos_contribution': 1.5717717409133911,
     'neg_contribution': 0.5712838172912598},
    {'loss': 2.018709182739258, 'pos_mean': 2.669168472290039,
     'neg_mean': 1.3682501316070557, 'pos_contribution': 1.3345842361450195,
     'neg_contribution': 0.6841250658035278},
)
NORMAL = {'loss': 1.0, 'pos_mean': 0.5, 'neg_mean': 1.5,
          'pos_contribution': 0.25, 'neg_contribution': 0.75}


class LossValidationTests(unittest.TestCase):
    def test_loss_three_original_false_alarms_pass(self):
        for row in REGRESSIONS:
            with self.subTest(loss=row['loss']):
                self.assertGreaterEqual(abs(row['loss'] - row['pos_contribution']
                                           - row['neg_contribution']), 1e-7)
                check_loss(row)

    def test_loss_normal_float32_records_pass(self):
        check_loss(NORMAL)
        check_loss(dict.fromkeys(NORMAL, 0.0))
        rng = np.random.default_rng(241)
        for pair in rng.uniform(0, 30, size=(500, 2)).astype(np.float32):
            positive, negative = np.float32(pair[0] * .5), np.float32(pair[1] * .5)
            row = {'loss': float(np.float32(positive + negative)),
                   'pos_mean': float(pair[0]), 'neg_mean': float(pair[1]),
                   'pos_contribution': float(positive), 'neg_contribution': float(negative)}
            check_loss(row)

    def test_loss_tampered_contribution_with_consistent_mean_fails(self):
        row = {**NORMAL, 'pos_contribution': 0.375, 'pos_mean': 0.75}
        with self.assertRaises(ValueError):
            check_loss(row)

    def test_loss_single_ulp_error_below_old_tolerance_fails(self):
        row = {**NORMAL, 'loss': 1.0 + 2 ** -23}
        row.update(pos_contribution=.125, neg_contribution=.375,
                   pos_mean=.25, neg_mean=.75, loss=.5 + 2 ** -24)
        self.assertLess(abs(row['loss'] - .5), 1e-7)
        with self.assertRaises(ValueError):
            check_loss(row)

    def test_loss_compensating_contribution_changes_fail(self):
        row = {**NORMAL, 'pos_contribution': .375, 'neg_contribution': .625}
        with self.assertRaises(ValueError):
            check_loss(row)

    def test_loss_python_float_sum_instead_of_float32_fails(self):
        row = dict(REGRESSIONS[0])
        row['loss'] = row['pos_contribution'] + row['neg_contribution']
        with self.assertRaises(ValueError):
            check_loss(row)

    def test_loss_nonfinite_or_unrepresentable_values_fail(self):
        for field in NORMAL:
            for value in (math.nan, math.inf, -math.inf, 1e100, .1, -1.0, -0.0):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    check_loss({**NORMAL, field: value})

    def test_loss_missing_or_nonnumeric_fields_fail(self):
        for field in NORMAL:
            row = dict(NORMAL)
            del row[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_loss(row)
            for value in ('1.0', True, None, [], {}):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    check_loss({**NORMAL, field: value})


class MetricsValidationTests(unittest.TestCase):
    def test_metrics_valid_confusion_counts_and_empty_cases_pass(self):
        check_metrics([{'tp': 2, 'fp': 1, 'fn': 1, 'pixels': 16,
                        'f1': 2 / 3, 'iou': .5, 'gt_positive_fraction': 3 / 16}])
        check_metrics([{'tp': 0, 'fp': 0, 'fn': 0, 'pixels': 16,
                        'f1': 1.0, 'iou': 1.0, 'gt_positive_fraction': 0.0}])

    def test_metrics_changed_f1_iou_or_confusion_counts_fail(self):
        normal = {'tp': 2, 'fp': 1, 'fn': 1, 'pixels': 16,
                  'f1': 2 / 3, 'iou': .5, 'gt_positive_fraction': 3 / 16}
        for field, value in (('f1', .7), ('iou', .6), ('tp', -1), ('fp', True),
                             ('fn', 1.5), ('pixels', 2), ('gt_positive_fraction', .4)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_metrics([{**normal, field: value}])


class OriginalLogicTests(unittest.TestCase):
    def test_training_v2_preserves_original_non_numeric_logic(self):
        from fecd.tools import check_stage4f1 as original
        from fecd.tools import check_stage4f1_v2 as revised

        before = ast.parse(inspect.getsource(original.check_training))
        after = ast.parse(inspect.getsource(revised.check_training))
        replacements = 0
        function = before.body[0]
        for node in ast.walk(function):
            if isinstance(node, ast.Assert) and '1e-07' in ast.unparse(node):
                if "step['loss']" in ast.unparse(node):
                    for parent in ast.walk(function):
                        for field, value in ast.iter_fields(parent):
                            if isinstance(value, list) and node in value:
                                value[value.index(node)] = ast.parse('check_loss(step)').body[0]
                                replacements += 1
        self.assertEqual(replacements, 1)
        self.assertEqual(ast.dump(before), ast.dump(after))


class StrictReaderTests(unittest.TestCase):
    def test_readers_corrupt_json_duplicate_keys_and_nonfinite_fail(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            for content in ('{"loss":NaN}', '{"loss":1,"loss":2}',
                            '{"loss":1e999}', '{"loss":', '{"loss":Infinity}'):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read(path)
            path.write_text('{"loss": 1.0}\n{"loss": NaN}\n')
            with self.assertRaises(ValueError):
                rows(path)


class CheckpointContentTests(unittest.TestCase):
    def test_checkpoint_tensor_and_memory_changes_change_content_hash(self):
        import torch
        from fecd.tools.stage4f1_v2_checks import checkpoint_hash

        payload = {'model': {'weight': torch.tensor([1.0, 2.0])},
                   'memory': {'attention': torch.tensor([3.0])}}
        original = checkpoint_hash(payload)
        changed_weight = {'model': {'weight': torch.tensor([1.0, 2.5])},
                          'memory': payload['memory']}
        changed_memory = {'model': payload['model'],
                          'memory': {'attention': torch.tensor([4.0])}}
        self.assertNotEqual(checkpoint_hash(changed_weight), original)
        self.assertNotEqual(checkpoint_hash(changed_memory), original)
        with self.assertRaises(ValueError):
            checkpoint_hash({'model': {'weight': torch.tensor([float('nan')])}, 'memory': {}})


class BoundaryTests(unittest.TestCase):
    def test_boundary_dataset_reads_and_foreign_writes_are_rejected(self):
        import os
        from fecd.tools.stage4f1_v2_io import ReadBoundary

        guard = ReadBoundary('/tmp/v2-output')
        for path in ('/home/datasets/raw/mask.png', '/hy-tmp/datasets/raw/image.jpg'):
            with self.assertRaises(PermissionError):
                guard('open', (path, 'r', os.O_RDONLY))
        with self.assertRaises(PermissionError):
            guard('open', ('/tmp/original-status.json', 'w', os.O_WRONLY))
        guard('open', ('/tmp/v2-output/result.json', 'x', os.O_CREAT | os.O_WRONLY))
        guard('open', ('/tmp/existing-run/final.pt', 'r', os.O_RDONLY))
        self.assertIn('/tmp/existing-run/final.pt', guard.reads)


class FailedDeliveryTests(unittest.TestCase):
    def test_incomplete_run_writes_failure_with_unchanged_historical_status(self):
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        from tempfile import TemporaryDirectory
        from fecd.tools.stage4f1_v2_io import sha
        from fecd.tools.stage4f1_v2_run import RUN_ID

        with TemporaryDirectory() as directory:
            base = Path(directory)
            run = base / RUN_ID
            run.mkdir()
            (run / 'artifacts').mkdir()
            status = run / 'status.json'
            status.write_text('{"status":"FAILED"}')
            digest = sha(status)
            output = base / 'v2'
            output.mkdir()
            (output / 'inputs_before.json').write_text(json.dumps({'files': {
                str(status): {'sha256': digest, 'bytes': status.stat().st_size}}}))
            completed = subprocess.run([sys.executable, '-m', 'fecd.tools.check_stage4f1_v2',
                                        '--input-run', str(run), '--output-dir', str(output)],
                                       capture_output=True, text=True,
                                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                            'PYTHONDONTWRITEBYTECODE': '1'})
            self.assertNotEqual(completed.returncode, 0)
            result = json.loads((output / 'validation_v2.json').read_text())
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['checks']['P0']['status'], 'FAIL')
            self.assertEqual(sha(status), digest)


if __name__ == '__main__':
    unittest.main()
