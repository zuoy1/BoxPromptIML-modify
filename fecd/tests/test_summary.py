"""Behavior checks for aggregation of already-produced diagnostic metrics."""
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from fecd.diagnostics.metrics import error_detection_metrics, reliability_histogram
from fecd.diagnostics.risk_coverage import risk_coverage
from fecd.diagnostics.protocol import PROTOCOL_HASH
from fecd.tools.summarize_stage3 import decide, summarize


def _metric(errors, scores):
    output = error_detection_metrics(errors, scores)
    risk = risk_coverage(errors, scores)
    risk.pop('curve')
    output.update(risk)
    return output


class SummaryTests(unittest.TestCase):
    def test_all_and_dataset_use_same_images_and_paired_groups(self):
        errors = np.tile([1, 0, 1, 0], 25)
        baseline = np.tile([.8, .2, .9, .1], 25)
        improved = 1 - baseline
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(20):
                regions = {region: {method: _metric(errors, baseline if method == 'S0' else improved)
                                    for method in ('S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S_HP')}
                           for region in ('all', 'pos', 'high_pos', 'neg')}
                segmentation = {'tp': 5, 'fp': 2, 'tn': 3, 'fn': 1, 'f1': 10 / 13, 'iou': 5 / 8}
                rows.append({'sample_key': str(index), 'sample_id': str(index), 'dataset': 'Toy',
                             'source_group_id': f'g{index // 2}', 'gt_area_fraction': .1,
                             'area_stratum': '10-30%', 'regions': regions,
                             'segmentation': {'K1': segmentation, 'K8': segmentation},
                             'trusted': {'S0': {'pos': {'selected_count': 10, 'precision': .9,
                                                       'conditional_coverage': .1, 'full_coverage': .1}}}})
            (root / 'per_image.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            manifest = {'status': 'passed', 'requested': len(rows), 'evaluated': len(rows),
                        'protocol_hash': PROTOCOL_HASH,
                        'evaluated_sample_keys_sha256': hashlib.sha256(
                            '\n'.join(sorted(row['sample_key'] for row in rows)).encode()).hexdigest()}
            (root / 'run_manifest.json').write_text(json.dumps(manifest))
            arrays, hist_index = {}, []
            for dataset in ('ALL', 'Toy'):
                for area in ('ALL', '10-30%'):
                    for region, methods in rows[0]['regions'].items():
                        for method in methods:
                            key = f'h{len(hist_index):06}'
                            arrays[key] = reliability_histogram(errors, baseline if method == 'S0' else improved) * len(rows)
                            hist_index.append({'key': key, 'dataset': dataset, 'area_stratum': area,
                                               'region': region, 'method': method})
            np.savez(root / 'histograms.npz', **arrays)
            (root / 'hist_index.json').write_text(json.dumps(hist_index))
            result = summarize(root)
            self.assertEqual(result['decision']['quantitative_decision'], 'GO')
            self.assertEqual(result['datasets']['ALL']['n_groups'], 10)
            self.assertEqual(result['datasets']['Toy']['macro']['pos']['S4']['auroc']['mean'], 1)
            self.assertEqual(result['datasets']['ALL']['micro']['pos']['S4']['auroc'], 1)
            self.assertEqual(result['datasets']['ALL']['segmentation']['K8']['confusion']['tp'], 100)
            self.assertAlmostEqual(result['datasets']['ALL']['trusted']['S0']['pos']['micro']['precision'], .9)
            self.assertTrue((root / 'summary.json').is_file())
            self.assertEqual(len(result['risk_curve_paths']), 6)
            manifest['status'] = 'running'
            (root / 'run_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                summarize(root)
            manifest.update(status='passed', evaluated=19)
            (root / 'run_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                summarize(root)
            manifest.update(evaluated=20, protocol_hash='changed')
            (root / 'run_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                summarize(root)
            manifest.update(protocol_hash=PROTOCOL_HASH, evaluated_sample_keys_sha256='changed')
            (root / 'run_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                summarize(root)
            manifest['evaluated_sample_keys_sha256'] = hashlib.sha256(
                '\n'.join(sorted(row['sample_key'] for row in rows)).encode()).hexdigest()
            (root / 'run_manifest.json').write_text(json.dumps(manifest))
            arrays['h000001'] *= 2
            np.savez(root / 'histograms.npz', **arrays)
            with self.assertRaises(ValueError):
                summarize(root)
            arrays['h000001'] //= 2
            np.savez(root / 'histograms.npz', **arrays)
            (root / 'hist_index.json').write_text(json.dumps(hist_index[:-1]))
            with self.assertRaises(ValueError):
                summarize(root)
            (root / 'hist_index.json').write_text(json.dumps(hist_index))
            (root / 'histograms.npz').write_bytes(b'corrupt npz')
            with self.assertRaises(ValueError):
                summarize(root)

    def test_empty_per_image_output_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'per_image.jsonl').write_text('')
            with self.assertRaises(ValueError):
                summarize(root)

    def test_decision_distinguishes_uncertainty_and_unfavorable_effect(self):
        primary = {'difference': .1, 'ci95': [-.1, .2]}
        secondary = {'difference': -.1, 'ci95': [-.2, .1]}
        summary = {'ALL': {'paired_differences': {'high_pos': {'S4': {'auroc': primary}},
                                                 'pos': {'S4': {'risk@0.5': secondary}}}}}
        self.assertEqual(decide(summary)['quantitative_decision'], 'CONDITIONAL GO')
        primary['difference'] = -.1
        self.assertEqual(decide(summary)['quantitative_decision'], 'NO-GO')


if __name__ == '__main__':
    unittest.main()
