"""Development aggregation keeps source groups and common candidate sets explicit."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from fecd.diagnostics.stage3ra_aggregate import summarize
from fecd.diagnostics.stage3ra_inference import paired_comparison, decide


def metric(auroc, risk, n=100, errors=10):
    return {'n': n, 'error_count': errors, 'error_prevalence': errors / n,
            'auroc': auroc, 'auprc': auroc, 'aurc': risk,
            'coverage': {str(c): {'risk': risk, 'precision': 1-risk,
                                 'selected_count': c*n, 'coverage': c}
                         for c in (.1, .25, .5, .75, .9)}}


class Stage3RASummaryTests(unittest.TestCase):
    def test_group_equal_does_not_overweight_many_images_from_one_source(self):
        rows = [{'dataset': 'Toy', 'source_group_id': group,
                 'regions': {'high_pos': {'c_sam': metric(.5, .1),
                                          'q_a': metric(score, .1)}}}
                for group, score in [('large', .9)]*3+[('small', .1)]]
        result = paired_comparison(rows, 'high_pos', 'q_a', 'c_sam', 'auroc')
        self.assertAlmostEqual(result['image_weighted']['difference'], .2)
        self.assertAlmostEqual(result['group_equal']['difference'], 0)
        self.assertEqual(result['image_weighted']['n_groups'], 2)

    def test_unfavorable_or_small_effects_do_not_request_confirmation(self):
        def comparison(delta, low, high):
            value = {'difference': delta, 'ci95': [low, high], 'n_groups': 30,
                     'n_images': 40}
            return {'image_weighted': value, 'group_equal': value}
        pairs = {'high_pos': {'q_a-minus-c_sam': {'auroc': comparison(-.1, -.2, -.01)}},
                 'pos': {'q_a-minus-c_sam': {'risk@0.5': comparison(-.01, -.02, -.001)}},
                 'neg': {'q_a-minus-c_sam': {'risk@0.5': comparison(0, -.0001, .0001)}}}
        result = decide({'ALL': {'paired_differences': pairs}})
        self.assertEqual(result['decision'], 'STOP')
        pairs['high_pos']['q_a-minus-c_sam']['auroc'] = comparison(.001, .0001, .002)
        self.assertEqual(decide({'ALL': {'paired_differences': pairs}})['decision'],
                         'INSUFFICIENT EVIDENCE')

    def test_aggregation_rejects_unfinished_or_nondev_membership(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows=[]
            for index in range(2):
                methods={name:metric(.8, .02) for name in
                         ('base_margin','mean_margin','variance','threshold','c_sam','q_a')}
                rows.append({'sample_key':str(index),'dataset':'Toy',
                             'source_group_id':str(index),'subset':'new_dev',
                             'regions':{r:methods for r in ('pos','high_pos','neg')}})
            (root/'stage3ra_per_image.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            split={'manifest_sha256':'split','records':[{'sample_key':str(i),'subset':'new_dev'}
                                                       for i in range(2)]}
            (root/'stage3ra_split_manifest.json').write_text(json.dumps(split))
            from fecd.diagnostics.stage3ra_protocol import PROTOCOL_HASH
            run={'status':'running','requested':2,'evaluated':2,'protocol_hash':PROTOCOL_HASH,
                 'split_hash':'split','evaluated_sample_keys_sha256':
                 hashlib.sha256(b'0\n1').hexdigest()}
            (root/'stage3ra_run_manifest.json').write_text(json.dumps(run))
            with self.assertRaises(ValueError): summarize(root)
            run['status']='passed'
            (root/'stage3ra_run_manifest.json').write_text(json.dumps(run))
            result=summarize(root)
            self.assertEqual(result['datasets']['ALL']['n_images'],2)
            self.assertEqual(result['datasets']['Toy']['macro']['pos']['q_a']['auroc']['mean'],.8)
            self.assertEqual(result['decision']['decision'],'STOP')
            self.assertTrue((root/'stage3ra_dev_summary.json').is_file())
            arrays, index = {}, []
            for dataset in ('ALL', 'Toy'):
                for region, methods in rows[0]['regions'].items():
                    for method in methods:
                        key=f'h{len(index):06}'
                        arrays[key]=np.zeros((2,4096),np.int64)
                        arrays[key][:,0]=[200,20]
                        index.append({'key':key,'dataset':dataset,'area_stratum':'ALL',
                                      'region':region,'method':method})
            np.savez(root/'stage3ra_histograms.npz',**arrays)
            (root/'stage3ra_hist_index.json').write_text(json.dumps(index))
            result=summarize(root)
            self.assertEqual(result['pooled_micro']['scopes'],36)
            self.assertEqual(result['datasets']['ALL']['micro']['pos']['q_a']['auroc'],.5)
            arrays['h000000'][0,0]+=1
            np.savez(root/'stage3ra_histograms.npz',**arrays)
            with self.assertRaises(ValueError): summarize(root)
            arrays['h000000'][0,0]-=1
            np.savez(root/'stage3ra_histograms.npz',**arrays)
            split['records'][0]['subset']='new_confirm'
            (root/'stage3ra_split_manifest.json').write_text(json.dumps(split))
            with self.assertRaises(ValueError): summarize(root)


if __name__ == '__main__':
    unittest.main()
