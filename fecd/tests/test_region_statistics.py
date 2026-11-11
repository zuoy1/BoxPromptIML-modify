"""来源代理组配对统计和预登记科研门槛的行为契约。"""
import copy
import unittest

from fecd.evaluation.region_statistics import analyze_records, classify_signal


def row(key, domain, group, tp, fp, fn, pixels=100):
    return {'sample_key': key, 'dataset': domain, 'source_group_id': group,
            'tp': tp, 'fp': fp, 'fn': fn, 'pixels': pixels,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1,
            'iou': tp/(tp+fp+fn) if tp+fp+fn else 1,
            'gt_positive_fraction': (tp+fn)/pixels,
            'prediction_positive_fraction': (tp+fp)/pixels}


class RegionStatisticsTests(unittest.TestCase):
    def test_identical_arms_zero_all_intervals_and_reproducible(self):
        rows = [row('a','A','g',1,2,0),row('b','A','h',20,10,10),row('c','B','g',2,1,0)]
        result = analyze_records(rows, rows, repeats=50)
        self.assertEqual(result, analyze_records(rows, rows, repeats=50))
        for stratum in result['paired'].values():
            for metric in stratum['metrics'].values():
                for value in metric.values():
                    if value['estimate'] is not None:
                        self.assertEqual(value['estimate'], 0)
                        self.assertEqual(value['ci95'], [0,0])
        self.assertEqual(result['research']['label'], 'NO GAIN')

    def test_group_equal_preserves_unequal_group_sizes_and_domain_namespace(self):
        left = [row('a','A','same',0,0,1), row('b','A','same',0,0,1),
                row('c','A','other',0,0,1), row('d','B','same',0,0,1)]
        right = [row('a','A','same',1,0,0),row('b','A','same',1,0,0),
                 row('c','A','other',0,0,1),row('d','B','same',0,0,1)]
        result = analyze_records(left,right,repeats=80)
        metrics = result['paired']['ALL']['metrics']['f1']
        self.assertEqual(metrics['image_equal']['estimate'], .5)
        self.assertAlmostEqual(metrics['group_equal']['estimate'], 1/3)
        self.assertAlmostEqual(metrics['dataset_equal']['estimate'], 1/3)
        self.assertEqual(result['paired']['ALL']['source_groups'],3)

    def test_stratum_filtered_after_shared_group_draws_reports_effective_counts(self):
        left = [row('a','A','g',1,0,0),row('b','A','h',30,0,0)]
        result = analyze_records(left,left,repeats=100)
        small = result['paired']['small']
        self.assertEqual(small['images'],1)
        self.assertEqual(small['source_groups'],1)
        valid=small['metrics']['recall']['image_equal']['valid_repeats']
        self.assertGreater(valid,0)
        self.assertLess(valid,100)
        self.assertEqual(result['paired']['large']['images'],1)

    def test_shared_draw_intervals_match_explicit_group_resampling_oracle(self):
        import numpy as np
        left = [row('a','A','g',0,0,1),row('b','A','g',0,0,1),
                row('c','A','h',0,0,1),row('d','B','g',0,0,1)]
        right = [row('a','A','g',1,0,0),row('b','A','g',1,0,0),
                 row('c','A','h',0,0,1),row('d','B','g',0,0,1)]
        result = analyze_records(left,right,repeats=50,seed=31)
        rng=np.random.default_rng(31)
        draws_a=[rng.integers(2,size=2) for _ in range(50)]
        draws_b=[rng.integers(1,size=1) for _ in range(50)]
        samples=[]
        for a,b in zip(draws_a,draws_b):
            values=[v for index in a for v in ([1.,1.] if index==0 else [0.])]
            values += [0. for _ in b]
            samples.append(np.mean(values))
        self.assertEqual(result['paired']['ALL']['metrics']['f1']['image_equal']['ci95'],
                         np.quantile(samples,[.025,.975]).tolist())

    def test_metrics_area_precision_and_micro_from_counts(self):
        rows=[row('a','A','g',1,3,1),row('b','B','g',0,0,2)]
        result=analyze_records(rows,rows,repeats=20)
        summary=result['arms']['R0']['ALL']['ALL']
        self.assertAlmostEqual(summary['precision'],.125)
        self.assertAlmostEqual(summary['recall'],.25)
        self.assertAlmostEqual(summary['area_bias'],0)
        self.assertAlmostEqual(summary['area_mae'],.02)
        self.assertAlmostEqual(summary['micro_f1'],.25)

    def test_empty_stratum_has_null_intervals_and_no_positive_signal(self):
        rows=[row('a','A','g',10,0,0)]
        result=analyze_records(rows,rows,repeats=20)
        interval=result['paired']['small']['metrics']['recall']['image_equal']
        self.assertIsNone(interval['estimate'])
        self.assertIsNone(interval['ci95'])
        self.assertEqual(interval['valid_repeats'],0)
        self.assertFalse(result['research']['positive_conditions']['small_recall_point'])

    def test_identity_duplicates_counts_and_metric_corruption_rejected(self):
        good=[row('a','A','g',1,0,0)]
        for broken in [good+good,[{**good[0],'source_group_id':'changed'}],
                       [{**good[0],'tp':-1}],[{**good[0],'f1':.2}],
                       [{**good[0],'gt_positive_fraction':float('nan')}]]:
            with self.assertRaises(ValueError):
                analyze_records(good,broken,repeats=20)
        with self.assertRaises(ValueError):
            analyze_records(good,good,repeats=0)

    def test_frozen_seven_positive_gates_and_safety_override(self):
        state={'main_f1_delta':.02,'main_f1_ci_low':.001,'small_recall_delta':-.01,
               'large_recall_delta':-.01,'small_recall_ci_low':-.04,'large_recall_ci_low':-.04,
               'domain_f1_deltas':{'A':-.02,'B':.04},'dataset_equal_f1_delta':.01,
               'area_mae_delta':0,'fp_delta':-1,'fn_delta':-1}
        self.assertEqual(classify_signal(state)['label'],'POSITIVE SIGNAL')
        for field,value in [('main_f1_delta',.009),('main_f1_ci_low',0),
                            ('small_recall_delta',-.021),('large_recall_ci_low',-.051),
                            ('dataset_equal_f1_delta',-.001),('area_mae_delta',.001)]:
            changed=copy.deepcopy(state);changed[field]=value
            self.assertNotEqual(classify_signal(changed)['label'],'POSITIVE SIGNAL')
        for changed in [{**state,'small_recall_delta':-.051},
                        {**state,'domain_f1_deltas':{'A':-.051}},
                        {**state,'main_f1_delta':0,'fp_delta':-2,'fn_delta':1}]:
            self.assertEqual(classify_signal(changed)['label'],'NEGATIVE SIGNAL')


if __name__=='__main__':
    unittest.main()
