"""Source proxies must never straddle the internal split."""
import unittest

from fecd.tools.prepare_stage3_split import assign_groups, source_tokens


class Stage3SplitTests(unittest.TestCase):
    def test_shared_donor_transitively_remains_in_one_group(self):
        rows = [dict(dataset='CASIAv2', sample_id=s, sample_key=str(i),
                     image_sha256=str(i), group=s) for i, s in enumerate([
            'Tp_D_CNN_M_N_ani00001_nat00002_00001',
            'Tp_D_CNN_M_N_nat00002_sec00003_00002',
            'Tp_S_CNN_M_N_sec00003_sec00003_00003'])]
        groups = assign_groups(rows)
        self.assertEqual(len(set(groups.values())), 1)

    def test_unknown_identity_is_marked_as_proxy(self):
        tokens, status = source_tokens(dict(dataset='CASIAv2', sample_id='unknown'))
        self.assertTrue(tokens[0].startswith('unresolved:'))
        self.assertIn('unresolved', status)

    def test_columbia_scene_proxy_ignores_camera_pair(self):
        a = source_tokens(dict(dataset='Columbia', sample_id='a_b_sub_01'))
        b = source_tokens(dict(dataset='Columbia', sample_id='c_d_sub_01'))
        self.assertEqual(a[0], b[0])


if __name__ == '__main__':
    unittest.main()
