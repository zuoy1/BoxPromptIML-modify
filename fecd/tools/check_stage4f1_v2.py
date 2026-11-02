"""4F1-V 独立版本化训练验收；仅替换原 Loss 分解断言。"""
from collections import Counter
from pathlib import Path

import torch

from fecd.pseudo.cache_schema import canonical_hash
from fecd.tools.check_stage4f1 import finite
from fecd.tools.preflight_formal_student import sha
from fecd.tools.stage4f1_v2_checks import read, rows, check_loss
from fecd.training.f1_protocol import DOMAINS, MEMBER_HASH, configuration
from fecd.training.f1_storage import resolve_output_root


def check_training(variant, schedule, selected, members, output_root=None):
    output = resolve_output_root(output_root)
    directory = output/variant
    config = configuration(variant)
    result = read(directory/'result.json')
    assert read(directory/'config.json') == config
    assert result['epochs'] == 20 and result['logical_updates'] == 17660 and result['gt_reads'] == 0
    assert result['membership_hash'] == MEMBER_HASH
    allow = read(output/'preflight_access.json')['opened_counts']
    allowed_rgb = {str(Path(r['image_path']).resolve()) for r in members.values()}
    cache_fragment = '/k1/' if variant == 'P0' else '/k8/'
    allowed = allowed_rgb | {p for p in allow if cache_fragment in p}
    assert len(allowed) == 7060
    for epoch in range(1,21):
        summary = read(directory/f'epoch{epoch:02d}.summary.json')
        steps = rows(directory/f'epoch{epoch:02d}.steps.jsonl')
        images = rows(directory/f'epoch{epoch:02d}.outputs.jsonl')
        access = read(directory/f'epoch{epoch:02d}.access.json')
        assert summary['status'] == 'PASS' and finite(summary) and finite(steps) and finite(images)
        assert summary['identity'] == schedule[epoch-1]
        assert summary['initial_hash'] == result['initial_hash']
        assert summary['dataset_forward_backward_counts'] == DOMAINS
        assert len(steps) == 883 and len(images) == 3530 and steps[-1]['batch_size'] == 2
        assert Counter(r['dataset'] for r in images) == Counter(DOMAINS)
        assert not access['denied'] and access['gt_reads'] == 0
        assert set(access['opened_counts']) == allowed
        keys = [key for step in steps for key in step['sample_keys']]
        assert len(set(keys)) == 3530 and set(keys) == set(members)
        assert keys == [r['sample_key'] for r in images]
        assert canonical_hash(keys) == schedule[epoch-1]['sample_order_hash']
        assert canonical_hash([r['operation'] for r in images]) == schedule[epoch-1]['geometry_hash']
        for i, step in enumerate(steps):
            assert step['update'] == (epoch-1)*883+i+1
            assert step['epoch_update'] == i+1 and step['epoch'] == epoch
            assert step['batch_size'] == (2 if i == 882 else 4)
            assert step['gradient_norm_before'] > 0 and 0 <= step['gradient_norm_after'] <= .50001
            check_loss(step)
            assert abs(step['pos_contribution']-.5*step['pos_mean']) < 1e-7
            assert abs(step['neg_contribution']-.5*step['neg_mean']) < 1e-7
        for row in images:
            key = row['sample_key']
            assert row['dataset'] == members[key]['dataset']
            assert row['candidate_positive_count']+row['candidate_negative_count'] == 224*224
            for short, sign in (('pos','positive'),('neg','negative')):
                n, k = row[f'candidate_{sign}_count'], row[f'selected_{sign}_count']
                assert k == (n if variant in ('P0','P1') else n//2)
                if variant != 'P0':
                    assert n == selected[key][f'candidate_{short}']
            if variant in ('P2F','P3F'):
                assert row['canonical_selection_hash'] == selected[key][f'{variant}_canonical_hash']
        assert abs(summary['mean_loss']-sum(r['loss'] for r in steps)/883) < 1e-12
    final = torch.load(directory/'final.pt', map_location='cpu', weights_only=False)
    assert final['config'] == config and final['update'] == 17660
    assert final['position'] == {'epoch':20,'next_batch':0}
    assert final['model_hash'] == result['final_hash'] == summary['model_memory_hash']
    assert sha(directory/'final.pt') == sha(directory/'last.pt')
    assert set(final['rng']) == {'python','numpy','torch','cuda'} and final['optimizer']['state']
    return {'status':'PASS', 'epochs':20, 'updates':17660, 'images':70600,
            'final_checkpoint_sha256':sha(directory/'final.pt'), 'gt_reads':0}



def main():
    from fecd.tools.stage4f1_v2_run import main as run
    run()


if __name__ == '__main__':
    main()
