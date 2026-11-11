"""阶段5C0限定单Epoch及独立磁盘恢复探针，不导入evaluator。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate

from fecd.tools.train_controlled_baseline import build, seeds
from fecd.training.checkpoint import restore_checkpoint, save_checkpoint, state_hash
from fecd.training.f1_protocol import DOMAINS, epoch_batches, epoch_identity, verify_frozen
from fecd.training.f1_runner import deterministic
from fecd.training.f1_state import training_state_hash
from fecd.training.region_access import RegionReadAudit
from fecd.training.region_data import RegionDataset
from fecd.training.region_engine import train_region_step

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('/hy-tmp/BoxPromtIML/fecd_runs/stage5c0')


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def configuration(variant):
    if variant not in ('R0', 'R1'):
        raise ValueError('仅允许R0/R1')
    config = json.loads((ROOT/f'fecd/configs/stage5c0/{variant}.json').read_text())
    from fecd.pseudo.cache_schema import canonical_hash
    if canonical_hash({k: v for k, v in config.items() if k != 'config_hash'}) != config['config_hash']:
        raise ValueError('配置hash错误')
    if config['epochs'] != 1 or config['variant'] != variant:
        raise ValueError('仅授权单Epoch预检')
    return config


def setup(variant, mode):
    verify_frozen()
    config = configuration(variant)
    deterministic()
    seeds(config['seed'])
    Image.init()
    dataset = RegionDataset(config['seed'])
    model, optimizer, info = build(config)
    initial = state_hash(model)
    directory = OUTPUT/mode/variant
    directory.mkdir(parents=True, exist_ok=False)
    write(directory/'initialization.json', {'initial_hash': initial, 'tinyvit': info,
                                          'fresh_initialization': True, 'config': config})
    batches = list(epoch_batches(len(dataset), config['seed'], 0, 4))
    return config, dataset, model, optimizer, directory, initial, batches


def boundary(dataset):
    return RegionReadAudit([r['image_path'] for r in dataset.samples],
                           [r['cache_path'] for r in dataset.samples], OUTPUT)


def probe(variant):
    config, dataset, model, optimizer, directory, initial, batches = setup(variant, 'probe')
    with boundary(dataset) as audit:
        first = default_collate([dataset[i] for i in batches[0]])
        first_result, _, _ = train_region_step(model, optimizer, first, config, 0)
        if state_hash(model) == initial:
            raise AssertionError('单batch参数未更新')
        save_checkpoint(directory/'update1.pt', model, optimizer, config, 1,
                        {'epoch': 0, 'next_batch': 1})
        second = default_collate([dataset[i] for i in batches[1]])
        expected, expected_rows, expected_masks = train_region_step(model, optimizer, second, config, 0)
        expected_hash = training_state_hash(model, optimizer)
        del model, optimizer
        torch.cuda.empty_cache()
        model, optimizer, _ = build(config)
        payload = restore_checkpoint(directory/'update1.pt', model, optimizer, config['config_hash'])
        if payload['update'] != 1 or payload['position'] != {'epoch': 0, 'next_batch': 1}:
            raise AssertionError('恢复训练位置不一致')
        actual, actual_rows, actual_masks = train_region_step(model, optimizer, second, config, 0)
        exact = (expected_hash == training_state_hash(model, optimizer)
                 and actual['loss'] == expected['loss'] and actual_rows == expected_rows
                 and np.array_equal(actual_masks, expected_masks))
        if not exact or audit.denied:
            raise AssertionError('恢复后的下一步轨迹不完全一致')
    result = {'status': 'PASS', 'variant': variant, 'initial_hash': initial,
              'single_batch': first_result, 'resume_loss': actual['loss'],
              'all_model_optimizer_memory_rng_exact': exact, 'selection_and_diagnostics_exact': exact,
              'position_exact': True, 'next_state_hash': expected_hash,
              'physical_probe_updates': 3, 'logical_probe_updates': 2,
              'gt_reads': 0, 'access_denied': audit.denied}
    write(directory/'result.json', result)
    write(directory/'access.json', {'opened': audit.opened, 'denied': audit.denied})
    print(json.dumps(result), flush=True)


def run_epoch(variant):
    probes = [json.loads((OUTPUT/'probe'/v/'result.json').read_text()) for v in ('R0', 'R1')]
    if any(r['status'] != 'PASS' for r in probes) or len({r['initial_hash'] for r in probes}) != 1:
        raise ValueError('共同初始化或恢复预检未通过')
    config, dataset, model, optimizer, directory, initial, batches = setup(variant, 'epoch')
    if initial != probes[0]['initial_hash']:
        raise AssertionError('Epoch未从共同全新初始化开始')
    start, counts, steps, image_rows, packed_masks = time.perf_counter(), Counter(), [], [], []
    input_hashes = {}
    torch.cuda.reset_peak_memory_stats()
    with boundary(dataset) as audit, (directory/'steps.jsonl').open('x') as step_log, (
            directory/'images.jsonl').open('x') as image_log:
        for index, ids in enumerate(batches):
            batch = default_collate([dataset[i] for i in ids])
            result, rows, masks = train_region_step(model, optimizer, batch, config, 0)
            result.update(update=index+1, batch_size=len(ids))
            steps.append(result)
            counts.update(batch['dataset'])
            step_log.write(json.dumps(result, allow_nan=False)+'\n')
            for row in rows:
                row['update'] = index+1
                image_log.write(json.dumps(row, allow_nan=False)+'\n')
            image_rows.extend(rows)
            packed_masks.extend(np.packbits(masks.reshape(len(ids), -1), axis=1))
            for i in ids:
                cache = Path(dataset.samples[i]['cache_path'])
                input_hashes[str(cache)] = hashlib.sha256(cache.read_bytes()).hexdigest()
            if (index+1) % 50 == 0 or index+1 == len(batches):
                step_log.flush()
                image_log.flush()
                print(json.dumps({'variant': variant, 'update': index+1, 'loss': result['loss'],
                                  'seconds': time.perf_counter()-start}), flush=True)
        if dict(counts) != DOMAINS or len(steps) != 883 or len(image_rows) != 3530 or steps[-1]['batch_size'] != 2:
            raise AssertionError('完整训练成员、更新或四域预算不一致')
        if audit.denied or any(r['selected_positive']+r['selected_negative_in']+r['selected_negative_out'] == 0
                               for r in image_rows):
            raise AssertionError('访问违规或异常空监督')
        save_checkpoint(directory/'epoch01.pt', model, optimizer, config, 883,
                        {'epoch': 1, 'next_batch': 0})
        np.savez_compressed(directory/'selection_masks.npz', masks=np.stack(packed_masks),
                            sample_keys=np.array([r['sample_key'] for r in image_rows]))
    identity = epoch_identity([s['sample_key'] for s in dataset.samples], config['seed'], 0)
    from fecd.pseudo.cache_schema import canonical_hash
    if (canonical_hash([r['sample_key'] for r in image_rows]) != identity['sample_order_hash']
            or canonical_hash([r['operation'] for r in image_rows]) != identity['geometry_hash']):
        raise AssertionError('实际数据顺序或几何变换不符')
    result = {'status': 'PASS', 'variant': variant, 'initial_hash': initial,
              'final_model_memory_hash': state_hash(model), 'complete_state_hash': training_state_hash(model, optimizer),
              'images': len(image_rows), 'logical_updates': len(steps), 'epochs': 1, 'identity': identity,
              'dataset_forward_backward_counts': dict(counts), 'mean_loss': float(np.mean([r['loss'] for r in steps])),
              'max_gradient_norm_after': max(r['gradient_norm_after'] for r in steps),
              'mean_prediction_area': float(np.mean([r['prediction_area'] for r in image_rows])),
              'empty_supervision': 0, 'all_zero_outputs': sum(r['all_zero'] for r in image_rows),
              'near_full_outputs': sum(r['near_full'] for r in image_rows),
              'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
              'elapsed_seconds': time.perf_counter()-start, 'gt_reads': 0, 'evaluation_performed': False}
    write(directory/'input_hashes.json', input_hashes)
    write(directory/'access.json', {'opened_counts': dict(Counter(audit.opened)), 'denied': audit.denied,
                                   'gt_reads': 0, 'training_allowlist_files': len(audit.allowed)})
    write(directory/'result.json', result)
    print(json.dumps(result), flush=True)
