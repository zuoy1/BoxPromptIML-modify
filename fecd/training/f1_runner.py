"""Formal 20-epoch loop; no evaluator imports or ground-truth interface."""
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data._utils.collate import default_collate

from fecd.tools.train_controlled_baseline import build, seeds
from fecd.training.access import DataReadAudit
from fecd.training.checkpoint import restore_checkpoint, state_hash
from fecd.training.f0_data import EpochDataset
from fecd.training.f0_diagnostics import summarize_outputs
from fecd.training.f0_engine import train_step
from fecd.training.f1_protocol import (ROOT, DOMAINS, MEMBER_HASH, configuration,
                                        epoch_batches, epoch_identity, verify_frozen)
from fecd.training.f1_state import save_last, training_state_hash
from fecd.training.f1_storage import resolve_output_root
from fecd.pseudo.cache_schema import canonical_hash


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def deterministic():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def restore_start(directory, model, optimizer, config):
    payload = restore_checkpoint(directory / 'last.pt', model, optimizer, config['config_hash'])
    epoch = payload['position']['epoch']
    if payload['position']['next_batch'] != 0 or not 1 <= epoch <= 20 or payload['update'] != epoch * 883:
        raise ValueError('Formal checkpoint is not a completed epoch boundary')
    for path in sorted(directory.glob('epoch*.*')):
        number = int(path.name.split('.')[0].removeprefix('epoch'))
        if number > epoch:
            recovery = directory / f'recovered_attempt_{time.time_ns()}'
            recovery.mkdir()
            path.rename(recovery / path.name)
    pending = directory / 'last.pending.pt'
    if pending.exists():
        pending.rename(directory / f'interrupted_pending_{time.time_ns()}.pt')
    return epoch


def run_epoch(model, optimizer, dataset, config, epoch, directory, initial_hash):
    dataset.set_epoch(epoch)
    batches = list(epoch_batches(len(dataset), config['seed'], epoch, config['batch_size']))
    diagnostics, steps = [], []
    counts = Counter()
    before = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    paths = {str(Path(row[field]).resolve()) for row in dataset.samples for field in ('image_path', 'cache_path')}
    with DataReadAudit([s['image_path'] for s in dataset.samples],
                       [s['cache_path'] for s in dataset.samples]) as audit, (
            directory / f'epoch{epoch+1:02d}.steps.jsonl').open('x') as log, (
            directory / f'epoch{epoch+1:02d}.outputs.jsonl').open('x') as outputs:
        for step, ids in enumerate(batches):
            batch = default_collate([dataset[i] for i in ids])
            result, images = train_step(model, optimizer, batch, config)
            result.update(epoch=epoch+1, epoch_update=step+1, update=epoch*883+step+1,
                          batch_size=len(ids), sample_keys=batch['sample_key'], operations=batch['operation'])
            steps.append(result)
            counts.update(batch['dataset'])
            log.write(json.dumps(result, allow_nan=False)+'\n')
            for image in images:
                image.update(epoch=epoch+1, update=result['update'])
                outputs.write(json.dumps(image, allow_nan=False)+'\n')
            diagnostics.extend(images)
            if (step+1) % 100 == 0 or step+1 == len(batches):
                log.flush()
                outputs.flush()
                print(json.dumps({'variant':config['variant'], 'epoch':epoch+1, 'update':step+1,
                                  'loss':result['loss'], 'seconds':time.perf_counter()-before}), flush=True)
    if (dict(counts) != DOMAINS or len(steps) != 883 or len(diagnostics) != 3530 or
            steps[-1]['batch_size'] != 2 or audit.denied or set(audit.opened) != paths):
        raise ValueError('Epoch data, update budget, or access boundary mismatch')
    keys = [s['sample_key'] for s in dataset.samples]
    identity = epoch_identity(keys, config['seed'], epoch)
    if identity['sample_order_hash'] != canonical_hash([r['sample_key'] for r in diagnostics]):
        raise ValueError('Actual sample order differs from frozen rule')
    if identity['geometry_hash'] != canonical_hash([r['operation'] for r in diagnostics]):
        raise ValueError('Actual geometry differs from frozen rule')
    summary = {'epoch':epoch+1, 'status':'PASS', 'initial_hash':initial_hash,
               'images':3530, 'updates':883, 'total_updates':(epoch+1)*883,
               'dataset_forward_backward_counts':dict(counts), 'identity':identity,
               'mean_loss':sum(r['loss'] for r in steps)/883,
               'mean_pos_contribution':sum(r['pos_contribution'] for r in steps)/883,
               'mean_neg_contribution':sum(r['neg_contribution'] for r in steps)/883,
               'gradient_norm_before_max':max(r['gradient_norm_before'] for r in steps),
               'gradient_norm_after_max':max(r['gradient_norm_after'] for r in steps),
               'step_seconds_sum':sum(r['step_seconds'] for r in steps),
               'peak_cuda_allocated_bytes':max(r['peak_cuda_allocated_bytes'] for r in steps),
               'peak_cuda_reserved_bytes':max(r['peak_cuda_reserved_bytes'] for r in steps),
               'outputs':summarize_outputs(diagnostics), 'gt_reads':0}
    write_json(directory / f'epoch{epoch+1:02d}.access.json',
               {'opened_counts':dict(Counter(audit.opened)), 'denied':audit.denied, 'gt_reads':0})
    save_last(directory/'last.pt', model, optimizer, config, (epoch+1)*883,
              {'epoch':epoch+1, 'next_batch':0})
    summary.update(elapsed_seconds=time.perf_counter()-before, model_memory_hash=state_hash(model),
                   complete_state_hash=training_state_hash(model, optimizer))
    write_json(directory/f'epoch{epoch+1:02d}.summary.json', summary)
    print(json.dumps({'variant':config['variant'], 'epoch_completed':epoch+1,
                      'mean_loss':summary['mean_loss'], 'elapsed_seconds':summary['elapsed_seconds']}), flush=True)
    return summary


def run(variant, resume=False, output_root=None):
    output = resolve_output_root(output_root)
    verify_frozen()
    config = configuration(variant)
    if json.loads((ROOT/f'fecd/configs/stage4f1/{variant}.json').read_text()) != config:
        raise ValueError('Formal configuration differs from frozen JSON')
    preflight = json.loads((output/'preflight.json').read_text())
    if preflight['status'] != 'PASS':
        raise ValueError('Formal engineering preflight has not passed')
    for name, digest in preflight['runtime_hashes'].items():
        import hashlib
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Formal code/protocol changed after preflight: {name}')
    directory = output/variant
    deterministic()
    seeds(config['seed'])
    dataset = EpochDataset(variant, seed=config['seed'], epoch=0)
    if len(dataset) != 3530 or dataset.membership_hash != MEMBER_HASH:
        raise ValueError('Student_train membership changed')
    model, optimizer, info = build(config)
    initial = state_hash(model)
    if initial != preflight['initial_hashes'][variant]:
        raise ValueError('Common fresh initialization mismatch')
    if resume:
        if json.loads((directory/'config.json').read_text()) != config:
            raise ValueError('Resume configuration mismatch')
        start_epoch = restore_start(directory, model, optimizer, config)
    else:
        directory.mkdir(exist_ok=False)
        write_json(directory/'config.json', config)
        write_json(directory/'initialization.json', {'initial_hash':initial, 'tinyvit':info,
                                                    'fresh_initialization':True})
        start_epoch = 0
    for epoch in range(start_epoch, 20):
        run_epoch(model, optimizer, dataset, config, epoch, directory, initial)
    if not (directory/'final.pt').exists():
        with (directory/'last.pt').open('rb') as source, (directory/'final.pt').open('xb') as target:
            shutil.copyfileobj(source, target)
    completed = [json.loads((directory/f'epoch{i:02d}.summary.json').read_text()) for i in range(1,21)]
    result = {'status':'PASS', 'variant':variant, 'epochs':20, 'logical_updates':17660,
              'physical_training_updates':17660, 'images':70600, 'initial_hash':initial,
              'final_hash':state_hash(model), 'membership_hash':dataset.membership_hash,
              'elapsed_seconds':sum(r['elapsed_seconds'] for r in completed), 'gt_reads':0,
              'dataset_forward_backward_counts':{d:n*20 for d,n in DOMAINS.items()},
              'config_hash':config['config_hash'], 'checkpoint':'final.pt'}
    write_json(directory/'result.json', result)
    print(json.dumps(result), flush=True)
    return result
