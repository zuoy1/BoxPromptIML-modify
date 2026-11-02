"""GT-free formal authorization, immutable inputs, and real CUDA resume gates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch
from torch.utils.data._utils.collate import default_collate

from fecd.tools.train_controlled_baseline import build, seeds
from fecd.tools.check_stage3 import analyze_source
from fecd.training.access import DataReadAudit
from fecd.training.checkpoint import restore_checkpoint, save_checkpoint, state_hash
from fecd.training.f0_data import EpochDataset
from fecd.training.f0_engine import train_step
from fecd.training.f1_protocol import (ROOT, NAMES, DOMAINS, MEMBER_HASH, FORMAL_HASH,
                                        EVAL_HASH, STATISTICS, configuration, epoch_batches,
                                        epoch_identity, verify_frozen)
from fecd.training.f1_runner import deterministic, write_json
from fecd.training.f1_state import training_state_hash
from fecd.training.f1_storage import resolve_output_root, parse_output_root
from fecd.pseudo.cache_schema import canonical_hash


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_hashes():
    paths = []
    for folder in ('training', 'evaluation', 'datasets', 'confidence', 'pseudo'):
        paths.extend((ROOT/f'fecd/{folder}').glob('*.py'))
    paths.extend(ROOT.glob('fecd/tools/*formal*student.py'))
    paths.extend(ROOT.glob('fecd/tests/test_stage4f1*.py'))
    paths.extend(ROOT.glob('fecd/configs/stage4f1/*.json'))
    paths.extend(ROOT.glob('fecd/configs/stage4f0/*.json'))
    paths.extend([ROOT/'fecd/model_loading.py', ROOT/'boxpromptiml.py',
                  ROOT/'fecd/tools/train_controlled_baseline.py'])
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}


def verify_protected():
    before = json.loads((ROOT/'fecd/assets/stage4f0/protected_before.json').read_text())
    delivery = json.loads((ROOT/'fecd/assets/stage4f0/delivery_manifest.json').read_text())
    for path, item in delivery['files'].items():
        before[path] = item['sha256']
    before['fecd/assets/stage4f0/delivery_manifest.json'] = sha(
        ROOT/'fecd/assets/stage4f0/delivery_manifest.json')
    for name, digest in before.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Historical file changed: {name}')
    return before


def probe(variant, dataset, directory):
    config = configuration(variant)
    seeds(config['seed'])
    model, optimizer, info = build(config)
    initial = state_hash(model)
    first_ids = next(epoch_batches(len(dataset), config['seed'], 0, 4))
    dataset.set_epoch(0)
    first = default_collate([dataset[i] for i in first_ids])
    train_step(model, optimizer, first, config)
    save_checkpoint(directory/'probe.pt', model, optimizer, config, 1, {'epoch':0, 'next_batch':1})
    saved_hash = training_state_hash(model, optimizer)
    dataset.set_epoch(1)
    next_ids = next(epoch_batches(len(dataset), config['seed'], 1, 4))
    second = default_collate([dataset[i] for i in next_ids])
    expected, _ = train_step(model, optimizer, second, config)
    complete = training_state_hash(model, optimizer)
    del model, optimizer
    torch.cuda.empty_cache()
    model, optimizer, _ = build(config)
    payload = restore_checkpoint(directory/'probe.pt', model, optimizer, config['config_hash'])
    if training_state_hash(model, optimizer) != saved_hash or payload['position'] != {'epoch':0,'next_batch':1}:
        raise ValueError('Full Adam/RNG/model/memory restore differs')
    actual, _ = train_step(model, optimizer, second, config)
    if training_state_hash(model, optimizer) != complete or actual['loss'] != expected['loss']:
        raise ValueError('Real restored continuation differs')
    return {'initial_hash':initial, 'full_restore_hash':saved_hash,
            'next_update_full_state_hash':complete, 'loss_exact':True,
            'all_model_adam_rng_memory_exact':True, 'probe_updates':3,
            'probe_policy':'two logical updates across changed epoch data; extra replay; discarded before formal init',
            'tinyvit':info}


def main(output_root=None):
    output = resolve_output_root(output_root)
    verify_frozen()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Formal preflight requires an empty output_root; refusing existing artifacts')
    (output/'logs').mkdir()
    before = time.perf_counter()
    deterministic()
    protected = verify_protected()
    write_json(output/'protected_before.json', protected)
    frozen_runtime = runtime_hashes()
    issues = {p:analyze_source(ROOT/p) for p in frozen_runtime if p.endswith('.py') and
              ('f1_' in p or 'formal_student' in p or 'test_stage4f1' in p)}
    if any(issues.values()):
        raise ValueError(f'Static analysis failed: {issues}')
    selection = json.loads((ROOT/'fecd/assets/stage4f0/selection_audit.json').read_text())
    expected_masks = {r['sample_key']:r for r in selection['per_image']}
    datasets = {name:EpochDataset(name) for name in NAMES}
    for dataset in datasets.values():
        if dataset.membership_hash != MEMBER_HASH or len(dataset) != 3530:
            raise ValueError('Frozen member hash differs')
    keys = [s['sample_key'] for s in datasets['P0'].samples]
    epochs = [epoch_identity(keys, 20261029, e) for e in range(20)]
    write_json(output/'epoch_schedule.json', epochs)
    all_rgb = {s['image_path'] for d in datasets.values() for s in d.samples}
    all_cache = {s['cache_path'] for d in datasets.values() for s in d.samples}
    with DataReadAudit(all_rgb, all_cache) as audit:
        for name in ('P0', 'P1'):
            dataset = datasets[name]
            for i in range(len(dataset)):
                dataset._validate_input(i)
                if (i+1) % 1000 == 0:
                    print(json.dumps({'identity':name, 'verified':i+1}), flush=True)
        for name in ('P2F','P3F'):
            dataset = datasets[name]
            for i in range(len(dataset)):
                item = dataset[i]
                if item['canonical_selection_hash'] != expected_masks[item['sample_key']][f'{name}_canonical_hash']:
                    raise ValueError('Canonical fixed selection differs from frozen full-data audit')
                for short, long in (('pos','positive'),('neg','negative')):
                    selected = item[f'selected_{short}']
                    candidate = item[f'candidate_{short}']
                    if selected != candidate//2 or selected != expected_masks[item['sample_key']][f'selected_{short}']:
                        raise ValueError(f'Supervision budget differs for {long}')
                if (i+1) % 1000 == 0:
                    print(json.dumps({'selection':name, 'verified':i+1}), flush=True)
        probes = {}
        with TemporaryDirectory(prefix='f1_resume_', dir=output) as scratch:
            for name, dataset in datasets.items():
                directory = Path(scratch)/name
                directory.mkdir()
                probes[name] = probe(name, dataset, directory)
                print(json.dumps({'resume_probe':name, 'status':'PASS'}), flush=True)
    initial = {name:row['initial_hash'] for name,row in probes.items()}
    if len(set(initial.values())) != 1:
        raise ValueError('Four common initial model/memory hashes differ')
    if dict(Counter(r['dataset'] for r in datasets['P0'].records)) != DOMAINS or audit.denied:
        raise ValueError('Dataset/access preflight mismatch')
    result = {'status':'PASS', 'formal_proposal_hash':FORMAL_HASH, 'evaluation_protocol_hash':EVAL_HASH,
              'member_hash':MEMBER_HASH, 'images':3530, 'initial_hashes':initial, 'resume_probes':probes,
              'epochs':epochs, 'protected_files':len(protected), 'runtime_hashes':frozen_runtime,
              'selection_full_data_matches_4f0':True,
              'selection_across20':'frozen epoch-independent selector plus 20-epoch synthetic tests and unchanged 4F0 audit',
              'static_issues':issues, 'gt_reads':0, 'new_confirm_gt_reads':0,
              'access_unique_files':len(set(audit.opened)), 'denied':audit.denied,
              'statistics':STATISTICS, 'elapsed_seconds':time.perf_counter()-before}
    result['result_hash'] = canonical_hash(result)
    write_json(output/'preflight_access.json', {'opened_counts':dict(Counter(audit.opened)), 'denied':audit.denied})
    write_json(output/'preflight.json', result)
    print(json.dumps({'status':'PASS', 'seconds':result['elapsed_seconds'], 'initial_hashes':initial}), flush=True)


if __name__ == '__main__':
    main(parse_output_root())
