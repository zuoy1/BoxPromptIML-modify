"""Storage-only probes: synthetic resume and untrained real-model evaluator IO, no GT."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from fecd.evaluation.memory import infer_one
from fecd.evaluation.strict import load_independent_model
from fecd.evaluation.data import file_sha
from fecd.tools.train_controlled_baseline import seeds, build
from fecd.training.access import DataReadAudit
from fecd.training.checkpoint import restore_checkpoint, state_hash
from fecd.training.f1_protocol import ROOT, configuration, verify_frozen
from fecd.training.f1_runner import deterministic, write_json
from fecd.training.f1_state import save_last, training_state_hash
from fecd.training.f1_storage import resolve_output_root, parse_output_root


def synthetic_update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(2)).square().sum()
    loss.backward()
    optimizer.step()
    model.memory_bank = model.memory_bank + 1
    return float(loss.detach())


def probe_resume(directory):
    config = configuration('P0')
    model = torch.nn.Linear(2,1)
    model.memory_bank = torch.ones(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    synthetic_update(model, optimizer)
    checkpoint = directory/'last.pt'
    save_last(checkpoint, model, optimizer, config, 883, {'epoch':1,'next_batch':0})
    before = training_state_hash(model, optimizer)
    expected_loss = synthetic_update(model, optimizer)
    expected = training_state_hash(model, optimizer)
    fresh = torch.nn.Linear(2,1)
    fresh.memory_bank = None
    restored_optimizer = torch.optim.Adam(fresh.parameters(), lr=.01)
    payload = restore_checkpoint(checkpoint, fresh, restored_optimizer, config['config_hash'])
    assert payload['position'] == {'epoch':1,'next_batch':0}
    assert training_state_hash(fresh, restored_optimizer) == before
    assert synthetic_update(fresh, restored_optimizer) == expected_loss
    assert training_state_hash(fresh, restored_optimizer) == expected
    save_last(checkpoint, fresh, restored_optimizer, config, 1766, {'epoch':2,'next_batch':0})
    assert torch.load(checkpoint, weights_only=False)['update'] == 1766
    assert not checkpoint.with_suffix('.pending.pt').exists()
    assert checkpoint.stat().st_dev == directory.stat().st_dev
    return {'atomic_replace':'PASS', 'same_filesystem':True, 'full_restore':'PASS',
            'model_adam_memory_all_rng_exact':True, 'next_synthetic_update_exact':True,
            'checkpoint_bytes':checkpoint.stat().st_size, 'checkpoint_sha256':file_sha(checkpoint)}


def probe_evaluator(directory):
    config = configuration('P0')
    seeds(config['seed'])
    model, optimizer, _ = build(config)
    checkpoint = directory/'untrained_real_student.pt'
    save_last(checkpoint, model, optimizer, config, 0, {'epoch':0,'next_batch':0})
    expected = state_hash(model)
    before = file_sha(checkpoint)
    size = checkpoint.stat().st_size
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    evaluator, snapshot, metadata = load_independent_model(checkpoint)
    assert metadata['checkpoint_model_hash'] == expected
    assert metadata['checkpoint_config'] == config
    image = torch.zeros(1,3,224,224,device=metadata['device'])
    first = infer_one(evaluator, image, snapshot)
    second = infer_one(evaluator, image, snapshot)
    assert torch.equal(first, second) and torch.isfinite(first).all()
    assert file_sha(checkpoint) == before
    return {'strict_loader':'PASS', 'config_identity':'PASS', 'model_memory_identity':'PASS',
            'checkpoint_sha256_unchanged':True, 'checkpoint_sha256':before,
            'checkpoint_bytes':size, 'synthetic_rgb_repeat_exact':True,
            'formal_training_updates':0, 'gt_reads':0}


def main(output_root=None):
    output = resolve_output_root(output_root)
    verify_frozen()
    if not output.is_dir():
        raise ValueError('Output directory must already exist after mount audit')
    deterministic()
    with DataReadAudit([],[]) as audit:
        with TemporaryDirectory(prefix='.storage_probe_',dir=output) as scratch:
            directory = Path(scratch)
            assert directory.stat().st_dev == output.stat().st_dev
            resume = probe_resume(directory)
            evaluator = probe_evaluator(directory)
    assert not audit.opened and not audit.denied
    result = {'status':'PASS', 'output_root':str(output), 'output_device':output.stat().st_dev,
              'root_device':Path('/').stat().st_dev, 'writable':os.access(output,os.W_OK),
              'resume':resume, 'evaluator':evaluator, 'gt_reads':0,
              'formal_training_updates':0, 'scratch_files_removed':True,
              'historical_files_moved_or_deleted':0}
    write_json(ROOT/'fecd/assets/stage4f1_storage_audit/storage_probe.json', result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main(parse_output_root())
