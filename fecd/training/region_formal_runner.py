"""正式R0/R1二十轮；未完成轮隔离，只恢复经过完整校验的边界。"""
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from fecd.tools.train_controlled_baseline import build, seeds
from fecd.training.checkpoint import restore_checkpoint, state_hash
from fecd.training.f1_protocol import verify_frozen
from fecd.training.f1_runner import deterministic
from fecd.training.f1_state import training_state_hash
from fecd.training.region_data import RegionDataset
from fecd.training.region_formal_epoch import run_epoch
from fecd.training.region_formal_protocol import (OUTPUT, configuration, validate_position,
                                                 collapse_gate, sha, verify_runtime, write)


def resume_boundary(directory, model, optimizer, config):
    if any(directory.parent.glob(f'BLOCKED_{directory.name}_*.json')):
        raise ValueError('组级BLOCKED工程失败不可通过恢复静默重放')
    completed = []
    for number in range(1,21):
        target = directory/f'epoch{number:02d}'
        if not target.exists():
            break
        if (target/'BLOCKED.json').exists():
            raise ValueError('失败Epoch不可静默重放或跳过')
        if not (target/'summary.json').exists():
            target.rename(directory/f'interrupted_epoch{number:02d}_{time.time_ns()}')
            break
        summary = json.loads((target/'summary.json').read_text())
        if (not summary.get('parameter_update_verified') or (target/'BLOCKED.json').exists()
                or collapse_gate(completed+[summary])):
            raise ValueError('历史Epoch工程门禁未通过，不可恢复')
        if (summary['status'] != 'PASS' or summary['epoch'] != number
                or summary['total_updates'] != number*883
                or sha(target/'checkpoint.pt') != summary['checkpoint_sha256']
                or sha(target/'selection.npz') != summary['selection_sha256']):
            raise ValueError('完整Epoch文件或位置hash不一致')
        completed.append(summary)
    if completed:
        epoch = len(completed)
        payload = restore_checkpoint(directory/f'epoch{epoch:02d}/checkpoint.pt', model,optimizer,
                                     config['config_hash'])
        if validate_position(payload['position'],payload['update']) != epoch:
            raise ValueError('checkpoint位置与完成目录轮次不一致')
        if training_state_hash(model,optimizer) != completed[-1]['complete_state_hash']:
            raise ValueError('恢复完整model/Adam/memory/RNG状态不符')
    return completed


def run(variant, resume=False):
    verify_frozen(); verify_runtime()
    pre = json.loads((OUTPUT/'results/preflight.json').read_text())
    if pre['status'] != 'PASS' or pre['automatic_shutdown']:
        raise ValueError('正式预检未通过或关机边界错误')
    config = configuration(variant)
    deterministic(); seeds(config['seed']); Image.init()
    dataset = RegionDataset(config['seed'])
    with np.load(OUTPUT/'results/candidates.npz',allow_pickle=False) as stored:
        if stored['sample_keys'].tolist() != [r['sample_key'] for r in dataset.samples]:
            raise ValueError('候选区域顺序不符')
        candidates = {k:stored[k].copy() for k in ('inside','outside')}
    model,optimizer,info = build(config)
    initial = state_hash(model)
    if initial != pre['initial_hashes'][variant]:
        raise ValueError('fresh共同初始化不符')
    directory = OUTPUT/variant
    if resume:
        if json.loads((directory/'config.json').read_text()) != config:
            raise ValueError('恢复配置改变')
        completed = resume_boundary(directory,model,optimizer,config)
    else:
        directory.mkdir(exist_ok=False)
        write(directory/'config.json',config)
        write(directory/'initialization.json',{'initial_hash':initial,'fresh':True,'tinyvit':info})
        completed = []
    for epoch in range(len(completed),20):
        verify_runtime()
        summary = run_epoch(model,optimizer,dataset,config,epoch,directory,candidates,completed)
        completed.append(summary)
    final = directory/'final.pt'
    source = directory/'epoch20/checkpoint.pt'
    if not final.exists():
        with source.open('rb') as inp,final.open('xb') as out:
            shutil.copyfileobj(inp,out)
    if sha(final) != completed[-1]['checkpoint_sha256']:
        raise ValueError('final不是epoch20 checkpoint')
    result = {'status':'PASS','variant':variant,'epochs':20,'logical_updates':17660,
              'images':70600,'initial_hash':initial,'final_hash':state_hash(model),
              'final_checkpoint_sha256':sha(final),'config_hash':config['config_hash'],
              'elapsed_seconds':sum(r['elapsed_seconds'] for r in completed),
              'gt_reads':0,'automatic_shutdown':False,'checkpoint':'epoch20 final'}
    if not (directory/'result.json').exists():
        write(directory/'result.json',result)
    print(json.dumps(result),flush=True)
