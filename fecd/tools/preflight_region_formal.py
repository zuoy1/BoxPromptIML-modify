"""5C1正式训练前：全成员缓存核验和独立少步完整状态恢复探针。"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate

from fecd.confidence.teacher_scores import sam_confidence
from fecd.pseudo.cache_schema import load_cache
from fecd.tools.train_controlled_baseline import build,seeds
from fecd.training.checkpoint import save_checkpoint,restore_checkpoint,state_hash
from fecd.training.data import CONFIGS
from fecd.training.f1_runner import deterministic
from fecd.training.f1_protocol import epoch_batches,epoch_identity,verify_frozen
from fecd.training.f1_state import training_state_hash
from fecd.training.region_access import RegionReadAudit
from fecd.training.region_data import RegionDataset
from fecd.training.region_engine import train_region_step
from fecd.training.region_negative_selection import select_region_pixels
from fecd.training.region_formal_protocol import OUTPUT,configuration,sha,write,verify_runtime


def main():
    verify_frozen(); verify_runtime()
    if Path('/hy-tmp').stat().st_dev == Path('/').stat().st_dev or shutil.disk_usage('/hy-tmp').free < 15*1024**3:
        raise ValueError('独立数据盘或15GB余量门禁失败')
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name():
        raise ValueError('已验证4090环境不可用')
    deterministic(); Image.init()
    dataset = RegionDataset()
    allowed_rgb = [r['image_path'] for r in dataset.samples]
    allowed_cache = [r['cache_path'] for r in dataset.samples]
    old = json.loads((OUTPUT.parent/'stage5c0/results/source_hashes.json').read_text())
    root = Path(__file__).resolve().parents[2]
    if any(sha(root/name) != digest for name,digest in old.items()):
        raise ValueError('5C0冻结实现或配置变化')
    inside,outside,hashes = [],[],{}
    budgets = Counter()
    with RegionReadAudit(allowed_rgb,allowed_cache,OUTPUT) as audit:
        for index,source in enumerate(dataset.samples):
            dataset._validate_input(index)
            cache = load_cache(source['cache_path'],CONFIGS[8])
            target = cache['mu'].astype(np.float32) >= .5
            box = cache['box_masks'].any(0)
            conf = sam_confidence(*(cache[k].astype(np.float32) for k in ('mu','var','agreement')))
            a,ca = select_region_pixels(target,box,'R0',20261029,source['sample_key'],0)
            b,cb = select_region_pixels(target,box,'R1',20261029,source['sample_key'],0,conf)
            if ca != cb or not np.array_equal(a&target,b&target):
                raise ValueError('正类或区域预算不公平')
            inside.append(np.packbits((~target&box).ravel()))
            outside.append(np.packbits((~target&~box).ravel()))
            hashes[source['cache_path']] = sha(source['cache_path'])
            budgets.update({k:v for k,v in ca.items() if k.startswith('selected_')})
            if (index+1)%500 == 0:
                print('预检缓存',index+1,flush=True)
    expected_hashes = json.loads((OUTPUT.parent/'stage5b/results/training_input_hashes.json').read_text())
    if any(expected_hashes[p] != h for p,h in hashes.items()):
        raise ValueError('训练K8缓存身份hash与历史变化')
    np.savez_compressed(OUTPUT/'results/candidates.npz',inside=np.stack(inside),outside=np.stack(outside),
                        sample_keys=np.array([s['sample_key'] for s in dataset.samples]))
    write(OUTPUT/'results/input_hashes.json',hashes)
    write(OUTPUT/'results/preflight_access.json',{'opened':dict(Counter(audit.opened)),
                                                'denied':audit.denied,'gt_reads':0})
    initial,probes = {},{}
    for variant in ('R0','R1'):
        deterministic(); seeds(20261029)
        config = configuration(variant)
        model,optimizer,_ = build(config)
        initial[variant] = state_hash(model)
        directory = OUTPUT/'probe'/variant
        directory.mkdir(parents=True,exist_ok=False)
        batches = list(epoch_batches(len(dataset),20261029,0,4))
        with RegionReadAudit(allowed_rgb,allowed_cache,OUTPUT) as probe_audit:
            first = default_collate([dataset[i] for i in batches[0]])
            train_region_step(model,optimizer,first,config,0)
            save_checkpoint(directory/'update1.pt',model,optimizer,config,1,{'epoch':0,'next_batch':1})
            batch = default_collate([dataset[i] for i in batches[1]])
            expected,er,em = train_region_step(model,optimizer,batch,config,0)
            expected_hash = training_state_hash(model,optimizer)
            del model,optimizer
            torch.cuda.empty_cache()
            model,optimizer,_ = build(config)
            payload = restore_checkpoint(directory/'update1.pt',model,optimizer,config['config_hash'])
            actual,ar,am = train_region_step(model,optimizer,batch,config,0)
            if (training_state_hash(model,optimizer) != expected_hash or actual['loss'] != expected['loss']
                    or er != ar or not np.array_equal(em,am)
                    or payload['position'] != {'epoch':0,'next_batch':1} or probe_audit.denied):
                raise ValueError('正式配置完整恢复探针不一致')
        probes[variant] = {'status':'PASS','all_state_next_step_exact':True,
                           'state_hash':expected_hash,'physical_updates':3,'gt_reads':0}
        write(directory/'result.json',probes[variant])
        del model,optimizer
        torch.cuda.empty_cache()
    if len(set(initial.values())) != 1:
        raise ValueError('两组fresh初始化不一致')
    keys = [s['sample_key'] for s in dataset.samples]
    write(OUTPUT/'results/epoch_schedule.json',[epoch_identity(keys,20261029,e) for e in range(20)])
    result = {'status':'PASS','initial_hashes':initial,'probes':probes,'images':3530,
              'membership_hash':dataset.membership_hash,'cache_identity_checks':3530,
              'selected_budget':dict(budgets),'automatic_shutdown':False,
              'shutdown_runner_used':False,'remaining_disk_bytes':shutil.disk_usage('/hy-tmp').free,
              'device':torch.cuda.get_device_name(),'gt_reads':0}
    write(OUTPUT/'results/preflight.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
