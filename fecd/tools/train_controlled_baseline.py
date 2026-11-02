"""Run only the authorized eight-update P0-P3 smoke, never formal training."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate

from fecd.model_loading import build_boxpromptiml_with_local_tinyvit
from fecd.pseudo.cache_schema import canonical_hash
from fecd.training.access import DataReadAudit
from fecd.training.checkpoint import save_checkpoint,restore_checkpoint,state_hash
from fecd.training.data import ControlledDataset
from fecd.training.engine import train_step
from fecd.training.protocol import configuration

PROJECT=Path(__file__).resolve().parents[2]
OUTPUT=PROJECT/'fecd/assets/stage4p'


def seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build(config):
    model,info=build_boxpromptiml_with_local_tinyvit(config['tinyvit_checkpoint'])
    model=model.to(config['device']).train()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],betas=tuple(config['betas']),
                               eps=config['eps'],weight_decay=config['weight_decay'])
    return model,optimizer,info


def run(variant):
    config=configuration(variant)
    path=PROJECT/f'fecd/configs/stage4p/{variant}.json'
    if json.loads(path.read_text())!=config:
        raise ValueError('Configuration source differs from checked-in JSON')
    directory=OUTPUT/variant
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this measured smoke')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seeds(config['seed'])
    Image.init()
    dataset=ControlledDataset(variant,seed=config['seed'],epoch=0)
    if len(dataset)!=3530:
        raise ValueError('Student train membership count differs')
    order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(config['seed'])).tolist()
    model,optimizer,info=build(config)
    initial_hash=state_hash(model)
    start=time.perf_counter()
    # The dataset exposes only frozen image/cache dictionaries, never labels.
    samples=dataset.samples
    rgb=[r['image_path'] for r in samples]
    cache=[r['cache_path'] for r in samples]
    records=[]
    torch.cuda.reset_peak_memory_stats()
    resumed=False
    saved=directory/'update4.pt'
    with DataReadAudit(rgb,cache) as audit, (directory/'steps.jsonl').open('x') as stream:
        for update in range(config['updates']):
            ids=order[update*config['batch_size']:(update+1)*config['batch_size']]
            batch=default_collate([dataset[i] for i in ids])
            if update==config['checkpoint_update']:
                # Compare one uninterrupted next step against a true disk restore.
                continuation=train_step(model,optimizer,batch,config['device'],config['gradient_clip_norm'])
                expected_hash=state_hash(model)
                del model,optimizer
                torch.cuda.empty_cache()
                model,optimizer,_=build(config)
                payload=restore_checkpoint(saved,model,optimizer,config['config_hash'])
                if payload['position']!={'epoch':0,'next_batch':update}:
                    raise AssertionError('Resume position mismatch')
                actual=train_step(model,optimizer,batch,config['device'],config['gradient_clip_norm'])
                if state_hash(model)!=expected_hash or continuation['loss']!=actual['loss']:
                    raise AssertionError('Restored next update differs from uninterrupted reference')
                resumed=True
                resume={'status':'passed','update':update+1,'loss_equal':True,
                        'model_and_memory_hash_equal':True,'uninterrupted_hash':expected_hash,
                        'loss':actual['loss']}
                (directory/'resume_check.json').write_text(json.dumps(resume,indent=2)+'\n')
            else:
                actual=train_step(model,optimizer,batch,config['device'],config['gradient_clip_norm'])
            actual.update(update=update+1,batch_indices=ids)
            records.append(actual)
            stream.write(json.dumps(actual,allow_nan=False)+'\n')
            stream.flush()
            print(json.dumps({'variant':variant,**actual}),flush=True)
            if update+1==config['checkpoint_update']:
                save_checkpoint(saved,model,optimizer,config,update+1,
                                {'epoch':0,'next_batch':update+1})
        save_checkpoint(directory/'final.pt',model,optimizer,config,config['updates'],
                        {'epoch':0,'next_batch':config['updates']})
    if audit.denied or not resumed or state_hash(model)==initial_hash:
        raise AssertionError('Access/resume/parameter update gate failed')
    result={'status':'PASS','variant':variant,'config_hash':config['config_hash'],
            'training_images':len(dataset),'consumed_images':len(records)*config['batch_size'],
            'logical_updates':len(records),'physical_updates_including_resume_probe':len(records)+1,
            'initial_model_hash':initial_hash,'final_model_hash':state_hash(model),
            'order_hash':canonical_hash(order),'train_member_keys_hash':canonical_hash(
                [r['sample_key'] for r in samples]),
            'tinyvit':info,'tinyvit_sha256':hashlib.sha256(Path(config['tinyvit_checkpoint']).read_bytes()).hexdigest(),
            'gt_files_read':0,'validation_performed':False,'confirmed_test_access':False,
            'access_open_count':len(audit.opened),'access_unique_files':len(set(audit.opened)),
            'access_denied':audit.denied,'access_allowed_rgb_count':len(set(rgb)),
            'elapsed_seconds':time.perf_counter()-start,
            'cuda_device':torch.cuda.get_device_name(),'torch_version':torch.__version__,
            'cuda_version':torch.version.cuda,'deterministic':True}
    (directory/'access.json').write_text(json.dumps({'opened':audit.opened,'denied':audit.denied},indent=2)+'\n')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('P0','P1','P2','P3'),required=True)
    args=parser.parse_args()
    run(args.variant)


if __name__=='__main__':
    main()
