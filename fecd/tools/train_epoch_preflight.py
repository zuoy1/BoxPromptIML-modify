"""Exactly one full student_train epoch, separate from frozen 4P and formal20."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from torch.utils.data._utils.collate import default_collate

from fecd.tools.train_controlled_baseline import seeds,build
from fecd.training.access import DataReadAudit
from fecd.training.checkpoint import save_checkpoint,restore_checkpoint,state_hash
from fecd.training.f0_data import EpochDataset
from fecd.training.f0_engine import train_step
from fecd.training.f0_diagnostics import summarize_outputs
from fecd.training.f0_protocol import configuration,EVAL_HASH
from fecd.pseudo.cache_schema import canonical_hash

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'fecd/assets/stage4f0'


def run(variant):
    config=configuration(variant)
    if variant not in ('P0','P1','P2F','P3F') or config['epochs']!=1:
        raise ValueError('Only the four authorized single-epoch preflights')
    if json.loads((ROOT/f'fecd/configs/stage4f0/{variant}.json').read_text())!=config:
        raise ValueError('Configuration differs from frozen version')
    gates=json.loads((OUTPUT/'gates.json').read_text())
    if gates['status']!='PASS' or gates['eval_protocol_hash']!=EVAL_HASH:
        raise ValueError('Selection/memory/protocol gate not passed')
    directory=OUTPUT/variant
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seeds(config['seed'])
    dataset=EpochDataset(variant,seed=config['seed'],epoch=0)
    assert len(dataset)==3530 and dataset.split_hash==config['split_hash']
    model,optimizer,info=build(config)
    initial=state_hash(model)
    order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(config['seed'])).tolist()
    counts=Counter()
    diagnostics=[]
    records=[]
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    with DataReadAudit([r['image_path'] for r in dataset.samples],
                       [r['cache_path'] for r in dataset.samples]) as audit, (
            directory/'steps.jsonl').open('x') as log, (
            directory/'train_outputs.jsonl').open('x') as outputs:
        for step,offset in enumerate(range(0,len(order),config['batch_size'])):
            ids=order[offset:offset+config['batch_size']]
            batch=default_collate([dataset[i] for i in ids])
            if step==config['resume_probe_update']:
                expected,ignored=train_step(model,optimizer,batch,config)
                expected_hash=state_hash(model)
                del model,optimizer
                torch.cuda.empty_cache()
                model,optimizer,_=build(config)
                saved=restore_checkpoint(directory/'update4.pt',model,optimizer,config['config_hash'])
                assert saved['position']=={'epoch':0,'next_batch':step}
                result,images=train_step(model,optimizer,batch,config)
                assert result['loss']==expected['loss'] and state_hash(model)==expected_hash
                (directory/'resume.json').write_text(json.dumps({
                    'status':'PASS','update':step+1,'loss_exact':True,'full_state_exact':True})+'\n')
            else:
                result,images=train_step(model,optimizer,batch,config)
            counts.update(batch['dataset'])
            result.update(update=step+1,batch_size=len(ids),sample_keys=batch['sample_key'],
                          operations=batch['operation'])
            records.append(result)
            log.write(json.dumps(result,allow_nan=False)+'\n')
            for image in images:
                image['update']=step+1
                outputs.write(json.dumps(image,allow_nan=False)+'\n')
            diagnostics.extend(images)
            if step+1==config['resume_probe_update']:
                save_checkpoint(directory/'update4.pt',model,optimizer,config,step+1,
                                {'epoch':0,'next_batch':step+1})
            if (step+1)%100==0 or offset+len(ids)==len(order):
                log.flush();outputs.flush()
                print(json.dumps({'variant':variant,'updates':step+1,'images':offset+len(ids),
                    'loss':result['loss'],'elapsed_seconds':time.perf_counter()-start}),flush=True)
        assert len(records)==883 and len(diagnostics)==3530 and records[-1]['batch_size']==2
        save_checkpoint(directory/'epoch1.pt',model,optimizer,config,883,{'epoch':1,'next_batch':0})
    assert not audit.denied and dict(counts)=={'CASIAv2':3278,'NIST16':172,'Columbia':52,'Coverage':28}
    result={'status':'PASS','variant':variant,'config_hash':config['config_hash'],
            'initial_hash':initial,'final_hash':state_hash(model),'order_hash':canonical_hash(order),
            'membership_hash':dataset.membership_hash,'images':3530,'logical_updates':883,
            'physical_updates':884,'last_batch':2,'dataset_forward_backward_counts':dict(counts),
            'elapsed_seconds':time.perf_counter()-start,'gt_reads':0,'tinyvit':info,
            'mean_loss':sum(r['loss'] for r in records)/len(records),
            'access_unique_files':len(set(audit.opened)),'denied':audit.denied}
    (directory/'training_summary.json').write_text(json.dumps(summarize_outputs(diagnostics),indent=2)+'\n')
    (directory/'access.json').write_text(json.dumps({'opened':audit.opened,'denied':audit.denied})+'\n')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',required=True,choices=('P0','P1','P2F','P3F'))
    run(parser.parse_args().variant)


if __name__=='__main__':
    main()
