"""Independent, immutable-checkpoint, full-new_dev evaluation entry point."""
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from fecd.model_loading import build_boxpromptiml_with_local_tinyvit
from fecd.pseudo.cache_schema import canonical_hash
from fecd.training.checkpoint import state_hash
from fecd.training.f0_protocol import EVAL_PROTOCOL, EVAL_HASH
from fecd.evaluation.data import dev_allowlist, file_sha, load_rgb, load_dev_gt, SPLIT_SHA
from fecd.evaluation.memory import deterministic, initial_memory, infer_one, reset_memory
from fecd.evaluation.metrics import output_diagnostics, segmentation_metrics, summarize_records

TINYVIT=Path('/home/weights/tiny_vit_5m_22kto1k_distill.pth')
TINYVIT_SHA='e4894bab91be2d8f1b5e9a2147b57f35ae84be46e195d6586308864fb06fad3d'


def load_independent_model(checkpoint,device='cuda:0'):
    deterministic()
    random.seed(EVAL_PROTOCOL['seed']);np.random.seed(EVAL_PROTOCOL['seed']);torch.manual_seed(EVAL_PROTOCOL['seed'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=payload['config']
    if (payload.get('schema')!='boxprompt-stage4p-checkpoint-v1' or config.get('split_hash')!=SPLIT_SHA
            or config.get('training_images')!=3530 or
            canonical_hash({k:v for k,v in config.items() if k!='config_hash'})!=config.get('config_hash')):
        raise ValueError('Checkpoint is not a verified own-run student_train checkpoint')
    if Path(config['tinyvit_checkpoint']).resolve()!=TINYVIT or file_sha(TINYVIT)!=TINYVIT_SHA:
        raise ValueError('Local TinyViT checkpoint identity mismatch')
    model,adapter=build_boxpromptiml_with_local_tinyvit(TINYVIT)
    model.load_state_dict(payload['model'],strict=True)
    model.to(device)
    snapshot={name:{'memory_bank':value} for name,value in payload['memory'].items()}
    reset_memory(model,snapshot)
    if state_hash(model)!=payload['model_hash']:
        raise ValueError('Student checkpoint model/memory digest mismatch')
    model.eval()
    return model,initial_memory(model),{'checkpoint_config':config,'checkpoint_model_hash':payload['model_hash'],
                                          'adapter':adapter,'device':device}


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def evaluate(checkpoint,output_dir):
    checkpoint=Path(checkpoint).resolve(strict=True);output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    before=file_sha(checkpoint)
    rows,allowlist,sources,split=dev_allowlist()
    metadata={'protocol':EVAL_PROTOCOL,'protocol_hash':EVAL_HASH,'checkpoint':str(checkpoint),
              'checkpoint_sha256':before,'split_hash':split['manifest_sha256'],
              'members_hash':canonical_hash([r['sample_key'] for r in rows]),
              'new_confirm_gt_access_count':0,'batch_size':1}
    write_json(output_dir/'evaluation_config.json',metadata)
    write_json(output_dir/'gt_allowlist.json',{'members':rows,'sources':sources,
                                             'members_hash':metadata['members_hash']})
    model,snapshot,construction=load_independent_model(checkpoint)
    records=[];start=time.perf_counter()
    with (output_dir/'per_image.jsonl').open('x',encoding='utf-8') as stream:
        for row in rows:
            image,native_hw=load_rgb(row)
            logits=infer_one(model,image.to(construction['device']),snapshot)
            if logits.shape!=(1,1,224,224):
                raise ValueError('Student logits do not follow frozen 224 grid')
            array=logits[0,0].numpy()
            gt=load_dev_gt(row,sources[row['sample_key']],native_hw,allowlist,output_dir/'gt_access.jsonl')
            record={'sample_key':row['sample_key'],'dataset':row['dataset'],'sample_id':row['sample_id'],
                    'source_group_id':row['source_group_id'],'native_hw':list(native_hw),
                    **segmentation_metrics(np.exp(-np.logaddexp(0.,-array.astype(np.float64)))>=EVAL_PROTOCOL['threshold'],gt),
                    **output_diagnostics(array)}
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');stream.flush()
            records.append(record)
    summary=summarize_records(records)
    after=file_sha(checkpoint)
    if before!=after:
        raise ValueError('Evaluation modified disk checkpoint')
    all_stats=summary['domains']['ALL']
    area_problem=sum(r['all_zero_prediction'] or r['near_full_prediction'] for r in records)/len(records)
    saturated=sum(r['saturated_probability_mean'] for r in records)/len(records)
    summary.update(metadata)
    summary.update(construction=construction,seconds=time.perf_counter()-start,
                   checkpoint_sha256_after=after,gt_access_count=len(records),
                   zero_or_near_full_fraction=area_problem,saturated_probability_fraction=saturated,
                   stop_formal_request=(area_problem>=.5 and all_stats['image_macro_f1']<=.05) or saturated>=.9,
                   per_image_sha256=file_sha(output_dir/'per_image.jsonl'),
                   gt_access_log_sha256=file_sha(output_dir/'gt_access.jsonl'))
    summary['result_hash']=canonical_hash(summary)
    write_json(output_dir/'summary.json',summary)
    return summary
