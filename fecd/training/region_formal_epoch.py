"""完整Epoch事务：真实区域选择、跨轮覆盖、完整checkpoint及提交标记。"""
from collections import Counter
import json
import time

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from fecd.pseudo.cache_schema import canonical_hash
from fecd.training.checkpoint import save_checkpoint, state_hash
from fecd.training.f1_protocol import DOMAINS, epoch_batches, epoch_identity
from fecd.training.f1_state import training_state_hash
from fecd.training.region_access import RegionReadAudit
from fecd.training.region_coverage import summarize_coverage
from fecd.training.region_engine import train_region_step
from fecd.training.region_formal_protocol import OUTPUT, sha, write, commit_summary
from fecd.training.region_formal_protocol import collapse_gate


def require_parameter_change(model, before):
    if not any(not torch.equal(parameter.detach(),old)
               for parameter,old in zip((p for p in model.parameters() if p.requires_grad),before)):
        raise ValueError('优化器未改变可训练参数，memory变化不能代替更新')


def run_epoch(model, optimizer, dataset, config, epoch, directory, candidates, completed):
    target = directory/f'epoch{epoch+1:02d}'
    target.mkdir(exist_ok=False)
    dataset.set_epoch(epoch)
    size = len(dataset)
    masks = np.zeros((size,6272), np.uint8)
    previous = cumulative = None
    if epoch:
        with np.load(directory/f'epoch{epoch:02d}/selection.npz', allow_pickle=False) as data:
            previous, cumulative = data['masks'].copy(), data['cumulative'].copy()
    steps, rows, domains = [], [], Counter()
    start = time.perf_counter()
    before_model = state_hash(model)
    torch.cuda.reset_peak_memory_stats()
    with RegionReadAudit([r['image_path'] for r in dataset.samples],
                         [r['cache_path'] for r in dataset.samples], OUTPUT) as audit, (
            target/'steps.jsonl').open('x') as logs, (target/'images.jsonl').open('x') as images:
        for index, ids in enumerate(epoch_batches(size, config['seed'], epoch, 4)):
            batch = default_collate([dataset[i] for i in ids])
            parameters = [p.detach().clone() for p in model.parameters() if p.requires_grad]
            record, details, selections = train_region_step(model, optimizer, batch, config, epoch)
            require_parameter_change(model,parameters)
            del parameters
            record.update(epoch=epoch+1, epoch_update=index+1, update=epoch*883+index+1,
                          batch_size=len(ids))
            if any(r['selected_positive']+r['selected_negative_in']+r['selected_negative_out'] == 0
                   for r in details):
                raise ValueError('异常空监督')
            masks[ids] = np.packbits(selections.reshape(len(ids),-1), axis=1)
            steps.append(record)
            rows.extend(details)
            domains.update(batch['dataset'])
            logs.write(json.dumps(record, allow_nan=False)+'\n')
            for row in details:
                row['update'] = record['update']
                images.write(json.dumps(row, allow_nan=False)+'\n')
            if (index+1)%200 == 0 or index+1 == 883:
                logs.flush(); images.flush()
                print(json.dumps({'variant':config['variant'], 'epoch':epoch+1, 'update':index+1,
                                  'loss':record['loss'], 'seconds':time.perf_counter()-start}), flush=True)
        expected = {str(p) for r in dataset.samples for p in (r['image_path'],r['cache_path'])}
        if len(steps) != 883 or len(rows) != 3530 or dict(domains) != DOMAINS or steps[-1]['batch_size'] != 2:
            raise ValueError('Epoch成员/更新/域预算异常')
        if audit.denied or set(audit.opened) != expected:
            raise ValueError('训练数据访问集合异常')
        write(target/'access.json', {'opened_counts':dict(Counter(audit.opened)),
                                     'denied':audit.denied, 'gt_reads':0})
        coverage, cumulative = summarize_coverage(masks, previous, cumulative,
                                                  candidates['inside'], candidates['outside'])
        for source, row in zip(dataset.samples, coverage):
            row.update(sample_key=source['sample_key'], dataset=source['dataset'], epoch=epoch+1)
        write(target/'coverage.json', coverage)
        np.savez_compressed(target/'selection.npz', masks=masks, cumulative=cumulative,
                            sample_keys=np.array([r['sample_key'] for r in dataset.samples]))
        save_checkpoint(target/'checkpoint.pt', model, optimizer, config, (epoch+1)*883,
                        {'epoch':epoch+1,'next_batch':0})
    identity = epoch_identity([s['sample_key'] for s in dataset.samples], config['seed'], epoch)
    if (canonical_hash([r['sample_key'] for r in rows]) != identity['sample_order_hash']
            or canonical_hash([r['operation'] for r in rows]) != identity['geometry_hash']):
        raise ValueError('实际顺序或几何不符')
    summary = {'status':'PASS','epoch':epoch+1,'images':len(rows),'updates':883,
               'total_updates':(epoch+1)*883,'domain_counts':dict(domains),'identity':identity,
               'mean_loss':float(np.mean([r['loss'] for r in steps])),
               'mean_pos_contribution':float(np.mean([r['pos_contribution'] for r in steps])),
               'mean_neg_contribution':float(np.mean([r['neg_contribution'] for r in steps])),
               'gradient_norm_after_max':max(r['gradient_norm_after'] for r in steps),
               'prediction_area_mean':float(np.mean([r['prediction_area'] for r in rows])),
               'all_zero_outputs':sum(r['all_zero'] for r in rows),
               'near_full_outputs':sum(r['near_full'] for r in rows),
               'peak_cuda_allocated':torch.cuda.max_memory_allocated(),
               'peak_cuda_reserved':torch.cuda.max_memory_reserved(),
               'elapsed_seconds':time.perf_counter()-start,'model_memory_hash':state_hash(model),
               'complete_state_hash':training_state_hash(model,optimizer),'gt_reads':0}
    summary['checkpoint_sha256'] = sha(target/'checkpoint.pt')
    summary['selection_sha256'] = sha(target/'selection.npz')
    summary['parameter_update_verified'] = summary['model_memory_hash'] != before_model
    if not summary['parameter_update_verified'] or collapse_gate(completed+[summary]):
        write(target/'BLOCKED.json',{'status':'BLOCKED','reason':'无法更新或连续严重全局塌缩'})
        raise ValueError('无法更新或连续两轮严重全局输出塌缩')
    commit_summary(target/'summary.json', summary)
    print(json.dumps({'variant':config['variant'],'epoch_completed':epoch+1,**summary}), flush=True)
    return summary
