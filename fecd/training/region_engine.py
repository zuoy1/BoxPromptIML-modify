"""单次原始学生更新；选择在无梯度的canonical网格上完成。"""
import time

import numpy as np
import torch

from fecd.training.losses import class_balanced_masked_bce
from fecd.training.region_data import spatial
from fecd.training.region_diagnostics import describe
from fecd.training.region_negative_selection import select_region_pixels


def train_region_step(model, optimizer, batch, config, epoch):
    device = config['device']
    cuda = str(device).startswith('cuda')
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch['image'].to(device))
    target = batch['target'].to(device)
    if logits.shape != target.shape or not torch.isfinite(logits).all():
        raise FloatingPointError('学生输出形状或数值异常')
    masks, rows, canonical_masks = [], [], []
    for i, key in enumerate(batch['sample_key']):
        op = batch['operation'][i]
        canonical = spatial(logits[i].detach(), op, inverse=True).cpu()
        probability = canonical.sigmoid()
        teacher, boxes = batch['canonical_target'][i], batch['box_union'][i]
        mask, _ = select_region_pixels(teacher, boxes, config['variant'], config['seed'], key,
                                      epoch, probability)
        reference, _ = select_region_pixels(teacher, boxes, 'R0', config['seed'], key,
                                           epoch, probability)
        next_mask, _ = select_region_pixels(teacher, boxes, config['variant'], config['seed'], key,
                                           epoch+1, probability)
        if not np.array_equal(mask & teacher.numpy().astype(bool),
                              reference & teacher.numpy().astype(bool)):
            raise AssertionError('R0/R1固定正类选择不一致')
        masks.append(spatial(torch.from_numpy(mask), op))
        canonical_masks.append(mask.squeeze())
        metadata = {'sample_key': key, 'dataset': batch['dataset'][i], 'operation': op,
                    'num_boxes': int(batch['num_boxes'][i]), 'epoch': epoch}
        rows.append(describe(canonical.numpy(), teacher.numpy(), boxes.numpy(), mask,
                             reference.squeeze(), next_mask.squeeze(), metadata,
                             batch['mu'][i].numpy(), batch['var'][i].numpy(),
                             batch['confidence'][i].numpy(), probability.numpy()))
    selection = torch.stack(masks).to(device)
    loss, details = class_balanced_masked_bce(logits, target, selection)
    if not torch.isfinite(loss):
        raise FloatingPointError('非有限损失')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('缺失或非有限梯度')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'], error_if_nonfinite=True)
    after = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))
    optimizer.step()
    if after > config['clip_norm']+1e-5 or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('梯度裁剪或更新后参数异常')
    if cuda:
        torch.cuda.synchronize()
    record = {k: float(v.item()) for k, v in details.items() if v.numel() == 1}
    record.update(loss=float(loss.item()), finite=True, gradient_norm_before=float(norm.item()),
                  gradient_norm_after=float(after.item()), step_seconds=time.perf_counter()-start,
                  sample_keys=list(batch['sample_key']), operations=list(batch['operation']))
    return record, rows, np.stack(canonical_masks)
