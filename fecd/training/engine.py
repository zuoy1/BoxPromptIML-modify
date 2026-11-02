"""FP32 update with observable supervision budget and finite-gradient checks."""
import time

import torch

from fecd.training.losses import class_balanced_masked_bce


def train_step(model,optimizer,batch,device,clip_norm):
    torch.cuda.synchronize()
    start=time.perf_counter()
    images=batch['image'].to(device)
    target=batch['target'].to(device)
    mask=batch['selection'].to(device)
    optimizer.zero_grad(set_to_none=True)
    logits=model(images)
    if logits.shape!=target.shape or not torch.isfinite(logits).all():
        raise ValueError('Student output grid or numeric mismatch')
    loss,details=class_balanced_masked_bce(logits,target,mask)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite masked BCE')
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('Missing/nonfinite gradients')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,error_if_nonfinite=True)
    after=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
    if after>clip_norm+1e-5:
        raise AssertionError('Gradient clipping exceeded configured bound')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Optimizer produced nonfinite parameters')
    torch.cuda.synchronize()
    report={k:float(v.item()) for k,v in details.items() if v.numel()==1}
    report.update(loss=float(loss.item()), gradient_norm_before=float(norm.item()),
                  gradient_norm_after=float(after.item()), finite=True,
                  prediction_positive_fraction=float((logits.sigmoid()>=.5).float().mean().item()),
                  teacher_positive_fraction=float(target.mean().item()),
                  selected_full_fraction=float(mask.float().mean().item()),
                  step_seconds=time.perf_counter()-start,
                  peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    for sign in ('pos','neg'):
        field=target==int(sign=='pos')
        candidate=field.flatten(1).sum(1).tolist()
        selected=(mask.bool()&field).flatten(1).sum(1).tolist()
        report[f'candidate_{sign}_per_image']=candidate
        report[f'selected_{sign}_per_image']=selected
    report['sample_keys']=batch['sample_key']
    report['operations']=batch['operation']
    return report
