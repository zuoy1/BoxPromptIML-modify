"""Single update with same 4P loss plus explicit per-image output diagnostics."""
import time

import torch

from fecd.training.losses import class_balanced_masked_bce
from fecd.training.f0_diagnostics import describe_output


def train_step(model,optimizer,batch,config):
    torch.cuda.synchronize()
    start=time.perf_counter()
    image=batch['image'].to(config['device'])
    target=batch['target'].to(config['device'])
    selection=batch['selection'].to(config['device'])
    optimizer.zero_grad(set_to_none=True)
    logits=model(image)
    if logits.shape!=target.shape or not torch.isfinite(logits).all():
        raise FloatingPointError('Bad student output')
    diagnostics=describe_output(logits,target,selection)
    loss,details=class_balanced_masked_bce(logits,target,selection)
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('Missing or nonfinite gradients')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
    after=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()) or after>config['clip_norm']+1e-5:
        raise FloatingPointError('Nonfinite parameters or invalid clip')
    torch.cuda.synchronize()
    record={k:float(v.item()) for k,v in details.items() if v.numel()==1}
    record.update(loss=float(loss.item()),gradient_norm_before=float(norm.item()),
                  gradient_norm_after=float(after.item()),step_seconds=time.perf_counter()-start,
                  peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    for i,row in enumerate(diagnostics):
        row.update(sample_key=batch['sample_key'][i],dataset=batch['dataset'][i],
                   canonical_selection_hash=batch['canonical_selection_hash'][i],
                   operation=batch['operation'][i])
    return record,diagnostics
