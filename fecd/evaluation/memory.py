"""Explicit restoration of mutable attention state before independent inference."""
import hashlib
import os

import torch


def deterministic():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def initial_memory(model):
    return {name:{'memory_bank':module.memory_bank.detach().cpu().clone() if module.memory_bank is not None else None}
            for name,module in model.named_modules() if hasattr(module,'memory_bank')}


def reset_memory(model, snapshot):
    modules=dict(model.named_modules())
    names={name for name,module in modules.items() if hasattr(module,'memory_bank')}
    if names!=set(snapshot):
        raise ValueError('Attention memory snapshot does not match model')
    device=next(model.parameters()).device
    for name,state in snapshot.items():
        value=state['memory_bank']
        modules[name].memory_bank=value.to(device).clone() if value is not None else None


def memory_hash(snapshot):
    digest=hashlib.sha256()
    for name,state in sorted(snapshot.items()):
        digest.update(name.encode());value=state['memory_bank']
        if value is None:
            digest.update(b'None')
        else:
            digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
            digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def infer_one(model, image, snapshot):
    if image.ndim!=4 or image.shape[0]!=1:
        raise ValueError('Strict inference requires batch size one')
    model.eval();reset_memory(model,snapshot)
    with torch.inference_mode():
        logits=model(image)
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite evaluation logits')
    return logits.detach().cpu().clone()
