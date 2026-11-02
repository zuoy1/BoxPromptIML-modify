"""Own-run checkpoints include mutable attention memory and all RNG states."""
import hashlib
from pathlib import Path
import random

import numpy as np
import torch


def state_hash(model):
    digest=hashlib.sha256()
    states=dict(model.state_dict())
    for name,module in model.named_modules():
        if hasattr(module,'memory_bank') and module.memory_bank is not None:
            states[f'{name}.memory_bank']=module.memory_bank
    for name,value in sorted(states.items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path,model,optimizer,config,update,position):
    path=Path(path)
    memory={name:module.memory_bank.detach().cpu().clone() if module.memory_bank is not None else None
            for name,module in model.named_modules() if hasattr(module,'memory_bank')}
    payload={'schema':'boxprompt-stage4p-checkpoint-v1','config':config,'update':update,
             'position':position,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
             'memory':memory,'model_hash':state_hash(model),
             'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
                    'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
    with path.open('xb') as stream:
        torch.save(payload,stream)


def restore_checkpoint(path,model,optimizer,expected_config_hash):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema')!='boxprompt-stage4p-checkpoint-v1' or (
            payload['config']['config_hash']!=expected_config_hash):
        raise ValueError('Checkpoint schema/config differs from requested run')
    model.load_state_dict(payload['model'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    modules=dict(model.named_modules())
    if set(payload['memory'])!={n for n,m in modules.items() if hasattr(m,'memory_bank')}:
        raise ValueError('Mutable attention memory contract mismatch')
    device=next(model.parameters()).device
    for name,value in payload['memory'].items():
        modules[name].memory_bank=value.to(device) if value is not None else None
    if state_hash(model)!=payload['model_hash']:
        raise ValueError('Restored model/memory hash mismatch')
    random.setstate(payload['rng']['python'])
    np.random.set_state(payload['rng']['numpy'])
    torch.set_rng_state(payload['rng']['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload['rng']['cuda'])
    return payload
