"""Atomic rolling checkpoints and model/Adam/all-RNG equality evidence."""
import hashlib
import os
from pathlib import Path
import random

import numpy as np
import torch

from fecd.training.checkpoint import save_checkpoint, state_hash


def _digest(digest, value):
    digest.update(type(value).__name__.encode())
    if isinstance(value, dict):
        for key in sorted(value, key=repr):
            _digest(digest, key)
            _digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(str(len(value)).encode())
        for item in value:
            _digest(digest, item)
    elif torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    else:
        digest.update(repr(value).encode())


def training_state_hash(model, optimizer):
    digest = hashlib.sha256()
    _digest(digest, {'model_memory': state_hash(model), 'optimizer': optimizer.state_dict(),
                     'python': random.getstate(), 'numpy': np.random.get_state(),
                     'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()})
    return digest.hexdigest()


def save_last(path, model, optimizer, config, update, position):
    path = Path(path)
    pending = path.with_suffix('.pending.pt')
    save_checkpoint(pending, model, optimizer, config, update, position)
    with pending.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)

