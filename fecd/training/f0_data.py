"""4F0 fixed-selection data; original 4P behavior remains untouched."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from fecd.confidence.teacher_scores import sam_confidence
from fecd.datasets.transforms import apply_spatial_transform
from fecd.training.data import ControlledDataset, OPERATIONS
from fecd.training.f0_selection import MODES, select_for_variant

BASE_VARIANTS={'P0':'P0','P1':'P1','P2F':'P2','P3F':'P3','P2E':'P2'}


class EpochDataset(ControlledDataset):
    """Frozen student identities with epoch-independent canonical masks for P2F/P3F."""
    def __init__(self,variant,seed=20261029,epoch=0):
        if variant not in MODES:
            raise ValueError('Unknown 4F0 baseline')
        super().__init__(BASE_VARIANTS[variant],seed,epoch)
        self.formal_variant=variant

    def __getitem__(self,index):
        self._validate_input(index)
        sample=self.dataset[index]
        if sample['k']!=self.k:
            raise ValueError('Wrong teacher K in 4F0 baseline')
        target=(sample['mu'].numpy()>=.5).astype(np.float32)
        confidence=sam_confidence(sample['mu'].numpy(),sample['var'].numpy(),
                                  sample['agreement'].numpy())
        selection,counts=select_for_variant(
            target,confidence,self.formal_variant,self.seed,sample['sample_key'],self.epoch)
        canonical_hash=hashlib.sha256(np.ascontiguousarray(selection).tobytes()).hexdigest()
        digest=hashlib.sha256(
            f'{self.seed}:{self.epoch}:{sample["sample_key"]}:geometry'.encode()).digest()
        operation=OPERATIONS[int.from_bytes(digest[:8],'big')%len(OPERATIONS)]
        image=sample['image'].numpy().transpose(1,2,0)
        image,target_2d,selection_2d=apply_spatial_transform(
            image,target[0],selection[0],operation=operation)
        result={'image':torch.from_numpy(image.transpose(2,0,1).copy()),
                'target':torch.from_numpy(target_2d[None].copy()),
                'selection':torch.from_numpy(selection_2d[None].copy()),
                'sample_key':sample['sample_key'],'sample_id':sample['sample_id'],
                'dataset':sample['dataset'],'operation':operation,'k':self.k,
                'canonical_selection_hash':canonical_hash,
                'selection_epoch':self.epoch if self.formal_variant=='P2E' else 0}
        result.update({key:int(value) for key,value in counts.items()})
        return result
