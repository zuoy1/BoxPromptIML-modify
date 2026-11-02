"""Frozen student_train-only hard-label data with deterministic pixel budgets."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from fecd.confidence.teacher_scores import sam_confidence
from fecd.datasets.fecd_dataset import FECDDataset
from fecd.datasets.transforms import apply_spatial_transform
from fecd.diagnostics.stage3ra_split import read_split
from fecd.pseudo.cache_schema import canonical_hash, load_cache
from fecd.training.selection import select_pixels

PROJECT=Path(__file__).resolve().parents[2]
K1_ROOT=PROJECT/'fecd/assets/stage2c_cache/k1/d7ccb769414f'
K8_ROOT=PROJECT/'fecd/assets/stage2c_cache/k8/2bff688546a7'
CONFIGS={1:'d7ccb769414fd8beb943888c81239e77e14e799c6300377673bbc2b6680f159b',
         8:'2bff688546a79079dcbac4ae29bb2c5169823704025c69b746a8cb57df99aafd'}
VARIANTS={'P0':(1,'all'),'P1':(8,'all'),'P2':(8,'random'),'P3':(8,'confidence')}
INDEX_FIELDS={'image_path','cache_path','dataset','split','sample_id','sample_key',
              'boxes_xyxy','image_sha1','box_hash'}
OPERATIONS=('identity','hflip','vflip','rot90:1','rot90:2','rot90:3')


def _build_samples(records, index_rows, cache_root):
    """Validate frozen index identities before exposing student-only metadata."""
    cache_root=Path(cache_root).resolve()
    index={}
    for row in index_rows:
        if set(row)!=INDEX_FIELDS or row['split']!='train':
            raise ValueError('Unexpected student cache index fields or role')
        key=row['sample_key']
        if key in index:
            raise ValueError('Duplicate cache index sample identity')
        index[key]=row
    selected=sorted((r for r in records if r['subset']=='student_train'),
                    key=lambda r:r['sample_key'])
    if len({r['sample_key'] for r in selected})!=len(selected):
        raise ValueError('Duplicate student training identity')
    samples=[]
    for row in selected:
        if row['sample_key'] not in index:
            raise ValueError('Missing student image in teacher cache index')
        source=index[row['sample_key']]
        if any(source[name]!=row[name] for name in
               ('sample_id','sample_key','dataset','image_path')):
            raise ValueError('Student/cache index identity mismatch')
        path=Path(source['cache_path'])
        path=(path if path.is_absolute() else PROJECT/path).resolve()
        if not path.is_relative_to(cache_root):
            raise ValueError('Student teacher cache path leaves frozen directory')
        sample={**source,'image_path':str(Path(source['image_path']).resolve()),
                'cache_path':str(path)}
        samples.append(sample)
    return samples


class ControlledDataset(Dataset):
    """Only the frozen 3,530 student images can enter a production constructor."""
    def __init__(self,variant,seed=20261029,epoch=0):
        if variant not in VARIANTS:
            raise ValueError('Unknown controlled baseline')
        split=read_split()
        records=sorted((r for r in split['records'] if r['subset']=='student_train'),
                       key=lambda r:r['sample_key'])
        if len(records)!=3530:
            raise ValueError('Frozen student_train must contain exactly 3530 images')
        k,_=VARIANTS[variant]
        root=K1_ROOT if k==1 else K8_ROOT
        index_rows=json.loads((root/'dataset_index.json').read_text())
        samples=_build_samples(records,index_rows,root)
        self._configure(variant,seed,epoch,records,samples,CONFIGS[k])
        self.split_hash=split['manifest_sha256']
        self.membership_hash=canonical_hash([r['sample_key'] for r in records])

    def _configure(self,variant,seed,epoch,records,samples,config_hash):
        if variant not in VARIANTS:
            raise ValueError('Unknown controlled baseline')
        self.variant=variant
        self.k,self.selection_mode=VARIANTS[variant]
        self.seed=int(seed)
        self.records=records
        self.samples=samples
        self.config_hash=config_hash
        self.dataset=FECDDataset(samples,expected_config_hash=config_hash,augment=False)
        self.set_epoch(epoch)

    def set_epoch(self,epoch):
        if not isinstance(epoch,int) or epoch<0:
            raise ValueError('Epoch must be a nonnegative integer')
        self.epoch=epoch

    def input_paths(self,index):
        row=self.samples[index]
        return [Path(row['image_path']),Path(row['cache_path'])]

    def __len__(self):
        return len(self.samples)

    def _validate_input(self,index):
        row,source=self.records[index],self.samples[index]
        contents=Path(source['image_path']).read_bytes()
        if hashlib.sha256(contents).hexdigest()!=row['image_sha256']:
            raise ValueError('Student image SHA256 mismatch')
        sha1=hashlib.sha1(contents).hexdigest()
        if sha1!=source['image_sha1']:
            raise ValueError('Student image SHA1 mismatch')
        with Image.open(io.BytesIO(contents)) as image:
            height,width=image.height,image.width
        load_cache(source['cache_path'],self.config_hash,expected={
            **row,'split':'train','k':self.k,'image_sha1':sha1,
            'box_hash':source['box_hash'],'image_h':height,'image_w':width})

    def __getitem__(self,index):
        self._validate_input(index)
        sample=self.dataset[index]
        if sample['k']!=self.k:
            raise ValueError('Wrong teacher K in controlled baseline')
        target=(sample['mu'].numpy()>=.5).astype(np.float32)
        confidence=sam_confidence(sample['mu'].numpy(),sample['var'].numpy(),
                                  sample['agreement'].numpy())
        selection,counts=select_pixels(target,confidence,self.selection_mode,
                                      self.seed,sample['sample_key'],self.epoch)
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
                'dataset':sample['dataset'],'operation':operation,'k':self.k}
        result.update({key:int(value) for key,value in counts.items()})
        return result
