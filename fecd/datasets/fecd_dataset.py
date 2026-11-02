"""Student-facing dataset: accepts only image/cache/box metadata, never GT paths."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from fecd.pseudo.cache_schema import load_cache
from fecd.datasets.transforms import RandomDiscreteGeometry,transform_boxes_xyxy

MEAN=np.asarray([.485,.456,.406],np.float32);STD=np.asarray([.229,.224,.225],np.float32)
class FECDDataset(Dataset):
    def __init__(self,index,expected_config_hash=None,augment=False,geometry=None,hard_single=False):
        """Index is a list or JSON containing image_path, cache_path, sample identity, native boxes only."""
        if isinstance(index,(str,Path)): self.samples=json.loads(Path(index).read_text())
        else:self.samples=list(index)
        allowed={'image_path','cache_path','dataset','split','sample_id','sample_key','boxes_xyxy','image_sha1','box_hash'}
        for s in self.samples:
            extra=set(s)-allowed
            if extra:raise ValueError(f'FECD student index has forbidden/unexpected fields: {sorted(extra)}')
            if s.get('split')!='train':raise ValueError('FECDDataset only accepts train split')
            if 'image_path' not in s or 'cache_path' not in s:raise ValueError('image and cache paths are required')
        self.expected_config_hash=expected_config_hash;self.augment=augment
        self.geometry=geometry or RandomDiscreteGeometry();self.hard_single=hard_single
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        s=self.samples[index]
        with Image.open(s['image_path']) as im:
            native_w,native_h=im.size
            im=im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR);image=np.asarray(im,dtype=np.uint8)
        c=load_cache(s['cache_path'],self.expected_config_hash)
        if (str(c['dataset']),str(c['split']),str(c['sample_id']),str(c['sample_key']),str(c['image_sha1']),str(c['box_hash']))!=(s['dataset'],s['split'],s['sample_id'],s['sample_key'],s['image_sha1'],s['box_hash']):raise ValueError('image/cache identity or source hash mismatch')
        maps=[c['mu'].astype(np.float32),c['var'].astype(np.float32),c['agreement'].astype(np.float32),np.moveaxis(c['box_masks'],0,-1)]
        boxes_native=np.asarray(s.get('boxes_xyxy',[]),dtype=np.float32).reshape(-1,4)[:3]
        boxes_224=boxes_native.copy()
        if len(boxes_224):boxes_224[:,[0,2]]*=224/native_w;boxes_224[:,[1,3]]*=224/native_h
        operation='identity'
        if self.augment:
            operation=self.geometry.sample_operation();image,*maps=self.geometry(image,*maps,operation=operation)
            boxes_224=transform_boxes_xyxy(boxes_224,(224,224),operation)
        image=(image.astype(np.float32)/255.-MEAN)/STD
        mu,var,agreement,box_masks=maps;box_masks=np.moveaxis(box_masks,-1,0)
        mu=mu.astype(np.float32);var=var.astype(np.float32);agreement=agreement.astype(np.float32);box_masks=box_masks.astype(np.float32)
        if self.hard_single:
            if int(c['k'])!=1:raise ValueError('hard_single is for the cached K=1 baseline')
            target=(mu>=.5).astype(np.float32)
        else:target=mu
        native_boxes=np.zeros((3,4),dtype=np.float32);native_boxes[:len(boxes_native)]=boxes_native
        boxes=np.zeros((3,4),dtype=np.float32);boxes[:len(boxes_224)]=boxes_224
        return {'image':torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))),
          'mu':torch.from_numpy(np.ascontiguousarray(mu[None])),'var':torch.from_numpy(np.ascontiguousarray(var[None])),
          'agreement':torch.from_numpy(np.ascontiguousarray(agreement[None])),'box_masks':torch.from_numpy(np.ascontiguousarray(box_masks)),
          'target':torch.from_numpy(np.ascontiguousarray(target[None])), 'boxes_xyxy':torch.from_numpy(boxes),'boxes_native_xyxy':torch.from_numpy(native_boxes),'num_boxes':int(c['num_boxes']),
          'sample_id':s['sample_id'],'dataset':s['dataset'],'split':s['split'],'sample_key':s['sample_key'],'k':int(c['k']),'cache_key':str(c['cache_key'])}
