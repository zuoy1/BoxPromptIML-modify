"""Synchronized discrete geometry for image and all cache maps."""
from __future__ import annotations
import random
import numpy as np

def transform_boxes_xyxy(boxes,image_hw,operation):
    """Transform half-open XYXY boxes in the same discrete image grid as the tensors."""
    out=np.asarray(boxes,dtype=np.float32).reshape(-1,4).copy();h,w=map(int,image_hw)
    if operation=='identity':return out
    if operation=='hflip':out[:,[0,2]]=w-out[:,[2,0]]
    elif operation=='vflip':out[:,[1,3]]=h-out[:,[3,1]]
    elif operation.startswith('rot90:'):
        k=int(operation.split(':')[1])%4
        for _ in range(k):
            x1,y1,x2,y2=out.T.copy();out=np.stack([y1,w-x2,y2,w-x1],axis=1);h,w=w,h
    else:raise ValueError(f'unsupported spatial op {operation}')
    return out

def apply_spatial_transform(image,*maps,operation):
    arrays=(image,)+maps; shape=image.shape[:2]
    if any(a.shape[:2]!=shape for a in maps):raise ValueError('image and maps must share grid before transform')
    if operation=='identity':out=arrays
    elif operation=='hflip':out=tuple(np.flip(a,axis=1) for a in arrays)
    elif operation=='vflip':out=tuple(np.flip(a,axis=0) for a in arrays)
    elif operation.startswith('rot90:'):
        k=int(operation.split(':')[1])%4;out=tuple(np.rot90(a,k=k,axes=(0,1)) for a in arrays)
    else:raise ValueError(f'unsupported spatial op {operation}')
    return tuple(np.ascontiguousarray(a) for a in out)

class RandomDiscreteGeometry:
    def __init__(self,hflip=True,vflip=True,rot90=True):self.ops=['identity']+(['hflip'] if hflip else [])+(['vflip'] if vflip else [])+([f'rot90:{k}' for k in (1,2,3)] if rot90 else [])
    def sample_operation(self):return random.choice(self.ops)
    def __call__(self,image,*maps,operation=None):return apply_spatial_transform(image,*maps,operation=operation or self.sample_operation())
