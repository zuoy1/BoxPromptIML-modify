"""K-prompt SAM inference; logits become probabilities before spatial resizing."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import cv2
import numpy as np
import torch
from fecd.pseudo.box_ops import PROMPT_VARIANTS, prompt_boxes
from fecd.coordinates import resize_probability

THRESHOLDS=(0.4,0.5,0.6)
def sigmoid_logits(logits):
    x=np.asarray(logits,dtype=np.float32)
    return (1.0/(1.0+np.exp(-np.clip(x,-80,80)))).astype(np.float32)

def summarize_prompt_probabilities(prompt_probs):
    p=np.asarray(prompt_probs,dtype=np.float32)
    if p.ndim!=3 or not p.shape[0]: raise ValueError('prompt_probs must be [K,H,W]')
    if not np.isfinite(p).all() or p.min()<0 or p.max()>1: raise ValueError('probabilities must be finite in [0,1]')
    mu=p.mean(0,dtype=np.float32); var=((p-mu[None])**2).mean(0,dtype=np.float32)
    agreement=np.mean(np.stack([(p>=t).astype(np.float32) for t in THRESHOLDS]),axis=(0,1),dtype=np.float32)
    return mu,var,agreement

def resize_prompt_probabilities(prompt_probs,target_hw=(224,224)):
    """Resize each sigmoid probability map before aggregating on the student grid."""
    p=np.asarray(prompt_probs,dtype=np.float32)
    if p.ndim!=3 or not p.shape[0]: raise ValueError('prompt_probs must be [K,H,W]')
    return np.stack([resize_probability(x,target_hw) for x in p],axis=0)

class SamEnsembleGenerator:
    def __init__(self, checkpoint_path, device='cuda:0', model_type='vit_h', predictor=None):
        self.checkpoint_path=Path(checkpoint_path).resolve()
        self.device=torch.device(device)
        if predictor is None:
            if not self.checkpoint_path.is_file(): raise FileNotFoundError(self.checkpoint_path)
            from segment_anything import SamPredictor,sam_model_registry
            self.model=sam_model_registry[model_type](checkpoint=str(self.checkpoint_path)).to(self.device).eval()
            predictor=SamPredictor(self.model)
        else: self.model=getattr(predictor,'model',None)
        self.predictor=predictor
        self.checkpoint_sha256=self._sha256(self.checkpoint_path) if self.checkpoint_path.is_file() else 'injected-test-predictor'
    @staticmethod
    def _sha256(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        return h.hexdigest()
    @property
    def load_info(self): return {'checkpoint':str(self.checkpoint_path),'checkpoint_sha256':self.checkpoint_sha256,'device':str(self.device)}
    def predict(self,image_rgb,boxes,k=8,target_hw=(224,224)):
        image=np.asarray(image_rgb)
        if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8: raise ValueError('expected uint8 RGB [H,W,3]')
        h,w=image.shape[:2]; b=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
        if len(b)>3: b=b[:3]
        variants=prompt_boxes(b,(h,w),k)
        if len(b)==0:
            zeros=np.zeros((h,w),np.float32); stats=(zeros,zeros.copy(),zeros.copy())
            resized_probs=np.zeros((k,*target_hw),np.float32); resized_stats=summarize_prompt_probabilities(resized_probs)
            return {'prompt_probs':np.zeros((k,h,w),np.float32),'prompt_probs_224':resized_probs,
                    'native':dict(zip(('mu','var','agreement'),stats)),
                    'maps':dict(zip(('mu','var','agreement'),resized_stats)),'maps_native_aggregate_then_resize':dict(zip(('mu','var','agreement'),[resize_probability(x,target_hw) for x in stats])),
                    'sam_quality_mean':0.0,'sam_quality_std':0.0,'base_mu':resized_probs[0],
                    'empty_boxes':True,'num_boxes':0,'prompt_shapes':[]}
        self.predictor.set_image(image)
        all_prompt=[]; qualities=[]; shapes=[]
        # This outermost operation holds encoder features stable; every component is a separate SAM box decode.
        for boxes_k in variants:
            component=[]
            for box in boxes_k:
                with torch.inference_mode():
                    # SamPredictor.predict returns (masks, iou_predictions, low_res_masks).
                    logits, quality, low_res_logits=self.predictor.predict(point_coords=None,box=box,
                        multimask_output=False,return_logits=True)
                logits=np.asarray(logits)
                if logits.shape!=(1,h,w): raise RuntimeError(f'expected native SAM logits [1,{h},{w}], got {logits.shape}')
                prob=sigmoid_logits(logits[0]); component.append(prob)
                q=np.asarray(quality,dtype=np.float32).reshape(-1)
                if q.size: qualities.extend(q.tolist())
                shapes.append({'logits':list(logits.shape),'quality':list(np.asarray(quality).shape),'low_res_logits':list(np.asarray(low_res_logits).shape)})
            all_prompt.append(np.maximum.reduce(component))
        p=np.stack(all_prompt)
        mu,var,agreement=summarize_prompt_probabilities(p)
        p224=resize_prompt_probabilities(p,target_hw)
        mu224,var224,agreement224=summarize_prompt_probabilities(p224)
        maps_native={name:resize_probability(value,target_hw) for name,value in zip(('mu','var','agreement'),(mu,var,agreement))}
        maps=dict(zip(('mu','var','agreement'),(mu224,var224,agreement224)))
        q=np.asarray(qualities,dtype=np.float32)
        result={'prompt_probs':p,'prompt_probs_224':p224,'native':{'mu':mu,'var':var,'agreement':agreement},'maps':maps,'maps_native_aggregate_then_resize':maps_native,
                'sam_quality_mean':float(q.mean()) if q.size else 0.,'sam_quality_std':float(q.std()) if q.size else 0.,
                'empty_boxes':False,'num_boxes':len(b),'prompt_shapes':shapes}
        for n,x in result['native'].items():
            if not np.isfinite(x).all(): raise RuntimeError(f'non-finite {n}')
        if maps['mu'].min()<0 or maps['mu'].max()>1 or maps['var'].min()<-1e-7 or maps['var'].max()>0.250001 or maps['agreement'].min()<0 or maps['agreement'].max()>1: raise RuntimeError('SAM summary range validation failed')
        # The Base prompt is persisted only in K=8 caches for a full-split K=1/K=8 audit.
        result['base_mu']=p224[0]
        return result
