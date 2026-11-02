"""Deterministic native-resolution Box Prompt perturbations (XYXY half-open)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PromptVariant:
    name: str
    dx: float
    dy: float
    scale: float

PROMPT_VARIANTS = (
    PromptVariant('Base', 0.0, 0.0, 1.0),
    PromptVariant('Left', -0.05, 0.0, 1.0),
    PromptVariant('Right', 0.05, 0.0, 1.0),
    PromptVariant('Up', 0.0, -0.05, 1.0),
    PromptVariant('Down', 0.0, 0.05, 1.0),
    PromptVariant('Small', 0.0, 0.0, 0.90),
    PromptVariant('Large', 0.0, 0.0, 1.10),
    PromptVariant('Mixed', 0.035, -0.035, 1.05),
)

def perturb_box_xyxy(box, image_hw: tuple[int,int], variant: PromptVariant) -> np.ndarray:
    """Offset box center by fractions of original size, scale around center, clip valid."""
    h,w=map(int,image_hw); b=np.asarray(box,dtype=np.float64)
    if h<=0 or w<=0 or b.shape!=(4,) or not np.isfinite(b).all(): raise ValueError('invalid dimensions/box')
    x1,y1,x2,y2=b
    if not (0<=x1<x2<=w and 0<=y1<y2<=h): raise ValueError(f'box must be half-open XYXY inside {(h,w)}: {b}')
    bw,bh=x2-x1,y2-y1; cx=(x1+x2)/2+variant.dx*bw; cy=(y1+y2)/2+variant.dy*bh
    nw=max(1.0,bw*variant.scale); nh=max(1.0,bh*variant.scale)
    ax1=max(0.0,cx-nw/2); ay1=max(0.0,cy-nh/2); ax2=min(float(w),cx+nw/2); ay2=min(float(h),cy+nh/2)
    # Clipping near edges can collapse a narrow box; shift the interval inward while preserving >=1px extent.
    if ax2<=ax1: ax1=min(max(cx-0.5,0),w-1); ax2=ax1+1
    if ay2<=ay1: ay1=min(max(cy-0.5,0),h-1); ay2=ay1+1
    ax1=np.clip(ax1,0,w-1); ay1=np.clip(ay1,0,h-1); ax2=np.clip(ax2,ax1+1,w); ay2=np.clip(ay2,ay1+1,h)
    return np.asarray([ax1,ay1,ax2,ay2],dtype=np.float32)

def prompt_boxes(boxes, image_hw, k=8):
    if not 1<=k<=len(PROMPT_VARIANTS): raise ValueError('K must be in [1,8]')
    b=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    return [[perturb_box_xyxy(box,image_hw,v) for box in b] for v in PROMPT_VARIANTS[:k]]
