"""Procedural unmanipulated-texture stress checks; not real-image validation."""
import json
from pathlib import Path

import cv2
import numpy as np

from fecd.forensic_reference.patches import extract_patches
from fecd.forensic_reference.matching import match_references
from fecd.forensic_reference.statistics import compare_regions
from fecd.forensic_reference.summary import distribution


def main():
    rng=np.random.default_rng(20261029)
    yy,xx=np.mgrid[:256,:384]
    noise=rng.normal(0,.01,(256,384))
    fields={
        'constant':np.full((256,384),.5),
        'stationary_noise':.5+noise,
        'continuous_varying_texture_same_noise':.5+noise+.09*np.sin(
            2*np.pi*(xx/24+xx*xx/(384*48))) * np.cos(2*np.pi*yy/27),
        'continuous_edges_same_noise':.5+noise+.08*np.tanh(4*np.sin(xx/8+yy/13)),
    }
    records=[]
    zero=np.zeros((224,224),np.float32)
    for name,field in fields.items():
        rgb=np.repeat(np.rint(np.clip(field,0,1)*255).astype(np.uint8)[...,None],3,axis=2)
        for size in (64,32):
            patches=extract_patches(rgb,zero,zero,zero,zero,patch_size=size)['patches']
            anchors=[patches[i] for i in np.linspace(0,len(patches)-1,min(16,len(patches)),dtype=int)]
            comparisons=[]
            for patch in anchors:
                match=match_references(patch,patches)
                stat=compare_regions(patch,match['references'])
                comparisons.append({'patch_id':patch['id'],'matching':{k:v for k,v in match.items()
                    if k!='references'},'statistics':stat})
            records.append({'case':name,'patch_size':size,'anchors':len(anchors),
                'valid':sum(r['statistics']['status']=='applicable' for r in comparisons),
                'distance':distribution([r['statistics']['standardized_quantile_distance']
                    for r in comparisons]),'comparisons':comparisons})
    out=Path('fecd/assets/stage3rb0/procedural_controls.json')
    with out.open('x') as stream:
        json.dump({'seed':20261029,'interpretation':'procedural content-only stress; no photographs or GT',
                   'imaging_noise_sigma':.01,'records':records},stream,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k!='comparisons'} for r in records],indent=2))


if __name__=='__main__':
    main()
