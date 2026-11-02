"""Deterministic, train-manifest-only coarse box export from prepared native GT masks."""
from __future__ import annotations
import argparse,csv,hashlib,json,random
from pathlib import Path
import cv2,numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SPLIT_ROOT=Path('/home/datasets/processed/boxprompt_paper_scale_20261016')
MAP=json.loads((ROOT/'configs/stage2a_path_mapping.json').read_text())['prefix_mappings'][0]
RATIOS=[.01,.02,.03,.04,.05,.06,.07,.08,.09,.10]
WEIGHTS=[.05,.05,.07,.08,.40,.10,.08,.07,.05,.05]
def resolve_declared_path(value):
    old,new=MAP['from'],Path(MAP['to'])
    if value==old:return new
    if value.startswith(old+'/'):return new/value[len(old)+1:]
    p=Path(value)
    if p.is_absolute():return p
    raise ValueError(f'not an explicitly mapped manifest path: {value}')
def sha(path,algo='sha256'):
    h=hashlib.new(algo)
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def sample_key(ds,split,sid,image_sha):return hashlib.sha256('\0'.join((ds,split,sid,image_sha)).encode()).hexdigest()
def mask_to_boxes(mask,seed,max_boxes=3):
    h,w=mask.shape[:2]; _,binary=cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rng=random.Random(int(seed)); items=[]
    # Mirrors original function: min_area is not applied; random ratio distribution, integer box expansion and clipping are retained.
    for contour in contours:
        area=float(cv2.contourArea(contour)); x,y,bw,bh=cv2.boundingRect(contour); cx,cy=x+bw//2,y+bh//2
        ratio=rng.choices(RATIOS,weights=WEIGHTS,k=1)[0]
        nw,nh=int(bw*(1+ratio)),int(bh*(1+ratio))
        x1=max(cx-nw//2,0); y1=max(cy-nh//2,0); x2=min(cx+nw//2,w); y2=min(cy+nh//2,h)
        # Keep the original half-open XYXY edge convention as implemented by train.py.
        if x2>x1 and y2>y1:items.append((area,[int(x1),int(y1),int(x2),int(y2)]))
    items.sort(key=lambda t:t[0],reverse=True)
    return [box for _,box in items[:max_boxes]]
def export_boxes(manifest_root=SPLIT_ROOT,output=ROOT/'assets/boxes_train.json',seed=20261023):
    root=Path(manifest_root); rows=list(csv.DictReader((root/'train_manifest.csv').open(newline='',encoding='utf-8')))
    records=json.loads((root/'manifest.json').read_text()); idx={(r['dataset'],r['id']):r for r in records}
    if len(idx)!=len(records):raise RuntimeError('duplicate identities in root manifest')
    output=Path(output); out=[]
    for row in rows:
        key=(row['dataset'],row['id']); rec=idx.get(key)
        if rec is None or row['split']!='train' or rec.get('split')!='train':raise RuntimeError(f'not a locked train identity: {key}')
        image=resolve_declared_path(rec['image']); maskpath=resolve_declared_path(rec['prepared_mask'])
        if sha(image)!=rec['image_sha256']:raise RuntimeError(f'image SHA256 mismatch {key}')
        with Image.open(image) as im: rgb=np.asarray(im.convert('RGB'))
        with Image.open(maskpath) as im: mask=np.asarray(im.convert('L'))
        if rgb is None or mask is None:raise RuntimeError(f'unreadable source {key}')
        if rgb.shape[:2]!=mask.shape[:2] or rgb.shape[1]!=rec['width'] or rgb.shape[0]!=rec['height']:raise RuntimeError(f'dimension mismatch {key}')
        # Per-identity seed prevents row ordering or partial restart from affecting boxes.
        ident=sample_key(row['dataset'],'train',row['id'],rec['image_sha256'])
        identity_seed=int.from_bytes(hashlib.sha256(f'{seed}:{ident}'.encode()).digest()[:8],'big')
        boxes=mask_to_boxes(mask,identity_seed,3)
        box_payload={'image_hw':[rgb.shape[0],rgb.shape[1]],'boxes_xyxy':boxes}
        box_hash=hashlib.sha256(json.dumps(box_payload,separators=(',',':')).encode()).hexdigest()
        out.append({'dataset':row['dataset'],'split':'train','sample_id':row['id'],'sample_key':ident,
                    'image_path':str(image),'image_sha256':rec['image_sha256'],'image_sha1':sha(image,'sha1'),
                    'image_hw':[rgb.shape[0],rgb.shape[1]],'group':rec.get('group',''),
                    'boxes_xyxy':boxes,'box_hash':box_hash,'num_boxes':len(boxes)})
    payload={'schema_version':'fecd-boxes-v1','seed':int(seed),'coordinate_convention':'native XYXY half-open pixel edges',
      'max_boxes':3,'source_manifest':str(root/'train_manifest.csv'),'source_gt':'prepared_mask resolved through stage2a explicit path mapping',
      'algorithm':'cv2 threshold >127; RETR_EXTERNAL; contour area descending; max 3; min_area intentionally not applied (matches train.py behavior); original weighted ratio distribution retained with identity-derived seed',
      'note':'Native GT boxes may differ from original online boxes generated after stochastic image augmentation. B1/B5 must share this export.', 'records':out}
    output.parent.mkdir(parents=True,exist_ok=True); tmp=output.with_name(output.name+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n'); tmp.replace(output)
    return payload

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest-root',type=Path,default=SPLIT_ROOT);p.add_argument('--output',type=Path,default=ROOT/'assets/boxes_train.json');p.add_argument('--seed',type=int,default=20261023);a=p.parse_args()
 d=export_boxes(a.manifest_root,a.output,a.seed); print(json.dumps({'output':str(a.output),'records':len(d['records']),'empty_boxes':sum(not x['num_boxes'] for x in d['records']),'sha256':sha(a.output)},indent=2))
if __name__=='__main__':main()
