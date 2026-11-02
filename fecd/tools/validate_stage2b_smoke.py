"""Cross-check K=1/K=8 caches, student dataset isolation and visual diagnostics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from fecd.pseudo.sam_ensemble import SamEnsembleGenerator
from fecd.pseudo.cache_schema import load_cache
from fecd.pseudo.generate_cache import build_config,make_box_masks
from fecd.pseudo.cache_schema import config_hash
from fecd.coordinates import resize_probability
from fecd.datasets.fecd_dataset import FECDDataset
from fecd.tools.export_boxes import sha

def color_heat(a):
 x=np.clip(np.asarray(a,dtype=np.float32)*255,0,255).astype(np.uint8);return cv2.cvtColor(cv2.applyColorMap(x,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
def validate(boxes_json,root,checkpoint,output):
 rows=json.loads(Path(boxes_json).read_text())['records']; ids=['Tp_D_CND_M_N_ani00018_sec00096_00138','Tp_D_CND_M_N_art00077_art00076_10290','Tp_D_CNN_M_N_art00052_arc00030_11853','089','canong3_nikond70_sub_17','NC2016_5131']; rows=[next(r for r in rows if r['sample_id']==sid) for sid in ids]
 manifests={}
 datasets={}
 for k in (1,8):
  paths=list((Path(root)/f'k{k}').glob('*/manifest.json')); m=json.loads(paths[0].read_text());manifests[k]=m
  assert m['succeeded']==len(rows) and not m['failed']
 # Verify every identity cached separately, K is distinct in the cache key and all read back.
 records_by_k={}
 for k in (1,8):
  idx=json.loads((Path(manifests[k]['successes'][0]['cache']).parent/'dataset_index.json').read_text())
  ds=FECDDataset(idx,expected_config_hash=manifests[k]['config_hash'],augment=False,hard_single=(k==1));datasets[k]=ds
  assert len(ds)==len(rows)
  records_by_k[k]={}
  for i,r in enumerate(rows):
   sample=ds[i];assert sample['image'].shape==(3,224,224) and sample['mu'].shape==(1,224,224) and sample['box_masks'].shape==(3,224,224)
   assert sample['split']=='train' and 'gt' not in sample and 'mask' not in sample and 'gt_path' not in sample
   assert bool(np.isfinite(sample['image'].numpy()).all())
   cache=load_cache(idx[i]['cache_path'],manifests[k]['config_hash']);records_by_k[k][r['sample_key']]=(idx[i],cache)
   if k==1: assert np.array_equal(sample['target'].numpy()[0],(cache['mu']>=.5).astype(np.float32))
 for i in range(len(rows)):
  assert np.array_equal(datasets[1][i]['image'].numpy(),datasets[8][i]['image'].numpy())
 # Actual Base variant output from the K=8 run must equal K=1 teacher output (up to float16 cache rounding).
 gen=SamEnsembleGenerator(checkpoint,device='cuda:0');comparison=[]
 for r in rows:
  with Image.open(r['image_path']) as im:image=np.asarray(im.convert('RGB'))
  result=gen.predict(image,r['boxes_xyxy'],k=8)
  base224=resize_probability(result['prompt_probs'][0],(224,224))
  c1=records_by_k[1][r['sample_key']][1]['mu'].astype(np.float32)
  err=float(np.max(np.abs(base224-c1)));comparison.append({'sample_id':r['sample_id'],'base_vs_k1_max_abs_error':err})
  assert err<.001
  entry=records_by_k[8][r['sample_key']][0];c8=records_by_k[8][r['sample_key']][1]
  with Image.open(r['image_path']) as im: original=im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR)
  canvas=original.copy();draw=ImageDraw.Draw(canvas)
  for box in r['boxes_xyxy']:
   h,w=r['image_hw'];x1,y1,x2,y2=box;xy=(x1*448/w,y1*448/h,x2*448/w,y2*448/h);draw.rectangle(xy,outline=(255,40,20),width=3)
  panel=Image.new('RGB',(1344,896),'white')
  c1=records_by_k[1][r['sample_key']][1]
  tiles=[original,canvas]+[Image.fromarray(color_heat(a)).resize((448,448),Image.Resampling.NEAREST) for a in ((c1['mu']>=.5).astype(np.float32),c8['mu'],c8['var']*4,c8['agreement'])]
  labels=['Original','Coarse boxes','K=1 hard mask','K=8 mean','K=8 variance x4','K=8 threshold vote']
  for j,(tile,label) in enumerate(zip(tiles,labels)):
   x=(j%3)*448;y=(j//3)*448;panel.paste(tile,(x,y));ImageDraw.Draw(panel).text((x+8,y+8),label,fill=(255,255,255),stroke_width=2,stroke_fill=(0,0,0))
  panel.save(Path(output)/(r['dataset']+'_'+r['sample_id'].replace('/','_')+'.jpg'),quality=92)
 report={'samples':len(rows),'k1_k8_base_comparison':comparison,'max_base_abs_error':max(x['base_vs_k1_max_abs_error'] for x in comparison),'student_samples':len(rows),'gt_leakage_keys_absent':True,'visualizations':[str(x) for x in Path(output).glob('*.jpg')]}
 (Path(output)/'validation.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--boxes',type=Path,default=Path(__file__).resolve().parents[1]/'assets/boxes_train.json');p.add_argument('--cache-root',type=Path,default=Path(__file__).resolve().parents[1]/'assets/stage2b_smoke/cache');p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'assets/stage2b_smoke/visualizations');p.add_argument('--checkpoint',type=Path,default=Path('/home/weights/sam_vit_h_4b8939.pth'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);print(json.dumps(validate(a.boxes,a.cache_root,a.checkpoint,a.output),indent=2))
if __name__=='__main__':main()
