"""Stage 2C preflight: aggregation order, quantization, source metadata and cache protocol."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from fecd.pseudo.sam_ensemble import SamEnsembleGenerator
from fecd.pseudo.generate_cache import build_config,make_box_masks
from fecd.pseudo.cache_schema import config_hash,build_cache_record,save_cache_atomic,load_cache,make_sample_key
from fecd.tools.export_boxes import sha

ROOT=Path(__file__).resolve().parents[1]
IDS=['Tp_D_CND_M_N_ani00018_sec00096_00138','Tp_D_CND_M_N_art00077_art00076_10290','Tp_D_CNN_M_N_art00052_arc00030_11853','089','canong3_nikond70_sub_17','NC2016_5131']
def color(x):
 x=np.clip(np.asarray(x,np.float32)*255,0,255).astype(np.uint8)
 import cv2
 return Image.fromarray(cv2.cvtColor(cv2.applyColorMap(x,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB))
def boundary_mask(mu,radius=5):
 hard=(mu>=.5).astype(np.uint8)
 import cv2
 edge=cv2.morphologyEx(hard,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
 return cv2.dilate(edge.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))>0
def metrics(a,b,region):
 d=np.abs(a-b);x=d[region]
 return {'max_abs':float(d.max()),'mean_abs':float(d.mean()),'boundary_max_abs':float(x.max()) if x.size else 0.,'boundary_mean_abs':float(x.mean()) if x.size else 0.}
def run(boxes_path=ROOT/'assets/boxes_train.json',out=ROOT/'assets/stage2c_preflight_comparison',checkpoint=Path('/home/weights/sam_vit_h_4b8939.pth')):
 out=Path(out);out.mkdir(parents=True,exist_ok=True); box_rows=json.loads(Path(boxes_path).read_text())['records'];rows=[next(r for r in box_rows if r['sample_id']==sid) for sid in IDS]
 g=SamEnsembleGenerator(checkpoint,device='cuda:0'); all_reports=[]; quant=[]; hashes={}
 for k in (1,8):
  cfg=build_config(g,k);ch=config_hash(cfg);hashes[str(k)]=ch;cache_dir=out/'caches'/f'k{k}'/ch[:12];cache_dir.mkdir(parents=True,exist_ok=True);success=[]
  for r in rows:
   with Image.open(r['image_path']) as im: image=np.asarray(im.convert('RGB'))
   if sha(r['image_path'],'sha1')!=r['image_sha1']:raise RuntimeError('image SHA1 differs from frozen boxes')
   if r['sample_key']!=make_sample_key(r['dataset'],'train',r['sample_id'],r['image_sha256']):raise RuntimeError('sample key/source identity mismatch')
   result=g.predict(image,r['boxes_xyxy'],k,target_hw=(224,224));a=result['maps_native_aggregate_then_resize'];b=result['maps']
   if k==8:
    boundary=boundary_mask(b['mu'])
    entry={'dataset':r['dataset'],'sample_id':r['sample_id'],'sample_key':r['sample_key'],'boundary_pixels':int(boundary.sum()),'scheme_A':{},'scheme_B':{}}
    for name in ('mu','var','agreement'):
     entry['scheme_A'][name]=a[name];entry['scheme_B'][name]=b[name];entry.setdefault('difference',{})[name]=metrics(a[name],b[name],boundary)
    all_reports.append(entry)
    panel=Image.new('RGB',(672,672),'white');draw=ImageDraw.Draw(panel)
    titles=[('Scheme A mean',a['mu']),('Scheme B mean',b['mu']),('Mean |diff| x20',np.abs(a['mu']-b['mu'])*20),('Scheme A var',a['var']*4),('Scheme B var',b['var']*4),('Vote |diff| x20',np.abs(a['agreement']-b['agreement'])*20)]
    for j,(title,v) in enumerate(titles):
     tile=color(v).resize((224,224),Image.Resampling.NEAREST);x=j%3*224;y=j//3*224;panel.paste(tile,(x,y));draw.text((x+5,y+5),title,fill='white',stroke_width=2,stroke_fill='black')
    panel.save(out/f"{r['dataset']}_{r['sample_id']}_aggregation.png")
   rec=build_cache_record(maps=b,box_masks=make_box_masks(r['boxes_xyxy'],image.shape[:2]),quality_mean=result['sam_quality_mean'],quality_std=result['sam_quality_std'],image_hw=image.shape[:2],schema_config_hash=ch,image_sha1=r['image_sha1'],box_hash=r['box_hash'],sample_id=r['sample_id'],dataset=r['dataset'],split='train',sample_key=r['sample_key'],num_boxes=r['num_boxes'],k=k,empty_boxes=not bool(r['num_boxes']),base_mu=result['base_mu'] if k==8 else None)
   path=cache_dir/(str(rec['cache_key'])+'.npz');save_cache_atomic(path,rec,ch,replace_corrupt=True)
   expected={'dataset':r['dataset'],'split':'train','sample_id':r['sample_id'],'sample_key':r['sample_key'],'image_sha1':r['image_sha1'],'image_sha256':r['image_sha256'],'box_hash':r['box_hash'],'num_boxes':r['num_boxes'],'k':k,'empty_boxes':not bool(r['num_boxes']),'image_h':image.shape[0],'image_w':image.shape[1]}
   loaded=load_cache(path,ch,expected)
   if not np.array_equal(loaded['box_masks'],rec['box_masks']):raise RuntimeError('box masks changed during cache roundtrip')
   qentry={'dataset':r['dataset'],'sample_id':r['sample_id'],'k':k,'fields':{}}
   for name in ('mu','var','agreement'):
    f32=np.asarray(b[name],np.float32);f16=loaded[name].astype(np.float32);d=np.abs(f32-f16);thresholds={}
    for t in (.25,.5,.75):
     near=np.abs(f32-t)<=.002
     thresholds[str(t)]={'near_count':int(near.sum()),'changed_count':int(((f32>=t)!=(f16>=t)).sum()),'changed_near_count':int((((f32>=t)!=(f16>=t))&near).sum()),'fraction_changed':float(((f32>=t)!=(f16>=t)).mean())}
    qentry['fields'][name]={'float32_min':float(f32.min()),'float32_max':float(f32.max()),'disk_float16_min':float(f16.min()),'disk_float16_max':float(f16.max()),'mean_abs_error':float(d.mean()),'max_abs_error':float(d.max()),'thresholds':thresholds}
   quant.append(qentry);success.append({'identity':r['dataset']+'/train/'+r['sample_id'],'cache':str(path),'cache_key':str(rec['cache_key'])})
  index=[]
  for r in rows:
   c=next(x for x in success if x['identity']==r['dataset']+'/train/'+r['sample_id'])
   index.append({'image_path':r['image_path'],'cache_path':c['cache'],'dataset':r['dataset'],'split':'train','sample_id':r['sample_id'],'sample_key':r['sample_key'],'boxes_xyxy':r['boxes_xyxy'],'image_sha1':r['image_sha1'],'box_hash':r['box_hash']})
  (cache_dir/'dataset_index.json').write_text(json.dumps(index,indent=2)+'\n')
 from fecd.datasets.fecd_dataset import FECDDataset
 loaded_by_k={}
 for k in (1,8):
  ch=hashes[str(k)];cache_dir=out/'caches'/f'k{k}'/ch[:12];idx=json.loads((cache_dir/'dataset_index.json').read_text())
  ds=FECDDataset(idx,expected_config_hash=ch,augment=False,hard_single=(k==1));assert len(ds)==6
  loaded_by_k[k]=[]
  for i in range(len(ds)):
   sample=ds[i]
   if 'gt' in sample or 'mask_path' in sample:raise RuntimeError('GT field leaked to student Dataset')
   loaded_by_k[k].append(load_cache(idx[i]['cache_path'],ch))
 base_errors=[]
 for i,(one,eight) in enumerate(zip(loaded_by_k[1],loaded_by_k[8])):
  d=np.abs(one['mu'].astype(np.float32)-eight['base_mu'].astype(np.float32));base_errors.append({'dataset':str(one['dataset']),'sample_id':str(one['sample_id']),'max_abs_error':float(d.max()),'mean_abs_error':float(d.mean())})
  if d.max()>.001:raise RuntimeError('K1/base K8 prediction mismatch')
 comparison=[{k:v for k,v in e.items() if k not in ('scheme_A','scheme_B')} for e in all_reports]
 report={'schema_version':'fecd-sam-cache-v2','k1_k8_base_consistency':base_errors,'dataset_read_count_per_k':{'1':len(loaded_by_k[1]),'8':len(loaded_by_k[8])},'gt_leakage_check':'passed','selected_aggregation':'scheme_B: resize every prompt probability to 224x224, then calculate population mean/variance and threshold vote','reason':'Uncertainty fields must share the student grid; interpolation and variance/vote aggregation do not commute, so scheme B defines uncertainty at the grid actually consumed by the student.','config_hashes':hashes,'sample_count':len(rows),'comparison':comparison,'float16_quantization':quant,'cache_protocol_checks':{'source_identity_and_hashes_checked':True,'k_and_config_checked':True,'six_each_k_roundtrip':True,'nonempty_box_masks_checked':True}}
 (out/'preflight_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');return report
if __name__=='__main__':print(json.dumps(run(),indent=2,ensure_ascii=False))
