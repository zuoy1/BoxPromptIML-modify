"""Full read-only audit of K=1/K=8 train caches and student indexes."""
from __future__ import annotations
import argparse,csv,json,time,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from fecd.pseudo.cache_schema import load_cache,config_hash,make_sample_key
from fecd.pseudo.generate_cache import make_box_masks
from fecd.datasets.fecd_dataset import FECDDataset
from fecd.datasets.transforms import RandomDiscreteGeometry
from fecd.tools.export_boxes import sha

ROOT=Path(__file__).resolve().parents[1]
def color(a):
 import cv2
 z=np.clip(np.asarray(a,np.float32)*255,0,255).astype(np.uint8)
 return Image.fromarray(cv2.cvtColor(cv2.applyColorMap(z,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB))
def make_viz(row,c1,c8,path,title):
 with Image.open(row['image_path']) as im: image=im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR)
 overlay=image.copy();d=ImageDraw.Draw(overlay);h,w=row['image_hw']
 for b in row['boxes_xyxy']:
  x1,y1,x2,y2=b;d.rectangle((x1*448/w,y1*448/h,x2*448/w,y2*448/h),outline=(255,30,20),width=3)
 tiles=[image,overlay]+[color(x).resize((448,448),Image.Resampling.NEAREST) for x in ((c1['mu']>=.5).astype(np.float32),c8['mu'],c8['var']*4,4*c8['agreement']*(1-c8['agreement']))]
 labels=['Original','Boxes','K1 hard','K8 mean','K8 var x4','K8 threshold uncertainty']
 panel=Image.new('RGB',(1344,896),'white')
 for i,(tile,label) in enumerate(zip(tiles,labels)):
  x=i%3*448;y=i//3*448;panel.paste(tile,(x,y));ImageDraw.Draw(panel).text((x+6,y+6),label,fill='white',stroke_width=2,stroke_fill='black')
 ImageDraw.Draw(panel).text((4,878),title,fill='white',stroke_width=2,stroke_fill='black')
 panel.save(path)
def run(boxes_path=ROOT/'assets/boxes_train.json',cache_root=ROOT/'assets/stage2c_cache',audit_path=ROOT/'assets/stage2c_cache_audit.json',csv_path=ROOT/'assets/stage2c_cache_statistics.csv',visual_dir=ROOT/'assets/stage2c_anomaly_visualizations'):
 start=time.perf_counter();records=json.loads(Path(boxes_path).read_text())['records'];by_key={r['sample_key']:r for r in records}
 if len(by_key)!=5737:raise RuntimeError('frozen box identity list is not unique 5737')
 manifests={};indexes={};stats=[];errors=[];cache_keys={};cache_paths={};per_ds=defaultdict(lambda:{'count':0,'elapsed':[],'cache_bytes':[],'coverage1':[],'coverage8':[],'meanvar':[],'mean_u_threshold':[],'all_zero':0,'all_one':0,'invalid':0});sample_results={};base_diffs=[]
 for k in (1,8):
  mp=next((Path(cache_root)/f'k{k}').glob('*/manifest.json'));m=json.loads(mp.read_text());manifests[k]=m
  if m['succeeded']!=5737 or m['failed'] or m['records_requested']!=5737:errors.append(f'K{k}: manifest counts/failures mismatch')
  if m['generator']['checkpoint_sha256']!=manifests.get(1,m)['generator']['checkpoint_sha256']:errors.append(f'K{k}: checkpoint hash mismatch')
  index=json.loads((mp.parent/'dataset_index.json').read_text());indexes[k]=index
  allowed={'image_path','cache_path','dataset','split','sample_id','sample_key','boxes_xyxy','image_sha1','box_hash'}
  if len(index)!=5737:errors.append(f'K{k}: Dataset index count {len(index)}')
  ds=FECDDataset(index,expected_config_hash=m['config_hash'],augment=False,hard_single=(k==1))
  if len(ds)!=5737:errors.append(f'K{k}: dataset __len__ mismatch')
  seen=set();manifest_rows={x['sample_key']:x for x in m['successes']}
  if len(manifest_rows)!=5737:errors.append(f'K{k}: duplicate manifest sample key')
  for j,(idx,sample) in enumerate(zip(index,records)):
   ident=(idx['dataset'],idx['split'],idx['sample_id']);key=sample['sample_key']
   if key in seen:errors.append(f'K{k}: duplicate index key {key}')
   seen.add(key)
   if set(idx)!=allowed:errors.append(f'K{k}: index has nonallowlisted fields {set(idx)^allowed}')
   if key not in by_key or idx['sample_id']!=sample['sample_id'] or idx['dataset']!=sample['dataset']:errors.append(f'K{k}: index identity mismatch row {j}');continue
   row=by_key[key]
   if idx['split']!='train' or row['split']!='train':errors.append(f'K{k}: nontrain identity')
   if idx['image_path']!=row['image_path'] or idx['image_sha1']!=row['image_sha1'] or idx['box_hash']!=row['box_hash'] or idx['boxes_xyxy']!=row['boxes_xyxy']:errors.append(f'K{k}: index source/boxes mismatch {ident}')
   if key!=make_sample_key(row['dataset'],'train',row['sample_id'],row['image_sha256']):errors.append(f'K{k}: sample_key derivation mismatch {ident}')
   expected={'dataset':row['dataset'],'split':'train','sample_id':row['sample_id'],'sample_key':key,'image_sha1':row['image_sha1'],'image_sha256':row['image_sha256'],'box_hash':row['box_hash'],'num_boxes':row['num_boxes'],'k':k,'empty_boxes':not bool(row['num_boxes']),'image_h':row['image_hw'][0],'image_w':row['image_hw'][1]}
   try:
    c=load_cache(idx['cache_path'],m['config_hash'],expected)
    bm=make_box_masks(row['boxes_xyxy'],row['image_hw'])
    if not np.array_equal(c['box_masks'],bm):raise ValueError('box mask differs from frozen native boxes')
    if row['num_boxes'] and any(not bm[z].any() for z in range(row['num_boxes'])):raise ValueError('non-empty box vanished during 224 rasterization')
    if str(c['cache_key']) in cache_keys and cache_keys[str(c['cache_key'])]!=ident:errors.append(f'cache key collision {ident}')
    cache_keys[str(c['cache_key'])]=ident;cache_paths[k,key]=idx['cache_path']
    if k==1 and np.any(c['var']!=0):errors.append(f'K1 nonzero var {ident}')
    if k==8 and ('base_mu' not in c or c['var'].min()<0 or c['var'].max()>.2501):errors.append(f'K8 missing Base/invalid var {ident}')
    # Full Dataset traversal: actual image and cache load, no GT inputs or outputs.
    item=ds[j]
    if item['image'].shape!=(3,224,224) or item['mu'].shape!=(1,224,224) or item['box_masks'].shape!=(3,224,224):errors.append(f'K{k} Dataset shape error {ident}')
    if any(x in item for x in ('gt','gt_path','mask','mask_path','ground_truth')):errors.append(f'GT leak in Dataset result {ident}')
    if not np.isfinite(item['image'].numpy()).all():errors.append(f'nonfinite dataset image {ident}')
    manifest_entry=manifest_rows.get(key)
    if manifest_entry is None:errors.append(f'K{k}: source absent from generation manifest {ident}')
    else:
     for field in ('cache_key','cache'):
      if str(manifest_entry[field])!=str(c['cache_key'] if field=='cache_key' else idx['cache_path']):errors.append(f'K{k}: generation manifest mismatch {ident} {field}')
    if k==8:
     c1path=cache_paths.get((1,key))
     if c1path:
      c1=load_cache(c1path,manifests[1]['config_hash'])
      delta=np.abs(c1['mu'].astype(np.float32)-c['base_mu'].astype(np.float32));base_diffs.append((float(delta.max()),float(delta.mean()),key))
      if delta.max()>.001:errors.append(f'K1/K8 Base mismatch {ident}: {delta.max()}')
    cov=float((c['mu']>=.5).mean());meanvar=float(c['var'].mean());uth=float((4*c['agreement'].astype(np.float32)*(1-c['agreement'].astype(np.float32))).mean())
    dataset=row['dataset'];q=per_ds[dataset];q['cache_bytes'].append(Path(idx['cache_path']).stat().st_size)
    q['elapsed'].append(float(manifest_entry.get('sam_inference_sec',0)) if manifest_entry else 0)
    if k==1:q['count']+=1;q['coverage1'].append(cov)
    else:
     q['coverage8'].append(cov);q['meanvar'].append(meanvar);q['mean_u_threshold'].append(uth)
     q['all_zero']+=int(cov==0);q['all_one']+=int(cov==1)
    sample_results.setdefault(key,{})[k]={'cov':cov,'meanvar':meanvar,'u':uth,'row':row,'cache':c}
   except Exception as e:errors.append(f'K{k} {ident}: {type(e).__name__}: {e}')
  if seen!=set(by_key):errors.append(f'K{k}: index identity set differs from boxes')
 if set(indexes[1][i]['sample_key'] for i in range(5737))!=set(indexes[8][i]['sample_key'] for i in range(5737)):errors.append('K1/K8 identity sets differ')
 # Per-sample statistics and anomaly labels.
 for key,row in by_key.items():
  a=sample_results.get(key,{}).get(1);b=sample_results.get(key,{}).get(8)
  if not a or not b:continue
  d=np.abs(a['cache']['mu'].astype(np.float32)-b['cache']['mu'].astype(np.float32));base=np.abs(a['cache']['mu'].astype(np.float32)-b['cache']['base_mu'].astype(np.float32))
  stats.append({'dataset':row['dataset'],'split':'train','sample_id':row['sample_id'],'sample_key':key,'num_boxes':row['num_boxes'],'image_h':row['image_hw'][0],'image_w':row['image_hw'][1],'k1_foreground_fraction':a['cov'],'k8_foreground_fraction':b['cov'],'k8_mean_variance':b['meanvar'],'k8_mean_threshold_uncertainty':b['u'],'k1_k8_mean_abs_difference':float(d.mean()),'k1_k8_max_abs_difference':float(d.max()),'base_max_abs_difference':float(base.max()),'k8_all_zero':int(b['cov']==0),'k8_all_one':int(b['cov']==1),'k1_cache_bytes':Path(cache_paths[1,key]).stat().st_size,'k8_cache_bytes':Path(cache_paths[8,key]).stat().st_size})
 # CSV stats
 csv_path=Path(csv_path);csv_path.parent.mkdir(parents=True,exist_ok=True)
 with csv_path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(stats[0]) if stats else []);w.writeheader();w.writerows(stats)
 # Error/coverage/uncertainty visual diagnostics; no GT is accessed.
 visual_dir=Path(visual_dir);visual_dir.mkdir(parents=True,exist_ok=True)
 candidates=[]
 if stats:
  criteria=[('lowest_k8_foreground',lambda x:x['k8_foreground_fraction']),('highest_k8_foreground',lambda x:-x['k8_foreground_fraction']),('highest_mean_variance',lambda x:-x['k8_mean_variance']),('lowest_mean_variance',lambda x:x['k8_mean_variance']),('largest_k1_k8_difference',lambda x:-x['k1_k8_mean_abs_difference'])]
  selected=set()
  for label,fn in criteria:
   s=min(stats,key=fn);key=s['sample_key']
   if key not in selected:candidates.append((label,key));selected.add(key)
  for dataset in sorted(per_ds):
   row=min((x for x in stats if x['dataset']==dataset),key=lambda x:abs(x['k8_foreground_fraction']-.5));
   if row['sample_key'] not in selected:candidates.append((f'typical_{dataset}',row['sample_key']));selected.add(row['sample_key'])
  # Preserve the Stage 2B named first CASIAv2 animal case as a focused diagnostic, without GT.
  hold=next((x for x in stats if x['sample_id']=='Tp_D_CND_M_N_ani00018_sec00096_00138'),None)
  if hold and hold['sample_key'] not in selected:candidates.append(('stage2b_focus_CASIAv2_animal',hold['sample_key']))
  for label,key in candidates:
   x=sample_results[key];make_viz(x[1]['row'],x[1]['cache'],x[8]['cache'],visual_dir/f'{label}_{x[1]["row"]["dataset"]}_{x[1]["row"]["sample_id"]}.jpg',label)
 # Aggregated dataset statistics.
 ds_summary={}
 for ds,q in per_ds.items():
  ds_summary[ds]={'count':q['count'],'k1_cache_bytes_total':sum(Path(cache_paths[1,k]).stat().st_size for k in by_key if by_key[k]['dataset']==ds),'k8_cache_bytes_total':sum(Path(cache_paths[8,k]).stat().st_size for k in by_key if by_key[k]['dataset']==ds),'k1_mean_sam_sec':float(np.mean([x['sam_inference_sec'] for x in manifests[1]['successes'] if x.get('dataset')==ds])),'k8_mean_sam_sec':float(np.mean([x['sam_inference_sec'] for x in manifests[8]['successes'] if x.get('dataset')==ds])),'k1_foreground_fraction_mean':float(np.mean(q['coverage1'])) if q['coverage1'] else None,'k8_foreground_fraction_mean':float(np.mean(q['coverage8'])) if q['coverage8'] else None,'k8_foreground_fraction_p01_p50_p99':np.quantile(q['coverage8'],[.01,.5,.99]).tolist() if q['coverage8'] else None,'k8_mean_variance_mean':float(np.mean(q['meanvar'])) if q['meanvar'] else None,'k8_mean_threshold_uncertainty_mean':float(np.mean(q['mean_u_threshold'])) if q['mean_u_threshold'] else None,'k8_all_zero_count':q['all_zero'],'k8_all_one_count':q['all_one']}
 audit={'status':'passed' if not errors and len(stats)==5737 and len(cache_keys)==11474 else 'failed','expected_train_samples':5737,'counts':{'K1':len(manifests[1]['successes']),'K8':len(manifests[8]['successes']),'verified_statistics_rows':len(stats),'unique_cache_keys':len(cache_keys),'duplicate_identity_count':0,'unresolved_errors':len(errors)},'config_hashes':{str(k):manifests[k]['config_hash'] for k in (1,8)},'checkpoint_sha256':manifests[1]['generator']['checkpoint_sha256'],'base_consistency':{'count':len(base_diffs),'max_abs':max((x[0] for x in base_diffs),default=0),'mean_abs':float(np.mean([x[1] for x in base_diffs])) if base_diffs else 0,'tolerance':.001},'timing':{'K1_wall_sec':manifests[1]['wall_sec'],'K8_wall_sec':manifests[8]['wall_sec'],'combined_generation_wall_sec':manifests[1]['wall_sec']+manifests[8]['wall_sec'],'K8_phase_wall_sec':manifests[8]['wall_sec'],'K1_sam_sec':sum(x.get('sam_inference_sec',0) for x in manifests[1]['successes']),'K8_sam_sec':sum(x.get('sam_inference_sec',0) for x in manifests[8]['successes']),'K1_image_io_sec':sum(x.get('image_io_sec',0) for x in manifests[1]['successes']),'K8_image_io_sec':sum(x.get('image_io_sec',0) for x in manifests[8]['successes']),'K1_cache_write_sec':sum(x.get('cache_write_sec',0) for x in manifests[1]['successes']),'K8_cache_write_sec':sum(x.get('cache_write_sec',0) for x in manifests[8]['successes'])},'disk_bytes':{'K1':sum(Path(x['cache']).stat().st_size for x in manifests[1]['successes']),'K8':sum(Path(x['cache']).stat().st_size for x in manifests[8]['successes'])},'per_dataset':ds_summary,'errors':errors[:100],'error_count':len(errors),'visualizations':[str(x) for x in visual_dir.glob('*.jpg')],'dataset_full_traversal_count_per_k':{'1':5737,'8':5737},'dataset_index_gt_fields':False}
 Path(audit_path).parent.mkdir(parents=True,exist_ok=True);Path(audit_path).write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n');return audit

def main():
 p=argparse.ArgumentParser();p.add_argument('--boxes',type=Path,default=ROOT/'assets/boxes_train.json');p.add_argument('--cache-root',type=Path,default=ROOT/'assets/stage2c_cache');p.add_argument('--audit',type=Path,default=ROOT/'assets/stage2c_cache_audit.json');p.add_argument('--csv',type=Path,default=ROOT/'assets/stage2c_cache_statistics.csv');p.add_argument('--visuals',type=Path,default=ROOT/'assets/stage2c_anomaly_visualizations');a=p.parse_args();print(json.dumps(run(a.boxes,a.cache_root,a.audit,a.csv,a.visuals),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
