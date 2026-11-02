"""Generate a selected subset or full train cache; default is no samples until --limit/--ids."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from fecd.pseudo.box_ops import PROMPT_VARIANTS
from fecd.pseudo.sam_ensemble import SamEnsembleGenerator
from fecd.pseudo.cache_schema import SCHEMA_VERSION,config_hash,build_cache_record,save_cache_atomic,load_cache,make_sample_key
from fecd.coordinates import rasterize_box_xyxy
from fecd.tools.export_boxes import sha

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=Path('/home/weights/sam_vit_h_4b8939.pth')
def make_box_masks(boxes,hw):
 out=np.zeros((3,224,224),np.uint8)
 for i,b in enumerate(boxes[:3]):
  h,w=hw
  scaled=[b[0]*224/w,b[1]*224/h,b[2]*224/w,b[3]*224/h]
  out[i]=rasterize_box_xyxy(scaled,(224,224))
 return out
def build_config(generator,k):
 return {'schema_version':SCHEMA_VERSION,'sam_type':'vit_h','sam_checkpoint_sha256':generator.checkpoint_sha256,
 'k':k,'prompt_variants':[v.__dict__ for v in PROMPT_VARIANTS[:k]],'max_component_boxes':3,'multimask_output':False,'return_logits':True,
 'logits_to_probability':'sigmoid at native resolution before resize','aggregation_grid':'resize each prompt probability to 224x224, then mean/variance/threshold vote','thresholds':[.4,.5,.6], 'resize_to':[224,224],'resize_interpolation':'bilinear align_corners=False (cv2 INTER_LINEAR)',
 'box_coordinates':'native XYXY half-open','box_fusion':'pixelwise maximum probability','box_export_version':'fecd-boxes-v1'}
def generate(boxes_json,output_root,k,limit=None,ids=None,device='cuda:0',checkpoint=CHECKPOINT,rebuild_corrupt=True):
 payload=json.loads(Path(boxes_json).read_text()); records=payload['records']
 wanted=set(ids or [])
 if wanted:records=[r for r in records if r['sample_id'] in wanted or r['sample_key'] in wanted]
 if limit is not None:records=records[:limit]
 generator=SamEnsembleGenerator(checkpoint,device=device); cfg=build_config(generator,k); ch=config_hash(cfg)
 outroot=Path(output_root)/f'k{k}'/ch[:12]; outroot.mkdir(parents=True,exist_ok=True)
 (outroot/'config.json').write_text(json.dumps({**cfg,'config_hash':ch},indent=2)+'\n')
 success=[];failed=[];log=[]
 for i,r in enumerate(records,1):
  started=time.perf_counter(); image_path=Path(r['image_path']); ident='/'.join((r['dataset'],r['split'],r['sample_id']))
  try:
   expected_box_hash=hashlib.sha256(json.dumps({'image_hw':r['image_hw'],'boxes_xyxy':r['boxes_xyxy']},separators=(',',':')).encode()).hexdigest()
   if expected_box_hash!=r['box_hash']:raise RuntimeError('frozen box_hash does not match native box coordinates')
   if sha(image_path,'sha1')!=r['image_sha1']:raise RuntimeError('image SHA1 changed since box export')
   expected={'dataset':r['dataset'],'split':'train','sample_id':r['sample_id'],'image_sha1':r['image_sha1'],'box_hash':r['box_hash'],'sample_key':r['sample_key'],'image_sha256':r['image_sha256'],'num_boxes':r['num_boxes'],'k':k,'empty_boxes':not bool(r['num_boxes']),'image_h':r['image_hw'][0],'image_w':r['image_hw'][1]}
   if r['sample_key']!=make_sample_key(r['dataset'],'train',r['sample_id'],r['image_sha256']):raise RuntimeError('sample_key does not match dataset/split/id/image SHA256')
   expected_masks=make_box_masks(r['boxes_xyxy'],r['image_hw'])
   if r['num_boxes'] and any(not expected_masks[j].any() for j in range(r['num_boxes'])):raise RuntimeError('non-empty native box vanished on 224 grid')
   path_candidate=None
   # Config-bound cache key can be computed without loading the teacher; verified cache enables safe resume.
   from fecd.pseudo.cache_schema import make_cache_key
   cache_key=make_cache_key(r['dataset'],'train',r['sample_id'],r['image_sha1'],r['box_hash'],ch)
   path_candidate=outroot/(cache_key+'.npz')
   if path_candidate.exists():
    try:
     cached=load_cache(path_candidate,ch,expected)
     if not np.array_equal(cached['box_masks'],expected_masks):raise ValueError('box_masks do not match frozen box export')
     success.append({'identity':ident,'dataset':r['dataset'],'sample_id':r['sample_id'],'sample_key':r['sample_key'],'cache':str(path_candidate),'cache_key':cache_key,'written':False,'bytes':path_candidate.stat().st_size,'num_boxes':r['num_boxes'],'empty_boxes':not bool(r['num_boxes']),'image_io_sec':0.0,'sam_inference_sec':0.0,'cache_write_sec':0.0,'elapsed_sec':time.perf_counter()-started,'resumed_without_inference':True})
     log.append({'ok':True,**success[-1]});print(json.dumps(log[-1]),flush=True);(outroot/'generation.jsonl').open('a').write(json.dumps(log[-1])+'\n');continue
    except Exception:
     # Validation failure is recorded and regenerated under the per-file lock below.
     pass
   io_started=time.perf_counter()
   with Image.open(image_path) as im: rgb=np.asarray(im.convert('RGB'))
   if list(rgb.shape[:2])!=r['image_hw']:raise RuntimeError('native dimensions differ from box export')
   image_io_sec=time.perf_counter()-io_started
   sam_started=time.perf_counter()
   result=generator.predict(rgb,r['boxes_xyxy'],k=k,target_hw=(224,224))
   sam_inference_sec=time.perf_counter()-sam_started
   rec=build_cache_record(maps=result['maps'],box_masks=expected_masks,quality_mean=result['sam_quality_mean'],quality_std=result['sam_quality_std'],image_hw=rgb.shape[:2],schema_config_hash=ch,image_sha1=r['image_sha1'],box_hash=r['box_hash'],sample_id=r['sample_id'],dataset=r['dataset'],split=r['split'],sample_key=r['sample_key'],num_boxes=r['num_boxes'],k=k,empty_boxes=result['empty_boxes'],base_mu=result['base_mu'] if k==8 else None)
   path=outroot/(rec['cache_key'].item()+'.npz')
   write_started=time.perf_counter();written=save_cache_atomic(path,rec,ch,replace_corrupt=rebuild_corrupt);cache_write_sec=time.perf_counter()-write_started
   loaded=load_cache(path,ch,expected)
   if not np.array_equal(loaded['box_masks'],expected_masks):raise RuntimeError('written cache box masks differ from frozen boxes')
   success.append({'identity':ident,'dataset':r['dataset'],'sample_id':r['sample_id'],'sample_key':r['sample_key'],'cache':str(path),'cache_key':str(loaded['cache_key']),'written':written,'bytes':path.stat().st_size,'num_boxes':r['num_boxes'],'empty_boxes':result['empty_boxes'],'mu_range':[float(result['maps']['mu'].min()),float(result['maps']['mu'].max())],'var_range':[float(result['maps']['var'].min()),float(result['maps']['var'].max())],'agreement_range':[float(result['maps']['agreement'].min()),float(result['maps']['agreement'].max())],'quality':[result['sam_quality_mean'],result['sam_quality_std']],'prompt_shapes':result['prompt_shapes'],'image_io_sec':image_io_sec,'sam_inference_sec':sam_inference_sec,'cache_write_sec':cache_write_sec,'elapsed_sec':time.perf_counter()-started})
   log.append({'ok':True,**success[-1]})
  except Exception as e:
   failed.append({'identity':ident,'error':f'{type(e).__name__}: {e}'}) ;log.append({'ok':False,**failed[-1]})
  print(json.dumps(log[-1],ensure_ascii=False),flush=True)
  (outroot/'generation.jsonl').open('a').write(json.dumps(log[-1],ensure_ascii=False)+'\n')
 index_rows=[]
 for r in records:
  hit=next((x for x in success if x['sample_key']==r['sample_key']),None)
  if hit:index_rows.append({'image_path':r['image_path'],'cache_path':hit['cache'],'dataset':r['dataset'],'split':'train','sample_id':r['sample_id'],'sample_key':r['sample_key'],'boxes_xyxy':r['boxes_xyxy'],'image_sha1':r['image_sha1'],'box_hash':r['box_hash']})
 (outroot/'dataset_index.json').write_text(json.dumps(index_rows,indent=2,ensure_ascii=False)+'\n')
 summary={'k':k,'config_hash':ch,'records_requested':len(records),'succeeded':len(success),'failed':failed,'successes':success,'generator':generator.load_info,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
 (outroot/'manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--boxes',type=Path,default=ROOT/'assets/boxes_train.json');p.add_argument('--output-root',type=Path,default=ROOT/'assets/stage2b_smoke/cache');p.add_argument('--k',type=int,choices=(1,3,5,8),required=True);p.add_argument('--limit',type=int);p.add_argument('--id',dest='ids',action='append');p.add_argument('--device',default='cuda:0');p.add_argument('--checkpoint',type=Path,default=CHECKPOINT);a=p.parse_args()
 print(json.dumps(generate(a.boxes,a.output_root,a.k,a.limit,a.ids,a.device,a.checkpoint),indent=2))
if __name__=='__main__':main()
