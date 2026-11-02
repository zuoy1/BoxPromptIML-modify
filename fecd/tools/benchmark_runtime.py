"""Real SAM throughput benchmark on explicitly selected training records."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from fecd.tools.export_boxes import sha
from fecd.pseudo.sam_ensemble import SamEnsembleGenerator
from fecd.pseudo.box_ops import prompt_boxes
from fecd.pseudo.generate_cache import build_config
from fecd.pseudo.cache_schema import config_hash,build_cache_record
from fecd.pseudo.generate_cache import make_box_masks

def benchmark(boxes_json,output,k_values=(1,3,5,8),limit=100,device='cuda:0',checkpoint=Path('/home/weights/sam_vit_h_4b8939.pth')):
 rows=json.loads(Path(boxes_json).read_text())['records'][:limit];g=SamEnsembleGenerator(checkpoint,device)
 report={'samples':len(rows),'results':{},'generator':g.load_info}
 for k in k_values:
  times=[];decoder=[];enc=[];nbox=[];sizes=[]
  if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
  for r in rows:
   with Image.open(r['image_path']) as im:image=np.asarray(im.convert('RGB'))
   boxes=r['boxes_xyxy'][:3];h,w=image.shape[:2]
   t=time.perf_counter();g.predictor.set_image(image)
   if torch.cuda.is_available():torch.cuda.synchronize()
   enc.append(time.perf_counter()-t)
   dt=0
   for group in prompt_boxes(boxes,(h,w),k):
    for box in group:
     st=time.perf_counter()
     with torch.inference_mode():g.predictor.predict(point_coords=None,box=box,multimask_output=False,return_logits=True)
     if torch.cuda.is_available():torch.cuda.synchronize()
     dt+=time.perf_counter()-st
   decoder.append(dt);times.append(enc[-1]+dt);nbox.append(len(boxes))
   # Store-equivalent approximate NPZ size measured on representative result.
   if len(sizes)<10:
    result=g.predict(image,boxes,k)
    cfg=config_hash(build_config(g,k));rec=build_cache_record(maps=result['maps'],box_masks=make_box_masks(boxes,(h,w)),quality_mean=result['sam_quality_mean'],quality_std=result['sam_quality_std'],image_hw=(h,w),schema_config_hash=cfg,image_sha1=r['image_sha1'],box_hash=r['box_hash'],sample_id=r['sample_id'],dataset=r['dataset'],split='train',sample_key=r['sample_key'],num_boxes=len(boxes),k=k,empty_boxes=result['empty_boxes'])
    tmp=Path(output)/f'bench-k{k}-{len(sizes)}.npz';tmp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(tmp,**rec);sizes.append(tmp.stat().st_size);tmp.unlink()
  avg=float(np.mean(times)); report['results'][str(k)]={'images':len(times),'mean_sec_per_image':avg,'image_encoder_mean_sec':float(np.mean(enc)),'prompt_decoder_total_mean_sec':float(np.mean(decoder)),'mean_component_boxes':float(np.mean(nbox)),'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,'mean_compressed_cache_bytes':float(np.mean(sizes)) if sizes else 0,'estimated_full_train_hours':avg*5737/3600}
 Path(output).mkdir(parents=True,exist_ok=True);(Path(output)/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--boxes',type=Path,default=Path(__file__).resolve().parents[1]/'assets/boxes_train.json');p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'assets/stage2b_smoke');p.add_argument('--limit',type=int,default=100);p.add_argument('--device',default='cuda:0');a=p.parse_args();print(json.dumps(benchmark(a.boxes,a.output,limit=a.limit,device=a.device),indent=2))
if __name__=='__main__':main()
