"""Run frozen K=1 and K=8 caches over exactly the locked 5,737 train identities."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from fecd.pseudo.generate_cache import generate

ROOT=Path(__file__).resolve().parents[1]
def run(boxes=ROOT/'assets/boxes_train.json',output=ROOT/'assets/stage2c_cache',device='cuda:0',checkpoint=Path('/home/weights/sam_vit_h_4b8939.pth')):
 source=json.loads(Path(boxes).read_text());rows=source['records']
 identities=[(r['dataset'],'train',r['sample_id']) for r in rows]
 if len(rows)!=5737 or len(set(identities))!=5737 or any(r['split']!='train' for r in rows):raise RuntimeError('frozen box file does not contain exactly 5,737 unique train identities')
 started=time.perf_counter();reports={}
 for k in (1,8):
  all_success={};pending=[];attempt_reports=[]
  summary=generate(boxes,output,k,device=device,checkpoint=checkpoint)
  attempt_reports.append({'attempt':1,'requested':summary['records_requested'],'succeeded':summary['succeeded'],'failed':summary['failed']})
  for x in summary['successes']:all_success[x['sample_key']]=x
  pending=summary['failed']
  for attempt in (2,3):
   if not pending:break
   retry=generate(boxes,output,k,ids=[x['sample_key'] for x in pending],device=device,checkpoint=checkpoint)
   attempt_reports.append({'attempt':attempt,'requested':retry['records_requested'],'succeeded':retry['succeeded'],'failed':retry['failed']})
   for x in retry['successes']:all_success[x['sample_key']]=x
   pending=retry['failed']
  ordered=[all_success[r['sample_key']] for r in rows if r['sample_key'] in all_success]
  config_hash=summary['config_hash'];cache_dir=Path(output)/f'k{k}'/config_hash[:12]
  final={'k':k,'config_hash':config_hash,'records_requested':5737,'succeeded':len(ordered),'failed':pending,'successes':ordered,'attempts':attempt_reports,'generator':summary['generator'],'wall_sec':time.perf_counter()-started}
  (cache_dir/'manifest.json').write_text(json.dumps(final,indent=2,ensure_ascii=False)+'\n')
  index=[]
  for r in rows:
   hit=next((x for x in ordered if x['sample_key']==r['sample_key']),None)
   if hit:index.append({'image_path':r['image_path'],'cache_path':hit['cache'],'dataset':r['dataset'],'split':'train','sample_id':r['sample_id'],'sample_key':r['sample_key'],'boxes_xyxy':r['boxes_xyxy'],'image_sha1':r['image_sha1'],'box_hash':r['box_hash']})
  (cache_dir/'dataset_index.json').write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
  reports[str(k)]=final
 report={'status':'generated_with_failures' if any(x['failed'] for x in reports.values()) else 'generated','sample_count':5737,'boxes_file':str(Path(boxes).resolve()),'wall_sec':time.perf_counter()-started,'k_runs':{k:{'succeeded':v['succeeded'],'failed':len(v['failed']),'wall_sec':v['wall_sec'],'attempts':v['attempts'],'manifest':str(Path(output)/f'k{k}'/v['config_hash'][:12]/'manifest.json')} for k,v in reports.items()}}
 (Path(output)/'stage2c_run_summary.json').write_text(json.dumps(report,indent=2)+'\n');return report
def main():
 p=argparse.ArgumentParser();p.add_argument('--boxes',type=Path,default=ROOT/'assets/boxes_train.json');p.add_argument('--output',type=Path,default=ROOT/'assets/stage2c_cache');p.add_argument('--device',default='cuda:0');p.add_argument('--checkpoint',type=Path,default=Path('/home/weights/sam_vit_h_4b8939.pth'));a=p.parse_args();print(json.dumps(run(a.boxes,a.output,a.device,a.checkpoint),indent=2))
if __name__=='__main__':main()
