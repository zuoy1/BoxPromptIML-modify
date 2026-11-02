"""Versioned and validated NPZ schema for SAM/box-only pseudo-label caches."""
from __future__ import annotations
import hashlib,json,os,tempfile,time
import fcntl
from pathlib import Path
import numpy as np
SCHEMA_VERSION='fecd-sam-cache-v2'
MAP_SHAPE=(224,224)
REQUIRED={'mu','var','agreement','box_masks','sam_quality_mean','sam_quality_std','image_h','image_w','schema_version','config_hash','image_sha1','box_hash','sample_id','dataset','split','sample_key','num_boxes','k','empty_boxes','cache_key'}
def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def make_sample_key(dataset,split,sample_id,image_sha256):
    return hashlib.sha256('\0'.join((dataset,split,sample_id,image_sha256)).encode()).hexdigest()
def make_cache_key(dataset,split,sample_id,image_sha1,box_hash,config_hash):
    return canonical_hash({'dataset':dataset,'split':split,'sample_id':sample_id,'image_sha1':image_sha1,'box_hash':box_hash,'config_hash':config_hash,'schema':SCHEMA_VERSION})
def config_hash(config):return canonical_hash(config)
def _scalar(x,dtype=None):return np.asarray(x,dtype=dtype)
def build_cache_record(*,maps,box_masks,quality_mean,quality_std,image_hw,schema_config_hash,image_sha1,box_hash,sample_id,dataset,split,sample_key,num_boxes,k,empty_boxes,base_mu=None):
    bh,bw=map(int,image_hw); bm=np.asarray(box_masks,dtype=np.uint8)
    if bm.shape!=(3,224,224):raise ValueError(f'box_masks must be [3,224,224], got {bm.shape}')
    rec={'mu':np.asarray(maps['mu'],np.float16),'var':np.asarray(maps['var'],np.float16),'agreement':np.asarray(maps['agreement'],np.float16),'box_masks':bm,
      'sam_quality_mean':_scalar(quality_mean,np.float32),'sam_quality_std':_scalar(quality_std,np.float32),'image_h':_scalar(bh,np.int32),'image_w':_scalar(bw,np.int32),
      'schema_version':_scalar(SCHEMA_VERSION),'config_hash':_scalar(schema_config_hash),'image_sha1':_scalar(image_sha1),'box_hash':_scalar(box_hash),'sample_id':_scalar(sample_id),'dataset':_scalar(dataset),'split':_scalar(split),'sample_key':_scalar(sample_key),'num_boxes':_scalar(num_boxes,np.int16),'k':_scalar(k,np.int8),'empty_boxes':_scalar(empty_boxes,np.bool_)}
    if base_mu is not None: rec['base_mu']=np.asarray(base_mu,np.float16)
    rec['cache_key']=_scalar(make_cache_key(dataset,split,sample_id,image_sha1,box_hash,schema_config_hash))
    validate_record(rec,expected_config_hash=schema_config_hash);return rec
def validate_record(rec,expected_config_hash=None,expected=None):
    missing=REQUIRED-set(rec)
    if missing:raise ValueError(f'cache missing fields: {sorted(missing)}')
    for k in ('mu','var','agreement'):
        a=np.asarray(rec[k])
        if a.shape!=MAP_SHAPE or a.dtype!=np.float16 or not np.isfinite(a).all():raise ValueError(f'invalid {k} shape/dtype/finite')
    if 'base_mu' in rec:
        a=np.asarray(rec['base_mu'])
        if a.shape!=MAP_SHAPE or a.dtype!=np.float16 or not np.isfinite(a).all() or a.min()<0 or a.max()>1: raise ValueError('invalid base_mu')
    if rec['box_masks'].shape!=(3,224,224) or rec['box_masks'].dtype!=np.uint8 or not np.isin(rec['box_masks'],(0,1)).all():raise ValueError('invalid box_masks')
    if any(not np.isfinite(np.asarray(rec[k])).all() for k in ('sam_quality_mean','sam_quality_std')):raise ValueError('non-finite quality')
    if int(rec['image_h'])<=0 or int(rec['image_w'])<=0:raise ValueError('invalid native image dimensions')
    if not 0<=int(rec['num_boxes'])<=3:raise ValueError('invalid component box count')
    if bool(rec['empty_boxes'])!=(int(rec['num_boxes'])==0):raise ValueError('empty box flag/count mismatch')
    active=np.asarray([bool(rec['box_masks'][i].any()) for i in range(3)])
    if int(active.sum())!=int(rec['num_boxes']) or (int(rec['num_boxes']) and not active[:int(rec['num_boxes'])].all()) or active[int(rec['num_boxes']):].any():raise ValueError('num_boxes does not agree with box_masks')
    if int(rec['k'])==1 and np.any(rec['var']!=0):raise ValueError('K=1 population variance must be exactly zero')
    for field in ('dataset','split','sample_id','sample_key','image_sha1','box_hash','config_hash'):
        if not str(rec[field]):raise ValueError(f'empty identity field: {field}')
    if np.min(rec['mu'])<0 or np.max(rec['mu'])>1 or np.min(rec['var'])<0 or np.max(rec['var'])>.2501 or np.min(rec['agreement'])<0 or np.max(rec['agreement'])>1:raise ValueError('probability statistic out of range')
    if int(np.asarray(rec['k'])) not in (1,3,5,8):raise ValueError('unsupported K value')
    if str(rec['schema_version'])!=SCHEMA_VERSION:raise ValueError('schema version mismatch')
    if expected_config_hash and str(rec['config_hash'])!=expected_config_hash:raise ValueError('configuration hash mismatch')
    if str(rec['cache_key'])!=make_cache_key(str(rec['dataset']),str(rec['split']),str(rec['sample_id']),str(rec['image_sha1']),str(rec['box_hash']),str(rec['config_hash'])):raise ValueError('cache key mismatch')
    if int(rec['k'])==8 and 'base_mu' not in rec: raise ValueError('K=8 cache is missing auditable base_mu')
    if int(rec['k'])!=8 and 'base_mu' in rec: raise ValueError('base_mu is only valid in K=8 caches')
    if expected:
        for key in ('dataset','split','sample_id','image_sha1','box_hash','sample_key','num_boxes','k','empty_boxes','image_h','image_w'):
            if key in expected and str(np.asarray(rec[key]).item()) != str(expected[key]): raise ValueError(f'{key} does not match expected source manifest')
        if all(x in expected for x in ('dataset','split','sample_id','image_sha256')):
            want=make_sample_key(*(str(expected[x]) for x in ('dataset','split','sample_id','image_sha256')))
            if str(rec['sample_key'])!=want: raise ValueError('sample_key does not match source identity')
        if 'k' in expected and int(rec['k'])!=int(expected['k']): raise ValueError('K does not match expected generation configuration')
    return True
def load_cache(path,expected_config_hash=None,expected=None):
    try:
      with np.load(path,allow_pickle=False) as npz:rec={k:npz[k].copy() for k in npz.files}
    except Exception as e:raise ValueError(f'corrupt/unreadable cache {path}: {e}') from e
    validate_record(rec,expected_config_hash,expected);return rec
def save_cache_atomic(path,record,expected_config_hash=None,lock_timeout=30,replace_corrupt=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);validate_record(record,expected_config_hash)
    lock=path.with_suffix(path.suffix+'.lock'); guard=path.parent/'.cache-lock-recovery.guard'; deadline=time.monotonic()+lock_timeout; fd=None
    while fd is None:
      # Serialize lock create/reclaim so no writer can create a new active lock
      # between another writer's stale-PID check and unlink.
      guard_fd=os.open(guard,os.O_CREAT|os.O_RDWR,0o600)
      fcntl.flock(guard_fd,fcntl.LOCK_EX)
      try:
        if lock.exists():
          # Reclaim only a provably stale lock. Never unlink a lock owned by a live PID.
          try:
            owner=int(lock.read_text().strip())
            try: os.kill(owner,0); alive=True
            except ProcessLookupError: alive=False
            except PermissionError: alive=True
            if not alive: lock.unlink(missing_ok=True)
          except (ValueError,OSError):
            # Unknown/malformed lock ownership is not proof of staleness.
            pass
        try:
          fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
          os.write(fd,str(os.getpid()).encode())
        except FileExistsError:
          fd=None
      finally:
        fcntl.flock(guard_fd,fcntl.LOCK_UN);os.close(guard_fd)
      if fd is None:
        if time.monotonic()>deadline:raise TimeoutError(f'cache write lock timeout: {lock}')
        time.sleep(.1)
    try:
      # Another process may have produced it while this process waited.
      if path.exists():
        try:existing=load_cache(path,expected_config_hash)
        except ValueError as exc:
          if (not replace_corrupt) or any(token in str(exc) for token in ('configuration hash mismatch','schema version mismatch')):raise
          path.unlink()
        else:
          if str(existing['cache_key'])==str(record['cache_key']):return False
          raise ValueError(f'existing cache identity/config differs: {path}')
      with tempfile.NamedTemporaryFile(prefix=path.name+'.',suffix='.tmp',dir=path.parent,delete=False) as f:temp=Path(f.name)
      try:
        with temp.open('wb') as out:np.savez_compressed(out,**record);out.flush();os.fsync(out.fileno())
        load_cache(temp,expected_config_hash);os.replace(temp,path)
      finally:
        if temp.exists():temp.unlink()
      return True
    finally:
      if fd is not None:os.close(fd)
      try:lock.unlink()
      except FileNotFoundError:pass
