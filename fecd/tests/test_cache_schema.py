import numpy as np
from fecd.pseudo.cache_schema import *

def record(ch='c'*64):
 return build_cache_record(maps={x:np.zeros((224,224),np.float32) for x in ('mu','var','agreement')},box_masks=np.zeros((3,224,224),np.uint8),quality_mean=0,quality_std=0,image_hw=(20,30),schema_config_hash=ch,image_sha1='a'*40,box_hash='b'*64,sample_id='same',dataset='D',split='train',sample_key='s'*64,num_boxes=0,k=1,empty_boxes=True)
def test_atomic_roundtrip_config_and_duplicate_names(tmp_path):
 a=record();p=tmp_path/'cache.npz';assert save_cache_atomic(p,a,'c'*64);assert not save_cache_atomic(p,a,'c'*64)
 b=load_cache(p,'c'*64);assert str(b['k'])=='1'
 try:load_cache(p,'d'*64)
 except ValueError:pass
 else:raise AssertionError('stale config was accepted')
 assert make_cache_key('D','train','same','a'*40,'b'*64,'c'*64)!=make_cache_key('E','train','same','a'*40,'b'*64,'c'*64)
def test_corrupt_cache_errors(tmp_path):
 p=tmp_path/'x.npz';p.write_bytes(b'not npz')
 try:load_cache(p)
 except ValueError as e:assert 'corrupt' in str(e)
 else:raise AssertionError('corrupt cache was accepted')

def test_tampered_metadata_and_stale_lock(tmp_path):
 from copy import deepcopy
 r=record()
 for key,value in [('sample_id','other'),('dataset','OTHER'),('k',np.asarray(8,np.int8)),('num_boxes',np.asarray(1,np.int16)),('empty_boxes',np.asarray(False)),('box_hash','c'*64),('image_sha1','d'*40),('sample_key','x'*64)]:
  bad=deepcopy(r);bad[key]=np.asarray(value)
  try:validate_record(bad,expected_config_hash='c'*64,expected={'dataset':'D','split':'train','sample_id':'same','image_sha1':'a'*40,'box_hash':'b'*64,'sample_key':'s'*64,'num_boxes':0,'k':1,'empty_boxes':True})
  except ValueError:pass
  else:raise AssertionError(f'tampered {key} metadata was accepted')
 # A live owner is preserved; a provably dead PID lock is reclaimed.
 live=tmp_path/'live.npz';live_lock=live.with_suffix('.npz.lock');live_lock.write_text(str(__import__('os').getpid()))
 try:save_cache_atomic(live,record(), 'c'*64,lock_timeout=.15)
 except TimeoutError:pass
 else:raise AssertionError('active lock was ignored')
 assert live_lock.exists()
 live_lock.unlink()
 stale=tmp_path/'stale.npz';stale.with_suffix('.npz.lock').write_text('2147483647')
 assert save_cache_atomic(stale,record(),'c'*64)
