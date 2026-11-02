"""Dedicated new_dev GT boundary; training never imports this module."""
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from fecd.diagnostics.stage3ra_split import read_split

SOURCE=Path('/home/datasets/processed/boxprompt_paper_scale_20261016/manifest.json')
SOURCE_SHA='7b82396c97d60a7463e26cdaad8e88e5d1593e47fb481939d54b0e03859c6f70'
SPLIT_SHA='022da143a26833f1613db722ae3be152117948330621e028f02b824f2d1680d0'


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def relocated(path):
    return Path(str(path).replace('/hy-tmp/datasets/','/home/datasets/'))


def dev_allowlist():
    split=read_split()
    if split['manifest_sha256']!=SPLIT_SHA:
        raise ValueError('Unrecognized frozen student/development split')
    rows=sorted([r for r in split['records'] if r['subset']=='new_dev'],key=lambda r:r['sample_key'])
    if len(rows)!=530 or len({r['sample_key'] for r in rows})!=530:
        raise ValueError('Strict evaluator requires all 530 unique new_dev members')
    if file_sha(SOURCE)!=SOURCE_SHA:
        raise ValueError('GT source metadata manifest SHA mismatch')
    manifest=json.loads(SOURCE.read_text())
    source_rows=manifest['records'] if isinstance(manifest,dict) else manifest
    train_rows=[r for r in source_rows if r['split']=='train']
    index={(r['dataset'],r['id']):r for r in train_rows}
    if len(index)!=len(train_rows):
        raise ValueError('Duplicate source training identity')
    sources={}
    for row in rows:
        source=index[(row['dataset'],row['sample_id'])]
        if source['image_sha256']!=row['image_sha256'] or relocated(source['image']).resolve()!=Path(row['image_path']).resolve():
            raise ValueError('Frozen RGB/source pairing mismatch')
        sources[row['sample_key']]=source
    return rows,{r['sample_key']:r for r in rows},sources,split


def load_rgb(row):
    content=Path(row['image_path']).read_bytes()
    if hashlib.sha256(content).hexdigest()!=row['image_sha256']:
        raise ValueError('Frozen RGB bytes changed')
    with Image.open(BytesIO(content)) as image:
        native_hw=(image.height,image.width)
        rgb=np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)/255.
    rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(rgb.transpose(2,0,1).copy()).unsqueeze(0),native_hw


def load_dev_gt(row,source,native_hw,allowed_rows,log_path):
    key=row.get('sample_key')
    if row.get('subset')!='new_dev' or key not in allowed_rows or row!=allowed_rows[key]:
        raise ValueError('GT access denied: exact frozen new_dev membership required')
    if (source.get('split')!='train' or source.get('dataset')!=row['dataset'] or source.get('id')!=row['sample_id']
            or source.get('image_sha256')!=row['image_sha256'] or
            tuple(native_hw)!=(source.get('height'),source.get('width'))):
        raise ValueError('GT source identity/native dimensions mismatch')
    path=relocated(source['prepared_mask'])
    content=path.read_bytes()
    entry={'sample_key':key,'role':'new_dev','path':str(path),'sha256':hashlib.sha256(content).hexdigest(),
           'native_hw':list(native_hw),'operation':'dedicated_strict_evaluator_gt_read'}
    with Path(log_path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(entry,sort_keys=True)+'\n')
    with Image.open(BytesIO(content)) as image:
        raw=np.asarray(image)
        if raw.ndim!=2 or raw.shape!=tuple(native_hw) or not np.isin(raw,[0,255]).all():
            raise ValueError('Prepared GT must be native binary 0/255 scalar mask')
        return np.asarray(image.resize((224,224),Image.Resampling.NEAREST))>127
