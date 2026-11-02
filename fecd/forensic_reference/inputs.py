"""Read-only, identity-checked RGB/cache inputs; no mask loading capability."""
import hashlib
import codecs
import io
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

from fecd.diagnostics.stage3ra_split import K8_CONFIG
from fecd.pseudo.cache_schema import load_cache

PROJECT = Path(__file__).resolve().parents[2]
SAM = PROJECT/'fecd/assets/stage2c_cache/k8/2bff688546a7'
Image.init()
codecs.lookup('cp437')
_warm = io.BytesIO()
np.savez_compressed(_warm, probe=np.zeros(1))
_warm.seek(0)
with np.load(_warm, allow_pickle=False) as _probe:
    _probe['probe']
_warm.close()


class ReadGuard:
    """Deny unallowlisted opens while active, including pathlib and NumPy IO."""
    def __init__(self, paths):
        self.allowed = {str(Path(p).resolve()) for p in paths}
        self.active = False
        self.opens = []
        self.denied = []

    def _audit(self, event, args):
        if event != 'open' or not self.active:
            return
        path = args[0]
        if isinstance(path, int):
            return
        resolved = str(Path(os.fsdecode(path)).resolve())
        mode = args[1] or ''
        flags = args[2] or 0
        writing = any(x in mode for x in ('w', 'a', '+', 'x')) or flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        if resolved not in self.allowed or writing:
            self.denied.append(resolved)
            raise PermissionError('GT-free input boundary rejected file access')
        self.opens.append(resolved)

    def __enter__(self):
        sys.addaudithook(self._audit)
        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False


def choose_rows(rows):
    return [row for dataset in sorted({r['dataset'] for r in rows})
            for row in sorted((r for r in rows if r['subset']=='new_dev'
                               and r['dataset']==dataset), key=lambda r:r['sample_key'])[:16]]


def load_input(row, index_row):
    if row['subset'] != 'new_dev':
        raise ValueError('B0 pilot may load only selected new_dev RGB')
    for key in ('sample_key', 'sample_id', 'dataset', 'image_path'):
        if row[key] != index_row[key]:
            raise ValueError('SAM index/sample identity mismatch')
    if index_row['split'] != 'train':
        raise ValueError('Nontraining sample rejected')
    image_path = Path(row['image_path']).resolve()
    cache_path = (PROJECT/index_row['cache_path']).resolve()
    if not cache_path.is_relative_to(SAM.resolve()):
        raise ValueError('K8 cache path escapes frozen directory')
    with ReadGuard([image_path, cache_path]) as guard:
        data = image_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['image_sha256']:
            raise ValueError('Image SHA256 mismatch')
        sha1 = hashlib.sha1(data).hexdigest()
        if sha1 != index_row['image_sha1']:
            raise ValueError('Image/SAM SHA1 mismatch')
        cache = load_cache(cache_path, K8_CONFIG, expected={
            **row, 'split':'train', 'k':8, 'image_sha1':sha1,
            'box_hash':index_row['box_hash']})
        with Image.open(io.BytesIO(data)) as image:
            info = {'format':image.format, 'native_hw':[image.height,image.width],
                    'jpeg_quantization':getattr(image,'quantization',None),
                    'pillow_info_keys':sorted(str(k) for k in image.info)}
            rgb = np.asarray(image.convert('RGB'))
        if info['native_hw'] != [int(cache['image_h']),int(cache['image_w'])]:
            raise ValueError('Native/cache shape mismatch')
    info.update(sample_key=row['sample_key'], gt_files_opened=0,
                allowed_opens=guard.opens, denied_opens=guard.denied,
                cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                image_sha256=row['image_sha256'])
    return rgb, cache, info
