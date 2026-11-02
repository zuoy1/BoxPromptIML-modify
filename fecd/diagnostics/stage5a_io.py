"""独立开发入口的精确身份允许名单与只读文件审计。"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from fecd.evaluation.data import dev_allowlist, relocated, SOURCE
from fecd.pseudo.cache_schema import load_cache
from fecd.training.data import K1_ROOT, K8_ROOT, CONFIGS
from fecd.tools.stage4f1_v2_io import sha, write_new

ROOT = Path(__file__).resolve().parents[2]
RUN = Path('/hy-tmp/BoxPromtIML/fecd_runs/stage4f1/formal20_seed20261029_noshutdown_20261029a')
OUTPUT = Path('/hy-tmp/BoxPromtIML/fecd_runs/stage5a/diagnosis_20261030')


def prepare():
    rows, allowed, sources, split = dev_allowlist()
    indices, paths = {}, {str(SOURCE.resolve()): ('metadata', None)}
    for k, root in ((1, K1_ROOT), (8, K8_ROOT)):
        values = json.loads((root / 'dataset_index.json').read_text())
        index = {r['sample_key']: r for r in values}
        if len(index) != len(values):
            raise ValueError('缓存索引身份重复')
        indices[k] = {}
        for row in rows:
            key = row['sample_key']
            entry = index[key]
            if any(entry[field] != row[field] for field in ('dataset', 'sample_id', 'image_path', 'sample_key')):
                raise ValueError('开发图像缓存身份不匹配')
            if entry['split'] != 'train':
                raise ValueError('缓存角色错误')
            path = (ROOT / entry['cache_path']).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('缓存路径越界')
            indices[k][key] = {**entry, 'cache_path': str(path)}
            paths[str(path)] = (f'K{k}', key)
    for row in rows:
        key = row['sample_key']
        paths[str(Path(row['image_path']).resolve())] = ('RGB', key)
        paths[str(relocated(sources[key]['prepared_mask']).resolve())] = ('GT', key)
    write_new(OUTPUT / 'allowlist.json', {'members': rows, 'sources': sources,
               'cache_indices': indices, 'allowed_paths': paths, 'split_hash': split['manifest_sha256']})
    return rows, allowed, sources, indices, paths


class DiagnosticBoundary:
    def __init__(self, paths, log):
        self.paths, self.log = paths, log
        self.denied, self.counts = [], {}

    def __call__(self, event, args):
        if event != 'open' or isinstance(args[0], int):
            return
        path = str(Path(os.fsdecode(args[0])).resolve())
        mode, flags = args[1:3]
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        writing = writing or bool(mode and any(c in mode for c in 'wax+'))
        data_path = path.startswith(('/home/datasets/', '/hy-tmp/datasets/')) or '/stage2c_cache/' in path
        if writing and not Path(path).is_relative_to(OUTPUT):
            self.denied.append(path)
            self.log.write(json.dumps({'denied_write': path}) + '\n')
            self.log.flush()
            raise PermissionError(f'禁止修改历史文件: {path}')
        if data_path and path not in self.paths and not path.endswith(('dataset_index.json', 'config.json')):
            self.denied.append(path)
            self.log.write(json.dumps({'denied_read': path}) + '\n')
            self.log.flush()
            raise PermissionError(f'开发诊断允许名单拒绝: {path}')
        if path in self.paths:
            role, key = self.paths[path]
            self.counts[role] = self.counts.get(role, 0) + 1
            self.log.write(json.dumps({'path': path, 'role': role, 'sample_key': key,
                                       'operation': 'stage5a_diagnostic_read'}) + '\n')
            self.log.flush()


def teacher(entry, row, k, native_hw):
    content = Path(row['image_path']).read_bytes()
    if hashlib.sha256(content).hexdigest() != row['image_sha256']:
        raise ValueError('RGB内容变化')
    if hashlib.sha1(content).hexdigest() != entry['image_sha1']:
        raise ValueError('缓存RGB SHA1身份错误')
    expected = {**row, 'split': 'train', 'image_sha1': entry['image_sha1'],
                'box_hash': entry['box_hash'], 'k': k,
                'image_h': native_hw[0], 'image_w': native_hw[1]}
    record = load_cache(entry['cache_path'], CONFIGS[k], expected)
    return record['mu'].astype(np.float64), np.any(record['box_masks'], axis=0)
