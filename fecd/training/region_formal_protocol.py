"""5C1独立正式配置；不修改已冻结5C0接口或配置。"""
import hashlib
import json
import os
from pathlib import Path

from fecd.pseudo.cache_schema import canonical_hash

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('/hy-tmp/BoxPromtIML/fecd_runs/stage5c1')


def configuration(variant):
    if variant not in ('R0', 'R1'):
        raise ValueError('仅正式R0/R1')
    base = json.loads((ROOT/'fecd/configs/stage5c0/R0.json').read_text())
    value = {k:v for k,v in base.items() if k not in
             ('config_hash','variant','selection','epoch_cap','diagnostic_only','gt_access')}
    value.update(schema='stage5c1-formal20-v1', variant=variant, epochs=20, steps=883,
                 updates=17660, automatic_shutdown=False, primary_checkpoint='epoch20 final',
                 selection='frozen region-stratified R0 random / R1 online hard',
                 checkpoint_policy='complete epoch boundaries only; all20 retained')
    return {**value, 'config_hash':canonical_hash(value)}


def validate_position(position, update):
    epoch = position.get('epoch')
    if (type(epoch) is not int or not 1 <= epoch <= 20 or position.get('next_batch') != 0
            or update != epoch*883):
        raise ValueError('只允许恢复完整且更新位置一致的Epoch边界')
    return epoch


def collapse_gate(summaries):
    if len(summaries) < 2:
        return False
    return all((s['all_zero_outputs']+s['near_full_outputs'])/s['images'] >= .9
               for s in summaries[-2:])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def commit_summary(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending.json')
    with pending.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError('禁止覆盖Epoch提交标记')
    os.replace(pending,path)


def verify_runtime():
    frozen = json.loads((OUTPUT/'results/runtime_frozen.json').read_text())
    if any(sha(path) != digest for path,digest in frozen.items()):
        raise ValueError('冻结运行文件发生变化')
