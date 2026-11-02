"""4F1-V 输入不变性、只读边界及独立结果持久化。"""
import hashlib
import json
import os
from pathlib import Path
import sys

from fecd.tools.stage4f1_v2_checks import read


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


class ReadBoundary:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.reads = set()
        self.denied = []

    def __call__(self, event, args):
        if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'):
            self.denied.append(event)
            raise PermissionError('复验禁止启动子进程')
        if event != 'open' or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(root) for root in
               ('/home/datasets', '/hy-tmp/datasets', '/hy-model', '/hy-netdisk')):
            self.denied.append(str(path))
            raise PermissionError(f'复验禁止打开数据集: {path}')
        mode, flags = args[1:3]
        writing = ((mode is not None and any(c in mode for c in 'wax+'))
                   or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if writing and not path.is_relative_to(self.output):
            self.denied.append(str(path))
            raise PermissionError(f'复验禁止写入输出目录以外: {path}')
        if not writing:
            self.reads.add(str(path))


def install_boundary(output):
    boundary = ReadBoundary(output)
    sys.addaudithook(boundary)
    return boundary


def verify_snapshot(snapshot, run):
    changes = []
    for name, item in snapshot['files'].items():
        path = Path(name)
        if not path.is_file():
            changes.append({'path': name, 'reason': 'missing'})
        elif path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            changes.append({'path': name, 'reason': 'hash_or_size_changed'})
    original = {name for name in snapshot['files'] if Path(name).is_relative_to(run)}
    current = {str(p.resolve()) for p in run.rglob('*') if p.is_file()}
    if original != current:
        changes.append({'run_files_added': sorted(current - original),
                        'run_files_removed': sorted(original - current)})
    if changes:
        raise ValueError(f'输入快照变化: {changes}')
    return {'files': len(snapshot['files']), 'original_run_files': len(original), 'changed': 0}


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / 'fecd/tools').glob('*stage4f1_v2*.py'))
    paths.extend((root / 'fecd/tools').glob('stage4f1_v2_*.py'))
    paths.append(root / 'fecd/tests/test_stage4f1_v2.py')
    return {str(p.relative_to(root)): sha(p) for p in sorted(set(paths))}


def boundary_evidence(boundary, snapshot, run):
    excluded = {str(p) for p in boundary.reads if not Path(p).is_file()}
    consumed = {name: sha(name) for name in sorted(boundary.reads - excluded)}
    return {'dataset_or_write_violations': boundary.denied,
            'dataset_images_opened': 0 if not boundary.denied else None,
            'read_files_sha256': consumed,
            'input_snapshot_check': verify_snapshot(snapshot, run),
            'scope': 'CPython open 审计；不是操作系统级沙箱或外部进程证明'}
