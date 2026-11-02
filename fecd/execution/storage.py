"""Exclusive runs and durable state, checked against data-mount disappearance."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re


def now():
    return datetime.now(timezone.utc).isoformat()


def guard_mount(path):
    path = Path(path)
    if path.is_relative_to('/hy-tmp'):
        if Path('/hy-tmp').stat().st_dev == Path('/').stat().st_dev:
            raise RuntimeError('DATA_MOUNT_LOST: refusing system-disk fallback')
        parent = path
        while not parent.exists():
            parent = parent.parent
        if parent.stat().st_dev != Path('/hy-tmp').stat().st_dev:
            raise RuntimeError('DATA_MOUNT_MISMATCH')


def atomic_json(path, value):
    path = Path(path)
    guard_mount(path)
    pending = path.with_name(path.name+'.pending')
    with pending.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_run(base, run_id):
    base = Path(base)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise ValueError('Unsafe run_id')
    guard_mount(base)
    base.mkdir(parents=True, exist_ok=True)
    with (base/'.stage4f1.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = base/run_id
        directory.mkdir(exist_ok=False)
        try:
            yield directory
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class RunStore:
    def __init__(self, directory, state, index_path=None):
        self.directory = Path(directory)
        self.state = state
        self.index_path = Path(index_path) if index_path else None

    def write(self, **changes):
        self.state.update(changes, updated_at=now())
        try:
            atomic_json(self.directory/'status.json', self.state)
        finally:
            if self.index_path:
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(self.index_path, self.state)
        return self.state

    def event(self, kind, **fields):
        guard_mount(self.directory)
        with (self.directory/'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time':now(), 'event':kind, **fields}, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def sync(self):
        guard_mount(self.directory)
        os.sync()
