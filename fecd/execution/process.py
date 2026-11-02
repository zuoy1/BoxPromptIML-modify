"""Synchronous children with durable progress and whole-process-group cancellation."""
import json
import os
from pathlib import Path
import signal
import subprocess

from fecd.execution.artifacts import command
from fecd.execution.storage import guard_mount
from fecd.training.f1_protocol import ROOT, NAMES


def real_stage(stage, artifacts, progress):
    logs = artifacts.parent/'logs'
    argv = command(stage, artifacts)
    guard_mount(artifacts)
    with (logs/f'{stage}.log').open('x') as stream:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'}, start_new_session=True)
        try:
            while True:
                try:
                    code = child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    guard_mount(artifacts)
                    if stage in NAMES:
                        completed = sorted((artifacts/stage).glob('epoch*.summary.json'))
                        checkpoint = artifacts/stage/'last.pt'
                        current = {'epoch':min(len(completed)+1,20), 'epoch_update':0,
                                   'completed_epochs':len(completed), 'total_update':len(completed)*883}
                        with (logs/f'{stage}.log').open('r') as reading:
                            reading.seek(max(0, reading.seek(0, os.SEEK_END)-8192))
                            for line in reading.read().splitlines():
                                if line.startswith('{'):
                                    try:
                                        item = json.loads(line)
                                    except json.JSONDecodeError:
                                        continue
                                    if 'epoch' in item and 'update' in item:
                                        current.update(epoch=item['epoch'], epoch_update=item['update'],
                                                       total_update=(item['epoch']-1)*883+item['update'])
                        progress(**current, latest_checkpoint=str(checkpoint) if checkpoint.exists() else None,
                                 child_pid=child.pid)
            if code:
                raise RuntimeError(f'Stage {stage} failed exit={code}; error details: {logs/stage}.log')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
