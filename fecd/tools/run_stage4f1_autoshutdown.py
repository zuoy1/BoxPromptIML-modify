"""One controlled run: cloud gate, fresh formal training, audit, sync, verified stop request."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys

from fecd.execution.orchestrator import run_job
from fecd.execution.storage import guard_mount
from fecd.execution.shutdown import evidence_verified
from fecd.training.f1_protocol import ROOT
from fecd.training.f1_storage import resolve_output_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', help='Base directory; each unique run_id gets its own new subdirectory')
    parser.add_argument('--run-id')
    parser.add_argument('--dry-run', action='store_true', help='Synthetic artifacts and mock shutdown only')
    parser.add_argument('--detach', action='store_true', help='Start this same controlled workflow in tmux')
    parser.add_argument('--shutdown-on-failure', action='store_true', default=False)
    parser.add_argument('--no-shutdown', action='store_true', help='Run training without cloud-stop verification; never stop the server')
    parser.add_argument('--shutdown-evidence', type=Path, help='Reviewed non-secret official evidence JSON')
    args = parser.parse_args()
    if args.no_shutdown and (args.shutdown_on_failure or args.shutdown_evidence):
        parser.error('--no-shutdown excludes shutdown-on-failure and shutdown-evidence')
    output = resolve_output_root(args.output_root)
    guard_mount(output)
    if not output.is_relative_to('/hy-tmp') or not output.is_dir():
        raise ValueError('Unattended execution requires an existing independent /hy-tmp output base')
    if not args.dry_run:
        if shutil.disk_usage('/').free < 5*1024**3 or shutil.disk_usage(output).free < 8*1024**3:
            raise RuntimeError('Insufficient system/data safety reserve')
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+secrets.token_hex(4)
    evidence = json.loads(args.shutdown_evidence.read_text()) if args.shutdown_evidence else None
    if args.detach and (args.no_shutdown or args.dry_run or evidence_verified(evidence or {})):
        if not shutil.which('tmux'):
            raise RuntimeError('tmux is unavailable')
        argv = [sys.executable,'-u','-m','fecd.tools.run_stage4f1_autoshutdown',
                '--output-root',str(output),'--run-id',run_id]
        if args.dry_run:
            argv.append('--dry-run')
        if args.no_shutdown:
            argv.append('--no-shutdown')
        if args.shutdown_on_failure:
            argv.append('--shutdown-on-failure')
        if args.shutdown_evidence:
            argv.extend(['--shutdown-evidence',str(args.shutdown_evidence.resolve())])
        subprocess.run(['tmux','new-session','-d','-s',f'f1_{run_id}','-c',str(ROOT),
                        shlex.join(['env','PYTHONDONTWRITEBYTECODE=1',*argv])],check=True)
        print(json.dumps({'detached_session':f'f1_{run_id}', 'run_id':run_id,
                          'status':'LAUNCH_REQUESTED; inspect durable index for actual state'}))
        return
    result = run_job(output, run_id, dry_run=args.dry_run, evidence=evidence,
                     shutdown_on_failure=args.shutdown_on_failure,
                     index_path=ROOT/f'fecd/assets/stage4f1_autoshutdown/index/{run_id}.json',
                     no_shutdown=args.no_shutdown)
    print(json.dumps(result,indent=2),flush=True)
    if result['status'] == 'FAILED':
        raise SystemExit(1)
    if result['status'] == 'BLOCKED' and not args.dry_run:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
