"""Non-destructive cloud/storage capability check; never executes the shutdown binary."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from fecd.execution.shutdown import file_sha, environment_fingerprint
from fecd.execution.storage import atomic_json, now, guard_mount
from fecd.training.f1_protocol import ROOT, verify_frozen
from fecd.training.f1_storage import resolve_output_root


def inspect_environment(output):
    guard_mount(output)
    status = Path('/proc/self/status').read_text()
    capability = int(next(line.split()[1] for line in status.splitlines() if line.startswith('CapEff:')),16)
    binary = Path('/usr/local/bin/shutdown')
    commands = {}
    for argv in (['df','-hT','/','/hy-tmp'], ['findmnt','-T','/hy-tmp'], ['systemd-detect-virt']):
        result = subprocess.run(argv,capture_output=True,text=True,check=False)
        commands[' '.join(argv)] = {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    return {'checked_at':now(), 'status':'SHUTDOWN_NOT_VERIFIED',
        'reason':'No verified official stop/billing/persistence/retention evidence; do not execute candidate binary',
        'pid1':Path('/proc/1/comm').read_text().strip(), 'environment_fingerprint':environment_fingerprint(),
        'cap_sys_boot':bool(capability & (1<<22)), 'systemd_bus_present':Path('/run/systemd/system').exists(),
        'candidate_executable':str(binary), 'candidate_sha256':file_sha(binary),
        'candidate_executed':False, 'official_provider_identity_verified':False,
        'platform_stop_verified':False, 'gpu_billing_stop_verified':False,
        'data_persistence_verified':False, 'long_shutdown_retention_verified':False,
        'output_root':str(output), 'data_device':output.stat().st_dev, 'system_device':Path('/').stat().st_dev,
        'system_free_bytes':shutil.disk_usage('/').free, 'data_free_bytes':shutil.disk_usage(output).free,
        'tmux_available':shutil.which('tmux') is not None, 'commands':commands,
        'formal_training_started':False,'shutdown_requested':False,'forbidden_gt_reads':0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root')
    parser.add_argument('--record-dir', type=Path, required=True)
    args = parser.parse_args()
    output = resolve_output_root(args.output_root)
    verify_frozen()
    args.record_dir.mkdir(parents=True,exist_ok=False)
    result = inspect_environment(output)
    atomic_json(args.record_dir/'capabilities.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
