"""No OS-poweroff fallback: only an evidence-bound platform wrapper may stop a run."""
import hashlib
from pathlib import Path
import subprocess
from urllib.parse import urlsplit


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evidence_verified(evidence):
    if evidence.get('status') != 'VERIFIED' or evidence.get('operation') != 'stop_instance_preserve_data':
        return False
    if evidence.get('argv') != ['/usr/local/bin/shutdown']:
        return False
    if not all(evidence.get(k) for k in ('provider','retention_rule','reviewed_by','environment_fingerprint')):
        return False
    for field in ('official_stop_url','billing_url','persistence_url'):
        url = urlsplit(evidence.get(field,''))
        if url.scheme != 'https' or not url.hostname or url.username or url.query:
            return False
    if evidence['environment_fingerprint'] != environment_fingerprint():
        return False
    return evidence.get('executable_sha256') == file_sha('/usr/local/bin/shutdown')


def environment_fingerprint():
    return hashlib.sha256(Path('/proc/1/cgroup').read_bytes()).hexdigest()


def request_shutdown(request):
    if request.get('dry_run'):
        return {'mock_only':True, 'requested':False, 'billing_stopped_verified':False}
    evidence = request['evidence']
    if not evidence_verified(evidence):
        raise RuntimeError('SHUTDOWN_NOT_VERIFIED')
    try:
        result = subprocess.run(evidence['argv'], capture_output=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('PLATFORM_STOP_TIMEOUT; provider/billing state unknown') from error
    if result.returncode:
        raise RuntimeError(f'PLATFORM_STOP_FAILED exit={result.returncode}; raw output withheld')
    return {'argv':evidence['argv'], 'executable_sha256':evidence['executable_sha256'],
            'returncode':result.returncode, 'requested':True, 'billing_stopped_verified':False,
            'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),
            'stderr_sha256':hashlib.sha256(result.stderr).hexdigest()}
