"""Explicit success/failure state machine; shutdown is never an EXIT trap."""
import json
from pathlib import Path

from fecd.execution.artifacts import STAGES, verify_stage, frozen_files
from fecd.execution.process import real_stage
from fecd.execution.shutdown import evidence_verified, request_shutdown, file_sha
from fecd.execution.simulation import simulate_stage
from fecd.execution.storage import exclusive_run, RunStore, now, atomic_json, guard_mount


def _blocked(store, reason):
    store.event('blocked', reason=reason)
    return store.write(status='BLOCKED', reason=reason, shutdown_status='SHUTDOWN_NOT_VERIFIED', ended_at=now())


def _manifest(directory):
    return {str(p.relative_to(directory)):{'sha256':file_sha(p), 'bytes':p.stat().st_size}
            for p in sorted(directory.rglob('*')) if p.is_file()}


def _shutdown(store, dry_run, requester, evidence, reason):
    request = {'dry_run':dry_run, 'reason':reason, 'evidence':evidence or {}}
    store.event('shutdown_intent', dry_run=dry_run, reason=reason,
                argv=(evidence or {}).get('argv',[]), billing_stopped_verified=False)
    store.write(shutdown_status='DRY_RUN_MOCK' if dry_run else 'REQUEST_PENDING',
                billing_stopped_verified=False, shutdown_attempted=True)
    store.sync()
    try:
        response = requester(request)
    except Exception as error:
        store.write(shutdown_status='FAILED', shutdown_error=str(error), billing_stopped_verified=False)
        store.event('shutdown_failed', error_type=type(error).__name__, error=str(error))
        store.sync()
        raise
    store.event('shutdown_response', response=response)
    store.write(shutdown_response=response, shutdown_status='DRY_RUN_MOCK' if dry_run else 'REQUESTED')
    store.sync()


def run_job(base, run_id, dry_run=False, evidence=None, stage_runner=None,
            shutdown_requester=None, shutdown_on_failure=False, index_path=None, no_shutdown=False):
    if no_shutdown and shutdown_on_failure:
        raise ValueError('no_shutdown cannot be combined with shutdown_on_failure')
    with exclusive_run(base, run_id) as directory:
        state = {'run_id':run_id, 'status':'RUNNING', 'dry_run':dry_run, 'started_at':now(),
                 'output_root':str(directory/'artifacts'), 'stage':None, 'epoch':0,
                 'latest_checkpoint':None, 'completed_stages':[],
                 'shutdown_on_failure':shutdown_on_failure, 'billing_stopped_verified':False,
                 'no_shutdown':no_shutdown, 'shutdown_status':'DISABLED' if no_shutdown else 'NOT_REQUESTED'}
        store = RunStore(directory, state, index_path)
        store.write()
        (directory/'logs').mkdir()
        if not dry_run and not no_shutdown and not evidence_verified(evidence or {}):
            return _blocked(store, 'SHUTDOWN_NOT_VERIFIED: stop/billing/data-retention evidence missing')
        execute = stage_runner or (simulate_stage if dry_run else real_stage)
        requester = shutdown_requester or request_shutdown
        artifacts = directory/'artifacts'
        try:
            before = {} if dry_run else frozen_files()
            atomic_json(directory/'runtime_and_history.json', before)
            for path, digest in before.items():
                if file_sha(path) != digest:
                    raise RuntimeError(f'Frozen file differs before start: {path}')
            for stage in STAGES:
                guard_mount(artifacts)
                store.write(stage=stage, epoch=0)
                store.event('stage_start', stage=stage, simulation=dry_run)
                execute(stage, artifacts, lambda **kw:store.write(**kw))
                gate = verify_stage(stage, artifacts, dry_run)
                completed = [*state['completed_stages'], stage]
                store.write(completed_stages=completed, last_gate=gate)
                store.event('stage_pass', stage=stage, gate=gate)
            for path, digest in before.items():
                if file_sha(path) != digest:
                    raise RuntimeError(f'Frozen file changed during run: {path}')
            report = {'engineering':'PASS', 'simulation':dry_run, 'completed_stages':state['completed_stages'],
                      'interpretation':'single seed new_dev development results; negative results are valid completion',
                      'validation':json.loads((artifacts/'validation.json').read_text()),
                      'paired_statistics':json.loads((artifacts/'analysis/paired_bootstrap.json').read_text()),
                      'forbidden_gt_reads':0, 'no_shutdown':no_shutdown, 'ended_at':now()}
            atomic_json(directory/'completion_report.json', report)
            from fecd.execution.report import write_report
            write_report(directory, report, dry_run)
            atomic_json(directory/'output_manifest.json', _manifest(artifacts))
            atomic_json(directory/'logs_manifest.json', _manifest(directory/'logs'))
            atomic_json(directory/'reports_manifest.json', {name:file_sha(directory/name) for name in
                        ('completion_report.json','completion_report.md','stage4f1_report.md','runtime_and_history.json')})
            store.sync()
            if dry_run:
                if not no_shutdown:
                    _shutdown(store, True, requester, None, 'success')
                return store.write(status='BLOCKED', dry_run_status='PASS',
                    simulated_status='SUCCESS' if no_shutdown else 'SUCCESS_SHUTDOWN_REQUESTED',
                    reason='DRY_RUN_ONLY; real cloud stop not executed', ended_at=now())
            atomic_json(directory/'SUCCESS.json', {'engineering':'PASS', 'run_id':run_id,
                        'completed_at':now(), 'billing_stopped_verified':False,
                        'output_manifest_sha256':file_sha(directory/'output_manifest.json')})
            if no_shutdown:
                store.write(status='SUCCESS', ended_at=now(), shutdown_status='DISABLED',
                            server_action='keep running; user stops instance manually')
                store.sync()
                return dict(state)
            store.write(status='SUCCESS_SHUTDOWN_REQUESTED', ended_at=now(),
                        shutdown_status='REQUEST_PENDING', billing_stopped_verified=False)
            store.sync()
            _shutdown(store, False, requester, evidence, 'success')
            return dict(state)
        except BaseException as error:
            state.update(status='FAILED', error_type=type(error).__name__, error=str(error), ended_at=now())
            try:
                store.write()
                store.event('failed', stage=state['stage'], error_type=type(error).__name__, error=str(error))
                store.sync()
                if not no_shutdown and shutdown_on_failure and not state.get('shutdown_attempted'):
                    _shutdown(store, dry_run, requester, evidence, 'failure')
            except Exception as save_error:
                print(f'FAILED_STATE_PERSISTENCE: {type(save_error).__name__}; last known run {directory}', flush=True)
            return dict(state)
