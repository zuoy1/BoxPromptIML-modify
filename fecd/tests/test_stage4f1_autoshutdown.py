"""File-backed orchestration contracts; all shutdown and training boundaries simulated."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fecd.execution.orchestrator import run_job
from fecd.execution.simulation import simulate_stage
from fecd.execution.storage import exclusive_run
from fecd.execution.shutdown import request_shutdown
from fecd.training.access import DataReadAudit


class AutoshutdownTests(unittest.TestCase):
    def test_no_shutdown_simulation_never_calls_shutdown_on_success(self):
        with TemporaryDirectory() as directory:
            with patch('fecd.execution.orchestrator.evidence_verified') as verify:
                stops = []
                result = run_job(Path(directory), 'run_only', dry_run=True, no_shutdown=True,
                                 shutdown_requester=lambda request:stops.append(request))
            self.assertEqual(result['dry_run_status'], 'PASS')
            self.assertEqual(result['simulated_status'], 'SUCCESS')
            self.assertEqual(result['shutdown_status'], 'DISABLED')
            self.assertFalse(stops)
            verify.assert_not_called()

    def test_no_shutdown_mode_rejects_failure_shutdown_switch(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_job(Path(directory), 'invalid', no_shutdown=True, shutdown_on_failure=True)

    def test_run_only_failure_never_calls_shutdown(self):
        with TemporaryDirectory() as directory:
            def fail(*args):
                raise RuntimeError('run-only training failure')
            stops = []
            result = run_job(Path(directory), 'failure', dry_run=True, no_shutdown=True,
                             stage_runner=fail, shutdown_requester=lambda request:stops.append(request))
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(result['shutdown_status'], 'DISABLED')
            self.assertFalse(stops)

    def run_case(self, fault=None, failure_shutdown=False):
        with TemporaryDirectory() as directory:
            calls, stops = [], []
            def stage(name, artifacts, progress):
                calls.append(name)
                if fault == name:
                    raise RuntimeError('injected stage failure')
                simulate_stage(name, artifacts, progress)
                if fault == 'missing_checkpoint' and name == 'P0':
                    (artifacts/'P0/final.pt').unlink()
                if fault == 'bad_hash' and name == 'P0':
                    with (artifacts/'P0/final.pt').open('ab') as stream:
                        stream.write(b'corruption')
            result = run_job(Path(directory), 'simulation', dry_run=True,
                             stage_runner=stage, shutdown_requester=lambda request:stops.append(request),
                             shutdown_on_failure=failure_shutdown)
            stored = json.loads((Path(directory)/'simulation/status.json').read_text())
            self.assertEqual(stored, result)
            self.assertFalse((Path(directory)/'simulation/SUCCESS.json').exists())
            return result, calls, stops

    def test_simulated_success_runs_all_stages_in_order_and_only_mock_shutdown(self):
        result, calls, stops = self.run_case()
        self.assertEqual(calls, ['preflight','P0','P1','P2F','P3F','evaluation',
                                 'statistics','curves','tests','validation'])
        self.assertEqual(result['dry_run_status'], 'PASS')
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertEqual(result['simulated_status'], 'SUCCESS_SHUTDOWN_REQUESTED')
        self.assertEqual(len(stops), 1)
        self.assertTrue(stops[0]['dry_run'])

    def test_training_failure_stops_later_stages_and_never_marks_success(self):
        result, calls, stops = self.run_case('P1')
        self.assertEqual(result['status'], 'FAILED')
        self.assertNotIn('P2F', calls)
        self.assertFalse(stops)

    def test_evaluation_failure_does_not_request_success_shutdown(self):
        result, calls, stops = self.run_case('evaluation')
        self.assertEqual(result['status'], 'FAILED')
        self.assertNotIn('statistics', calls)
        self.assertFalse(stops)

    def test_missing_checkpoint_or_corrupt_hash_blocks_success_shutdown(self):
        for fault in ('missing_checkpoint','bad_hash'):
            result, calls, stops = self.run_case(fault)
            self.assertEqual(result['status'], 'FAILED')
            self.assertNotIn('P1', calls)
            self.assertFalse(stops)

    def test_failure_shutdown_is_explicit_and_retains_failed_status(self):
        result, _, stops = self.run_case('P1', failure_shutdown=True)
        self.assertEqual(result['status'], 'FAILED')
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0]['reason'], 'failure')

    def test_unverified_cloud_blocks_before_any_real_child_or_shutdown(self):
        with TemporaryDirectory() as directory:
            calls = []
            result = run_job(Path(directory), 'blocked', dry_run=False,
                             stage_runner=lambda *args:calls.append(args), evidence={'status':'SHUTDOWN_NOT_VERIFIED'})
            self.assertEqual(result['status'], 'BLOCKED')
            self.assertEqual(result['shutdown_status'], 'SHUTDOWN_NOT_VERIFIED')
            self.assertFalse(calls)

    def test_existing_run_and_concurrent_lock_are_rejected_without_overwriting(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            with exclusive_run(base, 'first'):
                with self.assertRaises(BlockingIOError):
                    with exclusive_run(base, 'second'):
                        self.fail('concurrent run acquired the lock')
            with self.assertRaises(FileExistsError):
                with exclusive_run(base, 'first'):
                    self.fail('existing run was reused')

    def test_real_shutdown_boundary_is_never_executed_for_dry_run(self):
        with patch('fecd.execution.shutdown.subprocess.run') as execute:
            result = request_shutdown({'dry_run':True, 'reason':'success'})
            self.assertTrue(result['mock_only'])
            execute.assert_not_called()

    def test_data_mount_loss_blocks_before_any_run_or_child_creation(self):
        with patch('fecd.execution.storage.Path.stat', return_value=SimpleNamespace(st_dev=67)):
            with self.assertRaises(RuntimeError):
                with exclusive_run(Path('/hy-tmp/blocked-unmounted'), 'run'):
                    self.fail('unmounted run started')

    def test_simulation_opens_no_dataset_files_and_preserves_existing_outputs(self):
        with TemporaryDirectory() as directory:
            sentinel = Path(directory)/'old_P0.pt'
            sentinel.write_bytes(b'protected old system-disk artifact')
            with DataReadAudit([],[]) as audit:
                result = run_job(Path(directory), 'new', dry_run=True)
            self.assertEqual(result['dry_run_status'], 'PASS')
            self.assertFalse(audit.opened)
            self.assertFalse(audit.denied)
            self.assertEqual(sentinel.read_bytes(), b'protected old system-disk artifact')

    def test_failed_shutdown_is_failed_and_not_billing_confirmation(self):
        with TemporaryDirectory() as directory:
            def stop(request):
                raise RuntimeError('mock stop rejected')
            result = run_job(Path(directory), 'new', dry_run=True, shutdown_requester=stop)
            self.assertEqual(result['status'], 'FAILED')
            self.assertIn('mock stop rejected', result['error'])
            self.assertFalse(result['billing_stopped_verified'])
            self.assertEqual(result['shutdown_status'], 'FAILED')

    def test_shutdown_failure_is_not_retried_by_failure_shutdown_policy(self):
        with TemporaryDirectory() as directory:
            calls = []
            def stop(request):
                calls.append(request)
                raise RuntimeError('mock stop rejected')
            result = run_job(Path(directory), 'new', dry_run=True, shutdown_requester=stop,
                             shutdown_on_failure=True)
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(len(calls), 1)

    def test_failed_data_disk_write_retains_error_in_system_index(self):
        with TemporaryDirectory() as directory:
            index = Path(directory)/'index.json'
            from fecd.execution.storage import RunStore
            store = RunStore(Path(directory)/'missing', {'status':'FAILED','error':'mount lost'}, index)
            with self.assertRaises(FileNotFoundError):
                store.write()
            self.assertEqual(json.loads(index.read_text())['status'], 'FAILED')

