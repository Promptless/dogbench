"""Offline adapter contracts; never launch a CLI, container, or provider job."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dogbench import adapters
from dogbench.promptless_backend import load_promptless_backend


class AdapterContractTests(unittest.TestCase):
    def test_missing_private_backend_fails_before_any_network(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(adapters, '_gh_pr_diff_raw') as diff:
            result = adapters.run_promptless(
                prompt='Update docs', mirror_repo='owned/task', token='controller-token',
                api_trigger_key='trigger-key', runtime_base_url='https://configured.example',
                code_pr_url='https://github.com/owned/task/pull/1',
            )
        self.assertFalse(result.ok)
        self.assertIn('Promptless backend is not configured', result.error)
        diff.assert_not_called()

    def test_mintlify_requires_own_project_before_request(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(adapters, '_mintlify_request') as request:
            result = adapters.run_mintlify(prompt='Task', mirror_repo='owned/task',
                                           token='controller', mintlify_token='provided')
        self.assertFalse(result.ok)
        self.assertIn('MINTLIFY_PROJECT_ID', result.error)
        request.assert_not_called()

    def test_promptless_backend_preserves_trigger_keyed_pr_and_trace(self):
        class Backend:
            def ensure_collection(self, repo_url, **kwargs):
                self.repo_url = repo_url
                return 'owned-collection'
            def analysis_overlay(self, **kwargs):
                return None
            def dispatch_status(self, trigger_event_id):
                assert trigger_event_id == 'trigger-1'
                return {'suggestion_pr': 'https://github.com/owned/task/pull/12'}
            def export_trace(self, trigger_event_id):
                return {'trigger_event_id': trigger_event_id, 'events': []}
        backend = Backend()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
             patch.object(adapters, 'list_mirror_branches', return_value=['main']), \
             patch.object(adapters, '_post_api_trigger', return_value={'trigger_event_id': 'trigger-1'}) as trigger:
            result = adapters.run_promptless(
                prompt='Canonical task', mirror_repo='owned/task', token='controller',
                api_trigger_key='trigger-key', runtime_base_url='https://configured.example',
                log_dir=Path(tmp), backend=backend,
            )
            self.assertEqual(json.loads((Path(tmp)/'promptless_trace.json').read_text())['trigger_event_id'], 'trigger-1')
        self.assertTrue(result.ok)
        self.assertEqual(result.docs_pr_url, 'https://github.com/owned/task/pull/12')
        self.assertIn('Canonical task', trigger.call_args.args[2])
        self.assertEqual(trigger.call_args.kwargs['doc_collection_id'], 'owned-collection')

    def test_backend_contract_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, 'ensure_collection'):
            load_promptless_backend(object())

    def test_verdict_does_not_accept_invented_pr(self):
        output = subprocess.CompletedProcess([], 0, '[]', '')
        with patch.object(adapters.subprocess, 'run', return_value=output):
            verdict, url = adapters.extract_verdict_from_repo(
                'owned/task', 'token', agent_text='DOCS_PR_URL: https://github.com/other/task/pull/1')
        self.assertIsNone(url)
        self.assertTrue(verdict.startswith('UNKNOWN:'))

    def test_brokered_noop_marker_and_patch_intent(self):
        self.assertEqual(adapters.extract_brokered_verdict('NO_DOC_CHANGES_NEEDED')[0], 'NO_DOC_CHANGES_NEEDED')
        # Bare claims of a PR cannot substitute for controller patch capture.
        self.assertIsNone(adapters.extract_brokered_verdict('DOCS_PR_URL: https://github.com/owned/task/pull/3')[1])

    def test_container_isolation_uses_only_explicit_credential_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clone = root/'checkout'; clone.mkdir()
            auth = root/'codex-auth.json'; auth.write_text('{"tokens": {}}')
            configured = {'DOCBENCH_REQUIRE_AGENT_FS_ISOLATION': '1',
                          'DOCBENCH_AGENT_SANDBOX_IMAGE': 'operator-image:1',
                          'DOCBENCH_AGENT_SANDBOX_NETWORK': 'operator-internal',
                          'DOCBENCH_AGENT_EGRESS_PROXY': 'http://operator-proxy:3128',
                          'DOCBENCH_CODEX_AUTH_FILE': str(auth)}
            with patch.dict(os.environ, configured, clear=True), patch.object(adapters.sys, 'platform', 'linux'), \
                 patch.object(adapters.shutil, 'which', side_effect=lambda name: '/usr/bin/'+name):
                command, env, safe_home = adapters.isolate_local_agent_process(
                    ['/usr/bin/codex', 'exec', 'task'], {'GH_TOKEN': 'private-controller-token'}, clone)
            self.assertIn('--read-only', command)
            self.assertIn('operator-internal', command)
            self.assertIn('operator-image:1', command)
            self.assertNotIn('GH_TOKEN', env)
            self.assertEqual((safe_home/'.codex/auth.json').read_text(), auth.read_text())
            self.assertFalse((safe_home/'.claude/.credentials.json').exists())
            self.assertNotIn('private-controller-token', (safe_home.parent/'container.env').read_text())

    def test_unsealed_and_nonlinux_execution_fail_closed(self):
        with patch.dict(os.environ, {'DOCBENCH_REQUIRE_AGENT_FS_ISOLATION': '0'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'requires filesystem isolation'):
                adapters.isolate_local_agent_process(['codex'], {}, Path('.'))
        with patch.dict(os.environ, {}, clear=True), patch.object(adapters.sys, 'platform', 'darwin'):
            with self.assertRaisesRegex(RuntimeError, 'requires Linux'):
                adapters.isolate_local_agent_process(['codex'], {}, Path('.'))

    def test_codex_refresh_cannot_replace_another_configured_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); home = root/'isolated'; (home/'.codex').mkdir(parents=True)
            tokens = {'access_token':'access','account_id':'new','id_token':'id','refresh_token':'refresh'}
            (home/'.codex/auth.json').write_text(json.dumps({'tokens':tokens}))
            destination = root/'provided-auth.json'
            destination.write_text(json.dumps({'tokens':{'account_id':'existing'}}))
            with patch.dict(os.environ, {'DOCBENCH_CODEX_AUTH_FILE':str(destination)}, clear=True):
                self.assertFalse(adapters._persist_refreshed_codex_auth(home))
            self.assertEqual(json.loads(destination.read_text())['tokens']['account_id'], 'existing')


if __name__ == '__main__':
    unittest.main()
