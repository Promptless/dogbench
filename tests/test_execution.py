from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dogbench import adapters, execution
from dogbench.adapters import AgentResult
from dogbench.cli import main
from dogbench.items import load_items
from dogbench.predictions import load_records, validate_predictions
from dogbench.workspaces import prepare_items


def git(repo, *args):
    return subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,check=True).stdout.strip()


@pytest.fixture
def prepared(tmp_path):
    repo=tmp_path/'source'
    repo.mkdir()
    git(repo,'init','--quiet','--initial-branch=main')
    git(repo,'config','user.name','Fixture')
    git(repo,'config','user.email','fixture@example.invalid')
    git(repo,'config','commit.gpgsign','false')
    (repo/'guide.md').write_text('Existing documentation.\n')
    git(repo,'add','.')
    git(repo,'commit','--quiet','-m','snapshot')
    base=git(repo,'rev-parse','HEAD')
    path=tmp_path/'items.jsonl'
    path.write_text(json.dumps({
        'instance_id':'example-project-pr1',
        'source_url':'https://github.com/example/project/pull/1',
        'context':'Determine whether the described internal cleanup needs user documentation.',
        'docs':{'repo_url':str(repo),'base_sha':base,'head_sha':None,'paths':[]},
        'code':None,
    })+'\n')
    items=load_items(path)
    output=tmp_path/'prepared'
    prepare_items(items,output,offline=True)
    return path,output,items


def fake_adapter(monkeypatch, *, edit=False, ok=True, trace=None, verdict='NO_DOC_CHANGES_NEEDED'):
    def run(**kwargs):
        assert execution.os.environ['DOCBENCH_REQUIRE_AGENT_FS_ISOLATION']=='1'
        assert execution.os.environ['DOCBENCH_PATCH_BROKER_MODE']=='1'
        assert git(kwargs['clone_dir'],'remote')==''
        assert kwargs['token']==''
        if edit:
            (kwargs['clone_dir']/'guide.md').write_text('Updated user documentation.\n')
        kwargs['log_dir'].mkdir(parents=True,exist_ok=True)
        (kwargs['log_dir']/'codex.events.jsonl').write_text(trace or '{"type":"turn.completed"}\n')
        return AgentResult('codex','gpt-fixture',ok,verdict,None,0.1,error=None if ok else 'process timeout')
    monkeypatch.setattr(adapters,'run_codex',run)
    monkeypatch.setattr(execution.sys,'platform','linux')


@pytest.mark.parametrize('edit,ok,decision',[(False,True,'abstain'),(True,True,'patch'),(True,False,'patch')])
def test_local_preserves_patch_verdict_and_recovery(tmp_path,prepared,monkeypatch,edit,ok,decision):
    items,workspaces,_=prepared
    fake_adapter(monkeypatch,edit=edit,ok=ok)
    summary=execution.run_prepared(items,workspaces,tmp_path/'run',agent='codex',model='gpt-fixture')
    predictions=load_records(Path(summary['predictions']))
    validate_predictions(predictions,require_complete=True,item_ids={'example-project-pr1'})
    assert predictions[0]['decision']==decision
    candidate=json.loads((tmp_path/'run/items/example-project-pr1/candidate.json').read_text())
    assert candidate['extras']['sandbox_attestation']['recovered_after_agent_termination']==(edit and not ok)
    for workspace in (workspaces/'workspaces').iterdir():
        assert git(workspace/'docs','status','--porcelain')==''
    traces=tmp_path/'run/traces/example-project-pr1'
    assert [p.name for p in traces.iterdir()]==['codex.events.jsonl']
    assert 'DOCBENCH_AGENT_TASK_DIR' not in execution.os.environ


def test_trace_gate_wins_over_patch_recovery(tmp_path,prepared,monkeypatch):
    items,workspaces,_=prepared
    trace=json.dumps({'type':'item.completed','item':{'command':'curl https://github.com/example/project/pull/1','status':'completed','aggregated_output':'reference response'}})+'\n'
    fake_adapter(monkeypatch,edit=True,ok=False,trace=trace)
    summary=execution.run_prepared(items,workspaces,tmp_path/'run',agent='codex',model='gpt-fixture')
    row=load_records(Path(summary['predictions']))[0]
    assert row['status']=='invalid'
    assert 'decision' not in row and 'patch' not in row
    assert summary['stopped']=='contamination gate requested a batch stop'
    candidate=json.loads((tmp_path/'run/items/example-project-pr1/candidate.json').read_text())
    assert candidate['docs_diff']
    assert candidate['extras']['recovered_after_agent_termination']['ok'] is False


def test_model_contract_failure_cannot_recover():
    result=AgentResult('claude','wrong-model',False,'UNKNOWN',None,0,error='model mismatch',extras={'candidate_model_contract':{'ok':False}})
    assert execution.recover_brokered_patch_result(result,'+ patch') is False
    assert result.ok is False


def test_linux_requirement_precedes_dispatch(tmp_path,prepared,monkeypatch):
    items,workspaces,_=prepared
    monkeypatch.setattr(execution.sys,'platform','darwin')
    with pytest.raises(execution.ExecutionConfigurationError,match='Linux'):
        execution.run_prepared(items,workspaces,tmp_path/'run',agent='codex',model='gpt-fixture')
    assert not (tmp_path/'run').exists()


def test_mock_cli_run_emits_valid_bundle(tmp_path,prepared,monkeypatch,capsys):
    items,workspaces,_=prepared
    fake_adapter(monkeypatch,edit=True)
    assert main(['run',str(items),'--prepared',str(workspaces),'--output',str(tmp_path/'run'),'--agent','codex','--model','gpt-fixture','--json'])==0
    summary=json.loads(capsys.readouterr().out)
    assert summary['complete']
    assert main(['validate',summary['predictions'],'--format-only','--json'])==0


def test_cloud_refuses_source_repository_before_any_remote_call(tmp_path,monkeypatch):
    item={'docs':{'repo_url':'https://github.com/example/project'},'code':None}
    monkeypatch.setattr(execution,'_gh_json',lambda *a:pytest.fail('no remote call permitted'))
    with pytest.raises(execution.ExecutionConfigurationError,match='differ'):
        execution._validate_cloud_input(item,tmp_path,{'mirror_repo':'example/project'},'dummy')


@pytest.mark.parametrize('agent',['devin','mintlify','promptless'])
def test_cloud_dispatch_is_explicit_and_uses_original_adapter(tmp_path,monkeypatch,agent):
    config={
        'mirror_repo':'fixture/mirror','base_branch':'sandbox-base','docs_work_branch':'sandbox-work',
        'base_sha':'1'*40,'clone_dir':str(tmp_path),'github_token_env':'FIXTURE_GH',
        'devin_api_key_env':'FIXTURE_PROVIDER','mintlify_token_env':'FIXTURE_PROVIDER',
        'promptless_api_trigger_key_env':'FIXTURE_PROVIDER','project_id':'fixture-project',
        'project_mirror_confirmed':True,'runtime_base_url':'https://runtime.example.invalid',
    }
    monkeypatch.setenv('FIXTURE_GH','fixture-github')
    monkeypatch.setenv('FIXTURE_PROVIDER','fixture-provider')
    monkeypatch.setattr(execution,'_validate_cloud_input',lambda *a:{'ok':True})
    monkeypatch.setattr(adapters,'list_mirror_pr_numbers',lambda *a:[])
    calls=[]
    def run(**kwargs):
        calls.append(kwargs)
        return AgentResult(agent,agent,True,'NO_DOC_CHANGES_NEEDED',None,0.1)
    monkeypatch.setattr(adapters,'run_'+agent,run)
    result,patch,prompt,attestation=execution._run_cloud_adapter(
        agent,agent,{'context':'Inspect the requested internal change.','code':None},tmp_path,tmp_path,config,60,
    )
    assert result.ok and patch=='' and attestation['ok']
    assert calls[0]['mirror_repo']=='fixture/mirror'
    assert calls[0]['timeout_s']==60
    assert 'NO_DOC_CHANGES_NEEDED' in prompt


def test_rate_limit_leaves_no_candidate(tmp_path,prepared,monkeypatch):
    items,workspaces,_=prepared
    monkeypatch.setattr(execution.sys,'platform','linux')
    def limited(**kwargs):
        return AgentResult('claude','claude-fixture',False,'UNKNOWN',None,0.1,error='CLAUDE_RATE_LIMIT: fixture')
    monkeypatch.setattr(adapters,'run_claude',limited)
    summary=execution.run_prepared(items,workspaces,tmp_path/'run',agent='claude',model='claude-fixture')
    assert summary['exit_code']==2 and not summary['complete']
    assert not (tmp_path/'run/items/example-project-pr1/candidate.json').exists()
    assert Path(summary['predictions']).read_text()==''


def test_non_stopping_trace_quarantine_still_exit_three(tmp_path,prepared,monkeypatch):
    items,workspaces,_=prepared
    # No exact PR route: source behavior classifies this as blocked policy access.
    trace=json.dumps({'type':'item.completed','item':{'command':'curl https://github.com/example/project','status':'completed','aggregated_output':'HTTP 403'}})+'\n'
    fake_adapter(monkeypatch,edit=True,trace=trace)
    summary=execution.run_prepared(items,workspaces,tmp_path/'run',agent='codex',model='gpt-fixture')
    assert summary['stopped'] is None and summary['exit_code']==3
    assert summary['contamination_count']==1


def test_same_lane_alias_is_excluded_by_default(tmp_path,prepared,monkeypatch):
    items,workspaces,_=prepared
    fake_adapter(monkeypatch,edit=True)
    summary=execution.run_prepared(items,workspaces,tmp_path/'run',agent='codex',model='gpt-fixture')
    candidate=json.loads((tmp_path/'run/items/example-project-pr1/candidate.json').read_text())
    assert candidate['candidate_key']=='codex'


@pytest.mark.parametrize('repair',['retarget','republish'])
def test_cloud_safe_recovery_precedes_patch_fetch(tmp_path,monkeypatch,repair):
    from dogbench import cloud_results
    monkeypatch.setenv('FIXTURE_GH','fixture-token')
    config={'mirror_repo':'fixture/mirror','github_token_env':'FIXTURE_GH','base_branch':'frozen-base','base_sha':'1'*40,'docs_work_branch':'work-branch'}
    result=AgentResult('mintlify','mintlify',True,'DOCS_PR_URL', 'https://github.com/fixture/mirror/pull/2',0.1)
    calls=[]
    def validate(**kwargs):
        calls.append(('validate',kwargs['pr_url']))
        n=sum(c[0]=='validate' for c in calls)
        return {'ok':n>1,'errors':[] if n>1 else [{'code':'base_ref_mismatch'}],'warnings':[],'actual':{}}
    monkeypatch.setattr(cloud_results,'validate_pr_sha_context',validate)
    def retarget(**kwargs):
        calls.append(('retarget',kwargs['pr_url']))
        return repair=='retarget'
    monkeypatch.setattr(cloud_results,'retarget_pr_to_expected_base_if_safe',retarget)
    def republish(**kwargs):
        calls.append(('republish',kwargs['pr_url']))
        return 'https://github.com/fixture/mirror/pull/3'
    monkeypatch.setattr(cloud_results,'republish_pr_after_protected_base_write_if_safe',republish)
    def diff(url,token):
        calls.append(('diff',url))
        return '+ retained exact agent edit\n'
    monkeypatch.setattr(execution,'fetch_pr_diff',diff)
    patch=execution._finalize_cloud_result(result,config,tmp_path)
    assert patch and result.ok
    assert calls[-1][0]=='diff'
    assert result.extras['sha_validation']['ok']
    if repair=='republish':
        assert result.docs_pr_url.endswith('/3')
        assert [c[0] for c in calls]==['validate','retarget','republish','validate','diff']
    else:
        assert [c[0] for c in calls]==['validate','retarget','validate','diff']


def test_frozen_prompt_binding_reaches_adapter_and_rejects_drift(tmp_path, prepared, monkeypatch):
    import hashlib
    from dogbench import research_inputs
    from dogbench.contamination import canonical_json_bytes, sha256_bytes
    from dogbench.verify import verify_output

    items_path, old_prepared, items = prepared
    item = items[0]
    old_workspace = next((old_prepared / 'workspaces').iterdir())
    prompt = 'Decide whether documentation needs updating. Preserve these exact bytes.\n'
    record = {
        'item_sha256': research_inputs.item_digest(item),
        'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'docs_base_tree': git(old_workspace / 'docs', 'rev-parse', 'HEAD^{tree}'),
        'code_base_tree': None, 'code_blob_sha256': {}, 'logical_input_sha256': 'frozen-fixture',
    }
    monkeypatch.setattr(research_inputs, 'input_registry', lambda: {'items': {item['instance_id']: record}})
    with pytest.raises(RuntimeError, match='fresh workspace'):
        verify_output(old_prepared)
    fresh = tmp_path / 'frozen-prepared'
    prepare_items(items, fresh, offline=True)
    assert verify_output(fresh)['verified'] == 1
    workspace = next((fresh / 'workspaces').iterdir())
    assert (workspace / 'TASK.md').read_bytes() == prompt.encode()
    fake_adapter(monkeypatch)
    adapter = adapters.run_codex

    def checked_adapter(**kwargs):
        assert kwargs['prompt'].encode() == prompt.encode()
        return adapter(**kwargs)

    monkeypatch.setattr(adapters, 'run_codex', checked_adapter)
    execution.run_prepared(items_path, fresh, tmp_path / 'frozen-run',
                           agent='codex', model='gpt-fixture')
    assert (tmp_path / 'frozen-run/items/example-project-pr1/logs/research_input_binding.json').is_file()

    # Recomputing the self-reported attestation cannot bless changed prompt bytes.
    (workspace / 'TASK.md').write_text('Changed prompt\n')
    att_path = next((fresh / 'attestations').glob('*.json'))
    att = json.loads(att_path.read_text())
    content = (workspace / 'TASK.md').read_bytes()
    att['artifacts']['TASK.md'] = {'sha256': sha256_bytes(content), 'bytes': len(content)}
    att.pop('logical_input_sha256')
    att['logical_input_sha256'] = sha256_bytes(canonical_json_bytes(att))
    att_path.write_text(json.dumps(att))
    with pytest.raises(ValueError, match='frozen research prompt'):
        research_inputs.local_execution_prompt(item, workspace)
    with pytest.raises(ValueError, match='frozen research prompt'):
        verify_output(fresh)


def test_shipped_frozen_registry_and_item_identity():
    import hashlib
    from dogbench.research_inputs import input_registry, research_input, FrozenInputError
    root = Path(__file__).resolve().parents[1]
    registry = input_registry()
    reference = json.loads((root / 'data/release-reference.json').read_text())
    assert set(registry['items']) == set(reference['development_item_ids'])
    for record in registry['items'].values():
        assert hashlib.sha256(record['prompt'].encode()).hexdigest() == record['prompt_sha256']
        assert record['frozen_lane_verified']
    for item in load_items(root / 'data/sample-5/items.jsonl'):
        assert research_input(item, required=True)
        with pytest.raises(FrozenInputError, match='released item differs'):
            research_input({**item, 'context': item['context'] + ' changed'})


@pytest.mark.parametrize('drift', ['docs_tree', 'code_tree', 'code_blob', 'code_paths', 'prompt_hash'])
def test_frozen_workspace_checks_independent_reference(drift, tmp_path, monkeypatch):
    import hashlib
    from copy import deepcopy
    from dogbench import research_inputs

    workspace = tmp_path / 'workspace'
    for name in ('docs', 'code'):
        repo = workspace / name
        repo.mkdir(parents=True)
        git(repo, 'init', '--quiet', '--initial-branch=main')
        git(repo, 'config', 'user.name', 'Fixture')
        git(repo, 'config', 'user.email', 'fixture@example.invalid')
        git(repo, 'config', 'commit.gpgsign', 'false')
        (repo / 'guide.txt').write_text('before\n')
        git(repo, 'add', '.')
        git(repo, 'commit', '--quiet', '-m', 'snapshot')
    code = workspace / 'code'
    (code / 'guide.txt').write_text('after\n')
    git(code, 'add', '.')
    git(code, 'commit', '--quiet', '-m', 'change')
    item = {'instance_id': 'frozen-fixture'}
    record = {
        'item_sha256': research_inputs.item_digest(item), 'prompt': 'task\n',
        'prompt_sha256': hashlib.sha256(b'task\n').hexdigest(),
        'docs_base_tree': git(workspace / 'docs', 'rev-parse', 'HEAD^{tree}'),
        'code_base_tree': git(code, 'rev-parse', 'HEAD^^{tree}'),
        'code_blob_sha256': {'guide.txt': {
            'base_sha256': hashlib.sha256(b'before\n').hexdigest(),
            'head_sha256': hashlib.sha256(b'after\n').hexdigest(),
        }},
    }
    monkeypatch.setattr(research_inputs, 'input_registry', lambda: {'items': {'frozen-fixture': record}})
    research_inputs.verify_research_workspace(item, workspace, b'task\n')
    original = deepcopy(record)
    if drift == 'docs_tree':
        record['docs_base_tree'] = '0' * 40
    elif drift == 'code_tree':
        record['code_base_tree'] = '0' * 40
    elif drift == 'code_blob':
        record['code_blob_sha256']['guide.txt']['head_sha256'] = '0' * 64
    elif drift == 'code_paths':
        record['code_blob_sha256'] = {}
    else:
        record['prompt_sha256'] = '0' * 64
    with pytest.raises(research_inputs.FrozenInputError):
        research_inputs.verify_research_workspace(item, workspace, b'task\n')
    assert record != original
