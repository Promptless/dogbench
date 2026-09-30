"""Frozen source attestations retain the research base without weakening live checks."""
import pytest
from dogbench.sources import SourceAttestationError, attest_github_component

@pytest.mark.parametrize('frozen', [False, True])
def test_research_base_exception_is_explicit(monkeypatch, frozen):
    def github(path):
        if path.endswith('/pulls/7'):
            return {'base': {'sha': '3'*40}, 'head': {'sha': '2'*40}}
        if '/compare/' in path:
            return {'merge_base_commit': {'sha': '4'*40}}
        if '/files?' in path:
            return [{'filename': 'src/app.py'}]
        return []
    monkeypatch.setattr('dogbench.sources._github_json', github)
    kwargs=dict(source_url='https://github.com/example/project/pull/7', repo_url='https://github.com/example/project', base_sha='1'*40, head_sha='2'*40, paths=['src/app.py'], frozen_research_base=frozen)
    if not frozen:
        with pytest.raises(SourceAttestationError, match='not the PR range merge base'):
            attest_github_component(**kwargs)
        return
    result=attest_github_component(**kwargs)
    assert result['base_sha']=='1'*40
    assert result['github_merge_base_sha']=='4'*40
    assert result['base_policy']=='frozen_research_binding'
    with pytest.raises(SourceAttestationError, match='head SHA drifted'):
        attest_github_component(**{**kwargs, 'head_sha':'5'*40})
    with pytest.raises(SourceAttestationError, match='declared paths are absent'):
        attest_github_component(**{**kwargs, 'paths':['missing.py']})
