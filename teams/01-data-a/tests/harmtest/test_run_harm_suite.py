"""Regressions for Spec-v1-aware harm-suite coverage and fail-closed reporting."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent.parent / 'harmtest' / 'run_harm_suite.py'
spec = importlib.util.spec_from_file_location('run_harm_suite', MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def digest(data):
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    team = tmp_path / 'teams' / '01-data-a'
    team.mkdir(parents=True)
    monkeypatch.setattr(runner, 'TEAM', team)
    (team / 'harmtest').mkdir()
    (team / 'harmtest' / 'harm_test.py').write_text('')
    gold = team / 'gold'
    gold.mkdir()
    (gold / 'g1_pack_ids.txt').write_text('code-go\ncode-python\nmath-megamath\n')
    originals = tmp_path / 'packs'
    originals.mkdir()
    for pack in ('code-go', 'code-python'):
        (originals / (pack + '.parquet')).write_bytes(b'placeholder-' + pack.encode())
        (gold / (pack + '_gold.csv')).write_text('doc_id,reader_1,reader_2\nx,a,b\n')
    inv = []
    for pack, handle, code, state in [
        ('code-go', 'alice', 'B3', 'BLOCKED'),
        ('code-go', 'bob', 'B3', 'BLOCKED'),
        ('code-python', 'carol', 'A1', 'CANDIDATE'),
    ]:
        path = team / 'medicines' / 'submissions' / pack / handle / 'medicine.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f'# Medicine: {pack} / {code} / {handle}\n'
        path.write_text(content, encoding='utf-8', newline='')  # exact bytes: the spec stores their SHA-256 (Windows would write CRLF)
        inv.append((pack, handle, digest(content), code, state))
    lines = ['# G1 Data A — Spec v1: Parent Medicine (DRAFT)',
             '> **Status:** NOT APPROVED / NOT FROZEN',
             '> **G2 Indic:** PENDING',
             'Found **3 medicine files**, **3 proposed sections**, **2 pack/disease groups**.',
             '| Pack | Contributor | Medicine sections | State | Source SHA-256 |',
             '|---|---|---:|---|---|']
    for pack, handle, h, code, state in inv:
        lines.append(f'| {pack} | {handle} | 1 | MEDICINES | `{h}` |')
    lines += ['| Pack | Disease | Proposals | Decision | Resolution |',
              '|---|---|---:|---|---|',
              '| code-go | B3 | 2 | BLOCKED | needs review |']
    for i, (pack, handle, h, code, state) in enumerate(inv, 1):
        lines.append(f'### Medicine {i}: {pack} / {code} · {handle}\n\n> **Status:** {state}; not approved.')
    path = team / 'spec' / 'spec_v1.md'
    path.parent.mkdir()
    path.write_text('\n\n'.join(lines), encoding='utf-8')  # '—' and '·': not cp1252 on Windows
    registry = team / 'rules' / 'registry.json'
    registry.parent.mkdir()
    rules = [
        {'id': 'DA-B3-001', 'pack': 'code-go', 'handle': 'alice', 'code': 'B3', 'file': 'code-go/alice.py', 'withdrawn': False},
        {'id': 'DA-B3-002', 'pack': 'code-go', 'handle': 'bob', 'code': 'B3', 'file': 'code-go/bob.py', 'withdrawn': False},
        {'id': 'DA-A1-001', 'pack': 'code-python', 'handle': 'carol', 'code': 'A1', 'file': 'code-python/carol.py', 'withdrawn': False},
    ]
    registry.write_text(json.dumps({'rules': rules}))
    return SimpleNamespace(team=team, spec=path, registry=registry, gold=gold,
                           originals=originals, reports=tmp_path / 'reports', rules=rules)


def mock_success(monkeypatch, override=None):
    calls = []
    def fake(cmd, **kw):
        calls.append(cmd)
        def opt(name):
            return Path(cmd[cmd.index(name) + 1])
        rid = cmd[cmd.index('--rule') + 1]
        pack, gold, report = opt('--pack'), opt('--gold'), opt('--report')
        data = {'pack': pack.name, 'pack_sha256': runner.sha256(pack),
                'gold': gold.name, 'gold_sha256': runner.sha256(gold),
                'gold_count': 100, 'rules_file_sha256': 'abc123',
                'status': 'PASS', 'results': [{'rule': rid, 'result': 'PASS', 'changed': 1,
                                             'dropped': 0, 'errors': 0, 'affected_unique': 1,
                                             'percent': '1.000000'}]}
        if override:
            override(data)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(data))
        return SimpleNamespace(returncode={'PASS': 0, 'FAIL': 1, 'ERROR': 2}[data['status']])
    monkeypatch.setattr(runner.subprocess, 'run', fake)
    return calls


def execute(h):
    return runner.run(h.originals, h.reports, h.spec, h.registry,
                      h.team / 'gold' /'g1_pack_ids.txt', h.gold)


def records(h):
    return {row['pack_id']: row for row in json.loads((h.reports / 'g1_harm_summary.json').read_text())}


def test_parser_recognizes_conflicts_and_provenance(harness):
    s = runner.read_spec(harness.spec)
    assert len(s['inventory']) == 3
    assert sum(map(len, s['proposals'].values())) == 3
    assert ('code-go', 'B3') in s['conflicts']
    assert runner.source_drift(s, harness.team) == {}


def test_diagnostic_harm_pass_never_promotes_draft(harness, monkeypatch):
    calls = mock_success(monkeypatch)
    assert execute(harness) == 1
    r = records(harness)
    assert r['code-go']['status'] == 'BLOCKED_CONFLICT'
    assert r['code-python']['status'] == 'PENDING_APPROVAL'
    assert r['math-megamath']['status'] == 'NOT_EVALUATED'
    assert len(calls) == 3 and all('--rule' in c and '--rules' not in c for c in calls)
    assert all(x['harm_status'] == 'PASS' for x in r['code-go']['rules'])
    metadata = json.loads((harness.reports / 'g1_harm_metadata.json').read_text())
    assert metadata['combined_pipeline'] == 'NOT_RUN'
    assert metadata['g2_status'] == 'PENDING'
    assert metadata['release_readiness'] == 'INCOMPLETE'


def test_gold_harm_failure_reported(harness, monkeypatch):
    def fail(data):
        data['status'] = 'FAIL'
        data['results'][0].update(result='FAIL', changed=2, affected_unique=2, percent='2.000000')
    mock_success(monkeypatch, fail)
    assert execute(harness) == 1
    assert records(harness)['code-go']['status'] == 'HARM_FAIL'


def test_subprocess_report_cannot_be_reused(harness, monkeypatch):
    path = harness.reports / 'rules' / 'code-go_DA-B3-001_harm.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"status":"PASS"}')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    execute(harness)
    assert records(harness)['code-go']['status'] == 'ERROR'
    assert not path.exists()


def test_source_hash_drift_blocks_that_pack(harness, monkeypatch):
    calls = mock_success(monkeypatch)
    source = harness.team / 'medicines' / 'submissions' / 'code-go' / 'alice' / 'medicine.md'
    source.write_text('changed after Spec v1')
    execute(harness)
    assert records(harness)['code-go']['status'] == 'SPEC_STALE'
    assert len(calls) == 1


def test_unregistered_proposals_are_explicit_partial_coverage(harness, monkeypatch):
    calls = mock_success(monkeypatch)
    registry = json.loads(harness.registry.read_text())
    registry['rules'] = [x for x in registry['rules'] if x['handle'] != 'bob']
    harness.registry.write_text(json.dumps(registry))
    execute(harness)
    r = records(harness)['code-go']
    assert r['status'] == 'PARTIAL_COVERAGE'
    assert r['proposed_rules'] == 2 and r['mapped_rules'] == 1
    assert r['unextracted_proposals'] == [{'handle': 'bob', 'disease': 'B3', 'count': 1}]
    assert len(calls) == 2


def test_registry_rule_absent_from_spec_not_executed(harness, monkeypatch):
    calls = mock_success(monkeypatch)
    registry = json.loads(harness.registry.read_text())
    registry['rules'].append({'id':'DA-C5-001', 'pack':'code-go', 'handle':'nobody',
                              'code':'C5','file':'code-go/nobody.py','withdrawn':False})
    harness.registry.write_text(json.dumps(registry))
    execute(harness)
    assert len(calls) == 3
    md = json.loads((harness.reports / 'g1_harm_metadata.json').read_text())
    assert md['out_of_spec_registry_rule_ids'] == ['DA-C5-001']


def test_no_registered_rule_cannot_pass_even_if_inputs_exist(harness, monkeypatch):
    mock_success(monkeypatch)
    pack = 'math-megamath'
    (harness.originals / (pack + '.parquet')).write_bytes(b'placeholder')
    (harness.gold / (pack + '_gold.csv')).write_text('doc_id,reader_1,reader_2\nx,a,b\n')
    execute(harness)
    assert records(harness)['math-megamath']['status'] == 'PENDING_RULES'


def test_dry_run_reports_coverage_without_original_packs(harness, monkeypatch):
    calls = mock_success(monkeypatch)
    for path in harness.originals.iterdir():
        path.unlink()
    assert runner.run(harness.originals, harness.reports, harness.spec, harness.registry,
                      harness.team / 'gold' / 'g1_pack_ids.txt', harness.gold, dry_run=True) == 1
    assert not calls
    assert records(harness)['code-go']['status'] == 'DRY_RUN'
    assert len(records(harness)['code-go']['rules']) == 2


def test_report_hash_mismatch_does_not_pass(harness, monkeypatch):
    mock_success(monkeypatch, lambda d: d.update(gold_sha256='WRONG'))
    execute(harness)
    assert records(harness)['code-go']['status'] == 'ERROR'


def test_duplicate_registry_id_is_fatal(harness, monkeypatch):
    registry = json.loads(harness.registry.read_text())
    registry['rules'].append(dict(registry['rules'][0]))
    harness.registry.write_text(json.dumps(registry))
    assert execute(harness) == 2
    assert not (harness.reports / 'g1_harm_summary.json').exists()


def test_missing_source_in_spec_is_reported(harness, monkeypatch):
    mock_success(monkeypatch)
    missing = harness.team / 'medicines' / 'submissions' / 'code-python' / 'carol' / 'medicine.md'
    missing.unlink()
    execute(harness)
    assert records(harness)['code-python']['status'] == 'SPEC_STALE'


def test_real_spec_format_if_present():
    parent = HERE.parent / 'spec' / 'spec_v1.md'
    if not parent.is_file():
        pytest.skip('Run from checked-out repository to parse current parent spec')
    info = runner.read_spec(parent)
    assert info['inventory']
    assert info['proposals']


def test_json_paths_are_repo_relative(harness, monkeypatch):
    if harness.team.resolve().drive != runner.ROOT.resolve().drive:
        pytest.skip('Windows: the temp folder and the repository are on different drives, so no relative path exists')
    mock_success(monkeypatch)
    assert execute(harness) == 1
    metadata = json.loads((harness.reports / 'g1_harm_metadata.json').read_text())
    assert metadata['spec'] == runner.repo_relative(harness.spec)
    assert metadata['registry'] == runner.repo_relative(harness.registry)
    assert not Path(metadata['spec']).is_absolute()
    assert not Path(metadata['registry']).is_absolute()
    rows = records(harness)
    for record in rows.values():
        for rule in record['rules']:
            if 'report' in rule:
                expected = harness.reports / 'rules' / (record['pack_id'] + '_' + rule['id'] + '_harm.json')
                assert rule['report'] == runner.repo_relative(expected)
                assert not Path(rule['report']).is_absolute()
    # The original Parquet folder lives outside the repository.
    reason = rows['math-megamath']['reason']
    assert runner.repo_relative(harness.originals / 'math-megamath.parquet') in reason
    assert all(not Path(value).is_absolute() for value in reason.removeprefix('Missing input: ').split(', '))


def test_repo_relative_paths_include_outside_files(tmp_path):
    internal = runner.ROOT / 'teams' / '01-data-a' / 'spec' / 'spec_v1.md'
    assert runner.repo_relative(internal) == 'teams/01-data-a/spec/spec_v1.md'
    external = tmp_path / 'original-packs' / 'code-go.parquet'
    if external.resolve().drive != runner.ROOT.resolve().drive:
        pytest.skip('Windows: the temp folder and the repository are on different drives, so no relative path exists')
    rel = runner.repo_relative(external)
    assert not Path(rel).is_absolute()
    assert (runner.ROOT / rel).resolve() == external.resolve()
