"""Indic integration contracts. Every text is synthetic; no pack data is published."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = Path(__file__).resolve().parents[2] / "spec_v1"
sys.path.insert(0, str(SPEC))
import clean as c

spec = importlib.util.spec_from_file_location('indic_review', SPEC / 'build/integrate_indic.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
SOURCE = review.read_source()
SELECTED = [row for row in SOURCE if row['id'] in c._RULE]
DECISIONS = json.loads((SPEC.parent / 'results/spec_v1/cleaner/indic_integration.json').read_text(encoding='utf-8'))


def document(row, text):
    return dict(doc_id='synthetic-review', text=text, url='', pack_id=row['pack'], lang_label=row['lang'])


def test_inventory_and_overlap_decisions_are_explicit():
    assert len(SOURCE) == DECISIONS['ready_count'] == 260
    assert len(DECISIONS['overlaps']) == 74
    retired = review.read_retired()
    assert len(retired) == DECISIONS['retired_count'] == 10
    assert len(DECISIONS['rules']) == 270
    for row in retired:
        assert DECISIONS['rules'][row['id']]['retired'] is True
        assert c.DEFERRED_RULES[row['id']] == row['reason']
    assert not review.V1_EXCLUSIONS & c._RULE.keys()
    assert not review.V1_EXCLUSIONS & c.SCOPE.keys()
    for row in SOURCE:
        decision = DECISIONS['rules'][row['id']]
        assert (row['id'] in c._RULE) == (decision['status'] == 'CANDIDATE')
        if decision['status'] == 'DEFERRED':
            assert c.DEFERRED_RULES[row['id']] == decision['reason']
    groups = [(row['pack'], row['code']) for row in SELECTED]
    assert len(groups) == len(set(groups))
    assert list(c.RULES)[-1] == 'F1'
    assert {'D2', 'D5', 'F3', 'G1', 'G2', 'G4', 'H3'} <= c.RULES.keys()


@pytest.mark.parametrize('row', SELECTED, ids=lambda row: row['id'])
def test_factory_preserves_source_behavior_and_inputs(row):
    medicine = c._RULE[row['id']]()
    for text in review.probes(row['pack']).values():
        doc = document(row, text)
        original = dict(doc)
        actual = medicine(doc)
        assert actual == row['rule'](doc)
        assert actual[0] in {'keep', 'fix', 'drop'} and isinstance(actual[1], str)
        assert doc == original
        assert actual[0] != 'drop'
        assert all(actual[1].count(ch) == text.count(ch) for ch in ('\u200c', '\u200d'))
        assert medicine(dict(doc, text=actual[1])) == ('keep', actual[1])
        for pack in ('code-cpp', 'math-finemath', 'sci-finepdfs', 'en-dclm', 'ind-unknown', None):
            assert medicine(dict(doc, pack_id=pack)) == ('keep', text)


@pytest.mark.parametrize('pack', sorted({row['pack'] for row in SELECTED}))
def test_combined_pipeline_unicode_and_idempotence(pack):
    row = next(row for row in SELECTED if row['pack'] == pack)
    assert c._FAMILY[pack] == 'indic'
    assert pack in c.SCOPE['f1_rule.py']  # the shared F1's Indic path replaces the source F1 rules
    for text in review.probes(pack).values():
        doc = document(row, text)
        action, new = c.clean(doc)
        assert action != 'drop'
        assert all(new.count(ch) == text.count(ch) for ch in ('\u200c', '\u200d'))
        assert c.clean(dict(doc, text=new)) == ('keep', new)


def test_known_failure_and_benchmark_candidates_are_excluded(monkeypatch):
    monkeypatch.delenv('ITIHASA_TEST_SN', raising=False)
    for rid in ('DB-F2-001', 'DB-F2-002', 'DB-C3-003', 'DB-C3-005', 'DB-C3-009', 'DB-C3-012'):
        assert rid in c.DEFERRED_RULES
        assert rid not in c.SCOPE and rid not in c._RULE
    assert 'F2' not in c.RULES  # no keep-only detector masquerades as benchmark remediation


def test_indic_normalization_and_tag_independent_dispatch():
    row = dict(doc_id='synthetic', text='नदीचे पाणी\r\nशेतीसाठी वापरतात।\n', url='',
               pack_id='ind-mr-sangraha-v', lang_label='mr')
    tagged = dict(row, lang_label='mar_Deva')
    assert c.clean(row) == c.clean(tagged)


def test_unsupported_indic_pack_passes_through_but_is_not_registered():
    doc = dict(doc_id='synthetic', text='সাধারণ লেখা।', url='', pack_id='ind-ml-fw2', lang_label='ml')
    assert doc['pack_id'] not in c._FAMILY
    assert c.clean(doc) == ('keep', doc['text'])


def test_builder_does_not_silently_skip_a_bad_heading(tmp_path):
    source = (SPEC / 'sheets/specv1.md').read_text(encoding='utf-8')
    source = source.replace('— indic / A3 / `DB-A3-001`', '— indic / A3 / `XX-A3-001`', 1)
    sheet = tmp_path / 'specv1.md'
    sheet.write_text(source, encoding='utf-8', newline='\n')
    dest = tmp_path / 'clean.py'
    result = subprocess.run([sys.executable, '-I', str(SPEC / 'build/build_clean.py'), '--sheet', str(sheet), '--out', str(dest)], capture_output=True)
    assert result.returncode != 0 and b'unsupported medicine heading' in result.stderr
    assert not dest.exists()


def test_real_normalization_trigger_and_joiner_exception():
    row = dict(doc_id='synthetic', text='ত্ ', url='', pack_id='ind-bn-fw2', lang_label='bn')
    assert c.a5(row) == ('fix', 'ৎ ')
    for joiner in ('\u200c', '\u200d'):
        text = 'ত্' + joiner
        assert c.a5(dict(row, text=text)) == ('keep', text)


def test_review_falls_back_after_an_ineligible_overlap_candidate():
    rows = [{'id': 'first', 'pack': 'pack', 'code': 'C3'}, {'id': 'second', 'pack': 'pack', 'code': 'C3'}]
    decisions = {'first': {'review_issues': ['not idempotent']}, 'second': {'review_issues': []}}
    review.choose(rows, decisions)
    assert decisions['first']['status'] == 'DEFERRED'
    assert decisions['second']['status'] == 'CANDIDATE'


def test_review_detects_a_cross_rule_second_pass_action():
    def first(doc):
        text = doc['text'].replace('second', 'finished')
        return ('fix' if text != doc['text'] else 'keep', text)

    def second(doc):
        text = doc['text'].replace('start', 'second')
        return ('fix' if text != doc['text'] else 'keep', text)

    doc = dict(doc_id='synthetic', text='start', url='', pack_id='pack', lang_label='en')
    assert review.interaction_check([('first', first), ('second', second)], {'synthetic': doc}) == {'first': ['synthetic']}


@pytest.mark.parametrize('own_pack_inputs', ['missing_gold', 'empty_gold', 'missing_pack', 'valid'])
def test_integration_requires_nonempty_own_pack_gold(tmp_path, monkeypatch, own_pack_inputs):
    import pyarrow as pa
    import pyarrow.parquet as pq

    local = tmp_path / 'spec_v1'
    for folder in ('sheets', 'build'):
        (local / folder).mkdir(parents=True)
    (tmp_path / 'results/spec_v1/cleaner').mkdir(parents=True)
    packs, gold = tmp_path / 'packs', tmp_path / 'gold'
    packs.mkdir()
    gold.mkdir()
    # Exercise real source cards and the complete report/sheet writer in isolation.
    # Hindi HPLT has valid gold; it must not substitute for Hindi FW2's own gold.
    control, target = [next(row for row in SELECTED if row['pack'] == pack)
                       for pack in ('ind-hi-hplt', 'ind-hi-fw2')]
    (local / 'sheets/medicine_indic.md').write_text(
        '\n\n'.join(row['body'] for row in (control, target)) + '\n', encoding='utf-8')
    (local / 'sheets/specv1.md').write_bytes((SPEC / 'sheets/specv1.md').read_bytes())
    (local / 'build/extend_indic_siblings.py').write_bytes((SPEC / 'build/extend_indic_siblings.py').read_bytes())
    (local / 'build/f1_rule.py').write_bytes((SPEC / 'build/f1_rule.py').read_bytes())
    for row in (control, target):
        doc = document(row, review.probes(row['pack'])['prose'])
        pq.write_table(pa.Table.from_pylist([doc]), packs / (row['pack'] + '.parquet'))
        (gold / (row['pack'] + '_gold.csv')).write_text(
            'doc_id,reader_1,reader_2\nsynthetic-review,reader-one,reader-two\n', encoding='utf-8')
    target_gold = gold / (target['pack'] + '_gold.csv')
    if own_pack_inputs == 'missing_gold':
        target_gold.unlink()
    elif own_pack_inputs == 'empty_gold':
        target_gold.write_text('doc_id,reader_1,reader_2\n', encoding='utf-8')
    elif own_pack_inputs == 'missing_pack':
        (packs / (target['pack'] + '.parquet')).unlink()
    monkeypatch.setattr(review, 'SPEC', local)

    report = review.integrate(packs, gold)

    assert report['rules'][control['id']]['status'] == 'CANDIDATE'
    assert report['gold'][control['pack']]['result'] == 'PASS'
    decision = report['rules'][target['id']]
    if own_pack_inputs == 'valid':
        assert decision['status'] == 'CANDIDATE'
        assert decision['gold']['result'] == 'PASS'
    else:
        assert decision['status'] == 'DEFERRED'
        assert decision['gold']['result'] == 'NOT TESTABLE'
        evidence = report['gold'][target['pack']]
        assert evidence['result'] == 'NOT TESTABLE'
        assert evidence['reason'] in decision['reason']
        assert evidence['reason'] == decision['gold']['reason']
        if own_pack_inputs == 'empty_gold':
            assert evidence['reason'] == 'Gold set empty, not testable'
        else:
            assert ('gold' if own_pack_inputs == 'missing_gold' else 'pack') in evidence['missing']
        sheet = (local / 'sheets/specv1.md').read_text(encoding='utf-8')
        card = sheet.split(f'/ `{target["id"]}`\n', 1)[1]
        assert '```python deferred\n' in card
        assert 'STATUS = "DEFERRED"' in card
    assert json.loads((tmp_path / 'results/spec_v1/cleaner/indic_integration.json').read_text(encoding='utf-8')) == report
