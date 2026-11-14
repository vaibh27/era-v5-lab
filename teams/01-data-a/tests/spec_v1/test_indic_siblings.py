"""Receiving-pack scope and build contracts for Indic sibling reuse. Synthetic text only."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SPEC = Path(__file__).resolve().parents[2] / "spec_v1"
sys.path.insert(0, str(SPEC))
import clean as c


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


extension = module('sibling_extension_test', SPEC / 'build/extend_indic_siblings.py')
review = module('sibling_source_test', SPEC / 'build/integrate_indic.py')
ROWS = {row['id']: row for row in review.read_source()}
REPORT = json.loads((SPEC.parent / 'results/spec_v1/cleaner/indic_sibling_scopes.json').read_text(encoding='utf-8'))
BORROWED = [(pack, rid) for pack, result in REPORT['targets'].items() for rid in result['accepted']]


def test_policy_is_declared_in_source_and_native_ids_are_preserved():
    assert extension.policy_from_source() == REPORT['policy']
    assert len(ROWS) == 260
    for rid, packs in REPORT['rule_scopes'].items():
        assert c.SCOPE[rid] == frozenset(packs)
        assert ROWS[rid]['pack'] in packs
    assert {'ind-bn-sangraha-v', 'ind-hi-indiccorp', 'ind-hi-sangraha-s', 'ind-te-sangraha-v'} <= c._FAMILY.keys()
    indic = {pack for pack, family in c._FAMILY.items() if family == 'indic'}
    assert REPORT['privacy_missing_packs'] == []
    assert set(REPORT['shared_f1_packs']) == indic


def test_one_selected_rule_per_receiving_pack_and_disease():
    selected = [(pack, rid.split('-')[1]) for rid, packs in c.SCOPE.items() if rid.startswith('DB-') for pack in packs]
    assert len(selected) == len(set(selected))
    for pack, entry in REPORT['targets'].items():
        native_codes = {rid.split('-')[1] for rid in entry['native_rules']}
        assert not native_codes & {rid.split('-')[1] for rid in entry['accepted']}


@pytest.mark.parametrize('pack,rid', BORROWED, ids=[pack + '-' + rid for pack, rid in BORROWED])
def test_borrowed_callable_has_the_tested_scope_and_source_behavior(pack, rid):
    row = ROWS[rid]
    assert pack.split('-')[1] == row['pack'].split('-')[1]
    entry = REPORT['targets'][pack]
    attempt = next(a for a in entry['attempts'] if a['rule_id'] == rid and a['result'] == 'ACCEPTED')
    assert attempt['gold']['result'] == attempt['combined_gold']['result'] == 'PASS'
    assert entry['gold_n'] > 0
    fn = c._RULE[rid]()
    original = extension.target_rule(row, pack)
    for text in list(review.probes(pack).values()) + ['<p>' + review.SENTENCES[pack.split('-')[1]] + '</p>']:
        doc = dict(doc_id='synthetic', text=text, url='', lang_label=pack.split('-')[1], pack_id=pack)
        assert fn(doc) == original(doc)
        assert fn(dict(doc, pack_id='ind-unknown')) == ('keep', text)


def test_empty_or_missing_gold_never_creates_a_pipeline():
    for pack in ('ind-ta-fw2', 'ind-ta-sangraha-s'):
        assert REPORT['targets'][pack]['result'] == 'NOT TESTABLE'
        assert pack not in c._FAMILY
    for pack in ('ind-ml-fw2', 'ind-or-fw2'):
        assert REPORT['targets'][pack]['result'] == 'NO SAME-LANGUAGE DONOR'
        assert pack not in c._FAMILY


def test_new_hindi_sangraha_v_registration_requires_validated_receiving_scope():
    pack = 'ind-hi-sangraha-v'
    entry = REPORT['targets'][pack]
    assert entry['result'] == entry['combined_gold']['result'] == 'PASS'
    assert entry['gold_n'] > 0 and entry['accepted']
    assert c._FAMILY[pack] == 'indic' and pack in c.SCOPE['f1_rule.py']
    for text in review.probes(pack).values():
        outcomes = []
        for lang in ('hi', 'hin_Deva'):
            row = dict(doc_id='synthetic-hindi-receiver', text=text, url='', pack_id=pack, lang_label=lang)
            action, new = c.clean(row)
            assert action != 'drop'
            assert c.clean(dict(row, text=new)) == ('keep', new)
            outcomes.append((action, new))
        assert outcomes[0] == outcomes[1]


def test_rejected_combined_harm_scopes_are_not_emitted():
    failures = []
    for pack, entry in REPORT['targets'].items():
        for attempt in entry['attempts']:
            if attempt.get('combined_gold', {}).get('result') in {'FAIL', 'ERROR'}:
                failures.append((pack, attempt['rule_id']))
                assert attempt['result'] == 'REJECTED'
                assert pack not in c.SCOPE.get(attempt['rule_id'], frozenset())
                assert ROWS[attempt['rule_id']]['pack'] in c.SCOPE.get(attempt['rule_id'], frozenset())
    # Rebuilt Marathi gold exposes individually passing changes whose union exceeds the limit.
    assert ('ind-mr-fw2', 'DB-C1-012') in failures
    assert 'ind-mr-fw2' in c.SCOPE['DB-C3-024']  # preserve the native rule


def test_sibling_rule_factory_refuses_a_cross_language_target():
    with pytest.raises(ValueError, match='cross-language'):
        extension.target_rule(ROWS['DB-A5-001'], 'ind-hi-fw2')


def test_bengali_sibling_receives_the_real_normalization_fix():
    doc = dict(doc_id='synthetic', text='ত্ ', url='', lang_label='bn', pack_id='ind-bn-sangraha-v')
    assert c.a5(doc) == ('fix', 'ৎ ')
    assert 'ind-bn-sangraha-v' in c.SCOPE['f1_rule.py']  # every Indic pack gets the shared F1


def test_receiving_pipeline_checks_include_shared_f1_and_keep_pack_guards():
    pipeline = extension.compose([], set(), {}, review.shared_f1_rule(), 'ind-hi-fw2')
    doc = dict(doc_id='synthetic', text='संपर्क करें: 9876543210', url='',
               lang_label='hin_Deva', pack_id='ind-hi-fw2')
    assert pipeline(doc) == ('fix', 'संपर्क करें: <PHONE>')
    assert pipeline(dict(doc, pack_id='ind-unknown')) == ('keep', doc['text'])
    assert pipeline(dict(doc, text='संपर्क करें: <PHONE>')) == ('keep', 'संपर्क करें: <PHONE>')


@pytest.mark.parametrize('rid,own', [('DB-D4-011', 'ind-te-fw2'), ('DB-D5-001', 'ind-te-sangraha-s')])
def test_telugu_review_restrictions_override_fallback_borrowing(rid, own):
    assert c.SCOPE.get(rid, frozenset()) <= {own}
    assert REPORT['rule_scopes'].get(rid, []) in ([], [own])
    if rid not in c._RULE:
        assert rid in c.DEFERRED_RULES  # a ready source may still fail integration
    own_rule = extension.target_rule(ROWS[rid], own)
    for pack in ('ind-te-fw2', 'ind-te-sangraha-s', 'ind-te-sangraha-v'):
        if pack == own:
            continue
        # The restriction applies even when a receiving pack has no native rule.
        assert not extension.scope_allowed(ROWS[rid], pack, REPORT['policy'])
        with pytest.raises(ValueError, match='scope excluded'):
            extension.target_rule(ROWS[rid], pack)
        doc = dict(doc_id='synthetic', text='మరియు ' * 30, url='https://te.example.org/',
                   lang_label='tel_Telu', pack_id=pack)
        assert own_rule(doc) == ('keep', doc['text'])
        if rid in c._RULE:
            assert c._RULE[rid]()(doc) == ('keep', doc['text'])
    assert all(rid not in entry['accepted'] for pack, entry in REPORT['targets'].items() if pack != own)
    assert REPORT['policy']['allowed_rule_packs'][rid] == [own]


def test_telugu_sangraha_v_has_native_d4_and_d5():
    entry = REPORT['targets']['ind-te-sangraha-v']
    assert {'DB-D4-016', 'DB-D5-004'} <= set(entry['native_rules'])
    assert c.SCOPE['DB-D4-016'] >= {'ind-te-sangraha-v'}
    assert c.SCOPE['DB-D5-004'] >= {'ind-te-sangraha-v'}
