"""PR #240 regressions: synthetic text only, source and generated medicines."""
import importlib.util
import hashlib
from pathlib import Path
import re
import sys

import pytest

SPEC = Path(__file__).resolve().parents[2] / "spec_v1"
sys.path.insert(0, str(SPEC))
import clean as c

spec = importlib.util.spec_from_file_location('pr240_review', SPEC / 'build/integrate_indic.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
ROWS = {row['id']: row for row in review.read_source()}


def test_ready_source_matches_the_pinned_upstream_blob():
    source = (SPEC / 'sheets/medicine_indic.md').read_text(encoding='utf-8').split('\n\n', 1)[1]
    source = re.sub(r'\n\n<!-- INDIC SIBLING POLICY -->.*?<!-- END INDIC SIBLING POLICY -->\n\n',
                    '\n\n', source, count=1, flags=re.S).encode('utf-8')
    blob = hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
    assert blob == 'f972241202febc587a95953f27397721669e6243'


def document(row, text, **extra):
    return dict(doc_id='synthetic-pr240', text=text, pack_id=row['pack'],
                lang_label=row['lang'], url='') | extra


def functions(row):
    result = [row['rule']]
    if row['id'] in c._RULE:
        result.append(c._RULE[row['id']]())
    return result


@pytest.mark.parametrize('rid', [rid for rid, row in ROWS.items() if row['action'] == 'keep'])
def test_keep_wrappers_never_call_the_author_or_change_text(rid, monkeypatch):
    row = ROWS[rid]
    for fn in functions(row):
        def destructive_author(*args):
            pytest.fail('a label-only wrapper called its author function')
        monkeypatch.setitem(fn.__globals__, '_AUTHOR_RULE', destructive_author)
        for text in ['', 'English summary\nहिन्दी पंक्ति\n', 'RSS feed\n- #news\n- #update\n',
                     review.SENTENCES[row['pack'].split('-')[1]]]:
            assert fn(document(row, text)) == ('keep', text)
    if row['code'] == 'D3':
        assert row['action'] == 'keep'


def test_all_d3_medicines_are_label_only():
    rows = [row for row in ROWS.values() if row['code'] == 'D3']
    assert len(rows) == 22
    assert {row['action'] for row in rows} == {'keep'}
    assert 'def main(' not in ROWS['DB-D3-008']['block']


@pytest.mark.parametrize('rid', sorted(ROWS))
def test_visarga_and_its_spacing_survive_every_retained_rule(rid):
    row = ROWS[rid]
    lang = row['pack'].split('-')[1]
    sign, word = review.VISARGA[lang], review.SENTENCES[lang].split()[0]
    for gap in ['', ' ']:
        context = word + gap + sign + ' '
        text = (review.SENTENCES[lang] + ' ') * 2 + context + review.SENTENCES[lang]
        control = text.replace(context, word + gap + ': ')
        for fn in functions(row):
            action, new = fn(document(row, text))
            if action == 'drop':
                # A legitimate whole-document drop must not depend on visarga.
                assert fn(document(row, control))[0] == 'drop'
                continue
            assert new.count(sign) == text.count(sign)
            assert context in new
            colon_action, colon_text = fn(document(row, control))
            if colon_action != 'drop':
                assert colon_text.count(sign) == control.count(sign)


@pytest.mark.parametrize('pack', sorted(pack for pack, family in c._FAMILY.items() if family == 'indic'))
def test_visarga_survives_the_complete_receiving_pipeline(pack):
    lang = pack.split('-')[1]
    for gap in ['', ' ']:
        context = review.SENTENCES[lang].split()[0] + gap + review.VISARGA[lang] + ' '
        text = (review.SENTENCES[lang] + ' ') * 2 + context + review.SENTENCES[lang]
        # A receiving-only pack need not have a native source card.
        action, new = c.clean(dict(doc_id='synthetic-receiver-visarga', text=text,
                                  url='', pack_id=pack, lang_label=lang))
        assert action != 'drop'
        assert context in new
        assert new.count(review.VISARGA[lang]) == text.count(review.VISARGA[lang])


def test_telugu_d4_uses_url_and_narrowed_thresholds():
    row = ROWS['DB-D4-011']
    text = review.SENTENCES['te']
    for fn in functions(row):
        for url in ['', 'https://te.wikipedia.org/a', 'https://te.m.wikipedia.org/a',
                    'https://te.vikaspedia.in/a', 'https://example.org/a']:
            assert fn(document(row, text, url=url)) == ('keep', text)
        assert fn(document(row, text, url='https://te.example.org/a')) == ('drop', text)
        for marks, words, action in [(5, 10, 'keep'), (6, 200, 'keep'), (6, 100, 'drop')]:
            probe = ' '.join(['మరియు'] * marks + ['తెలుగు'] * (words - marks))
            assert fn(document(row, probe))[0] == action


def test_hindi_nfc_nukta_conjunct_is_not_mojibake():
    row = ROWS['DB-G1-012']
    assert row['action'] == 'drop'
    assert 'MOJI' not in row['block']
    text = 'हमें ज़्यादा जानकारी चाहिए।\n'
    for fn in functions(row):
        assert fn(document(row, text)) == ('keep', text)


def test_retired_code_is_parsed_without_execution(tmp_path, monkeypatch):
    archived = review.read_retired()[0]
    body = archived['body'].replace('```python deferred\n',
                                   '```python deferred\nraise AssertionError("retired code ran")\n')
    (tmp_path / 'sheets').mkdir()
    (tmp_path / 'sheets/medicine_indic_retired.md').write_text(body, encoding='utf-8')
    monkeypatch.setattr(review, 'SPEC', tmp_path)
    rows = review.read_retired()
    assert len(rows) == 1 and rows[0]['id'] == archived['id']
    assert 'rule' not in rows[0]


def test_synthetic_review_rejects_joiner_addition():
    def medicine(doc):
        text = doc['text'].replace('नदी', 'न\u200dदी')
        return ('fix' if text != doc['text'] else 'keep', text)
    assert any('joiner count' in issue for issue in review.synthetic_check(medicine, 'ind-hi-fw2', 'hi'))
