"""Code speed changes must equal frozen PR34 rules on contacts, secrets and misses.

All inputs are synthetic; corpus equivalence/harm checks are recorded separately.
"""
import random
from pathlib import Path
import sys

import pytest

SPEC = Path(__file__).resolve().parents[2] / 'spec_v1'
sys.path.insert(0, str(SPEC))
from code_speed_reference import e3_reference, f1_hits_reference
import clean as c

parametrize = pytest.mark.parametrize
OLD_E3 = e3_reference()
NEW_E3 = c._RULE['DA-E3-004']()
PHONE = '212' + '-555-' + '0199'
EMAIL = 'reader' + '@' + 'private.example'
VALUE = 'A7b2C9d3E5f8G1h4'


def doc(text, pack='code-python', lang='python'):
    return dict(doc_id='synthetic', pack_id=pack, lang_label=lang, text=text, url='')


def secret_cases():
    cases = ['', 'No contacts or assignments here.', 'key, token and secret are ordinary words.',
             'KEY = "dummy_value_123456789"', 'new Pusher("' + 'abc123' * 4 + '")']
    formats = [('AIza', 'A', 35), ('AKIA', 'A', 16), ('gh' + 'p_', 'a', 30),
               ('github_' + 'pat_', 'a', 40), ('s' + 'k-', 'a', 20),
               ('rk_' + 'live_', 'a', 16), ('sk_' + 'live_', 'a', 16),
               ('xox' + 'b-', 'a', 10)]
    for prefix, char, size in formats:
        for n in (size - 1, size, size + 1):
            for left, right in [('', ''), ('é', ''), (' ', 'z'), ('\n', '\n')]:
                cases.append(left + prefix + char * n + right)
    cases += ['SG.' + 'a' * 16 + '.' + 'b' * 16,
              'eyJ' + 'a' * 10 + '.eyJ' + 'b' * 10 + '.' + 'c' * 10,
              '-----' + 'BEGIN PRIVATE KEY' + '-----\nsynthetic\n-----' + 'END PRIVATE KEY' + '-----']
    for name in ('secret', 'TOKEN', 'api_key', 'app-key', 'authKey', 'access_key',
                 'clientKey', 'private_key', 'password', 'passwd', 'pwd', 'key',
                 'prefixSecretSuffix', 'Key', 'ſecret', 'prıvate_Key'):
        for operator in ('=', ':', ':=', '=>'):
            for quote in ('"', "'", '`'):
                cases.append(name + operator + quote + VALUE + quote)
    cases += ['SECRET: &\'static str = "' + VALUE + '"', 'key: String = "' + VALUE + '"']
    for n in (15, 16, 17, 199, 200, 201):
        cases.append('key = "' + (VALUE * 13)[:n] + '"')
    return cases


@parametrize('text', secret_cases())
def test_secret_formats_boundaries_names_and_unicode_match_old_rule(text):
    assert NEW_E3(doc(text)) == OLD_E3(doc(text))


def test_secret_substitution_order_on_random_mixed_documents():
    rng = random.Random(34)
    pieces = secret_cases() + ['EXAMPLE_SECRET', '\n', ' ', 'identifier', 'KEY', 'gh', 'AIza']
    changed = unchanged = 0
    for _ in range(4000):
        text = ' '.join(rng.choice(pieces) for _ in range(rng.randint(1, 8)))
        row = doc(text)
        expected = OLD_E3(row)
        assert NEW_E3(row) == expected
        changed += expected[0] == 'fix'
        unchanged += expected[0] == 'keep'
    assert changed > 500 and unchanged > 50


@parametrize('pack', ['en-dclm', 'math-finemath', 'ind-hi-fw2', None])
def test_code_secret_scope_stays_unchanged(pack):
    text = 'secret = "' + VALUE + '"'
    assert NEW_E3(doc(text, pack=pack)) == OLD_E3(doc(text, pack=pack)) == ('keep', text)


def contact_cases():
    return ['', 'plain prose', 'company without a contact', 'identifier123', EMAIL,
            PHONE, 'Phone: ' + PHONE, '12 Cedar Road', EMAIL + ' ' + PHONE,
            'company ' + PHONE + ' ' + EMAIL, '12 Cedar Road ' + PHONE,
            PHONE.translate(str.maketrans('0123456789', '०१२३४५६७८९')),
            PHONE.translate(str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')),
            'Phone ²²²-555-0199', 'info' + '@' + 'private.example',
            'user' + '@' + 'example.com', '12 Clinic Road',
            ('x' * 4000) + ' ' + PHONE]


@parametrize('previous', ['', 'Company', 'University office', 'some prose', EMAIL])
@parametrize('line', contact_cases())
def test_contact_candidates_and_previous_line_organisation_match_old_rule(line, previous):
    assert c._f1_hits(line, previous) == f1_hits_reference(line, previous)


def test_contact_spans_match_old_rule_on_random_lines():
    rng = random.Random(35)
    pieces = contact_cases()[:-1] + ['inc', 'support', '\t', ' ', 'é', 'K', 'NaN', '42']
    kinds = set()
    empty = 0
    for _ in range(4000):
        line = ' '.join(rng.choice(pieces) for _ in range(rng.randint(0, 5)))
        previous = rng.choice(pieces)
        expected = f1_hits_reference(line, previous)
        assert c._f1_hits(line, previous) == expected
        empty += not expected
        kinds.update(tag for _, _, tag in expected)
    assert empty > 500 and kinds == {'<PHONE>', '<EMAIL>', '<ADDRESS>'}


@parametrize('pack,lang', [('code-cpp', 'cpp'), ('code-go', 'go'), ('code-java', 'java'),
                         ('code-js-ts', 'javascript'), ('code-js-ts', 'typescript'),
                         ('code-python', 'python'), ('code-rust', 'rust')])
def test_shared_f1_keeps_parser_boundaries_and_personal_listing_result(monkeypatch, pack, lang):
    for text in ['// ' + EMAIL + '\nvalue = ' + PHONE,
                 '# ' + EMAIL + '\nvalue = "' + PHONE + '"',
                 '/* company */\n// ' + PHONE + '\nconst value = "' + EMAIL + '";',
                 '\n'.join(['// ' + EMAIL + ' ' + PHONE] * 8)]:
        row = doc(text, pack, lang)
        actual = c._shared_f1(row)
        with monkeypatch.context() as reference:
            reference.setattr(c, '_f1_hits', f1_hits_reference)
            assert actual == c._shared_f1(row)
