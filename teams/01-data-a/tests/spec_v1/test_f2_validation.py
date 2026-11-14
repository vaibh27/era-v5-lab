"""A full-set validation must report missing/empty own-pack gold as a failure."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = Path(__file__).resolve().parents[2] / 'spec_v1'
spec = importlib.util.spec_from_file_location('validate_f2', SPEC / 'validate/validate_f2.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
parametrize = pytest.mark.parametrize


@parametrize('empty', [False, True])
def test_missing_or_empty_gold_is_never_a_pass(tmp_path, monkeypatch, empty):
    gold = tmp_path / 'gold'
    gold.mkdir()
    if empty:
        (gold / 'en-dclm_gold.csv').write_text('doc_id,reader_1,reader_2\n')
    fp = SimpleNamespace(manifest={'version': 'v2', 'normalize_id': 'g9-v2'},
                         set_sha256='test-only', benchmarks=['toy'])
    monkeypatch.setattr(validation.f2_rule, 'fingerprints', lambda: fp)
    monkeypatch.setattr(validation.clean, '_FAMILY', {'en-dclm': 'english'})
    out = tmp_path / 'result.json'
    monkeypatch.setattr('sys.argv', ['validate_f2.py', '--packs', str(tmp_path),
                                   '--gold', str(gold), '--out', str(out)])
    assert validation.main() == 1
    report = json.loads(out.read_text())
    assert report['result'] == 'FAIL'
    assert report['packs']['en-dclm']['result'] == ('NOT TESTABLE' if empty else 'ERROR')
