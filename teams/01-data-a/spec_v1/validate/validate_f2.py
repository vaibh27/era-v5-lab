"""Validate the full pinned F2 set alone and after clean.py on every supported pack's own gold.

Run from the repository root:
  python -I teams/01-data-a/spec_v1/validate/validate_f2.py --packs <diagnosis-packs>

Missing inputs, invalid/empty gold and runtime errors fail the run. Existing v1
reports are preserved; the output records the actual set, normalizer and code hashes.
This checks diagnosis gold, not the full tokenizer sample or final deduped shards.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC))
sys.path.insert(0, str(SPEC.parent / 'harmtest'))
import clean
import f2_rule
import harm_test as ht


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_then_f2(doc):
    action, text = clean.clean(doc)
    if action == 'drop' or f2_rule.check(text)[0]:
        return 'drop', doc['text']
    return action, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packs', type=Path, required=True)
    parser.add_argument('--gold', type=Path, default=SPEC.parent / 'gold')
    parser.add_argument('--indic-gold', type=Path, default=SPEC.parents[1] / '02-data-b/gold')
    parser.add_argument('--out', type=Path, default=SPEC.parent / 'results/spec_v1/f2/v2_validation.json')
    args = parser.parse_args()
    fp = f2_rule.fingerprints()  # Check all hashes and warm the full set before per-doc timeouts.
    report = {
        'set_version': fp.manifest['version'], 'set_sha256': fp.set_sha256,
        'normalize_id': fp.manifest['normalize_id'], 'scope': 'full pinned G9 set',
        'benchmarks': sorted(fp.benchmarks), 'itihasa_covered': False,
        'clean_sha256': sha(SPEC / 'clean.py'), 'f2_sha256': sha(Path(f2_rule.__file__)),
        'validator_sha256': sha(Path(__file__)), 'harm_test_sha256': sha(Path(ht.__file__)),
        'g9_code_sha256': {name: sha(f2_rule._g9() / name)
                           for name in ('fpset.py', 'normalize.py', 'g1_spec_v1_normalize.py')},
        'limit_percent': '1', 'packs': {},
    }
    failed = False
    for pack, family in sorted(clean._FAMILY.items()):
        path = args.packs / (pack + '.parquet')
        gold = (args.indic_gold if family == 'indic' else args.gold) / (pack + '_gold.csv')
        try:
            ids = ht.read_gold(gold)
            docs = ht.read_pack_docs(path, ids)
            for doc in docs.values():
                doc['pack_id'] = pack
            raw = ht._result('F2 alone', ht.run_rule(f2_rule.f2, docs, 5), len(ids), Fraction(1))
            combined = ht._result('clean then F2', ht.run_rule(clean_then_f2, docs, 5), len(ids), Fraction(1))
            passed = raw['result'] == combined['result'] == 'PASS'
            entry = {'family': family, 'gold_n': len(ids), 'gold_sha256': sha(gold),
                     'pack_sha256': sha(path), 'limit_documents': ht.allowed_documents(len(ids), Fraction(1)),
                     'f2_alone': raw, 'combined': combined, 'result': 'PASS' if passed else 'FAIL',
                     'raw_gold_hits': {doc_id: fp.check(docs[doc_id]['text'])[1]
                                       for doc_id in raw['dropped_ids']}}
            print(pack, 'F2', raw['result'], raw['dropped'], 'combined', combined['result'],
                  combined['changed'], combined['dropped'], flush=True)
        except (ht.Fatal, OSError, ValueError) as exc:
            passed = False
            entry = {'family': family, 'result': 'NOT TESTABLE' if isinstance(exc, ht.NotTestable) else 'ERROR',
                     'error_type': type(exc).__name__}
            print(pack, entry['result'], entry['error_type'], flush=True)
        report['packs'][pack] = entry
        failed |= not passed
    report['result'] = 'FAIL' if failed else 'PASS'
    report['gold_n'] = sum(entry.get('gold_n', 0) for entry in report['packs'].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
