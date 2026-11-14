"""Measure the optional G9 F2 stage on each supported Indic pack's own gold.

Loads the pinned full fingerprint set once. Reports F2 alone and clean.py then F2,
counting each changed/dropped gold ID once. Does not enable F2 in the default cleaner.
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
import harm_test as ht
import f2_rule


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_then_f2(doc):
    action, text = clean.clean(doc)
    if action == 'drop' or f2_rule.check(text)[0]:
        return ('drop', doc['text'])
    return (action, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packs', type=Path, required=True)
    parser.add_argument('--gold', type=Path, default=SPEC.parents[1] / '02-data-b/gold')
    args = parser.parse_args()
    fingerprint_set = f2_rule.fingerprints()  # warm/check before entering per-document timeouts
    report = {'set_sha256': fingerprint_set.set_sha256, 'clean_sha256': sha(SPEC / 'clean.py'),
              'f2_sha256': sha(Path(f2_rule.__file__)), 'scope': 'full pinned G9 set; Itihasa is not covered', 'packs': {}}
    failed = False
    for pack in sorted(p for p, family in clean._FAMILY.items() if family == 'indic'):
        path, gold = args.packs / (pack + '.parquet'), args.gold / (pack + '_gold.csv')
        ids = ht.read_gold(gold)
        docs = ht.read_pack_docs(path, ids)
        for d in docs.values():
            d['pack_id'] = pack
        raw = ht._result('F2 alone', ht.run_rule(f2_rule.f2, docs, 5), len(ids), Fraction(1))
        combined = ht._result('clean then F2', ht.run_rule(clean_then_f2, docs, 5), len(ids), Fraction(1))
        report['packs'][pack] = {'gold_n': len(ids), 'pack_sha256': sha(path), 'gold_sha256': sha(gold), 'f2_alone': raw, 'combined': combined}
        failed |= raw['result'] != 'PASS' or combined['result'] != 'PASS'
        print(pack, 'F2 alone', raw['result'], raw['changed'], raw['dropped'], 'combined', combined['result'], combined['changed'], combined['dropped'], flush=True)
    (SPEC.parent / 'results/spec_v1/f2/indic_gold_validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
