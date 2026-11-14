"""Check a generated cleaner against a saved baseline and available Indic diagnosis/gold packs.

Reports IDs/counts/digests only. --baseline is the pre-integration clean.py; it is never modified.
Exit 1 means a regression/harm/error; exit 2 means some Indic diagnosis packs were missing.
"""
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC.parent / 'harmtest'))
import harm_test as ht


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docs(path, pack):
    rows = pq.read_table(path).to_pylist()
    assert all(row.get('pack_id', pack) == pack for row in rows), 'pack metadata mismatch'
    result = {row['doc_id']: dict(row, pack_id=pack) for row in rows}
    assert len(rows) == len(result), 'duplicate doc_id'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packs', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--gold', type=Path, default=SPEC.parents[1] / '02-data-b/gold')
    args = parser.parse_args()
    baseline, cleaner = load('baseline', args.baseline), load('generated', SPEC / 'clean.py')
    report = {'baseline_clean_sha256': sha(args.baseline), 'clean_sha256': sha(SPEC / 'clean.py'),
              'existing_families': {}, 'indic': {}}
    failed = missing = False
    for pack in sorted(p for p, family in baseline._FAMILY.items() if family != 'indic'):
        differences, matching_errors = [], []
        rows = docs(args.packs / (pack + '.parquet'), pack)
        for doc_id, row in rows.items():
            outcomes = []
            for module in (baseline, cleaner):
                try:
                    outcomes.append(('result', module.clean(row)))
                except Exception as exc:
                    outcomes.append(('error', type(exc).__name__))
            if outcomes[0] != outcomes[1]:
                differences.append(doc_id)
            elif outcomes[0][0] == 'error':
                matching_errors.append([doc_id, outcomes[0][1]])
        report['existing_families'][pack] = {'documents': len(rows), 'different_ids': differences,
                                             'matching_baseline_errors': matching_errors}
        failed |= bool(differences)
        print(pack, 'differences', len(differences), 'matching baseline errors', len(matching_errors), flush=True)
    for pack in sorted(p for p, family in cleaner._FAMILY.items() if family == 'indic'):
        path, gold = args.packs / (pack + '.parquet'), args.gold / (pack + '_gold.csv')
        if not path.exists() or not gold.exists():
            report['indic'][pack] = {'result': 'NOT TESTABLE', 'missing': [kind for kind, p in [('pack', path), ('gold', gold)] if not p.exists()]}
            missing = True
            print(pack, 'NOT TESTABLE', flush=True)
            continue
        ids = ht.read_gold(gold)
        gold_docs = ht.read_pack_docs(path, ids)
        for row in gold_docs.values():
            row['pack_id'] = pack
        changed, dropped, errors = ht.run_rule(cleaner.clean, gold_docs, 5)
        rows = docs(path, pack)
        fixed, removed, full_errors = ht.run_rule(cleaner.clean, rows, 5)
        again = []
        for doc_id in fixed:
            _, text = cleaner.clean(rows[doc_id])
            if cleaner.clean(dict(rows[doc_id], text=text)) != ('keep', text):
                again.append(doc_id)
        result = 'ERROR' if errors or full_errors else 'PASS' if ht.harm_percent(len(changed), len(dropped), len(ids), Fraction(1))[1] else 'FAIL'
        report['indic'][pack] = {'pack_sha256': sha(path), 'gold_sha256': sha(gold), 'result': result,
            'gold_n': len(ids), 'gold_changed': changed, 'gold_dropped': dropped, 'gold_errors': errors,
            'full_n': len(rows), 'fixed': fixed, 'dropped': removed, 'errors': full_errors, 'second_run_changes': again}
        failed |= result != 'PASS' or bool(again)
        print(pack, result, 'gold', len(changed), len(dropped), 'full', len(fixed), len(removed),
              'errors', len(full_errors), 'second-run changes', len(again), flush=True)
    (SPEC.parent / 'results/spec_v1/cleaner/indic_clean_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 1 if failed else 2 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
