"""Validate sibling scopes declared in medicine_indic.md and refresh consolidated Indic cards.

Usually called by integrate_indic.py. Standalone mode reuses native decisions only when
all source Python blocks and native diagnosis/gold snapshots still match their digests.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re

SPEC = Path(__file__).resolve().parents[1]


def source_code_digest(rows):
    return hashlib.sha256('\n'.join(ast.dump(ast.parse(row['block'])) for row in rows).encode()).hexdigest()


def policy_from_source():
    source = (SPEC / 'sheets/medicine_indic.md').read_text(encoding='utf-8')
    match = re.search(r'<!-- INDIC SIBLING POLICY -->.*?```json\n(.*?)\n```', source, re.S)
    return json.loads(match.group(1)) if match else {'enabled': False}


def scope_allowed(row, pack, policy):
    allowed = policy.get('allowed_rule_packs', {}).get(row['id'])
    return allowed is None or pack in allowed


def target_rule(row, pack, policy=None):
    if pack.split('-')[1] != row['pack'].split('-')[1]:
        raise ValueError('cross-language sibling scope is forbidden')
    if not scope_allowed(row, pack, policy if policy is not None else policy_from_source()):
        raise ValueError('rule/pack scope excluded by review policy')
    ns = {'__name__': 'indic_sibling_candidate'}
    exec(compile(row['block'], row['id'], 'exec'), ns)
    # The only runtime scope used by the copied source wrapper. The author function is unchanged.
    ns['_PACKS'] = frozenset({pack})
    return ns['medicine']


def compose(rows, chosen, functions, shared_f1, pack):
    order = [(row['id'], functions[row['id']]) for row in rows if row['id'] in chosen]

    def medicine(doc):
        if doc.get('pack_id') != pack:
            return ('keep', doc['text'])
        d = dict(doc)
        for _, fn in order:
            action, text = fn(d)
            if action == 'drop':
                return ('drop', doc['text'])
            d['text'] = text
        action, text = shared_f1(d)
        if action == 'drop':
            return ('drop', doc['text'])
        d['text'] = text
        return ('fix' if d['text'] != doc['text'] else 'keep', d['text'])
    return medicine


def extend(review, packs, gold_dir, report=None):
    policy = policy_from_source()
    if not policy.get('enabled'):
        return
    if not all(policy.get(name) is True for name in ('same_language_only', 'native_rules_take_priority',
                                                    'fill_missing_diseases_only', 'require_receiving_gold')):
        raise ValueError('unsupported sibling scope policy')
    rows = review.read_source()
    for row in rows:
        if not scope_allowed(row, row['pack'], policy):
            raise ValueError('native rule scope excluded by review policy: ' + row['id'])
    shared_f1 = review.shared_f1_rule()
    primary_path = SPEC.parent / 'results/spec_v1/cleaner/indic_integration.json'
    report = report or json.loads(primary_path.read_text(encoding='utf-8'))
    if report.get('source_code_sha256') != source_code_digest(rows):
        raise ValueError('source algorithms changed: rerun integrate_indic.py before extending scopes')
    if report.get('f1_rule_sha256') != review.sha(SPEC / 'build/f1_rule.py'):
        raise ValueError('shared F1 changed: rerun integrate_indic.py before extending scopes')
    for pack, evidence in report['gold'].items():
        if evidence.get('pack_sha256'):
            if review.sha(packs / (pack + '.parquet')) != evidence['pack_sha256'] or review.sha(gold_dir / (pack + '_gold.csv')) != evidence['gold_sha256']:
                raise ValueError(f'{pack}: native inputs changed; rerun integrate_indic.py')
    selected = [row for row in rows if report['rules'][row['id']]['status'] == 'CANDIDATE']
    scopes = {row['id']: {row['pack']} for row in selected}
    targets = {}
    import pyarrow.parquet as pq
    for pack in policy['target_pack_ids']:
        lang = pack.split('-')[1]
        donors = [row for row in selected if row['pack'].split('-')[1] == lang and row['pack'] != pack]
        native = [row for row in selected if row['pack'] == pack]
        native_codes = {row['code'] for row in native}
        candidates = [row for row in donors if row['code'] not in native_codes]
        entry = {'native_rules': [r['id'] for r in native], 'accepted': [], 'attempts': [], 'result': 'PENDING'}
        targets[pack] = entry
        if not native and not donors:
            entry.update(result='NO SAME-LANGUAGE DONOR')
            print(pack, entry['result'], flush=True)
            continue
        path, gold = packs / (pack + '.parquet'), gold_dir / (pack + '_gold.csv')
        if not path.exists() or not gold.exists():
            entry.update(result='NOT TESTABLE', missing=[kind for kind, p in [('pack', path), ('gold', gold)] if not p.exists()])
            print(pack, entry['result'], flush=True)
            continue
        try:
            ids = review.ht.read_gold(gold)
        except review.ht.NotTestable:
            entry.update(result='NOT TESTABLE', missing=['nonempty gold'])
            continue
        gold_docs = review.ht.read_pack_docs(path, ids)
        data = pq.read_table(path).to_pylist()
        if not all(d.get('pack_id', pack) == pack for d in data):
            raise ValueError(f'{pack}: metadata mismatch')
        docs = {d['doc_id']: dict(d, pack_id=pack) for d in data}
        if len(docs) != len(data):
            raise ValueError(f'{pack}: duplicate IDs')
        for d in gold_docs.values():
            d['pack_id'] = pack
        chosen = {row['id'] for row in native}
        codes = set(native_codes)
        functions = {row['id']: target_rule(row, pack, policy) for row in native}
        for row in candidates:
            if row['code'] in codes:
                continue
            rid = row['id']
            attempt = {'rule_id': rid, 'source_pack': row['pack'], 'code': row['code']}
            entry['attempts'].append(attempt)
            if not scope_allowed(row, pack, policy):
                attempt.update(result='REJECTED', reasons=['rule/pack scope excluded by review policy'])
                continue
            functions[rid] = target_rule(row, pack, policy)
            fn = functions[rid]
            issues = review.synthetic_check(fn, pack, lang)
            single = review.ht._result(rid, review.ht.run_rule(fn, gold_docs, 5), len(ids), review.Fraction(1))
            attempt['gold'] = single
            if single['result'] != 'PASS':
                issues.append('receiving-pack gold: ' + single['result'])
            if not issues:
                check = review.full_rule_check(fn, docs)
                attempt['full_rule'] = check
                if check['errors'] or check['nonidempotent_ids']:
                    issues.append('receiving-pack runtime/idempotence failure')
            if not issues:
                pipeline = compose(rows, chosen | {rid}, functions, shared_f1, pack)
                issues += review.synthetic_check(pipeline, pack, lang)
                combined = review.ht._result('combined', review.ht.run_rule(pipeline, gold_docs, 5), len(ids), review.Fraction(1))
                attempt['combined_gold'] = combined
                if combined['result'] != 'PASS':
                    issues.append('combined receiving-pack gold: ' + combined['result'])
                if not issues:
                    check = review.full_rule_check(pipeline, docs)
                    attempt['combined_full'] = check
                    if check['errors'] or check['nonidempotent_ids']:
                        issues.append('combined receiving-pack runtime/idempotence failure')
            attempt.update(result='REJECTED' if issues else 'ACCEPTED', reasons=issues)
            if not issues:
                chosen.add(rid)
                codes.add(row['code'])
                scopes[rid].add(pack)
                entry['accepted'].append(rid)
        if not chosen:
            entry['result'] = 'NO ELIGIBLE RULES'
            continue
        pipeline = compose(rows, chosen, functions, shared_f1, pack)
        entry.update(result='PASS', gold_n=len(ids), pack_sha256=review.sha(path), gold_sha256=review.sha(gold),
                     combined_gold=review.ht._result('combined', review.ht.run_rule(pipeline, gold_docs, 5), len(ids), review.Fraction(1)),
                     final_full=review.full_rule_check(pipeline, docs))
        assert entry['combined_gold']['result'] == 'PASS' and not entry['final_full']['errors'] and not entry['final_full']['nonidempotent_ids']
        print(pack, 'accepted sibling scopes', len(entry['accepted']), 'gold', len(ids), 'PASS', flush=True)
    supported = sorted({p for values in scopes.values() for p in values})
    privacy_missing = []  # Every registered Indic pack runs shared F1 last.
    result = {'privacy_missing_packs': privacy_missing, 'shared_f1_packs': supported,
              'f1_rule_sha256': review.sha(SPEC / 'build/f1_rule.py'),
              'combined_execution': 'selected native/sibling medicines followed by shared Indic F1',
              'source_sha256': review.sha(SPEC / 'sheets/medicine_indic.md'), 'source_code_sha256': source_code_digest(rows),
              'policy': policy, 'targets': targets, 'rule_scopes': {rid: sorted(p) for rid, p in scopes.items()}}
    destination = SPEC.parent / 'results/spec_v1/cleaner/indic_sibling_scopes.json'
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    source = (SPEC / 'sheets/specv1.md').read_text(encoding='utf-8')
    # Refresh scope metadata only, preserving algorithms, native decisions, card numbering and IDs.
    heads = list(re.finditer(r'^## Medicine \d+ — indic / \w+ / `(DB-[^`]+)`\n', source, re.M))
    for i in reversed(range(len(heads))):
        head = heads[i]
        end = heads[i+1].start() if i+1 < len(heads) else len(source)
        body = source[head.start():end]
        rid = head.group(1)
        if rid not in scopes:
            continue
        original = next(row['pack'] for row in rows if row['id'] == rid)
        body = re.sub(r'(?m)^SOURCE_ORIGIN = .*\n', '', body)
        origin = ', '.join([original] + sorted(scopes[rid] - {original}))
        body = re.sub(r'(?m)^ORIGIN = .*$', lambda m: f'ORIGIN = {json.dumps(origin)}\nSOURCE_ORIGIN = {json.dumps(original)}', body)
        body = re.sub(r'(?m)^SPEC_PACKS = .*$', lambda m: 'SPEC_PACKS = {' + ', '.join(repr(p) for p in sorted(scopes[rid])) + '}', body)
        source = source[:head.start()] + body + source[end:]
    source = re.sub(r'(?s)<!-- VALIDATED SIBLING SCOPES -->.*?<!-- END VALIDATED SIBLING SCOPES -->\n*', '', source)
    supported = sorted({p for values in scopes.values() for p in values})
    summary = '''<!-- VALIDATED SIBLING SCOPES -->
## Validated Indic sibling scopes

The scope policy starts in [medicine_indic.md](medicine_indic.md). Integration borrows selected native candidates only within the same language, preserves native disease selections, and tests every receiving scope on its own gold and full diagnosis pack. Explicit review restrictions override fallback borrowing: `DB-D4-011` runs only on `ind-te-fw2`, and `DB-D5-001` only on `ind-te-sangraha-s`. `SOURCE_ORIGIN` preserves the donor; `ORIGIN` and `SPEC_PACKS` contain validated execution scopes. No source algorithm or rule ID changes. Results: [indic_sibling_scopes.json](../../results/spec_v1/cleaner/indic_sibling_scopes.json).

'''
    summary += f'Current Indic coverage: **{len(supported)} packs**; **{sum(len(v["accepted"]) for v in targets.values())} additional rule/pack scopes**. All are CANDIDATE. Packs without nonempty receiving gold remain unregistered; Malayalam/Odia have no same-language donor.\n\n'
    summary += '| Receiving pack | Borrowed scopes | Result |\n|---|---:|---|\n'
    for pack, entry in targets.items():
        summary += f'| {pack} | {len(entry["accepted"])} | {entry["result"]} |\n'
    summary += '\n<!-- END VALIDATED SIBLING SCOPES -->\n\n'
    first = source.index('## Medicine 1 ')
    source = source[:first] + summary + source[first:]
    source = re.sub(r'The \d+ covered packs are partitioned[^\n]*', f'The {13+len(supported)} covered packs are partitioned into code (six), English prose (four), math (two), science (one), and Indic ({len(supported)}). Indic sibling scopes are validated separately below.', source)
    source = source.replace('Nine additional Indic sample packs have no candidates in this source sheet and remain unsupported.', 'Remaining Indic packs without validated scopes are unsupported.')
    source = re.sub(r'Source SHA-256: `[0-9a-f]+`', 'Source SHA-256: `' + result['source_sha256'] + '`', source)
    source = re.sub(r'No selected F1 candidate exists for: [^.]*\.', 'No selected F1 candidate exists for: ' + ', '.join(privacy_missing) + '.', source)
    source = source.replace('Rules use original `ORIGIN` pack guards', 'Native rules use original `ORIGIN` pack guards; accepted sibling guards are broadened only as documented below. Both operate')
    (SPEC / 'sheets/specv1.md').write_text(source, encoding='utf-8')
    report['source_sha256'] = result['source_sha256']
    report['privacy_missing_packs'] = privacy_missing
    report['sibling_scopes_report'] = destination.name
    report['rule_scopes'] = result['rule_scopes']
    primary_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packs', type=Path, required=True)
    parser.add_argument('--gold', type=Path, default=SPEC.parents[1] / '02-data-b/gold')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('native_indic_review', SPEC / 'build/integrate_indic.py')
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    extend(review, args.packs, args.gold)


if __name__ == '__main__':
    main()
