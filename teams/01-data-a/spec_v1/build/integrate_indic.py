"""Review and collate Indic candidates into sheets/specv1.md without changing their source sheet.

Run before build_clean.py. --packs points to week-1 diagnosis files, not week-2 shards.
Missing inputs are NOT TESTABLE. Source-order overlap selection is provisional, not approval.
"""
import argparse
import ast
import importlib.util
from types import SimpleNamespace
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import signal
import sys
import textwrap
import unicodedata

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC.parent / 'harmtest'))
import harm_test as ht

if ht.HAS_ALARM:
    signal.signal(signal.SIGALRM, ht._on_alarm)

SENTENCES = {
    'bn': 'বাংলা ভাষায় নদী ও কৃষির ইতিহাস নিয়ে এই নিবন্ধটি লেখা হয়েছে।',
    'gu': 'આ લેખમાં નદી અને ખેતીના ઇતિહાસ વિશે માહિતી આપવામાં આવી છે.',
    'hi': 'इस लेख में नदी और खेती के इतिहास के बारे में जानकारी दी गई है।',
    'kn': 'ಈ ಲೇಖನದಲ್ಲಿ ನದಿ ಮತ್ತು ಕೃಷಿಯ ಇತಿಹಾಸದ ಬಗ್ಗೆ ಮಾಹಿತಿ ನೀಡಲಾಗಿದೆ.',
    'mr': 'या लेखात नदी आणि शेतीच्या इतिहासाबद्दल माहिती दिली आहे.',
    'sa': 'अस्मिन् लेखे नद्याः कृषेः च इतिहासस्य विवरणं विद्यते।',
    'ta': 'இந்த கட்டுரையில் நதி மற்றும் விவசாய வரலாறு பற்றிய தகவல்கள் உள்ளன.',
    'te': 'ఈ వ్యాసంలో నది మరియు వ్యవసాయ చరిత్ర గురించి సమాచారం ఉంది.',
}
CLUSTERS = {'bn': 'ক্', 'gu': 'ક્', 'hi': 'क्', 'kn': 'ಕ್', 'mr': 'क्', 'sa': 'क्', 'ta': 'க்', 'te': 'క్'}
FENCE = re.compile(r'^```python\n(.*?)^```', re.S | re.M)
CARD_FENCE = re.compile(r'^```python(?: deferred)?\n(.*?)^```', re.S | re.M)
VISARGA = {'bn': 'ঃ', 'gu': 'ઃ', 'hi': 'ः', 'kn': 'ಃ', 'mr': 'ः', 'sa': 'ः', 'ta': 'ஃ', 'te': 'ః'}
V1_EXCLUSIONS = {
    'DB-G2-001', 'DB-G2-002', 'DB-F1-005', 'DB-F2-001', 'DB-F2-002',
    'DB-G4-001', 'DB-G4-005', 'DB-A6-010', 'DB-A6-011', 'DB-A6-012',
    'DB-A6-014', 'DB-A6-018',
}


F1_REPLACED = "replaced by the shared F1, spec_v1/build/f1_rule.py (Indic path): the TSAI ruling's <PHONE>/<EMAIL> placeholders with organisation and public-office contacts kept; this source rule used other placeholders or deleted values, and replaced organisation contacts"

# Added by Data A after PR #28: preserve reserved privacy markers while keeping
# the original contributor functions unchanged. Embedded in the affected cards,
# so the sheet harm runner and generated cleaner execute the same adapter.
PLACEHOLDER_RULES = {'DB-B2-001', 'DB-B2-003', 'DB-B2-004'}
PLACEHOLDER_COMPAT = r'''
# Added by Data A: protect TSAI privacy placeholders from HTML stripping.
import re as _da_privacy_re
_da_privacy_tokens = _da_privacy_re.compile(r"<(?:PHONE|EMAIL|ADDRESS)>")

def _da_keep_privacy_placeholders(rule):
    def medicine(doc):
        original = doc["text"]
        if not _da_privacy_tokens.search(original):
            return rule(doc)
        marker = "\ufdd0"
        while marker in original:
            marker += "\ufdd0"
        tokens = []
        def shield(match):
            tokens.append(match.group(0))
            return marker + str(len(tokens) - 1) + marker
        protected = _da_privacy_tokens.sub(shield, original)
        action, text = rule(dict(doc, text=protected))
        if action == "drop":
            return ("drop", original)
        for index, token in enumerate(tokens):
            text = text.replace(marker + str(index) + marker, token)
        return ("fix" if text != original else "keep", text)
    return medicine

medicine = _da_keep_privacy_placeholders(medicine)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shared_f1_rule():
    spec = importlib.util.spec_from_file_location('indic_shared_f1', SPEC / 'build/f1_rule.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.f1


def probes(pack):
    lang = pack.split('-')[1]
    sentence, cluster = SENTENCES[lang], CLUSTERS[lang]
    result = {
        'prose': sentence + '\n' + sentence.replace('।', '.') + '\n' + sentence + '\n',
        'joiners': (sentence + ' ') * 5 + cluster + '\u200d' + cluster[0] + ' ' + cluster + '\u200c' + cluster[0],
        'nfc': unicodedata.normalize('NFC', sentence),
    }
    word = sentence.split()[0]
    # Match the ordinary-prose probe's length. Repeating six Telugu sentences
    # would trigger D4's conjunction-density threshold instead of testing signs.
    for gap, label in [('', 'visarga'), (' ', 'spaced-visarga')]:
        result[label] = (sentence + ' ') * 2 + word + gap + VISARGA[lang] + ' ' + sentence
    result['colon'] = (sentence + ' ') * 2 + word + ': ' + sentence
    return result


def synthetic_check(rule, pack, lang):
    issues = []
    for label, text in probes(pack).items():
        doc = dict(doc_id='synthetic-review', text=text, url='', pack_id=pack, lang_label=lang)
        try:
            action, new = ht._invoke(rule, doc, 2)
            if action == 'drop':
                issues.append(label + ': drops ordinary synthetic prose')
                continue
            if any(new.count(c) != text.count(c) for c in ('\u200c', '\u200d')):
                issues.append(label + ': changes script joiner count')
            if label in ('visarga', 'spaced-visarga'):
                script = pack.split('-')[1]
                gap = ' ' if label == 'spaced-visarga' else ''
                context = SENTENCES[script].split()[0] + gap + VISARGA[script] + ' '
                if new.count(VISARGA[script]) != text.count(VISARGA[script]) or context not in new:
                    issues.append(label + ': changes visarga or adjacent spacing')
            action2, new2 = ht._invoke(rule, dict(doc, text=new), 2)
            if action2 == 'drop' or new2 != new:
                issues.append(label + ': not idempotent')
            for outside in ('code-cpp', None, 'ind-unknown'):
                if ht._invoke(rule, dict(doc, pack_id=outside), 2) != ('keep', text):
                    issues.append(label + ': out-of-pack action')
        except Exception as exc:
            issues.append(label + ': ' + type(exc).__name__)
    return issues


def read_source():
    source = SPEC / 'sheets/medicine_indic.md'
    src = source.read_text(encoding='utf-8')
    heads = list(re.finditer(r'^## \d+\. ', src, re.M))
    rows = []
    for i, head in enumerate(heads):
        body = src[head.start():heads[i + 1].start() if i + 1 < len(heads) else len(src)].rstrip()
        blocks = FENCE.findall(body)
        assert len(blocks) == 1
        block = textwrap.dedent(blocks[0])
        ns = {'__name__': 'indic_source'}
        exec(compile(block, source.name, 'exec'), ns)
        if ns['RULE_ID'] in PLACEHOLDER_RULES:
            block = block.rstrip() + '\n' + PLACEHOLDER_COMPAT
            exec(compile(PLACEHOLDER_COMPAT, 'Data A privacy compatibility', 'exec'), ns)
        packs = ns['ORIGIN'].split(', ')
        assert len(packs) == 1, 'review currently requires one pack per Indic source rule'
        rows.append(dict(id=ns['RULE_ID'], code=ns['CODE'], pack=packs[0], action=ns['ACTION'],
                         lang=sorted(ns['LANGS'])[0], rule=ns['medicine'], body=body, block=block))
    assert len({r['id'] for r in rows}) == len(rows)
    assert not V1_EXCLUSIONS & {r['id'] for r in rows}, 'out-of-v1 rule in ready source'
    return rows


def read_retired():
    """Read historical metadata without importing or executing retired Python."""
    source = SPEC / 'sheets/medicine_indic_retired.md'
    if not source.exists():
        return []
    src = source.read_text(encoding='utf-8')
    heads = list(re.finditer(r'^## \d+\. ', src, re.M))
    rows = []
    for i, head in enumerate(heads):
        body = src[head.start():heads[i + 1].start() if i + 1 < len(heads) else len(src)].rstrip()
        blocks = re.findall(r'^```python deferred\n(.*?)^```', body, re.S | re.M)
        assert len(blocks) == 1, 'retired card must have one deferred block'
        block = textwrap.dedent(blocks[0])
        names = {'RULE_ID', 'CODE', 'ACTION', 'LANGS', 'ORIGIN', 'STATUS', 'DEFER_REASON'}
        metadata = {}
        for node in ast.parse(block).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        assert target.id not in metadata, 'duplicate retired metadata'
                        metadata[target.id] = ast.literal_eval(node.value)
        assert names <= metadata.keys() and metadata['STATUS'] == 'DEFERRED'
        assert metadata['RULE_ID'] in V1_EXCLUSIONS and metadata['DEFER_REASON']
        rows.append(dict(id=metadata['RULE_ID'], code=metadata['CODE'], pack=metadata['ORIGIN'],
                         action=metadata['ACTION'], lang=sorted(metadata['LANGS'])[0],
                         reason=metadata['DEFER_REASON'], body=body, block=block))
    assert len(rows) == len({row['id'] for row in rows})
    return rows


def full_rule_check(rule, docs):
    nonidempotent, errors = [], []
    for doc_id, doc in docs.items():
        try:
            action, text = ht._invoke(rule, doc, 5)
            if action != 'drop':
                action2, new = ht._invoke(rule, dict(doc, text=text), 5)
                if action2 != 'keep' or new != text:
                    nonidempotent.append(doc_id)
        except Exception as exc:
            errors.append([doc_id, type(exc).__name__])
    return dict(nonidempotent_ids=nonidempotent, errors=errors)


def choose(rows, decisions):
    selected = {}
    for row in rows:
        d = decisions[row['id']]
        key = row['pack'], row['code']
        reason = '; '.join(d['review_issues'])
        if not reason and key in selected:
            reason = 'overlap alternative; provisional source-order selection is ' + selected[key]
        d.update(status='DEFERRED' if reason else 'CANDIDATE',
                 reason=reason or 'first eligible candidate in source order; independent approval pending')
        if d['status'] == 'CANDIDATE':
            selected[key] = row['id']


def interaction_check(active, docs):
    implicated = defaultdict(list)
    for doc_id, doc in docs.items():
        d = dict(doc)
        dropped = False
        for rid, rule in active:
            action, text = ht._invoke(rule, d, 5)
            if action == 'drop':
                dropped = True
                break
            d['text'] = text
        if dropped:
            continue
        for rid, rule in active:
            action, text = ht._invoke(rule, d, 5)
            if action != 'keep':
                implicated[rid].append(doc_id)
            if action == 'drop':
                break
            d['text'] = text
    return implicated


def integrate(packs, gold_dir):
    rows = read_source()
    retired = read_retired()
    assert not {r['id'] for r in rows} & {r['id'] for r in retired}
    shared_f1 = shared_f1_rule()
    gold_cache, gold_reports, pack_cache = {}, {}, {}
    for pack in sorted({r['pack'] for r in rows}):
        path, gold = packs / (pack + '.parquet'), gold_dir / (pack + '_gold.csv')
        if not path.exists() or not gold.exists():
            missing = [name for name, p in [('pack', path), ('gold', gold)] if not p.exists()]
            gold_reports[pack] = {'result': 'NOT TESTABLE', 'missing': missing,
                                  'reason': 'Required own-pack inputs missing: ' + ', '.join(missing)}
            continue
        try:
            ids = ht.read_gold(gold)
        except ht.NotTestable:
            gold_reports[pack] = {'result': 'NOT TESTABLE', 'missing': ['nonempty gold'],
                                  'reason': 'Gold set empty, not testable', 'gold_sha256': sha(gold)}
            continue
        import pyarrow.parquet as pq
        pack_rows = pq.read_table(path).to_pylist()
        assert all(row.get('pack_id', pack) == pack for row in pack_rows), 'pack metadata mismatch'
        pack_cache[pack] = {row['doc_id']: dict(row, pack_id=pack) for row in pack_rows}
        assert len(pack_cache[pack]) == len(pack_rows), 'duplicate diagnosis IDs'
        docs = ht.read_pack_docs(path, ids)
        for doc in docs.values():
            doc['pack_id'] = pack
        gold_cache[pack] = docs
        gold_reports[pack] = {'result': 'PENDING', 'n': len(ids), 'pack_sha256': sha(path), 'gold_sha256': sha(gold)}
    decisions, groups = {}, defaultdict(list)
    for number, r in enumerate(rows, 1):
        if number % 25 == 0:
            print(f'Reviewing Indic candidate {number}/{len(rows)}', flush=True)
        rid, pack = r['id'], r['pack']
        issues = synthetic_check(r['rule'], pack, r['lang'])
        synthetic_issues = list(issues)
        full = {'result': 'NOT TESTABLE'}
        if pack in pack_cache and not issues and r['code'] != 'F2':
            full = full_rule_check(r['rule'], pack_cache[pack])
            if full['nonidempotent_ids']:
                issues.append('not idempotent on own diagnosis pack')
            if full['errors']:
                issues.append('runtime errors on own diagnosis pack')
        result = {'result': 'NOT TESTABLE'}
        if pack in gold_cache:
            changed, dropped, errors = ht.run_rule(r['rule'], gold_cache[pack], 5)
            n = len(gold_cache[pack])
            result = dict(n=n, changed=changed, dropped=dropped, errors=errors,
                          result='ERROR' if errors else 'PASS' if ht.harm_percent(len(changed), len(dropped), n, Fraction(1))[1] else 'FAIL')
            if result['result'] != 'PASS':
                issues.append('current own-pack gold: ' + result['result'])
        else:
            result['reason'] = gold_reports[pack]['reason']
            issues.append('current own-pack gold: ' + result['reason'])
        if rid == 'DB-F2-001':
            issues.append('legacy Itihasa rule requires ITIHASA_TEST_SN; optional G9 F2 does not cover Itihasa')
        if r['code'] == 'F1':
            issues.append(F1_REPLACED)
        if r['code'] == 'F2':
            issues.append('legacy sentence-based F2 is excluded; G9 decontamination is separate and opt-in through run_clean.py --f2')
        key = pack, r['code']
        groups[key].append(rid)
        decisions[rid] = dict(pack=pack, code=r['code'], synthetic_issues=synthetic_issues,
                              review_issues=issues, gold=result, full_rule=full)
    choose(rows, decisions)
    # Resolve interactions monotonically: remove only rules that act again on a
    # retained document, then try the next eligible overlap alternative.
    for _ in range(len(rows) + 1):
        conflicts = {}
        for pack, docs in pack_cache.items():
            active = [(r['id'], r['rule']) for r in rows if r['pack'] == pack and decisions[r['id']]['status'] == 'CANDIDATE']
            conflicts.update(interaction_check(active + [('f1_rule.py', shared_f1)], docs))
        if not conflicts:
            break
        print(f'Interaction review: deferring {len(conflicts)} second-pass rules', flush=True)
        for rid, ids in conflicts.items():
            if rid == 'f1_rule.py':
                raise RuntimeError('shared Indic F1 acts again after the complete pipeline; fix F1 before building')
            decisions[rid]['interaction_ids'] = ids
            decisions[rid]['review_issues'].append('acts again on retained documents after the full pipeline')
        choose(rows, decisions)
    else:
        raise RuntimeError('Indic interaction review did not converge')
    for pack, docs in gold_cache.items():
        active = [(r['id'], r['rule']) for r in rows if r['pack'] == pack and decisions[r['id']]['status'] == 'CANDIDATE']
        changed, dropped, errors = ht.run_sheet(active + [('f1_rule.py', shared_f1)], docs, 5)
        result = 'ERROR' if errors else 'PASS' if ht.harm_percent(len(changed), len(dropped), len(docs), Fraction(1))[1] else 'FAIL'
        gold_reports[pack].update(result=result, changed=changed, dropped=dropped, errors=errors)
        if result != 'PASS':
            # No active family is emitted with a known combined harm failure.
            for rid, _ in active:
                decisions[rid].update(status='DEFERRED', reason='combined own-pack gold: ' + result)
    for row in retired:
        decisions[row['id']] = dict(pack=row['pack'], code=row['code'], status='DEFERRED',
                                    reason=row['reason'], review_issues=[row['reason']],
                                    gold={'result': 'NOT RUN', 'reason': 'retired; policy excludes execution'},
                                    full_rule={'result': 'NOT RUN'}, synthetic_issues=[], retired=True)
    report = {'source_sha256': sha(SPEC / 'sheets/medicine_indic.md'),
              'ready_count': len(rows), 'retired_count': len(retired),
              'retired_sha256': sha(SPEC / 'sheets/medicine_indic_retired.md') if retired else None,
              'f1_rule_sha256': sha(SPEC / 'build/f1_rule.py'),
              'combined_execution': 'selected native medicines followed by shared Indic F1',
              'placeholder_compatibility_rules': sorted(PLACEHOLDER_RULES),
              'source_code_sha256': hashlib.sha256('\n'.join(ast.dump(ast.parse(row['block'])) for row in rows).encode()).hexdigest(),
              'policy': 'one eligible candidate per pack/disease, first in source order; F2 deferred; no approvals inferred',
              'privacy_missing_packs': [],  # the shared F1 (Indic path) runs on every Indic pack of the family
              'rules': decisions, 'overlaps': [dict(pack=p, code=c, ids=ids,
                 selected=next((rid for rid in ids if decisions[rid]['status'] == 'CANDIDATE'), None))
                 for (p, c), ids in groups.items() if len(ids) > 1], 'gold': gold_reports}
    path = SPEC.parent / 'results/spec_v1/cleaner/indic_integration.json'
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    sheet = SPEC / 'sheets/specv1.md'
    base = re.split(r'^## Medicine \d+ — indic /', sheet.read_text(encoding='utf-8'), maxsplit=1, flags=re.M)[0].rstrip()
    base = re.sub(r'(?s)<!-- INDIC INTEGRATION -->.*?<!-- END INDIC INTEGRATION -->\n*', '', base)
    base = base.replace('Consolidates the four local medicine sheets', 'Consolidates the five local medicine sheets')
    base = base.replace('The four source sheets', 'The five source sheets')
    base = base.replace('The 13 packs are partitioned into code (six), English prose (four), math (two), and science (one).',
                        'The 26 covered packs are partitioned into code (six), English prose (four), math (two), science (one), and Indic (13). Nine additional Indic sample packs have no candidates in this source sheet and remain unsupported.')
    active_count = sum(d['status'] == 'CANDIDATE' for d in decisions.values())
    native_count = len({row['pack'] for row in rows})
    summary = f'''<!-- INDIC INTEGRATION -->
## Indic integration and deferred inventory

[medicine_indic.md](medicine_indic.md) contributes {len(rows)} ready source candidates on {native_count} diagnosis packs; **{active_count} are selected and {len(rows) - active_count} deferred**. Another {len(retired)} retired local cards are preserved in [medicine_indic_retired.md](medicine_indic_retired.md) as policy-deferred history and are never executed. Source IDs and the pinned Data B code, including its labelled visarga trims, remain in the source sheets. All local selections are CANDIDATE, not APPROVED. Source SHA-256: `{report['source_sha256']}`.

Each overlapping pack/disease group uses the first candidate in source section order that passes the local checks. Later alternatives remain deferred; this avoids stacking competing fixes/drop heuristics and makes no claim of superior recall. Review all {len(report['overlaps'])} groups and positive cases before freeze. The four #19 medicines are sample-hash-specific; a gold pass does not establish general pack coverage. All D3 wrappers keep the original text. The full per-rule decision and available gold evidence is in [indic_integration.json](../../results/spec_v1/cleaner/indic_integration.json).

F1 on the Indic packs is the shared rule, `build/f1_rule.py` (its Indic path: Indian phone formats and digits, Indian organisation domains and words), run last on every Indic pack; the source F1 rules are deferred, because they did not follow the TSAI ruling's placeholders and organisation exception. Rules use original `ORIGIN` pack guards regardless of the spelling of language tags. `SPEC_PACKS` supplies equivalent scope to the builder. Original Indic order is retained within its family; the disease registry keeps F1 last without reordering each pack pipeline.

Data A adds a labelled compatibility wrapper to `DB-B2-001`, `DB-B2-003` and `DB-B2-004` to preserve reserved `<PHONE>`, `<EMAIL>` and `<ADDRESS>` tokens while removing HTML. The original contributor functions remain unchanged in the source sheet and in the generated cards. Native and sibling combined validation includes shared F1 last; records contain its SHA-256. Gold checks use only the affected pack's current nonempty gold. Missing or empty own-pack inputs defer the rules; flag-only entries and pending author-approved format rewrites are not made executable by this integration.

Deferred cards use `python deferred` fences and `STATUS = "DEFERRED"`: they are inventory, not executable harm-runner blocks. `build_clean.py` validates them but omits them from `SCOPE`, `_RULE`, and pipelines, recording their reasons in `DEFERRED_RULES`. `DB-F2-001` requires an approved benchmark and `ITIHASA_TEST_SN`; both F2 candidates remain deferred rather than implying benchmark protection. Synthetic Unicode/ordinary-prose failures, own-pack non-idempotence and rules that act again after the full pipeline are also deferred. The available full packs are checked until no selected second-run action remains. Re-enabling requires changing the integration decision and rerunning the review/build, not silently returning keep after an exception.

Only locally available week-1 diagnosis packs are eligible for gold tests. Week-2 train/heldout shards cannot substitute for diagnosis packs: their IDs intentionally differ. Missing inputs are NOT TESTABLE. Existing Data A validation tables below are historical evidence for those four families; see the Indic report for its own snapshots. No combined Indic gold pass is claimed for missing packs.

| Pack | Native medicines plus shared F1: local own-pack gold result |
|---|---|
'''
    for pack, result in gold_reports.items():
        summary += f'| {pack} | {result["result"]}' + (f' ({result["n"]} gold docs)' if 'n' in result else ' (missing ' + ', '.join(result['missing']) + ')') + ' |\n'
    summary += '\n<!-- END INDIC INTEGRATION -->\n\n'
    first = base.index('## Medicine 1 ')
    base = base[:first] + summary + base[first:]
    n = len(re.findall(r'^## Medicine \d+ — ', base, re.M))
    appended = []
    for i, row in enumerate(rows + retired, n + 1):
        decision = decisions[row['id']]
        block = row['block']
        block = re.sub(r'^STATUS = .*$', f'STATUS = "{decision["status"]}"', block, flags=re.M)
        block = re.sub(r'^DEFER_REASON = .*\n', '', block, flags=re.M)
        additions = 'SPEC_PACKS = ' + repr({row['pack']}) + '\n'
        if decision['status'] == 'DEFERRED':
            additions += 'DEFER_REASON = ' + json.dumps(decision['reason']) + '\n'
        block = re.sub(r'^(STATUS = .*\n)', lambda m: m.group(0) + additions, block, count=1, flags=re.M)
        fence = 'python deferred' if decision['status'] == 'DEFERRED' else 'python'
        body = CARD_FENCE.sub(lambda m: '```' + fence + '\n' + block.rstrip() + '\n```', row['body'])
        heading = f'## Medicine {i} — indic / {row["code"]} / `{row["id"]}`'
        body = re.sub(r'^## .*', lambda m: heading, body, count=1)
        note = ('\n\n**Integration decision:** ' + decision['status'] + '. ' + decision['reason'] +
                '. Source harm statements below are inherited; current local own-pack gold: ' + decision['gold']['result'] +
                '. The integration status above supersedes the source card status below.\n')
        body = body.replace('\n\n', note + '\n', 1)
        appended.append(body)
    sheet.write_text(base.rstrip() + '\n\n' + '\n\n'.join(appended) + '\n', encoding='utf-8')
    print(f'Indic: {active_count} selected, {len(rows)-active_count} deferred, {len(retired)} retired; {len(gold_cache)}/{native_count} diagnosis packs testable')
    extension_spec = importlib.util.spec_from_file_location('indic_sibling_extension', SPEC / 'build/extend_indic_siblings.py')
    extension = importlib.util.module_from_spec(extension_spec)
    extension_spec.loader.exec_module(extension)
    api = SimpleNamespace(read_source=read_source, sha=sha, ht=ht, probes=probes, synthetic_check=synthetic_check, full_rule_check=full_rule_check, Fraction=Fraction, shared_f1_rule=shared_f1_rule)
    extension.extend(api, packs, gold_dir, report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--packs', required=True, type=Path)
    ap.add_argument('--gold', type=Path, default=SPEC.parents[1] / '02-data-b/gold')
    args = ap.parse_args()
    integrate(args.packs, args.gold)


if __name__ == '__main__':
    main()
