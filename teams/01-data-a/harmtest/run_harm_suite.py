"""Legacy registry-inventory audit of G1 Data A candidate rules against refreshed gold sets.

This runner requires the former non-executable inventory format and staged registry source files.
Its historical default spec/spec_v1.md is absent from this checkout. Supply --spec for a staged
legacy inventory; the current spec_v1/sheets/specv1.md is not compatible with this parser.
For current sheet/cleaner validation, use spec_v1/validate/validate_specv1.py or make_harm_tests.py.

Run from any directory:
  python teams/01-data-a/harmtest/run_harm_suite.py --parquet-dir /path/to/original-packs --spec /path/to/legacy-inventory.md

Spec v1 is a *non-executable* five-field review document: its code examples
are ```text fences, NOT a --rules input. This runner reads the Spec inventory,
proposal statuses and conflict ledger, then tests *individual* active rules
from rules/registry.json via harm_test.py --rule RULE_ID. In particular:
  - A harm PASS means the rule touched <=1% of gold and had no errors.
  - It does NOT mean the proposed medicine is approved for production.
  - BLOCKED clashes and unreviewed candidate rules are never combined.
  - Absent rules/packs/gold, stale sources, and G2 handoff remain pending.

Outputs JSON reports with document IDs, counts, and hashes, never document text.
Executing registry rules runs contributor Python; review their code first.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEAM = ROOT / 'teams' / '01-data-a'
SPEC = TEAM / 'spec' / 'spec_v1.md'
REGISTRY = TEAM / 'rules' / 'registry.json'
G1_PACKS = TEAM / 'gold' / 'g1_pack_ids.txt'

# All matching formats below are the documented output of build_spec_v1.py.
INVENTORY_ROW = re.compile(
    r'^\|\s*([\w-]+)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*(TEMPLATE|MEDICINES)\s*\|\s*`([a-f0-9]{64})`\s*\|$', re.M)
PROPOSAL = re.compile(
    r'^### Medicine \d+: ([\w-]+) / ([A-H][1-9]|UNCLASSIFIED) · ([^\n]+)\n\n'
    r'> \*\*Status:\*\* ([A-Z_]+);', re.M)
CONFLICT_ROW = re.compile(
    r'^\|\s*([\w-]+)\s*\|\s*([A-H][1-9])\s*\|\s*\d+\s*\|\s*(BLOCKED|DUPLICATE)\s*\|', re.M)
RULE_ID = re.compile(r'^D[AB]-[A-H][1-9]-\d{3}$')
EXIT = {'PASS': 0, 'FAIL': 1, 'ERROR': 2}


def repo_relative(path: Path) -> str:
    """Serialize paths relative to the repository, including external inputs."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()
    except ValueError:  # Windows: the file is on another drive than the repository, so there is no relative path
        return Path(path).resolve().as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_spec(path: Path) -> dict:
    """Read only metadata from Spec v1; never execute its example code."""
    src = path.read_text(encoding='utf-8')
    if 'NOT APPROVED / NOT FROZEN' not in src:
        raise ValueError('Unrecognized Spec v1 approval banner; manual review required')
    inventory = {}
    for pack, handle, count, state, digest in INVENTORY_ROW.findall(src):
        key = (pack, handle.strip())
        if key in inventory:
            raise ValueError(f'Duplicate medicine source in spec: {key!r}')
        inventory[key] = {'sections': int(count), 'state': state, 'sha256': digest}
    if not inventory:
        raise ValueError('Spec v1 has no source inventory')
    proposals = {}
    for pack, code, handle, state in PROPOSAL.findall(src):
        if state not in ('CANDIDATE', 'BLOCKED', 'DUPLICATE'):
            raise ValueError(f'Unsupported proposal state: {state}')
        key = (pack, handle.strip(), code)
        proposals.setdefault(key, []).append(state)
    conflicts = {(pack, code) for pack, code, state in CONFLICT_ROW.findall(src) if state == 'BLOCKED'}
    expected = re.search(r'Found \*\*(\d+) medicine files\*\*, \*\*(\d+) proposed sections\*\*', src)
    if expected and (len(inventory), sum(len(v) for v in proposals.values())) != tuple(map(int, expected.groups())):
        raise ValueError('Spec v1 source/medicine counts do not match its own inventory')
    for pack, handle, code in proposals:
        if (pack, handle) not in inventory:
            raise ValueError(f'Spec v1 proposal has no inventory entry: {pack}/{handle}/{code}')
    for (pack, handle, code), states in proposals.items():
        if (pack, code) in conflicts and any(s != 'BLOCKED' for s in states):
            raise ValueError(f'Conflict ledger and proposal disagree: {pack}/{code}')
    return {'inventory': inventory, 'proposals': proposals, 'conflicts': conflicts,
            'sha256': sha256(path)}


def read_pack_ids(path: Path) -> list[str]:
    ids = [s.strip() for line in path.read_text(encoding='utf-8').splitlines()
           if (s := line.split('#', 1)[0]).strip()]
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r'[\w-]+', x) for x in ids):
        raise ValueError('Invalid G1 pack ID list')
    return ids


def read_registry(path: Path) -> list[dict]:
    entries = json.loads(path.read_text(encoding='utf-8'))['rules']
    if not isinstance(entries, list):
        raise ValueError('Registry rules must be a list')
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(k), str) and entry[k].strip()
            for k in ('id', 'pack', 'handle', 'code', 'file')
        ) or not RULE_ID.fullmatch(entry['id']):
            raise ValueError('Invalid registry rule entry')
        if entry['id'] in ids:
            raise ValueError(f'Duplicate registry ID: {entry["id"]}')
        ids.add(entry['id'])
    return entries


def source_drift(spec_info: dict, team: Path) -> dict[str, list[str]]:
    """Prevent an out-of-date parent spec from silently certifying modified sources."""
    bad = {}
    base = team / 'medicines' / 'submissions'
    for (pack, handle), info in spec_info['inventory'].items():
        source = base / pack / handle / 'medicine.md'
        if not source.is_file():
            bad.setdefault(pack, []).append(f'{handle}: medicine.md missing')
        elif sha256(source) != info['sha256']:
            bad.setdefault(pack, []).append(f'{handle}: medicine.md changed since spec build')
    # Spec v1 must include all current medicines, not only the original snapshot.
    if base.is_dir():
        for source in base.glob('*/*/medicine.md'):
            if (source.parent.parent.name, source.parent.name) not in spec_info['inventory']:
                bad.setdefault(source.parent.parent.name, []).append(f'{source.parent.name}: absent from spec')
    return bad


def check_report(path: Path, rc: int, id_: str, pack: str, gold: Path, parquet: Path) -> dict:
    """Trust no stale, missing or inconsistent subprocess report as a PASS."""
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = data['results']
    status = data['status']
    if (status not in EXIT or rc != EXIT[status] or not isinstance(rows, list)
            or len(rows) != 1 or rows[0].get('rule') != id_
            or rows[0].get('result') != status or data.get('pack') != parquet.name
            or data.get('gold') != gold.name or data.get('pack_sha256') != sha256(parquet)
            or data.get('gold_sha256') != sha256(gold) or not data.get('rules_file_sha256')):
        raise ValueError('Inconsistent process result or source hashes')
    item = rows[0]
    changed, dropped, errors = (item[k] for k in ('changed', 'dropped', 'errors'))
    n = data['gold_count']
    if not all(isinstance(x, int) and x >= 0 for x in (changed, dropped, errors, n)) or not n:
        raise ValueError('Invalid report counters')
    if int(item['affected_unique']) != changed + dropped or changed + dropped > n:
        raise ValueError('Invalid affected-document count')
    if (status == 'PASS' and errors) or (status == 'ERROR' and not errors):
        raise ValueError('Invalid ERROR/PASS state')
    return {'harm_status': status, 'changed': changed, 'dropped': dropped,
            'errors': errors, 'percent': item['percent'], 'gold_count': n,
            'gold_sha256': data['gold_sha256'], 'pack_sha256': data['pack_sha256'],
            'rules_file_sha256': data['rules_file_sha256'], 'report': repo_relative(path)}


def run(parquet_dir: Path, report_dir: Path, spec: Path,
        registry: Path | None = None, pack_ids: Path | None = None,
        gold_dir: Path | None = None, *, limit: str = '1', timeout: float = 5.0,
        dry_run: bool = False) -> int:
    registry = registry or TEAM / 'rules' / 'registry.json'
    pack_ids = pack_ids or TEAM / 'gold' / 'g1_pack_ids.txt'
    gold_dir = gold_dir or TEAM / 'gold'
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        info = read_spec(spec)
        g1_packs = read_pack_ids(pack_ids)
        entries = read_registry(registry)
        drift = source_drift(info, TEAM)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        print(f'error: invalid G1 inputs: {exc}', file=sys.stderr)
        return 2

    spec_packs = {pack for pack, _ in info['inventory']}
    all_packs = list(dict.fromkeys([*g1_packs, *sorted(spec_packs - set(g1_packs))]))
    by_pack = {}
    out_of_spec = []
    for e in entries:
        if e.get('withdrawn'):
            continue
        key = (e['pack'], e['handle'], e['code'])
        if key not in info['proposals']:
            out_of_spec.append(e['id'])
        else:
            by_pack.setdefault(e['pack'], []).append(e)
    results = []
    for pack in all_packs:
        core = pack in g1_packs
        record = {'pack_id': pack, 'scope': 'G1' if core else 'SPEC_EXTRA',
                  'spec_sections': sum(v['sections'] for (p, _), v in info['inventory'].items() if p == pack),
                  'conflicts': sorted(code for p, code in info['conflicts'] if p == pack),
                  'rules': []}
        parquet = parquet_dir / (pack + '.parquet')
        gold = gold_dir / (pack + '_gold.csv')
        pack_entries = by_pack.get(pack, [])
        proposed = {key: states for key, states in info['proposals'].items() if key[0] == pack}
        count_by_key = {}
        for entry in pack_entries:
            key = (entry['pack'], entry['handle'], entry['code'])
            count_by_key[key] = count_by_key.get(key, 0) + 1
        # A registry entry can cover only one proposal; unextracted medicines
        # must never vanish from the pack's release/coverage report.
        unextracted = [
            {'handle': h, 'disease': code, 'count': len(states) - count_by_key.get((p, h, code), 0)}
            for (p, h, code), states in proposed.items()
            if len(states) > count_by_key.get((p, h, code), 0)
        ]
        record['proposed_rules'] = sum(len(states) for states in proposed.values())
        record['mapped_rules'] = len(pack_entries)
        record['unextracted_proposals'] = unextracted
        missing = [repo_relative(path) for path in (parquet, gold) if not path.is_file()]
        if drift.get(pack):
            record.update(status='SPEC_STALE', reason='; '.join(drift[pack]))
        elif missing and not dry_run:
            record.update(status='NOT_EVALUATED', reason='Missing input: ' + ', '.join(missing))
        elif not pack_entries:
            record.update(status='PENDING_RULES', reason='No active registry rules matched to Spec v1 medicine proposals')
        else:
            for entry in sorted(pack_entries, key=lambda e: e['id']):
                rule_id = entry['id']
                states = info['proposals'][(pack, entry['handle'], entry['code'])]
                spec_state = 'BLOCKED' if 'BLOCKED' in states else 'CANDIDATE'
                rule_record = {'id': rule_id, 'handle': entry['handle'], 'disease': entry['code'],
                               'spec_status': spec_state}
                if dry_run:
                    rule_record.update(harm_status='NOT_RUN', reason='--dry-run')
                else:
                    report = report_dir / 'rules' / (pack + '_' + rule_id + '_harm.json')
                    report.parent.mkdir(parents=True, exist_ok=True)
                    report.unlink(missing_ok=True)
                    cmd = [sys.executable, str(TEAM / 'harmtest' / 'harm_test.py'),
                           '--pack', str(parquet), '--gold', str(gold),
                           '--registry', str(registry), '--rule', rule_id,
                           '--limit', str(limit), '--timeout', str(timeout), '--report', str(report)]
                    try:
                        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
                        if not report.is_file():
                            raise ValueError('Harm process did not write a report')
                        rule_record.update(check_report(report, proc.returncode, rule_id, pack, gold, parquet))
                    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
                        # Avoid printing arbitrary code exceptions or source text in public artifacts.
                        rule_record.update(harm_status='ERROR', reason=type(exc).__name__)
                record['rules'].append(rule_record)
            statuses = {r['harm_status'] for r in record['rules']}
            if 'ERROR' in statuses:
                record.update(status='ERROR', reason='One or more rule harm tests could not be validated')
            elif 'FAIL' in statuses:
                record.update(status='HARM_FAIL', reason='One or more candidate rules exceed the harm limit')
            elif 'NOT_RUN' in statuses:
                record.update(status='DRY_RUN', reason='Commands not executed')
            elif unextracted:
                record.update(status='PARTIAL_COVERAGE', reason='Some medicine proposals have no active registry implementation')
            elif record['conflicts']:
                record.update(status='BLOCKED_CONFLICT', reason='Individual results are diagnostic only; conflict adjudication pending')
            else:
                record.update(status='PENDING_APPROVAL', reason='Individual rules passed, but parent Spec v1 remains NOT APPROVED')
        results.append(record)
        print(f'{pack:<22} {record["status"]:<19} {len(record["rules"])} rules')

    (report_dir / 'g1_harm_summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    (report_dir / 'g1_harm_metadata.json').write_text(json.dumps({
        'spec': repo_relative(spec), 'spec_sha256': info['sha256'],
        'registry': repo_relative(registry), 'registry_sha256': sha256(registry),
        'g1_pack_count': len(g1_packs), 'spec_extra_pack_count': len(spec_packs - set(g1_packs)),
        'out_of_spec_registry_rule_ids': sorted(out_of_spec),
        'g2_status': 'PENDING', 'combined_pipeline': 'NOT_RUN',
        'spec_status': 'DRAFT / NOT APPROVED', 'release_readiness': 'INCOMPLETE',
        'dry_run': dry_run,
    }, indent=2) + '\n', encoding='utf-8')
    print('G1 release readiness: INCOMPLETE (draft spec, G2 pending, no approved combined pipeline)')
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, default=TEAM / 'reports')
    parser.add_argument('--gold-dir', type=Path, default=TEAM / 'gold')
    parser.add_argument('--spec', type=Path, default=SPEC,
                        help='legacy inventory format, not the current medicine sheet; historical default is absent from this checkout')
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    parser.add_argument('--pack-ids', type=Path, default=G1_PACKS)
    parser.add_argument('--limit', default='1')
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--dry-run', action='store_true', help='plan test coverage without executing contributor code')
    args = parser.parse_args(argv)
    return run(args.parquet_dir, args.report_dir, args.spec, args.registry,
               args.pack_ids, args.gold_dir, limit=args.limit, timeout=args.timeout,
               dry_run=args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
