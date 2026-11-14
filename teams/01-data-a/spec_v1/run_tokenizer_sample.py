"""Run the pinned Week-2 tokenizer sample through clean.py, with sequential per-rule counts.

Offline; only the manifest's sample parts are accepted. Writes exclusively beneath
./out/. Train and heldout stay separate. No source text is printed or added to reports.
The original clean.clean() and _run() execute; factories are wrapped only for counting.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from functools import cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
try:
    import resource  # Unix only: peak memory for the job log
except ImportError:  # Windows: the run works, peak memory is not recorded
    resource = None
import time

import pyarrow as pa
import pyarrow.parquet as pq

SPEC = Path(__file__).resolve().parent
REPO = SPEC.parents[2]
sys.path.insert(0, str(SPEC))
import clean


class JobError(Exception):
    pass


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def peak_rss_mib():
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 1024 * 1024 if sys.platform == 'darwin' else 1024
    return round(rss / unit, 3)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def safe_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value:
        raise JobError('non-relative manifest path')
    if len(path.parts) != 3 or path.parts[0] not in {'train', 'heldout'}:
        raise JobError('unexpected sample split/path')
    return path


def output_folder(value):
    allowed = (Path.cwd() / 'out').resolve()
    path = Path(value).resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise JobError('output must be a named subfolder of ./out/')
    return path


def manifest(config):
    path = REPO / config['manifest']
    if digest(path) != config['manifest_sha256']:
        raise JobError('sample manifest checksum mismatch')
    data = read_json(path)
    shards = data['shards']
    counts = Counter(s['split'] for s in shards)
    if dict(counts) != config['expected_parts'] or len(shards) != 86:
        raise JobError('only tokenizer sample v0 (48 train + 38 heldout parts) is allowed')
    seen = set()
    for shard in shards:
        p = safe_path(shard['path'])
        if shard['path'] in seen or p.parts[0] != shard['split'] or p.parts[1] != shard['pack_id']:
            raise JobError('duplicate or inconsistent sample manifest entry')
        seen.add(shard['path'])
    if sum(s['rows'] for s in shards) != data['total_docs']:
        raise JobError('manifest row totals disagree')
    return data


def resolve_input(root, shard, hashes):
    p = safe_path(shard['path'])
    # Official layout split/pack/part, or the user's organized pack/split/part.
    candidates = [root / p, root / p.parts[1] / p.parts[0] / p.parts[2]]
    for folder in (root / p.parts[0] / p.parts[1], root / p.parts[1] / p.parts[0]):
        candidates.extend(sorted(folder.glob('*.parquet')))
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file() or candidate.stat().st_size != shard['bytes']:
            continue
        if candidate not in hashes:
            hashes[candidate] = digest(candidate)
        if hashes[candidate] == shard['sha256']:
            return candidate
    raise JobError('missing or checksum-mismatched sample part: ' + shard['path'])


class RuleCounter:
    def __init__(self):
        self.counts = defaultdict(Counter)

    @contextmanager
    def instrument(self):
        original_run = clean._run
        original_ids = dict(clean._ID)
        wrappers = {}

        def wrap(make, rid):
            @cache
            def factory():
                rule = make()

                def counted(doc):
                    try:
                        result = rule(doc)
                        if (not isinstance(result, tuple) or len(result) != 2
                                or result[0] not in {'keep', 'fix', 'drop'} or not isinstance(result[1], str)):
                            raise JobError('invalid rule result')
                    except Exception:
                        self.counts[rid]['error'] += 1
                        raise
                    self.counts[rid][result[0]] += 1
                    return result
                return counted
            return factory

        for rid, make in clean._RULE.items():
            wrapped = wrap(make, rid)
            wrappers[make] = wrapped
            clean._ID[wrapped] = rid

        def counted_run(doc, *factories):
            return original_run(doc, *(wrappers[make] for make in factories))

        clean._run = counted_run
        try:
            yield
        finally:
            clean._run = original_run
            clean._ID.clear()
            clean._ID.update(original_ids)


def applicable_rules(pack):
    return [rid for rid in clean._RULE if pack in clean.SCOPE[rid]]


def process_part(source, shard, output, batch_size):
    started = time.monotonic()
    relative, pack = shard['path'], shard['pack_id']
    destination = output / relative
    report_path = output / 'reports' / (relative + '.json')
    if destination.exists() or report_path.exists():
        raise JobError('refusing to overwrite existing part: ' + relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    source_file = pq.ParquetFile(source)
    if source_file.metadata.num_rows != shard['rows']:
        raise JobError('input row count differs from sample manifest: ' + relative)
    if not {'doc_id', 'text', 'pack_id', 'lang_label'}.issubset(source_file.schema_arrow.names):
        raise JobError('missing required input columns: ' + relative)
    counter, actions, rows, tags, seen = RuleCounter(), Counter(), 0, set(), set()
    supported = pack in clean._FAMILY
    try:
        with counter.instrument(), pq.ParquetWriter(temporary, source_file.schema_arrow, compression='zstd') as writer:
            for batch in source_file.iter_batches(batch_size=batch_size):
                kept = []
                for row in batch.to_pylist():
                    doc_id = row['doc_id']
                    if not isinstance(doc_id, str) or doc_id in seen or not isinstance(row['text'], str):
                        raise JobError('invalid text or duplicate doc_id: ' + relative)
                    if row['pack_id'] != pack:
                        raise JobError('pack_id differs from manifest: ' + relative)
                    seen.add(doc_id)
                    tags.add(row['lang_label'])
                    try:
                        action, text = clean.clean(row)
                    except Exception as exc:
                        write_json(output / 'failure.json', {'part': relative, 'doc_id': doc_id,
                                   'error_type': type(exc).__name__, 'processed_rows': rows,
                                   'rule_counts': dict(counter.counts)})
                        raise JobError('cleaner error: ' + relative + ' doc_id=' + doc_id + ' type=' + type(exc).__name__) from None
                    actions[action] += 1
                    if action != 'drop':
                        kept.append(dict(row, text=text))
                    rows += 1
                if kept:
                    writer.write_table(pa.Table.from_pylist(kept, schema=source_file.schema_arrow))
        if rows != shard['rows']:
            raise JobError('processed rows disagree with manifest: ' + relative)
        if digest(source) != shard['sha256']:
            raise JobError('input changed during processing: ' + relative)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    kept_count = actions['keep'] + actions['fix']
    if pq.ParquetFile(destination).metadata.num_rows != kept_count:
        raise JobError('output row count mismatch: ' + relative)
    per_rule = {}
    for rid in applicable_rules(pack):
        counts = counter.counts[rid]
        invoked = sum(counts.values())
        per_rule[rid] = {kind: counts[kind] for kind in ('keep', 'fix', 'drop', 'error')}
        per_rule[rid].update(invocations=invoked, eligible_by_pack=rows, not_reached=rows-invoked)
    result = {'part': relative, 'pack_id': pack, 'split': shard['split'],
              'status': 'cleaned' if supported else 'unsupported_passthrough',
              'input_sha256': shard['sha256'], 'output_sha256': digest(destination),
              'input_rows': rows, 'output_rows': kept_count,
              'actions': dict(actions), 'errors': 0, 'rule_counts': per_rule,
              'language_tags': sorted(tags, key=str), 'seconds': round(time.monotonic()-started, 3)}
    assert sum(actions.values()) == rows and kept_count + actions['drop'] == rows
    write_json(report_path, result)
    print(json.dumps({k: result[k] for k in ('part', 'status', 'input_rows', 'output_rows', 'actions', 'errors', 'seconds')}, sort_keys=True), flush=True)
    return result


def write_aggregate(output, reports, provenance, expected_count):
    aggregate = {rid: Counter() for rid in clean._RULE}
    for report in reports:
        for rid, counts in report['rule_counts'].items():
            aggregate[rid].update(counts)
    with (output / 'rule_counts.csv').open('w', newline='', encoding='utf-8') as stream:
        columns = ['rule_id', 'eligible_by_pack', 'invocations', 'keep', 'fix', 'drop', 'error', 'not_reached']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for rid, counts in aggregate.items():
            writer.writerow(dict(rule_id=rid, **{key:counts[key] for key in columns[1:]}))
    summary = dict(provenance, completed_parts=len(reports), expected_parts=expected_count,
                   input_rows=sum(r['input_rows'] for r in reports), output_rows=sum(r['output_rows'] for r in reports),
                   actions=dict(sum((Counter(r['actions']) for r in reports), Counter())),
                   errors=sum(r['errors'] for r in reports),
                   unsupported_parts=[r['part'] for r in reports if r['status']=='unsupported_passthrough'],
                   parts=reports,
                   rule_count_semantics='sequential invocations inside actual clean.clean/_run; fixes on later-dropped docs still count; not independent rule hits')
    write_json(output / 'job_summary.json', summary)
    write_json(output / 'cleaned_manifest.json', {'sample':provenance['sample'], 'complete_sample':len(reports)==expected_count,
               'parts':[{k:r[k] for k in ('part','pack_id','split','status','output_rows','output_sha256')} for r in reports]})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, help='override TSAI sample mount for a local smoke test')
    parser.add_argument('--out', type=Path, help='override output, still restricted to ./out/<job>/')
    parser.add_argument('--only', action='append', default=[], help='manifest part path for local smoke tests; repeatable')
    args = parser.parse_args(argv)
    started = time.monotonic()
    config = read_json(args.config)
    if config.get('optional_f2') is not False:
        raise JobError('this request runs clean.py only, without optional F2')
    if config.get('unsupported_policy') != 'passthrough_with_explicit_status':
        raise JobError('unsupported pack policy must be explicit')
    if digest(Path(clean.__file__)) != config['clean_sha256']:
        raise JobError('clean.py differs from the validated job pin')
    data = manifest(config)
    known = {s['path'] for s in data['shards']}
    if set(args.only)-known:
        raise JobError('requested part is outside tokenizer sample v0')
    shards = [s for s in data['shards'] if not args.only or s['path'] in args.only]
    output = output_folder(args.out or config['output_root'])
    if output.exists():
        raise JobError('output folder already exists; choose a fresh ./out/<job>/')
    input_root = (args.input_root or Path(config['input_root'])).resolve()
    # All selected inputs are checksummed before creating outputs.
    hashes = {}
    resolved = [(s, resolve_input(input_root, s, hashes)) for s in shards]
    output.mkdir(parents=True)
    provenance = {'sample':data['name'], 'manifest_sha256':config['manifest_sha256'],
                  'clean_sha256':digest(Path(clean.__file__)), 'runner_sha256':digest(Path(__file__)),
                  'config_sha256':digest(args.config), 'scope':'partial smoke test' if args.only else 'full sample only',
                  'optional_f2':False, 'python':sys.version.split()[0],
                  'packages':{name:importlib.metadata.version(name) for name in ('pyarrow','ftfy','tree-sitter','tree-sitter-language-pack')},
                  'rule_scope':{rid:sorted(packs) for rid,packs in clean.SCOPE.items()}}
    print(json.dumps({'event':'start', 'selected_parts':len(shards), 'manifest_parts':len(data['shards']),
                      'sample_rows':data['total_docs'], 'clean_sha256':provenance['clean_sha256'],
                      'runner_sha256':provenance['runner_sha256'], 'scope':provenance['scope']}, sort_keys=True), flush=True)
    reports = []
    for shard, source in resolved:
        reports.append(process_part(source, shard, output, config['batch_size']))
    provenance.update(wall_seconds=round(time.monotonic()-started,3),
                      peak_rss_mib=peak_rss_mib())
    summary = write_aggregate(output, reports, provenance, len(data['shards']))
    print(json.dumps({'event':'complete', **{key:summary[key] for key in ('completed_parts','input_rows','output_rows','errors','actions','wall_seconds','peak_rss_mib')}}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except JobError as exc:
        print('ERROR: '+str(exc), file=sys.stderr)
        raise SystemExit(1)
