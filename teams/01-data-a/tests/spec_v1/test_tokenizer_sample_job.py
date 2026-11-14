"""Streaming sample-job tests, using made-up rows only."""
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SPEC = Path(__file__).resolve().parents[2] / "spec_v1"
JOBS = SPEC.parent / 'jobs' / '2026-11-02-tokenizer-sample-v0-clean'
spec = importlib.util.spec_from_file_location('sample_job_test', SPEC / 'run_tokenizer_sample.py')
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


MANIFEST_REL = (JOBS / 'tokenizer_sample_v0_manifest.json').relative_to(job.REPO).as_posix()


def submitted_config():
    """The submitted config as committed (never edited), with its manifest path pointing at where the manifest now lives."""
    config = json.loads((JOBS / 'tokenizer_sample_v0_clean.json').read_text(encoding='utf-8'))
    return dict(config, manifest=MANIFEST_REL)


def test_submitted_config_matches_its_manifest():
    # A submitted job is a record: its clean_sha256 pins the cleaner that ran, not the current one, so it is not compared with clean.py.
    config = submitted_config()
    assert len(config['clean_sha256']) == 64
    assert config['manifest_sha256'] == job.digest(JOBS / 'tokenizer_sample_v0_manifest.json')
    assert config['expected_parts'] == {'train': 48, 'heldout': 38}
    assert config['optional_f2'] is False
    assert config['unsupported_policy'] == 'passthrough_with_explicit_status'


def test_pr244_rerun_config_pins_current_cleaner_and_recorded_smoke():
    rerun = SPEC.parent / 'jobs' / '2026-11-12-pr244-rerun'
    path = rerun / 'tokenizer_sample_v0_clean.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    smoke = json.loads((rerun / 'logs/job_summary.json').read_text(encoding='utf-8'))
    assert config['clean_sha256'] == job.digest(SPEC / 'clean.py') == smoke['clean_sha256']
    assert smoke['config_sha256'] == job.digest(path)
    assert config['manifest'] == MANIFEST_REL
    assert len(job.manifest(config)['shards']) == 86
    assert config['optional_f2'] is False
    assert config['unsupported_policy'] == 'passthrough_with_explicit_status'
    assert config['output_root'] != submitted_config()['output_root']
    assert smoke['scope'] == 'partial smoke test' and smoke['completed_parts'] == 2
    assert smoke['errors'] == 0 and smoke['input_rows'] == smoke['output_rows'] + smoke['actions']['drop']


@pytest.mark.parametrize('platform,rss', [('darwin', 512 * 1024 * 1024), ('linux', 512 * 1024)])
def test_peak_memory_uses_platform_units(platform, rss, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(job, 'resource', SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=rss)))
    monkeypatch.setattr(job.sys, 'platform', platform)
    assert job.peak_rss_mib() == 512.0


def test_peak_memory_is_unavailable_without_resource(monkeypatch):
    monkeypatch.setattr(job, 'resource', None)
    assert job.peak_rss_mib() is None


def part(tmp_path, pack='en-dclm', split='train'):
    rows = [dict(doc_id='d1', text='Area of a circle.\n', pack_id=pack, lang_label='en', url=''),
            dict(doc_id='d2', text='Area\u200b of a circle.\n', pack_id=pack, lang_label='en', url=''),
            dict(doc_id='d3', text='Subscribe to our newsletter\n', pack_id=pack, lang_label='en', url='')]
    path = tmp_path / 'input' / split / pack / 'part-00000.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    shard = dict(path=f'{split}/{pack}/part-00000.parquet', pack_id=pack, split=split,
                 rows=len(rows), bytes=path.stat().st_size, sha256=job.digest(path))
    return path, shard, rows


def test_counting_runs_actual_cleaner_with_identical_results():
    rows = [dict(doc_id='d', text=t, pack_id='en-dclm', lang_label='en', url='')
            for t in ('Area of a circle.\n', 'Area\u200b of a circle.\n', 'Subscribe to our newsletter\n')]
    expected = [job.clean.clean(row) for row in rows]
    original_run, original_ids = job.clean._run, dict(job.clean._ID)
    counter = job.RuleCounter()
    with counter.instrument():
        assert [job.clean.clean(row) for row in rows] == expected
    assert job.clean._run is original_run and job.clean._ID == original_ids
    assert counter.counts['DA-A1-004']['fix'] == 1
    assert sum(counter.counts['DA-F1-005'].values()) == 2  # drop short-circuits later rules


def test_streaming_output_and_rule_counts_match_cleaner(tmp_path):
    path, shard, rows = part(tmp_path)
    out = tmp_path / 'out'
    result = job.process_part(path, shard, out, batch_size=1)
    expected = [(row['doc_id'], job.clean.clean(row)) for row in rows]
    written = pq.read_table(out / shard['path']).to_pylist()
    assert [(row['doc_id'], row['text']) for row in written] == [(d, value[1]) for d, value in expected if value[0] != 'drop']
    assert result['actions'] == {'keep':1, 'fix':1, 'drop':1}
    assert result['rule_counts']['DA-F1-005']['not_reached'] == 1
    assert result['input_rows'] == result['output_rows'] + result['actions']['drop']
    with pytest.raises(job.JobError, match='overwrite'):
        job.process_part(path, shard, out, 1)


def test_unknown_pack_is_explicit_passthrough(tmp_path):
    path, shard, rows = part(tmp_path, 'cadj-commits')
    out = tmp_path / 'out'
    result = job.process_part(path, shard, out, 2)
    assert result['status'] == 'unsupported_passthrough'
    assert result['rule_counts'] == {} and result['actions'] == {'keep':3}
    assert pq.read_table(out / shard['path']).to_pylist() == rows


def test_heldout_and_train_outputs_are_separate(tmp_path):
    out = tmp_path / 'out'
    for split in ('train', 'heldout'):
        path, shard, _ = part(tmp_path, split=split)
        job.process_part(path, shard, out, 2)
    assert (out/'train/en-dclm/part-00000.parquet').is_file()
    assert (out/'heldout/en-dclm/part-00000.parquet').is_file()


def test_hash_resolution_handles_browser_rename_but_rejects_wrong_bytes(tmp_path):
    path, shard, _ = part(tmp_path)
    target = tmp_path/'organized/en-dclm/train/part-00000_part2.parquet'
    target.parent.mkdir(parents=True)
    path.rename(target)
    assert job.resolve_input(tmp_path/'organized', shard, {}) == target
    target.write_bytes(b'wrong data')
    with pytest.raises(job.JobError, match='checksum'):
        job.resolve_input(tmp_path/'organized', shard, {})


def test_errors_are_counted_and_partial_output_is_not_published(tmp_path, monkeypatch):
    path, shard, _ = part(tmp_path)
    make = job.clean._RULE['DA-A1-004']
    original = make()

    def broken(doc):
        raise ValueError('synthetic private content that must not appear in reports')

    monkeypatch.setattr(job.clean, '_PIPELINES', {'english':(lambda: broken,)})
    # Replace the registered factory consistently so the actual clean() path raises.
    failed_factory = job.clean._PIPELINES['english'][0]
    monkeypatch.setattr(job.clean, '_RULE', {'DA-A1-004':failed_factory, 'f1_rule.py':job.clean._shared_f1_rule})
    monkeypatch.setattr(job.clean, '_ID', {failed_factory:'DA-A1-004', job.clean._shared_f1_rule:'f1_rule.py'})
    out = tmp_path/'out'
    with pytest.raises(job.JobError, match='ValueError'):
        job.process_part(path, shard, out, 1)
    text = (out/'failure.json').read_text()
    assert 'synthetic private content' not in text
    assert json.loads(text)['rule_counts']['DA-A1-004']['error'] == 1
    assert not (out/shard['path']).exists()


def test_manifest_and_output_scope_are_bounded(tmp_path, monkeypatch):
    config = submitted_config()
    data = job.manifest(config)
    assert len(data['shards']) == 86
    assert sum(s['rows'] for s in data['shards']) == 1444766
    with pytest.raises(job.JobError, match='checksum'):
        job.manifest(dict(config, manifest_sha256='0'*64))
    monkeypatch.chdir(tmp_path)
    assert job.output_folder('out/job') == tmp_path/'out/job'
    with pytest.raises(job.JobError, match='output'):
        job.output_folder('../elsewhere')
    with pytest.raises(job.JobError, match='manifest path'):
        job.safe_path('../train/escape.parquet')


def test_all_dropped_part_still_has_a_valid_parquet_schema(tmp_path):
    path, shard, rows = part(tmp_path)
    for row in rows:
        row['text'] = 'Subscribe to our newsletter\n'
    pq.write_table(pa.Table.from_pylist(rows), path)
    shard.update(bytes=path.stat().st_size, sha256=job.digest(path))
    out = tmp_path/'out'
    result = job.process_part(path, shard, out, 1)
    table = pq.read_table(out/shard['path'])
    assert table.num_rows == result['output_rows'] == 0
    assert table.schema == pq.read_schema(path)
    assert result['actions'] == {'drop':3}


def test_aggregate_includes_zero_rules_and_explicit_partial_manifest(tmp_path):
    path, shard, _ = part(tmp_path, 'cadj-commits')
    out = tmp_path/'out'
    report = job.process_part(path, shard, out, 2)
    summary = job.write_aggregate(out, [report], {'sample':'synthetic'}, 86)
    assert summary['unsupported_parts'] == [shard['path']]
    assert json.loads((out/'cleaned_manifest.json').read_text())['complete_sample'] is False
    import csv
    counts = list(csv.DictReader((out/'rule_counts.csv').open()))
    assert len(counts) == len(job.clean._RULE)
    assert all(row['invocations'] == '0' for row in counts)
