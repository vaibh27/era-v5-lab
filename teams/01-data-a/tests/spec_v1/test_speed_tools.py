"""Acceptance helpers must reject incomplete/erroring measurements and failed jobs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

TEAM = Path(__file__).resolve().parents[2]
CHECK = TEAM / 'oneoff/speed_check.py'
SHELL = TEAM / 'oneoff/speed_compare.sh'
parametrize = pytest.mark.parametrize


def manifest():
    return dict(limit=0, clean_py_sha256='a' * 64, packs={
        'code-cpp': dict(docs=2, errors=0, actions={'keep': 2}, input_sha256='b' * 64,
                         output_sha256='c' * 64, docs_per_s=100.0)})


def compare(tmp_path, before, after, expected=True):
    b, a = tmp_path / 'before.json', tmp_path / 'after.json'
    b.write_text(json.dumps(before)); a.write_text(json.dumps(after))
    command = [sys.executable, '-I', str(CHECK), 'compare', str(b), str(a)]
    if expected:
        command += ['--expected-packs', 'code-cpp']
    return subprocess.run(command, capture_output=True, text=True)


@parametrize('problem', ['errors', 'missing_both', 'missing_after', 'rows', 'zero_rows',
                        'input', 'output', 'actions', 'limit', 'zero_speed'])
def test_compare_rejects_invalid_equivalence(tmp_path, problem):
    b = manifest(); a = copy.deepcopy(b)
    if problem == 'errors':
        b['packs']['code-cpp']['errors'] = a['packs']['code-cpp']['errors'] = 2
    elif problem == 'missing_both':
        b['packs'] = {}; a['packs'] = {}
    elif problem == 'missing_after':
        a['packs'] = {}
    elif problem == 'rows':
        a['packs']['code-cpp']['docs'] = 1
    elif problem == 'zero_rows':
        b['packs']['code-cpp']['docs'] = a['packs']['code-cpp']['docs'] = 0
    elif problem == 'input':
        a['packs']['code-cpp']['input_sha256'] = 'd' * 64
    elif problem == 'output':
        a['packs']['code-cpp']['output_sha256'] = 'd' * 64
    elif problem == 'actions':
        a['packs']['code-cpp']['actions'] = {'drop': 2}
    elif problem == 'limit':
        a['limit'] = 1
    else:
        a['packs']['code-cpp']['docs_per_s'] = 0
    assert compare(tmp_path, b, a).returncode != 0


def test_compare_accepts_complete_zero_error_measurements(tmp_path):
    result = compare(tmp_path, manifest(), manifest())
    assert result.returncode == 0 and '1 of 1' in result.stdout
    assert compare(tmp_path, manifest(), manifest(), expected=False).returncode != 0


def test_shell_propagates_pytest_no_tests_exit_code(tmp_path):
    fake = tmp_path / 'python'
    fake.write_text('#!/bin/sh\necho "no tests ran"\nexit 5\n')
    fake.chmod(0o700)
    result = subprocess.run(['bash', str(SHELL), '--work', str(tmp_path / 'work'), 'pytest'],
                            env=dict(os.environ, PY=str(fake)), capture_output=True, text=True)
    assert result.returncode != 0 and 'RESULT: FAIL' in result.stdout


def test_run_rejects_empty_pack_folder(tmp_path):
    result = subprocess.run([sys.executable, '-I', str(CHECK), 'run', '--clean-dir',
                            str(TEAM / 'spec_v1'), '--packs', str(tmp_path),
                            '--out', str(tmp_path / 'run.json')], capture_output=True, text=True)
    assert result.returncode != 0 and 'no registered packs' in result.stderr
