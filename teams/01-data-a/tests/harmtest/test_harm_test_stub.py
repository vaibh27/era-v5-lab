"""tools/harm_test.py is a forwarder to harmtest/harm_test.py (other teams use the old path). Delete with the stub in Week 4."""
import subprocess
import sys
from pathlib import Path

TEAM = Path(__file__).resolve().parents[2]
STUB = TEAM / "tools" / "harm_test.py"
REAL = TEAM / "harmtest" / "harm_test.py"


def run(script, *args):
    return subprocess.run([sys.executable, "-I", str(script), *args], capture_output=True, text=True)


def test_stub_help_matches_the_real_file():
    stub, real = run(STUB, "--help"), run(REAL, "--help")
    assert stub.returncode == real.returncode == 0
    assert stub.stdout == real.stdout
    assert "harmtest/harm_test.py" in stub.stderr  # the moved note goes to stderr only


def test_stub_passes_the_exit_code_through():
    stub, real = run(STUB), run(REAL)  # no arguments: argparse exits with 2
    assert stub.returncode == real.returncode == 2
