"""Moved to ../harmtest/harm_test.py. This forwarder will be removed in Week 4."""
import runpy, sys
from pathlib import Path
sys.stderr.write("note: harm_test.py moved to teams/01-data-a/harmtest/harm_test.py\n")
runpy.run_path(str(Path(__file__).resolve().parent.parent / "harmtest" / "harm_test.py"), run_name="__main__")
