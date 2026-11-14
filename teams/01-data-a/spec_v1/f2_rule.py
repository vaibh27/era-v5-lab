"""F2, benchmark leak: G9's decontamination check (teams/09-evaluation/decontam, PR 31) as a rule.

A doc is dropped when it shares one run of words (13 words, 20 tokens for code) with an item of G9's fingerprint set,
or holds a benchmark canary string. The check is G9's own code and set, unchanged: only the full set is the formal
check, and a report gives its set_sha256.

  python spec_v1/run_clean.py --pack <pack>.parquet --out out --f2      clean.py, then F2 on each doc it keeps
  python -I tools/harm_test.py --pack <pack>.parquet --gold gold/<pack>_gold.csv --rules spec_v1/tools/f2_rule.py --rule F2 --timeout 300

--timeout 300: the set loads inside the first rule call, past harm_test.py's default of 5 s.
The set is pinned: SET names its folder and SET_SHA256 its set_sha256, and a set with another set_sha256 is refused.
F2_FINGERPRINTS can point to a copy of the same set elsewhere. A future set needs its own harm test, then both
SET and SET_SHA256 changed here. The v2 migration evidence is results/spec_v1/f2/v2_validation.json.
The full set needs about 1.1 GB of memory and loads in about 20 s, once, on the first doc.
It finds copies only (G9's README): a reworded, translated or very short item is not found, nor a benchmark outside the set.
"""
import hashlib
import importlib.util
import os
import sys
from functools import cache
from pathlib import Path

SET = "v2"  # G9's fingerprint set: teams/09-evaluation/fingerprints/<SET>
SET_SHA256 = "2826a3b883210c653979103c09ce1c79e40fc4f7034b54027f2836482370d22a"


def _g9():
    """teams/09-evaluation/decontam: from this file, or from the working folder when harm_test.py runs this file's source."""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "teams" / "09-evaluation" / "decontam" / "fpset.py").is_file():
            return p / "teams" / "09-evaluation" / "decontam"
    raise FileNotFoundError("teams/09-evaluation/decontam not found: run from inside the lab repository")


@cache
def fingerprints(benchmarks=None):
    """G9's fingerprint set, checked against the pin and its SHA256SUMS, loaded once. benchmarks (a tuple of IDs) is for the tests only."""
    g9 = _g9()
    path = Path(os.environ.get("F2_FINGERPRINTS") or g9.parent / "fingerprints" / SET)
    if not (path / "SHA256SUMS").is_file():
        raise FileNotFoundError(f"no fingerprint set at {path}: SET or F2_FINGERPRINTS names a missing folder")
    got = hashlib.sha256((path / "SHA256SUMS").read_text(encoding="utf-8").encode("utf-8")).hexdigest()  # as fpset.py computes it
    if got != SET_SHA256:
        raise ValueError(f"the fingerprint set at {path} has set_sha256 {got}, not the pinned {SET_SHA256}. "
                         "A new set needs its own harm test before SET and SET_SHA256 change.")
    # fpset imports a module named normalize, and spec_v1/tools has one too (Team 1's). Load G9's for this import only.
    saved = sys.modules.pop("normalize", None)
    sys.path.insert(0, str(g9))
    try:
        spec = importlib.util.spec_from_file_location("g9_fpset", g9 / "fpset.py")
        fpset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fpset)
    finally:
        sys.path.remove(str(g9))
        sys.modules.pop("normalize", None)
        if saved is not None:
            sys.modules["normalize"] = saved
    return fpset.Fingerprints(path, list(benchmarks) if benchmarks else None)


def check(text):
    """(flagged, hits). A hit names the benchmark item, the runs shared with it and the character offset, never the text."""
    return fingerprints().check(text)


def f2(doc):
    return ("drop" if check(doc["text"])[0] else "keep", doc["text"])


RULES = {"F2": f2}  # disease code -> function, the form harm_test.py reads
