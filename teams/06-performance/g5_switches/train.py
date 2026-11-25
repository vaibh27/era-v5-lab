"""Run G5's trainer (teams/05-architecture/harness/train_week2.py) with G6 speed switches.

Same arguments as train_week2.py, plus any number of `--g6 key=value` (see g5_switches/__init__.py):

    PYTHONPATH=teams/07-infrastructure python teams/06-performance/g5_switches/train.py --g6 attn_impl=sdpa \\
        --config teams/05-architecture/configs/arms/dryrun2.yaml ...the rest of the G5 command...

With no --g6 option (or only defaults) this is G5's trainer exactly, same run identity. Any other switch is
recorded in the manifest under config.provenance.g6_switches and is part of the run identity.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
G5_HARNESS = ROOT / "teams" / "05-architecture" / "harness"
sys.path.insert(0, str(G5_HARNESS))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import g5_switches  # noqa: E402


def split_args(argv):
    """(G6 switch pairs, remaining argv for G5). Accepts `--g6 k=v` and `--g6=k=v`."""
    pairs, rest, i = [], [], 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--g6":
            if i + 1 >= len(argv):
                raise SystemExit("--g6 needs key=value")
            pairs.append(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--g6="):
            pairs.append(arg[len("--g6="):])
        else:
            rest.append(arg)
        i += 1
    return pairs, rest


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pairs, rest = split_args(argv)
    try:
        switches = g5_switches.parse(pairs)
    except ValueError as e:
        raise SystemExit(str(e))
    import train_week2

    g5_switches.install(switches)
    record = g5_switches.provenance(switches)
    g5_validate = train_week2.validate
    if switches != g5_switches.DEFAULTS:

        def validate(cfg):
            # G5 validates its own config first; the G6 record is added after, so G5's checks are unchanged.
            errors = g5_validate(cfg)
            if not errors:
                cfg.setdefault("provenance", {})["g6_switches"] = record
            return errors

        train_week2.validate = validate
    sys.argv = [str(G5_HARNESS / "train_week2.py"), *rest]
    print(f"[g5_switches] {record['switches']}", flush=True)
    try:
        return train_week2.main()
    finally:   # leave G5's modules as found (matters when called in-process, e.g. tests)
        train_week2.validate = g5_validate
        g5_switches.uninstall()


if __name__ == "__main__":
    main()
