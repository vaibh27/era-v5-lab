#!/usr/bin/env python3
"""
extra_compare.py
================

Chooses and reports the extra-merge candidates (v2_spec idea 1: C3 sample-v0 plus `extra` merges per script on
more pruned o200k base ids, slots_p768.json / slots_p1536.json) against C3 sample-v0 and BrahmicTokenizer-131K.

Splits are the ones mix_search.py used for C3 sample-v0. FLORES is never read.
    choose:  FineWeb-2 dev + sample v0 held-out half a   (fetch_heldout.py dev/, sample_to_corpus.py --heldout-out a/)
    report:  FineWeb-2 check + sample v0 held-out half b  (check/, b/)

Indic and English: tokens per whitespace word. Code: tokens per character. A candidate is eligible when, on the
choose splits, no Indic language is worse than C3 sample-v0 and English and code are within the COMPARE.md pass
bar (at most 2% worse than BrahmicTokenizer-131K). The largest eligible one is chosen; the report splits show
whether it holds, and every English / code change against C3 is listed.

    python extra_compare.py --heldout <fetch_heldout.py out> --heldout-s0 <heldout_s0> --brahmic <tokenizer.json> \\
        --c3 <C3 sample-v0 tokenizer.json> --cand p768=<tokenizer.json> --cand p1536=<tokenizer.json> --out compare.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slot_search import INDIC, measure  # noqa: E402

CHOOSE, REPORT = ("dev", "s0a"), ("check", "s0b")
PASS_BAR = 0.02     # COMPARE.md: at most 2% worse than BrahmicTokenizer-131K on English and code


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--heldout-s0", required=True)
    ap.add_argument("--brahmic", required=True)
    ap.add_argument("--c3", required=True)
    ap.add_argument("--cand", action="append", required=True, help="name=path/to/tokenizer.json, smallest first")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    splits = {"dev": Path(args.heldout) / "dev", "check": Path(args.heldout) / "check",
              "s0a": Path(args.heldout_s0) / "a", "s0b": Path(args.heldout_s0) / "b"}
    toks = {"brahmic": args.brahmic, "c3": args.c3, **dict(c.split("=", 1) for c in args.cand)}
    res, files = {}, {}
    for name, path in toks.items():
        tok = Tokenizer.from_file(path)
        files[name] = {"path": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        res[name] = {}
        for split, d in splits.items():
            res[name][split] = {lang: measure(tok, d / f"{lang}.txt", "char" if lang == "Code" else "word")
                                for lang in INDIC + ["En", "Code"] if (d / f"{lang}.txt").exists()}
        print(f"measured {name}", flush=True)
    for split, d in splits.items():
        files[f"heldout/{split}"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.glob("*.txt"))}

    def rel(name, ref, sets):
        return {f"{s}/{l}": res[name][s][l] / res[ref][s][l] - 1 for s in sets for l in res[name][s]}

    def ok(name, sets):
        indic = all(v <= 0 for k, v in rel(name, "c3", sets).items() if k.split("/")[1] in INDIC)
        bar = all(v <= PASS_BAR for k, v in rel(name, "brahmic", sets).items() if k.split("/")[1] in ("En", "Code"))
        return indic and bar

    def mean_indic(name, s):
        return sum(res[name][s][l] for l in INDIC if l in res[name][s]) / sum(l in res[name][s] for l in INDIC)

    cands = [c.split("=", 1)[0] for c in args.cand]
    summary = {}
    for c in cands:
        choose, report = rel(c, "c3", CHOOSE), rel(c, "c3", REPORT)
        summary[c] = {"choose_worse_than_c3": {k: v for k, v in choose.items() if v > 0},
                      "report_worse_than_c3": {k: v for k, v in report.items() if v > 0},
                      "eligible": ok(c, CHOOSE), "holds_on_report": ok(c, REPORT),
                      "mean_indic_vs_c3": {s: mean_indic(c, s) / mean_indic("c3", s) - 1 for s in splits},
                      "mean_indic_vs_brahmic": {s: mean_indic(c, s) / mean_indic("brahmic", s) - 1 for s in splits},
                      "en_code_vs_brahmic": {f"{s}/{l}": res[c][s][l] / res["brahmic"][s][l] - 1
                                             for s in splits for l in ("En", "Code") if l in res[c][s]}}
    eligible = [c for c in cands if summary[c]["eligible"]]
    chosen = eligible[-1] if eligible else None
    json.dump({"chosen": chosen, "rule": "largest candidate with no Indic language worse than C3 sample-v0 and En / code within the "
                       "pass bar (<= +2% vs BrahmicTokenizer-131K) on dev + s0a",
               "summary": summary, "fertility": res, "files": files}, open(args.out, "w"), indent=1)

    print(f"\n{'lang':6}" + "".join(f"{n + ' ' + s:>16}" for n in ["c3"] + cands for s in ("dev", "check")))
    for lang in INDIC + ["En", "Code"]:
        row = f"{lang:6}"
        for n in ["c3"] + cands:
            for s in ("dev", "check"):
                v = res[n][s].get(lang)
                row += f"{'':>16}" if v is None else (f"{v:16.4f}" if n == "c3" else f"{v:8.4f} {v / res['c3'][s][lang] - 1:+6.2%} ")
        print(row)
    for c in cands:
        m = summary[c]["mean_indic_vs_c3"]
        print(f"{c}: eligible={summary[c]['eligible']} holds_on_report={summary[c]['holds_on_report']}  mean Indic vs C3 " + "  ".join(f"{s} {v:+.2%}" for s, v in m.items())
              + "  | En/code vs Brahmic " + "  ".join(f"{k} {v:+.2%}" for k, v in summary[c]["en_code_vs_brahmic"].items()))
    print(f"chosen: {chosen}  (wrote {args.out})")


if __name__ == "__main__":
    main()
