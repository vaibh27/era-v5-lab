#!/usr/bin/env python3
"""sa_marker_build.py - the extra Sanskrit marker words of lang_label_config.json (#675), from grammar tables.

The forms come from sa_paradigms.py (pronoun declensions, verb paradigms, indeclinables), not from any corpus. The
corpus is used only as a safety filter: a form is dropped when it occurs in more than MAX_HI_MR of the Hindi and
Marathi diagnosis documents (the packs of diag_packs_dir), or when a reader excluded it by hand (EXCLUDE, with the
reason). Counts are documents that contain the form, words split as lang_id.WORD does, after NFC.

  -> sa_marker_words.txt  the kept forms, one per line (markers.sa.words_file in the config)
  -> sa_marker_words.csv  every form: paradigm, documents in ind-sa-fw2 and in Hindi/Marathi, kept or why dropped

Setup as lang_label_check.py (pyarrow):  $PY sa_marker_build.py [--packs-dir <folder with ind-*.parquet>]
"""
import argparse
import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path

import lang_id
from sa_paradigms import build

HERE = Path(__file__).resolve().parent
MAX_HI_MR = 1  # a form in 2 or more of the 3,500 Hindi and Marathi documents is a Hindi or Marathi word too
HI_MR_PACKS = ["ind-hi-fw2", "ind-hi-hplt", "ind-hi-indiccorp", "ind-hi-sangraha-s", "ind-hi-sangraha-v",
               "ind-mr-fw2", "ind-mr-sangraha-v"]
SA_PACK = "ind-sa-fw2"  # for information only: a form unseen in the 500 Sanskrit documents is kept
EXCLUDE = {  # form: reason (a grammar form that is also a Hindi or Marathi word, found after the filter)
    "लिखत": "Hindi noun (as in लिखत-पढ़त); 8 hits in one Hindi document",
}


def doc_counts(packs_dir, packs):
    """Number of documents of the packs that contain each word."""
    import pyarrow.parquet as pq  # noqa: PLC0415
    counts = Counter()
    for pack in packs:
        for text in pq.read_table(Path(packs_dir) / f"{pack}.parquet", columns=["text"]).column("text").to_pylist():
            counts.update(set(lang_id.WORD.findall(unicodedata.normalize("NFC", text or ""))))
    return counts


def decide(form, hi_mr):
    if form in EXCLUDE:
        return "dropped: " + EXCLUDE[form]
    if hi_mr > MAX_HI_MR:
        return f"dropped: in {hi_mr} Hindi/Marathi documents"
    return "kept"


def main():
    cfg = json.loads((HERE / "lang_label_config.json").read_text(encoding="utf-8"))  # not load_config: no words file yet
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--packs-dir", default=str(HERE / cfg["diag_packs_dir"]))
    a = ap.parse_args()
    sa, hi_mr = doc_counts(a.packs_dir, [SA_PACK]), doc_counts(a.packs_dir, HI_MR_PACKS)
    rows = []
    for form, group, source in build():
        form = unicodedata.normalize("NFC", form)
        rows.append({"form": form, "group": group, "paradigm": source, "docs_sa": sa[form], "docs_hi_mr": hi_mr[form],
                     "decision": decide(form, hi_mr[form])})
    with open(HERE / "sa_marker_words.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    kept = [r["form"] for r in rows if r["decision"] == "kept"]
    (HERE / "sa_marker_words.txt").write_text(
        "# Sanskrit marker words from grammar tables (#675), built by sa_marker_build.py; do not edit by hand\n"
        + "\n".join(kept) + "\n", encoding="utf-8")
    print(f"{len(rows)} forms, {len(kept)} kept -> sa_marker_words.txt, sa_marker_words.csv")


if __name__ == "__main__":
    main()
