"""Tests for scripts/reading_recall.py: the parts that need no pack. Every row here is made up.

  python3 -m unittest discover -s spec_v1_1/tests      (from teams/01-data-a)
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC / "scripts"))
import reading_recall as r  # noqa: E402

FIELDS = ("domain", "pack", "reader", "file", "from", "doc_id", "kind", "code", "offsets", "quote", "why", "source")


def classify(text, out):
    return out[0] if out[1] != text or out[0] == "drop" else "keep"


class Codes(unittest.TestCase):
    def test_rule_to_reading_code(self):
        self.assertEqual(r.parse_codes(["DA-C9-003=NEW-template-pages", " DA-A7-001 = NEW-hyphen-split-words "]),
                         {"DA-C9-003": "NEW-template-pages", "DA-A7-001": "NEW-hyphen-split-words"})
        self.assertEqual(r.parse_codes(None), {})

    def test_bad_pairs(self):
        for bad in ("DA-C9-003", "=NEW-x", "DA-C9-003="):
            with self.assertRaises(ValueError):
                r.parse_codes([bad])


class ReadingDocs(unittest.TestCase):
    def test_new_rows_of_the_domain_and_codes_only(self):
        path = Path(tempfile.mkdtemp()) / "findings_all.csv"
        rows = [
            {"domain": "math", "pack": "math-finemath", "doc_id": "d1", "kind": "new", "code": "B1"},
            {"domain": "math", "pack": "math-finemath", "doc_id": "d1", "kind": "new", "code": "B1"},
            {"domain": "math", "pack": "math-megamath", "doc_id": "d2", "kind": "new", "code": "NEW-x"},
            {"domain": "math", "pack": "math-finemath", "doc_id": "d3", "kind": "missed", "code": "B1"},
            {"domain": "science", "pack": "sci-finepdfs", "doc_id": "d4", "kind": "new", "code": "B1"},
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in FIELDS})
        got = r.reading_docs(path, "math", {"B1", "NEW-x"})
        self.assertEqual(dict(got), {"B1": {"d1": "math-finemath"}, "NEW-x": {"d2": "math-megamath"}})


class Tally(unittest.TestCase):
    def test_counts_and_fills_the_pack_from_the_reading_row(self):
        def rule(doc):
            if doc["pack_id"] != "math-finemath":
                return ("keep", doc["text"])
            if "bad" in doc["text"]:
                return ("drop", doc["text"])
            if "x" in doc["text"]:
                return ("fix", doc["text"].replace("x", "y"))
            if "boom" in doc["text"]:
                raise ValueError("boom")
            return ("keep", doc["text"])

        docs = {"a": {"doc_id": "a", "text": "bad"}, "b": {"doc_id": "b", "text": "x"},
                "c": {"doc_id": "c", "text": "ok"}, "e": {"doc_id": "e", "text": "boom"}}
        wanted = {"a": "math-finemath", "b": "math-finemath", "c": "math-finemath", "d": "math-finemath",
                  "e": "math-finemath"}
        t = r.tally(rule, classify, docs, wanted)
        self.assertEqual((t["drop"], t["fix"], t["keep"], t["missing"]), (["a"], ["b"], ["c"], ["d"]))
        self.assertEqual(t["error"], ["e (ValueError)"])

    def test_the_parts_pack_id_wins_over_the_reading_row(self):
        docs = {"a": {"doc_id": "a", "text": "t", "pack_id": "math-megamath"}}
        seen = []
        r.tally(lambda d: seen.append(d["pack_id"]) or ("keep", d["text"]), classify, docs, {"a": "math-finemath"})
        self.assertEqual(seen, ["math-megamath"])


if __name__ == "__main__":
    unittest.main()
