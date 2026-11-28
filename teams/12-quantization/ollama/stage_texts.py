#!/usr/bin/env python3
"""Stage plain text for measure_export_gaps.py once, WITH network: a few long Wikipedia articles per language.

    python stage_texts.py --out ./inputs/texts [--langs en hi bn ta te kn ml mr gu pa]

Writes <out>/<lang>.txt (one paragraph per line) and <out>/sources.json (titles, revision ids, sha256 of each file).
Wikipedia text is CC BY-SA 4.0; it is read by the job and never committed. One request per article, with a pause,
to stay inside Wikipedia's API rate limit. The job itself (run_ollama_job.py) needs no network.
"""
from __future__ import annotations

import argparse, hashlib, json, time, urllib.parse, urllib.request
from pathlib import Path

ARTICLES = {
    "en": ["India", "Bengaluru", "Indian Institute of Science"],
    "hi": ["भारत", "दिल्ली", "हिन्दी"],
    "bn": ["বাংলাদেশ", "কলকাতা", "রবীন্দ্রনাথ ঠাকুর"],
    "ta": ["இந்தியா", "சென்னை", "தமிழ்"],
    "te": ["భారతదేశం", "హైదరాబాదు", "తెలుగు"],
    "kn": ["ಭಾರತ", "ಬೆಂಗಳೂರು", "ಕನ್ನಡ"],
    "ml": ["ഇന്ത്യ", "കേരളം", "മലയാളം"],
    "mr": ["भारत", "मुंबई", "मराठी भाषा"],
    "gu": ["ભારત", "અમદાવાદ", "ગુજરાતી ભાષા"],
    "pa": ["ਪੰਜਾਬ", "ਅੰਮ੍ਰਿਤਸਰ", "ਗੁਰੂ ਨਾਨਕ"],
}
UA = {"User-Agent": "era-v5-g12-export-gap-check/0.1 (theschoolofai/era-v5-lab, teams/12-quantization)"}


def fetch(lang: str, title: str) -> tuple[str, int]:
    q = urllib.parse.urlencode({"action": "query", "titles": title, "prop": "extracts|revisions", "rvprop": "ids",
                                "explaintext": 1, "format": "json", "redirects": 1})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"https://{lang}.wikipedia.org/w/api.php?{q}",
                                                               headers=UA), timeout=60) as r:
                page = next(iter(json.load(r)["query"]["pages"].values()))
                return page.get("extract", ""), page.get("revisions", [{}])[0].get("revid", 0)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"rate-limited on {lang}:{title}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--langs", nargs="+", default=list(ARTICLES), choices=list(ARTICLES))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sources = {}
    for lang in a.langs:
        lines, revs = [], {}
        for title in ARTICLES[lang]:
            text, rev = fetch(lang, title)
            revs[title] = rev
            lines += [l.strip() for l in text.split("\n") if len(l.strip()) > 20 and not l.strip().startswith("==")]
            time.sleep(2)
        path = a.out / f"{lang}.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        sources[lang] = {"revisions": revs, "paragraphs": len(lines),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(lang, len(lines), "paragraphs", flush=True)
    (a.out / "sources.json").write_text(json.dumps(sources, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
