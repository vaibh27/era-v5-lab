"""
Refresh docs/ke2_week1_explainer.html, an interactive explainer of the team's week-1 work, from
the committed results, the figures of proof_explained.md and the pinned tokenizer. The page is one
self-contained HTML file: open it in any browser.

The page's text is written by hand. Its playgrounds run on a data block that this script fills:
  - the raw bytes of every token id, the ids whose decode([id]) differs, and the script class of
    every token (ke2.script_class, as in the collision audit)
  - real tokenizations of the example sentences
  - results/: collisions.csv, collisions_summary.csv, collisions_meta.json, proof_checks.json,
    tables/manifest.json, tokenizer_decode_cpu_*, codec_v1_cpu_mps_results.json
  - proof.md's check tables, and the test functions in tests/

Before writing, it serializes the embedded bytes the way the team's scripts do and requires the
committed hashes: tokenizer_decode_cpu_sha256.txt for the printouts, and proof_checks.json's
raw_table_sha256 for the raw bytes. So the playgrounds run on exactly the audited bytes.

It also inlines the SVG figures of docs/figures/ into every <div class="fig-scroll" data-fig="NAME">
slot, turning their fixed colours into the page's theme tokens so they follow light and dark mode.

Run from the lab repo root, with the tokenizer found as in README.md:
  python teams/04-ke2/src/docs/build_explainer.py
"""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make ke2 importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ke2.config import RESULTS_DIR, TEAM_DIR, TOKENIZER_SHA256
from ke2.script_class import script_class
from ke2.tokenizer import decode_bytes_for_id, load_tokenizer, raw_bytes_for_id

PAGE = TEAM_DIR / "docs" / "ke2_week1_explainer.html"
FIGS = TEAM_DIR / "docs" / "figures"
BLOCK = re.compile(r'(<script type="application/json" id="ke2-data">)(.*?)(</script>)', re.S)
FIG_SLOT = re.compile(r'(<div class="fig-scroll" data-fig="(\w+)">)(.*?)(</div>)', re.S)
TAG = re.compile(r'<(\w+)((?:\s+[\w:-]+="[^"]*")*)\s*(/?>)')
# proof_figures.py colour -> page token (defined for light and dark in the page's :root blocks)
THEME = {
    "#ffffff": "--fig-bg",
    "#0f2230": "--ink",
    "#3f5260": "--ink-2",
    "#667884": "--muted",
    "#d8e0e4": "--rule",
    "#b7c4cb": "--rule-strong",
    "#f3f6f7": "--paper",
    "#2a78d6": "--raw",
    "#e2edfb": "--raw-wash",
    "#eb6834": "--dec",
    "#fde8dd": "--dec-wash",
    "#ffd675": "--mark",
    "#c98d0c": "--mark-edge",
    "#0a8a0a": "--good",
    "#e1f3e1": "--good-bg",
    "#b42318": "--bad-ink",
    "#e2e8eb": "--fig-grid",
    "#d5dde1": "--fig-grid",
    "#e6ecef": "--fig-grid",
}

EXAMPLES = [
    ("Hindi (token A)", "Devanagari", "भारत एक देश है"),
    ("Tamil (tokens B and C)", "Tamil", "மிஸ் ம்ம்"),
    ("Bengali", "Bengali", "আমি বাংলায় গান গাই"),
    ("Tamil", "Tamil", "தமிழ் மொழி"),
    ("Telugu", "Telugu", "తెలుగు భాష"),
    ("Punjabi", "Gurmukhi", "ਪੰਜਾਬੀ ਬੋਲੀ"),
    ("Gujarati", "Gujarati", "ગુજરાતી ભાષા"),
    ("Odia", "Odia", "ଓଡ଼ିଆ ଭାଷା"),
    ("Kannada", "Kannada", "ಕನ್ನಡ ಭಾಷೆ"),
    ("Malayalam", "Malayalam", "മലയാളം"),
    ("English", "Latin", "Hello, world!"),
    ("Python", "Latin", "def f(x): return x**2"),
    ("Santali", "Ol Chiki", "ᱚᱞ ᱪᱤᱠᱤ"),
    ("Manipuri", "Meetei Mayek", "ꯃꯤꯇꯩ ꯃꯌꯦꯛ"),
    ("Sinhala", "Sinhala", "අකුරු"),
    ("Japanese", "Kanji and kana", "日本語のテキスト"),
    ("Emoji", "Emoji", "😀🙂"),
]

# The check tables of proof.md, sections 1 to 3 (synthetic strings; proof.md reports "pass" for each).
PROOF_TABLES = [
    (
        "Lemma 1",
        "exactly 2 distinct values, fp32 and bf16",
        "P = 32, 64; every L from 1 to P; 3 random strings each",
        "tests/test_lemma1.py",
    ),
    ("Lemma 1", "the two values equal the closed form h, u of step 3", "P = 32, 64; every L", "tests/test_lemma1.py"),
    (
        "Lemma 1",
        "gap h − u after rounding to bf16 is above 8 (smallest 16.0)",
        "P = 32, 64; every L",
        "tests/test_lemma1.py",
    ),
    (
        "Lemma 1",
        "bytes read back from the code equal the input, fp32 and bf16",
        "edge strings and 3,000 random strings per P",
        "tests/test_lemma1.py",
    ),
    (
        "Lemma 1",
        "the empty string gives the zero code, no non-empty string does",
        "200 random strings per P",
        "tests/test_lemma1.py",
    ),
    (
        "Lemma 1",
        "distinct strings give distinct codes, fp32 and bf16",
        "at least 20,000 strings per P, with near neighbours",
        "tests/test_lemma1.py",
    ),
    (
        "Lemma 2",
        "(a), (b), (c), (e) for both cut rules, on raw bytes",
        "P = 32, 64; 5,000 strings each",
        "tests/test_lemma2.py",
    ),
    (
        "Lemma 2",
        "(d): valid UTF-8 stays valid and is never emptied; both rules agree with utf8_safe_truncate",
        "P = 32, 64; 5,000 strings each",
        "tests/test_lemma2.py",
    ),
    (
        "Lemma 2",
        "(c): a cut token is empty or valid; the empty case really occurs; the proposed rule never empties",
        "P = 32, 64; 5,000 strings each",
        "tests/test_lemma2.py",
    ),
    (
        "Lemma 2",
        "the 36-byte example: the repository rule keeps 0 bytes, the proposed rule 30",
        "one token, P = 32",
        "tests/test_lemma2.py",
    ),
    (
        "Theorem 1",
        "grouping by code equals grouping by cut string, fp32 and bf16",
        "1,717 synthetic tokens with every cause; P = 32, 64; both rules",
        "tests/test_theorem1.py",
    ),
    (
        "Theorem 1",
        "every pair sharing a code gets one of the five causes; all five occur",
        "the same vocabulary",
        "tests/test_theorem1.py",
    ),
    ("Theorem 1", "pair_cause on small hand-made cases", "hand-made pairs", "tests/test_theorem1.py"),
]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def decode_cpu_hash(dec: list[bytes]) -> str:
    """The hash of src/determinism/tokenizer_decode_cpu.py (format ke2-decode-cpu-v1)."""
    h = hashlib.sha256(b"ke2-decode-cpu-v1\n")
    for i, b in enumerate(dec):
        h.update(struct.pack(">II", i, len(b)) + b)
    return h.hexdigest()


def raw_table_hash(raw: list[bytes]) -> str:
    """raw_table_sha256 of src/proofs/proof_checks.py: id as 4 bytes, length as 2 bytes, then the bytes."""
    h = hashlib.sha256()
    for i, b in enumerate(raw):
        h.update(struct.pack(">IH", i, len(b)) + b)
    return h.hexdigest()


def test_functions() -> list[str]:
    out = []
    for f in sorted((TEAM_DIR / "tests").glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        out += [
            f"tests/{f.name}::{n.name}"
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
        ]
    return out


def proof_data() -> dict:
    pc = json.loads((RESULTS_DIR / "proof_checks.json").read_text(encoding="utf-8"))
    checks = [
        {"part": p, "check": c, "size": s, "test": t, "result": "PASS", "ok": True} for p, c, s, t in PROOF_TABLES
    ]
    real_ok = True
    for P in (32, 64):
        for kind in ("raw", "decode"):
            r = pc[f"P{P}_{kind}"]
            ok = all(r[k]["partitions_equal"] for k in ("float32", "bfloat16"))
            real_ok &= ok
            checks.append(
                {
                    "part": "Real tokenizer",
                    "check": f"{kind} table, P = {P}: groups by code equal groups by cut string, fp32 and bf16",
                    "size": f"{r['float32']['distinct_codes']:,} distinct codes, {r['groups']} groups",
                    "test": "tests/test_proof_checks.py",
                    "result": "PASS" if ok else "FAIL",
                    "ok": ok,
                }
            )
    return {
        "checks": checks,
        "real_ok": real_ok,
        "raw_table_sha256": pc["raw_table_sha256"],
        "not_valid_utf8": pc["not_valid_utf8"],
        "max_bytes": pc["max_bytes"],
        "bf16_min_gap": pc.get("P32_bf16_min_gap"),
    }


def collision_data(V: int) -> dict:
    meta = json.loads((RESULTS_DIR / "collisions_meta.json").read_text(encoding="utf-8"))
    tables = []
    for key, t in meta["totals"].items():
        kind, P = key.split("_P")
        tables.append(
            {
                "table": key,
                "kind": kind,
                "P": int(P),
                "vocab_size": V,
                "groups": t["n_groups"],
                "colliding_tokens": t["n_tokens_in_groups"],
                "codes_lost": t["n_excess_tokens"],
                "distinct_codes": V - t["n_excess_tokens"],
                "causes": t["causes"],
                "other": t["causes"].get("other", 0),
            }
        )
    groups: dict[str, list] = {}
    for r in read_csv(RESULTS_DIR / "collisions.csv"):
        row = bytes.fromhex(r["bytes_hex"])
        groups.setdefault(f"{r['extraction']}_P{r['P']}", []).append(
            {
                "group": int(r["group_id"]),
                "size": int(r["n_tokens"]),
                "cause": r["cause"],
                "row": r["bytes_hex"],
                "text": row.decode("utf-8", "replace"),
                "ids": [int(x) for x in re.findall(r"\d+", r["token_ids"])],
                "classes": r["member_classes"] or r["script_class"],
            }
        )
    by_script: dict[str, list] = {}
    for r in read_csv(RESULTS_DIR / "collisions_summary.csv"):
        by_script.setdefault(f"{r['extraction']}_P{r['P']}", []).append(
            {"cls": r["script_class"], "tokens": int(r["n_tokens"]), "colliding": int(r["n_colliding_tokens"])}
        )
    return {"tables": tables, "groups": groups, "by_script": by_script}


def determinism_data(sha: str) -> dict:
    meta = json.loads((RESULTS_DIR / "tokenizer_decode_cpu_metadata.json").read_text(encoding="utf-8"))
    codec = json.loads((RESULTS_DIR / "codec_v1_cpu_mps_results.json").read_text(encoding="utf-8"))
    lines = [
        ln.strip()
        for ln in (RESULTS_DIR / "tokenizer_decode_cpu_sha256.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    runs = [
        {k: r.get(k) for k in ("system", "machine", "python_version", "tokenizers_version", "token_count", "sha256")}
        for r in meta["runs"]
    ]
    mps = {}
    for line in (RESULTS_DIR / "codec_v1_cpu_mps_sha256.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            P, _, m, _ = line.split()
            mps.setdefault(P, set()).add(m)
    hashes = {"tokenizer decode, 3 CPU runs": {"sha256": sha, "match": meta["all_match"] and len(set(lines)) == 1}}
    for P, s in sorted(mps.items()):
        hashes[f"forward codec P = {P}, 3 MPS runs"] = {"sha256": sorted(s)[0], "match": len(s) == 1}
    summary = {P: {k: v for k, v in d.items() if k != "strict_probe_statuses"} for P, d in codec["summary"].items()}
    return {
        "file_lines": [
            {
                "sha256": h,
                "run": f"pass {k + 1}",
                "settings": f"{r['system']} {r['machine']}, Python {r['python_version']}, tokenizers {r['tokenizers_version']}",
            }
            for k, (h, r) in enumerate(zip(lines, runs, strict=True))
        ],
        "runs": runs,
        "all_match": meta["all_match"],
        "hashes": hashes,
        "codec": summary,
        "process_policy": meta.get("process_policy", ""),
    }


def inline_svg(name: str) -> str:
    """One figure as inline SVG: scalable, with its own marker id, coloured by theme tokens."""
    s = (FIGS / f"{name}.svg").read_text(encoding="utf-8").strip()
    s, n = re.subn(
        r'<svg xmlns="http://www.w3.org/2000/svg" width="\d+" height="\d+"',
        '<svg xmlns="http://www.w3.org/2000/svg" class="pfig"',
        s,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"{name}.svg: unexpected root element")
    s = s.replace('id="ah"', f'id="ah-{name}"').replace("url(#ah)", f"url(#ah-{name})")

    def themed(m):
        attrs, style = m.group(2), []
        for prop in ("fill", "stroke"):
            a = re.search(rf' {prop}="(#[0-9a-f]{{6}})"', attrs)
            if a:
                if a.group(1) not in THEME:
                    raise SystemExit(f"{name}.svg: no theme token for {a.group(1)}")
                style.append(f"{prop}:var({THEME[a.group(1)]})")
                attrs = attrs.replace(a.group(0), "", 1)
        return f"<{m.group(1)}{attrs}" + (f' style="{";".join(style)}"' if style else "") + m.group(3)

    return TAG.sub(themed, s)


def main() -> int:
    tb = load_tokenizer(expected_sha256=TOKENIZER_SHA256)
    V = tb.vocab_size
    raw = [raw_bytes_for_id(tb, i) for i in range(V)]
    dec = [decode_bytes_for_id(tb, i) for i in range(V)]
    committed = (RESULTS_DIR / "tokenizer_decode_cpu_sha256.txt").read_text(encoding="utf-8").split()[0]
    if decode_cpu_hash(dec) != committed:
        raise SystemExit("printout bytes differ from results/tokenizer_decode_cpu_sha256.txt")
    proof = proof_data()
    if raw_table_hash(raw) != proof["raw_table_sha256"]:
        raise SystemExit("raw bytes differ from raw_table_sha256 in results/proof_checks.json")

    classes = [script_class(b, is_special=i in tb.special_ids, detail=True) for i, b in enumerate(raw)]
    names = sorted(set(classes))
    if len(names) > 255:
        raise SystemExit("too many script classes for one byte each")
    index = {n: i for i, n in enumerate(names)}
    manifest = json.loads((RESULTS_DIR / "tables" / "manifest.json").read_text(encoding="utf-8"))
    data = {
        "V": V,
        "tokens": {
            "lengths": b64(np.array([len(r) for r in raw], dtype=np.uint8)),
            "blob": base64.b64encode(b"".join(raw)).decode("ascii"),
            "decode_diff": {str(i): d.hex() for i, (r, d) in enumerate(zip(raw, dec, strict=True)) if r != d},
            "special": sorted(tb.special_ids),
            "class_names": names,
            "classes": b64(np.array([index[c] for c in classes], dtype=np.uint8)),
        },
        "examples": [
            {"label": lab, "script": scr, "text": t, "ids": tb.tok.encode(t, add_special_tokens=False).ids}
            for lab, scr, t in EXAMPLES
        ],
        "collisions": collision_data(V),
        "proof": proof,
        "determinism": determinism_data(committed),
        "manifest": {
            f"{t['kind']}_P{t['pos_dim']}": {
                "distinct_rows": t["n_unique_rows"],
                "colliding": t["n_colliding_tokens"],
                "truncated": t["n_truncated"],
                "rows_with_u_fffd": t["n_rows_containing_U+FFFD"],
                "sha256": t["sha256"],
            }
            for t in manifest["tables"].values()
        },
        "tests": test_functions(),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/").replace("<!--", "<\\!--").replace(chr(0xFFFD), "\\" + "ufffd")
    html = PAGE.read_text(encoding="utf-8")
    if not BLOCK.search(html):
        raise SystemExit(f"data block not found in {PAGE}")
    html = BLOCK.sub(lambda m: m.group(1) + payload + m.group(3), html, count=1)
    slots = [m.group(2) for m in FIG_SLOT.finditer(html)]
    missing = sorted({p.stem for p in FIGS.glob("*.svg")} - set(slots))
    if missing:
        raise SystemExit(f"figures with no slot on the page: {missing}")
    html = FIG_SLOT.sub(lambda m: m.group(1) + inline_svg(m.group(2)) + m.group(4), html)
    PAGE.write_text(html, encoding="utf-8", newline="\n")
    print(
        f"wrote docs/{PAGE.name}: {len(html) / 1e6:.2f} MB, {len(slots)} figures, {len(data['tests'])} test functions listed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
