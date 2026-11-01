"""
Draw the figures of proof_explained.md (a companion to proof.md) as SVG files in docs/figures/.

Every number on a figure is computed here, from the upstream kronecker_codec and the closed
forms of proof.md, Lemma 1 step 3, and the script stops if the two disagree. The three example
tokens (A = 2420, B = 39329, C = 74642) are read from the pinned tokenizer, so a figure can only
show bytes the tokenizer really has. Numbering and cause names follow proof.md.

  fig1_pipeline.svg     token -> bytes -> cut string -> lit cells -> code -> read back
  fig2_proof_map.svg    which result of proof.md uses which, with the test files
  fig3_cut_rule.svg     Lemma 2: the repository cut rule on a valid and on a broken long token
  fig4_marks_grid.svg   Lemma 1, step 1: the lit cells of A, and of B and C, in the 256 x 32 grid
  fig5_two_values.svg   Lemma 1, step 3: the two values h and u for every length, P = 32 and 64
  fig6_margins.svg      room for error of a fixed cut-off of 8 and of the sign rule
  fig7_cause_tree.svg   Theorem 1 corollary: pair_cause as a decision tree
  fig8_theorem_BC.svg   Theorem 1 on B and C, raw table against decode table

The output is deterministic: the same inputs give byte-identical files.

Run from the lab repo root, with the tokenizer found as in README.md (KE2_TOKENIZER or the
Hugging Face cache):  python teams/04-ke2/src/docs/proof_figures.py
"""

from __future__ import annotations

import math
import sys
from html import escape
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make ke2 and the upstream codec importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ke2.upstream import ensure_kronecker_embeddings

    ensure_kronecker_embeddings()

import torch
from kronecker_embeddings.codec import kronecker_codec

from ke2.config import TEAM_DIR, TOKENIZER_SHA256
from ke2.tokenizer import decode_bytes_for_id, load_tokenizer, raw_bytes_for_id
from ke2.truncation import repo_truncate

OUT = TEAM_DIR / "docs" / "figures"
EPS = 1e-6
A_ID, B_ID, C_ID = 2420, 39329, 74642

INK, INK2, MUTED = "#0f2230", "#3f5260", "#667884"
RULE, RULE2, PAPER = "#d8e0e4", "#b7c4cb", "#f3f6f7"
RAW, RAW_W, DEC, DEC_W = "#2a78d6", "#e2edfb", "#eb6834", "#fde8dd"
MARK, MARK_E = "#ffd675", "#c98d0c"
GOOD, GOOD_W, BAD = "#0a8a0a", "#e1f3e1", "#b42318"
SANS = "'Segoe UI', 'Helvetica Neue', Arial, 'Nirmala UI', 'Noto Sans', sans-serif"
MONO = "Consolas, Menlo, 'DejaVu Sans Mono', monospace"


# ------------------------------------------------------------------ numbers
def closed_form(L: int, P: int) -> tuple[float, float]:
    """(h, u) of proof.md, Lemma 1 step 3, for a string of L >= 1 bytes."""
    D = 256 * P
    sigma = math.sqrt((1 - L / D) / (D - 1)) + EPS
    m = math.sqrt(L) / D
    return (1 / math.sqrt(L) - m) / sigma, -m / sigma


def codes(rows: list[bytes], P: int) -> torch.Tensor:
    buf = torch.zeros((len(rows), P), dtype=torch.long)
    for k, r in enumerate(rows):
        buf[k, : len(r)] = torch.tensor(list(r), dtype=torch.long)
    return kronecker_codec(buf, torch.tensor([len(r) for r in rows]), pos_dim=P)


def cut_tries(r: bytes, P: int) -> int:
    """Length the repository rule keeps, trying P, P-1, P-2, P-3 (checked against repo_truncate)."""
    keep = len(r)
    if len(r) > P:
        keep = 0
        for c in range(P, P - 4, -1):
            try:
                r[:c].decode("utf-8")
                keep = c
                break
            except UnicodeDecodeError:
                pass
    assert repo_truncate(r, P) == r[:keep]
    return keep


def facts() -> dict:
    tb = load_tokenizer(expected_sha256=TOKENIZER_SHA256)
    f = {
        "A": raw_bytes_for_id(tb, A_ID),
        "B": raw_bytes_for_id(tb, B_ID),
        "C": raw_bytes_for_id(tb, C_ID),
        "Bd": decode_bytes_for_id(tb, B_ID),
        "Cd": decode_bytes_for_id(tb, C_ID),
    }
    assert f["A"] == bytes.fromhex("20e0a48fe0a495") and decode_bytes_for_id(tb, A_ID) == f["A"]
    assert f["B"] == bytes.fromhex("b8e0af8d") and f["C"] == bytes.fromhex("aee0af8d")
    assert f["Bd"] == f["Cd"] == bytes.fromhex("efbfbde0af8d")
    E = codes([f["A"], f["B"], f["C"], f["Bd"]], 32)
    for k, row in enumerate((f["A"], f["B"], f["C"], f["Bd"])):
        h, u = closed_form(len(row), 32)
        assert set((E[k] > 8).nonzero().flatten().tolist()) == {v * 32 + p for p, v in enumerate(row)}, "lit cells"
        assert abs(E[k].max().item() - h) < 1e-4 and abs(E[k].min().item() - u) < 1e-6, "two values"
    f["hA"], f["uA"] = E[0].max().item(), E[0].min().item()
    f["diffBC"] = (E[1] != E[2]).nonzero().flatten().tolist()
    assert f["diffBC"] == [0xAE * 32, 0xB8 * 32]
    # proof_explained.md: B and C share 3 of 4 lit cells, so their unnormalized codes have overlap 3/4
    x = kronecker_codec(torch.tensor([list(f["B"]), list(f["C"])]), torch.tensor([4, 4]), pos_dim=4, z_normalize=False)
    assert abs((x[0] @ x[1]).item() - 0.75) < 1e-6
    # proof_explained.md: noise of up to 7.9 per number on A's code; a cut-off of 8 still reads A back, the sign rule does not
    g = torch.Generator().manual_seed(7)
    noisy = E[0] + (torch.rand(E[0].shape, generator=g) * 2 - 1) * 7.9
    assert set((noisy > 8).nonzero().flatten().tolist()) == {v * 32 + p for p, v in enumerate(f["A"])}
    f["sign_rule_extra"] = int((noisy > 0).sum().item()) - len(f["A"])
    assert f["sign_rule_extra"] > 1000
    f["curves"] = {P: [closed_form(L, P) for L in range(1, P + 1)] for P in (32, 64)}
    for P in (32, 64):  # the codec agrees with the closed forms at every length, also after bf16 and fp16 casts
        D = 256 * P
        E = codes([b"a" * L for L in range(1, P + 1)], P)
        for L in range(1, P + 1):
            h, u = f["curves"][P][L - 1]
            assert abs(E[L - 1].max().item() - h) / h < 1e-5 and abs(E[L - 1].min().item() - u) / -u < 1e-5
            for dt in (torch.bfloat16, torch.float16):
                c = E[L - 1].to(dt).float()
                assert c.max().item() > 15.87 and c.min().item() < 0 and len(torch.unique(c)) == 2
        hP = f["curves"][P][-1][0]
        assert hP > 15.93 and min(u for _, u in f["curves"][P]) > -(math.sqrt(P) / D) / math.sqrt((1 - P / D) / (D - 1))
        assert min(h for h, _ in f["curves"][P]) - 8 > 7.93
    return f


# ------------------------------------------------------------------ drawing helpers
def text(x, y, s, size=13, anchor="start", weight=400, fill=INK, mono=False) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{MONO if mono else SANS}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>'
    )


def lines(x, y, rows: list[str], size=13, lh=None, **kw) -> str:
    lh = lh or size * 1.35
    return "".join(text(x, y + i * lh, r, size, **kw) for i, r in enumerate(rows))


def rect(x, y, w, h, fill="#ffffff", stroke=RULE2, rx=8, sw=1.2, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def line(x1, y1, x2, y2, stroke=INK2, sw=1.5, arrow=False, dash=None, both=False) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += ' marker-end="url(#ah)"' if arrow else ""
    extra += ' marker-start="url(#ah)"' if both else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'
    )


def path(d, stroke=INK2, sw=1.5) -> str:
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" marker-end="url(#ah)"/>'


def circle(x, y, r, fill, stroke=INK, sw=1.2) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def tag(x, y, s, fill=PAPER) -> str:
    w = 10 + 7 * len(s)
    return rect(x, y, w, 20, fill, RULE2, 10, 1) + text(x + w / 2, y + 14, s, 11.5, "middle", 600, INK2, mono=True)


def byte_box(x, y, s: str, fill="#ffffff", stroke=RULE2, w=33, h=26, ink=INK, sw=1.2) -> str:
    return rect(x, y, w, h, fill, stroke, 4, sw) + text(
        x + w / 2, y + h / 2 + 4.5, s, 12, "middle", 600, ink, mono=True
    )


def svg(w, h, title, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="{escape(title)}">\n<title>{escape(title)}</title>\n'
        f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>\n'
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="12" fill="#ffffff" stroke="{RULE}"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def num(v: float, d: int) -> str:
    return f"{v:.{d}f}".replace("-", "−")


# ------------------------------------------------------------------ figure 1: the pipeline
def fig1(f) -> str:
    b = [
        text(24, 34, "From a token to its code, and back", 18, weight=700),
        text(
            24,
            56,
            "Token A (id 2420) at P = 32. Each box is real code; the brackets name the part of proof.md that covers each step.",
            13,
            fill=MUTED,
        ),
    ]
    xs = [24 + i * 164 for i in range(6)]
    boxes = [
        ("TOKEN ID", [("2420", 20, True), ('" एक"', 15, False)], "#ffffff", RULE2),
        ("RAW BYTES", [("20 e0 a4 8f", 14, True), ("e0 a4 95", 14, True), ("7 bytes", 12, False)], RAW_W, RAW),
        (
            "CUT STRING",
            [("7 of 32 slots", 14, False), ("kept whole,", 12, False), ("since 7 ≤ 32", 12, False)],
            RAW_W,
            RAW,
        ),
        ("LIT CELLS", [("7 entries of 0.378", 13, False), ("8,185 zeros", 13, False)], "#ffffff", RULE2),
        (
            "CODE (NORMALIZED)",
            [(f"7 × {f['hA']:.2f}", 13, True), (f"8,185 × {num(f['uA'], 4)}", 13, True)],
            MARK,
            MARK_E,
        ),
        (
            "READ BACK",
            [("cells above the", 13, False), ("midpoint → 7 bytes", 12, False), ("→ token 2420", 12, False)],
            GOOD_W,
            GOOD,
        ),
    ]
    for i, (lab, rows, fill, stroke) in enumerate(boxes):
        x = xs[i]
        b.append(rect(x, 76, 140, 96, fill, stroke))
        b.append(text(x + 70, 96, lab, 10.5, "middle", 700, MUTED))
        for k, (r, s, m) in enumerate(rows):
            b.append(text(x + 70, 121 + k * 19, r, s, "middle", 700 if k == 0 else 400, INK, mono=m))
        if i < 5:
            b.append(line(x + 142, 124, x + 162, 124, arrow=True))
    b.append(path(f"M{xs[5] + 70},172 V192 H{xs[2] + 70} V175", stroke=GOOD, sw=1.8))
    b.append(
        text((xs[2] + xs[5]) / 2 + 70, 210, "the code gives the cut string back exactly", 12.5, "middle", 600, GOOD)
    )
    for i, lab in (
        (1, "Tables: raw or decode"),
        (2, "Lemma 2: cut rule"),
        (3, "Lemma 1, step 1"),
        (4, "Lemma 1, steps 2–3"),
        (5, "Lemma 1, steps 4–5"),
    ):
        x = xs[i]
        b.append(
            line(x + 6, 228, x + 134, 228, stroke=RULE2)
            + line(x + 6, 223, x + 6, 233, stroke=RULE2)
            + line(x + 134, 223, x + 134, 233, stroke=RULE2)
        )
        b.append(text(x + 70, 248, lab, 12.5, "middle", 600, INK2))
    b.append(
        line(xs[2] + 6, 268, xs[5] + 134, 268, stroke=INK, sw=1.8)
        + line(xs[2] + 6, 262, xs[2] + 6, 274, stroke=INK, sw=1.8)
        + line(xs[5] + 134, 262, xs[5] + 134, 274, stroke=INK, sw=1.8)
    )
    b.append(
        text(
            (xs[2] + xs[5] + 140) / 2,
            292,
            "Theorem 1: two tokens get the same code exactly when they have the same cut string",
            13.5,
            "middle",
            700,
        )
    )
    return svg(1000, 312, "The forward encode of token A and the part of proof.md that covers each step", b)


# ------------------------------------------------------------------ figure 2: proof map
def fig2(f) -> str:
    b = [text(24, 32, "How proof.md is built", 18, weight=700)]
    top = [
        (
            24,
            "LEMMA 2 · THE CUT RULE",
            [
                "(a)–(f): keeps a start of the token,",
                "never splits a letter, and makes",
                "every cut string fit in P bytes",
            ],
            ["test_lemma2.py"],
        ),
        (
            350,
            "LEMMA 1, STEP 1 · THE LIT CELLS",
            ["each byte lights one cell, in its", "own position, so the lit cells", "spell out the string"],
            ["test_lemma1.py"],
        ),
        (
            676,
            "LEMMA 1, STEPS 2–5 · NORMALIZING",
            ["two values: lit cells high, the", "rest slightly below zero; the lit", "cells give the string back"],
            ["test_lemma1.py"],
        ),
    ]
    for x, head, rows, tags in top:
        b.append(rect(x, 84, 300, 116))
        b.append(text(x + 14, 106, head, 11, weight=700, fill=MARK_E))
        b.append(lines(x + 14, 128, rows, 13))
        tx = x + 14
        for t in tags:
            b.append(tag(tx, 170, t))
            tx += 18 + 7 * len(t)
    b.append(path("M174,84 C174,62 440,62 440,82"))
    b.append(text(307, 58, "(b), (c): every cut string fits in P", 12, "middle", 600, INK2))
    b.append(path("M560,84 C560,62 826,62 826,82"))
    b.append(text(693, 58, "the sum of squares is 1", 12, "middle", 600, INK2))
    b.append(rect(190, 252, 610, 122, MARK, MARK_E, sw=1.6))
    b.append(text(206, 274, "THEOREM 1 · THE CODEC NEVER CAUSES A COLLISION", 11, weight=700))
    b.append(
        lines(
            206,
            296,
            [
                "code(τ(r)) = code(τ(r′)) exactly when τ(r) = τ(r′)",
                "Corollary: a shared code has one of five causes, the first that applies:",
                "empty, cut_to_empty, same_raw, bad_bytes_in, truncation",
            ],
            13,
        )
    )
    b.append(tag(206, 344, "test_theorem1.py", fill="#ffffff"))
    b.append(tag(354, 344, "test_proof_checks.py", fill="#ffffff"))
    b.append(line(174, 200, 300, 250, arrow=True))
    b.append(text(40, 232, "(a), (b), (f): the causes", 12, "start", 600, INK2))
    b.append(line(500, 200, 500, 250, arrow=True))
    b.append(text(508, 230, "injective", 12, "start", 600, INK2))
    b.append(line(826, 200, 700, 250, arrow=True))
    b.append(text(772, 232, "step 5: read back", 12, "start", 600, INK2))
    b.append(rect(826, 262, 150, 92, "#ffffff", RULE2, dash="5 4"))
    b.append(lines(838, 282, ["CHECKED FACT", "printout = raw with", "U+FFFD, all 131,072 ids"], 12))
    b.append(tag(838, 326, "proof.md §4"))
    b.append(line(826, 308, 802, 308, arrow=True))
    b.append(rect(330, 406, 340, 40, GOOD_W, GOOD, sw=1.6))
    b.append(text(500, 431, "Gate 1 holds (proof.md §4)", 15, "middle", 700, GOOD))
    b.append(line(500, 374, 500, 404, arrow=True))
    return svg(1000, 466, "Dependency map of Lemma 1, Lemma 2 and Theorem 1 of proof.md, and Gate 1", b)


# ------------------------------------------------------------------ figure 3: the cut rule
def cut_row(b: list[str], y: int, data: bytes, P: int, title: str, letters, notes: list[str], stray0=False) -> int:
    end = cut_tries(data, P)
    b.append(text(24, y, title, 14, weight=700))
    show = [0, 1, 2, *range(24, len(data))]
    pos_x, x = {}, 40
    for i in show:
        if i == 24:
            b.append(text(x + 18, y + 59, "…", 18, "middle", 700, MUTED))
            x += 40
        pos_x[i] = x
        x += 37
    for i in show:
        kept, stray = i < end, stray0 and i == 0
        b.append(
            byte_box(
                pos_x[i],
                y + 40,
                f"{data[i]:02x}",
                RAW_W if kept else PAPER,
                BAD if stray else (RAW if kept else RULE2),
                ink=INK if kept else MUTED,
                sw=2.2 if stray else 1.2,
            )
        )
        if i in (0, 24, 29, 31, 35, 41) and i < len(data):
            b.append(text(pos_x[i] + 16.5, y + 82, str(i), 10.5, "middle", fill=MUTED))
    for lo, hi, lab in letters:
        xa, xb = pos_x[lo] + 2, pos_x[hi] + 31
        b.append(line(xa, y + 32, xb, y + 32, stroke=MARK_E, sw=1.5))
        b.append(text((xa + xb) / 2, y + 26, lab, 13, "middle", 600, MARK_E))
    xP = pos_x[P - 1] + 35
    b.append(line(xP, y + 34, xP, y + 72, stroke=BAD, sw=2, dash="4 3"))
    for c in range(P, P - 4, -1):
        if end and c < end:
            break
        xc, ok = pos_x[c - 1] + 35, c == end
        b.append(line(xc, y + 88, xc, y + 96, stroke=GOOD if ok else BAD, sw=2))
        b.append(text(xc, y + 112, f"{c} {'✓' if ok else '✗'}", 12, "middle", 700, GOOD if ok else BAD))
    b.append(lines(24, y + 138, notes, 13, 18, fill=INK2))
    return end


def fig3(f) -> str:
    b = [
        text(
            24,
            32,
            "Lemma 2 at P = 32: try 32, 31, 30 and 29 bytes, keep the first start that is valid text",
            17,
            weight=700,
        ),
        text(
            24,
            54,
            "The repository rule (utf8_safe_truncate). Blue bytes are kept; the red dashed line is P = 32; each tick is a place the rule tries to cut.",
            13,
            fill=MUTED,
        ),
    ]
    e1 = cut_row(
        b,
        96,
        b"a" * 30 + "भारत".encode(),
        32,
        "A valid token of 42 bytes: 30 × a, then भारत",
        [(30, 32, "भ"), (33, 35, "ा"), (36, 38, "र"), (39, 41, "त")],
        [
            "Cutting at 32 or 31 would split भ, so those starts are not valid text. 30 is valid: the cut string keeps 30 bytes (Lemma 2 (c), (d))."
        ],
    )
    e2 = cut_row(
        b,
        266,
        b"\x95" + b"a" * 20 + "ಕನ್ನಡ".encode(),
        32,
        "A broken token of 36 bytes: 95, then 20 × a, then ಕನ್ನಡ",
        [(24, 26, "ನ"), (27, 29, "್"), (30, 32, "ನ"), (33, 35, "ಡ")],
        [
            "Every start begins with 95, a continuation byte with nothing before it, so no cut is valid and the cut string is empty (Lemma 2 (f)).",
            "The proposed rule τ′ in proof.md would keep 30 bytes of this token instead.",
        ],
        stray0=True,
    )
    assert (e1, e2) == (30, 0)
    b.append(
        text(
            24,
            456,
            "No BrahmicTokenizer token is longer than 32 bytes, so at P = 32 and P = 64 neither rule cuts anything (Lemma 2 (b)). "
            "Both cases matter only for another tokenizer or a smaller P.",
            12,
            fill=MUTED,
        )
    )
    return svg(1000, 476, "The repository cut rule on a valid 42-byte token and on a broken 36-byte token", b)


# ------------------------------------------------------------------ figure 4: the lit cells in the grid
def grid_panel(b: list[str], ox: float, title: list[str], marks, labels):
    gx, gy, cw, rh, P = ox + 46, 104, 9, 2.4, 32
    b.append(text(ox, 34, title[0], 15, weight=700))
    b.append(text(ox, 54, title[1], 12.5, fill=MUTED))
    b.append(rect(gx, gy, cw * P, rh * 256, PAPER, RULE2, rx=2, sw=1))
    for p in range(1, P):
        b.append(line(gx + p * cw, gy, gx + p * cw, gy + rh * 256, stroke="#e2e8eb", sw=0.6))
    for v in range(16, 256, 16):
        b.append(line(gx, gy + v * rh, gx + cw * P, gy + v * rh, stroke="#d5dde1", sw=0.6))
    for v in range(0, 256, 32):
        b.append(text(gx - 6, gy + v * rh + 4, f"{v:02x}", 11, "end", fill=MUTED, mono=True))
    for p in (0, 8, 16, 24, 31):
        b.append(text(gx + p * cw + cw / 2, gy - 8, str(p), 11, "middle", fill=MUTED))
    b.append(text(gx, gy - 26, "position (column) →", 11.5, fill=INK2))
    b.append(text(ox, gy + rh * 256 + 24, "byte value (row) ↓, 256 rows", 11.5, fill=INK2))
    for v, p, _ in marks:
        y = gy + v * rh + rh / 2
        b.append(line(gx + p * cw + cw / 2 + 6, y, gx + cw * P + 6, y, stroke=MARK_E, sw=0.8, dash="2 2"))
    for v, p, kind in marks:
        b.append(
            circle(
                gx + p * cw + cw / 2,
                gy + v * rh + rh / 2,
                5.5,
                {"both": MARK, "B": INK, "C": "#ffffff"}[kind],
                MARK_E if kind == "both" else INK,
                1.6,
            )
        )
    for v, dy, s in labels:
        b.append(text(gx + cw * P + 10, gy + v * rh + rh / 2 + 4 + dy, s, 12, mono=True))


def fig4(f) -> str:
    b: list[str] = []
    grid_panel(
        b,
        24,
        ["Token A: 7 bytes, 7 lit cells", "one per position; e0 and a4 appear twice, in different positions"],
        [(v, p, "both") for p, v in enumerate(f["A"])],
        [
            (0x20, 0, "20 · pos 0"),
            (0x8F, -3, "8f · pos 3"),
            (0x95, 4, "95 · pos 6"),
            (0xA4, 0, "a4 · pos 2, 5"),
            (0xE0, 0, "e0 · pos 1, 4"),
        ],
    )
    grid_panel(
        b,
        520,
        ["Tokens B and C: 4 lit cells each, 3 shared", "they differ only at position 0: b8 (B) against ae (C)"],
        [(0xB8, 0, "B"), (0xAE, 0, "C")] + [(v, p, "both") for p, v in enumerate(f["B"]) if p > 0],
        [
            (0x8D, 0, "8d · pos 3"),
            (0xAE, -6, "ae · pos 0 (C)"),
            (0xAF, 9, "af · pos 2"),
            (0xB8, 5, "b8 · pos 0 (B)"),
            (0xE0, 0, "e0 · pos 1"),
        ],
    )
    ly = 104 + 2.4 * 256 + 52
    b.append(
        text(
            24,
            ly,
            "Cell number = byte × 32 + position, so each lit cell gives back its byte and its position.",
            12,
            fill=MUTED,
        )
    )
    b.append(circle(566, ly - 4, 5.5, MARK, MARK_E, 1.6) + text(578, ly, "lit (B and C: shared)", 12))
    b.append(circle(762, ly - 4, 5.5, INK, INK, 1.6) + text(774, ly, "only B", 12))
    b.append(circle(852, ly - 4, 5.5, "#ffffff", INK, 1.6) + text(864, ly, "only C", 12))
    return svg(1000, int(ly + 22), "The lit cells of tokens A, B and C in the 256 by 32 grid", b)


# ------------------------------------------------------------------ figure 5: the two values
def chart(b: list[str], x0, x1, y0, y1, ymin, ymax, yticks, fy, title, series, refs, rings, caption):
    def X(L):
        return x0 + (L - 1) / 63 * (x1 - x0)

    def Y(v):
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    b.append(text(x0 - 50, y0 - 22, title, 14.5, weight=700))
    for t in yticks:
        b.append(line(x0, Y(t), x1, Y(t), stroke="#e6ecef", sw=1))
        b.append(text(x0 - 8, Y(t) + 4, fy(t), 11, "end", fill=MUTED))
    for L in (1, 8, 16, 24, 32, 40, 48, 56, 64):
        b.append(text(X(L), y1 + 18, str(L), 11, "middle", fill=MUTED))
    b.append(line(x0, y1, x1, y1, stroke=RULE2, sw=1.2))
    b.append(text((x0 + x1) / 2, y1 + 38, "string length L (bytes)", 12, "middle", fill=INK2))
    for v, lab, col, at in refs:
        b.append(line(x0, Y(v), x1, Y(v), stroke=col, sw=1.6, dash="6 4"))
        if isinstance(at, tuple):  # a key in an empty corner, where no curve passes
            kx, ky = X(at[0]), Y(at[1])
            b.append(line(kx, ky - 4, kx + 22, ky - 4, stroke=col, sw=1.6, dash="6 4"))
            b.append(text(kx + 28, ky, lab, 11.5, "start", 700, col))
        else:
            b.append(text(x0 + 6, Y(v) + at, lab, 11.5, "start", 700, col))
    for vals, col in series:
        pts = " ".join(f"{X(L):.1f},{Y(v):.2f}" for L, v in enumerate(vals, start=1))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        b.extend(circle(X(L), Y(v), 2.2, col, col, 0.5) for L, v in enumerate(vals, start=1))
    for L, v in rings:
        b.append(circle(X(L), Y(v), 6.5, "none", INK, 1.6))
    b.append(text(x0 - 50, y1 + 62, caption, 12, weight=600))


def fig5(f) -> str:
    b = [text(24, 30, "Lemma 1, step 3: every code has just two values, set by the string length L", 17, weight=700)]
    c32, c64 = f["curves"][32], f["curves"][64]
    chart(
        b,
        84,
        470,
        92,
        330,
        0,
        140,
        [0, 20, 40, 60, 80, 100, 120, 140],
        str,
        "On each of the L lit cells: h(L)",
        [([h for h, _ in c32], RAW), ([h for h, _ in c64], MARK_E)],
        [(15.93, "floor 15.93, for every L", BAD, (30, 124)), (8, "a fixed cut-off of 8", GOOD, (30, 110))],
        [(7, c32[6][0]), (4, c32[3][0])],
        f"Ringed at P = 32: A (L = 7) {c32[6][0]:.2f}, and B, C (L = 4) {c32[3][0]:.2f}",
    )
    chart(
        b,
        584,
        970,
        92,
        330,
        -0.07,
        0.005,
        [-0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0],
        lambda t: num(t, 2),
        "On every other cell: u(L)",
        [([u for _, u in c32], RAW), ([u for _, u in c64], MARK_E)],
        [(0, "0, the sign rule's cut-off", BAD, -6), (-0.06262, "floor −0.0627, for every L", BAD, 15)],
        [(7, c32[6][1]), (4, c32[3][1])],
        f"Ringed at P = 32: A {num(c32[6][1], 4)}, and B, C {num(c32[3][1], 4)}",
    )
    b.append(line(24, 428, 44, 428, stroke=RAW, sw=2.4) + text(50, 432, "P = 32 (L up to 32, D = 8,192)", 12))
    b.append(line(290, 428, 310, 428, stroke=MARK_E, sw=2.4) + text(316, 432, "P = 64 (L up to 64, D = 16,384)", 12))
    b.append(text(560, 432, "The two values never meet: the gap h − u is always above 15.99.", 12, fill=INK2))
    return svg(1000, 452, "The two values of a normalized code for every string length, at P = 32 and P = 64", b)


# ------------------------------------------------------------------ figure 6: room for error
def fig6(f) -> str:
    b = [
        text(
            24,
            30,
            "Room for error: how far each number of a code may move before reading it back fails",
            17,
            weight=700,
        ),
        text(
            24,
            52,
            "Every value of a code lies in one of two zones. A cut-off works as long as noise cannot push a value across it.",
            13,
            fill=MUTED,
        ),
    ]

    def X(v):
        return 60 + (v + 2) * 36

    ya = 176
    b.append(line(X(-2), ya, X(22), ya, stroke=INK2, sw=1.2))
    for t in range(-2, 23, 2):
        b.append(
            line(X(t), ya, X(t), ya + 5, stroke=INK2, sw=1)
            + text(X(t), ya + 20, str(t).replace("-", "−"), 11, "middle", fill=MUTED)
        )
    b.append(rect(X(15.93), 128, X(22) - X(15.93), 48, MARK, MARK_E, rx=3))
    b.append(text(X(15.93) + 10, 148, "every lit cell ≥ 15.93", 12.5, weight=700))
    b.append(text(X(15.93) + 10, 166, "(up to 128) →", 12, fill=INK2))
    b.append(rect(X(-0.0627), 128, X(0) - X(-0.0627), 48, INK, INK, rx=0, sw=0.5))
    b.append(line(X(-0.03), 182, X(-1.2), 214, stroke=INK2, sw=1))
    b.append(text(X(-2), 230, "every other cell: between −0.063 and 0", 12.5, weight=700))
    b.append(line(X(8), 84, X(8), 186, stroke=GOOD, sw=2.5))
    b.append(text(X(8), 78, "fixed cut-off 8", 12.5, "middle", 700, GOOD))
    b.append(line(X(0) + 4, 112, X(8) - 3, 112, stroke=GOOD, sw=1.5, arrow=True, both=True))
    b.append(text((X(0) + X(8)) / 2, 106, "room > 8", 12.5, "middle", 700, GOOD))
    b.append(line(X(8) + 3, 112, X(15.93) - 3, 112, stroke=GOOD, sw=1.5, arrow=True, both=True))
    b.append(text((X(8) + X(15.93)) / 2, 106, "room ≥ 7.93", 12.5, "middle", 700, GOOD))
    b.append(line(X(0), 84, X(0), 186, stroke=BAD, sw=2))
    b.append(text(X(0) + 6, 78, "sign rule (cut-off 0)", 12.5, "middle", 700, BAD))

    def Z(v):
        return 60 + (v + 0.08) * 8800

    yz = 352
    b.append(text(24, 270, "Zoomed in 1,000 times around 0: the sign rule's room is tiny", 14, weight=700))
    b.append(line(Z(-0.08), yz, Z(0.02), yz, stroke=INK2, sw=1.2))
    for t in (-0.08, -0.06, -0.04, -0.02, 0, 0.02):
        b.append(
            line(Z(t), yz, Z(t), yz + 5, stroke=INK2, sw=1) + text(Z(t), yz + 20, num(t, 2), 11, "middle", fill=MUTED)
        )
    b.append(rect(Z(-0.0627), 310, Z(-0.0078) - Z(-0.0627), 42, RULE2, INK2, rx=3, sw=1))
    b.append(
        text((Z(-0.0627) + Z(-0.0078)) / 2, 336, "u of every non-empty string: −0.0627 to −0.0078", 12.5, "middle", 700)
    )
    b.append(line(Z(0), 296, Z(0), yz + 4, stroke=BAD, sw=2))
    b.append(line(Z(-0.0078) + 2, 302, Z(0) - 2, 302, stroke=BAD, sw=1.5, arrow=True, both=True))
    b.append(text((Z(-0.0078) + Z(0)) / 2, 296, "0.0078", 12, "middle", 700, BAD))
    b.append(circle(Z(0), yz, 5, BAD, BAD))
    b.append(
        lines(
            Z(0) + 12,
            318,
            ["empty string: every cell is", "exactly 0, so the sign rule", "has no room at all"],
            12,
            fill=BAD,
        )
    )
    return svg(1000, 394, "Room for error of a fixed cut-off of 8 and of the sign rule", b)


# ------------------------------------------------------------------ figure 7: the cause tree (pair_cause)
def qbox(b, cx, y, w, s):
    b.append(rect(cx - w / 2, y, w, 40, PAPER, INK2, rx=20, sw=1.4))
    b.append(text(cx, y + 24.5, s, 13, "middle", 600))


def leaf(b, cx, y, name, rows, real, hit=False):
    w, h = 180, 122
    b.append(rect(cx - w / 2, y, w, h, GOOD_W if hit else "#ffffff", GOOD if hit else RULE2, sw=1.8 if hit else 1.2))
    b.append(text(cx, y + 22, name, 12.5, "middle", 700, mono=True))
    b.append(lines(cx, y + 44, rows, 11.5, 15.5, anchor="middle", fill=INK2))
    b.append(text(cx, y + h - 11, real, 11.5, "middle", 700, GOOD if hit else MUTED))


def edge(b, x1, y1, x2, y2, lab, lx, ly, anchor="middle"):
    b.append(line(x1, y1, x2, y2, arrow=True))
    b.append(text(lx, ly, lab, 12, anchor, 700, INK2))


def fig7(f) -> str:
    b = [
        text(24, 30, "Theorem 1 corollary: a shared code has the first of five causes that applies", 17, weight=700),
        text(
            24,
            52,
            "This is pair_cause in src/proofs/forward.py. Green: the only cause BrahmicTokenizer-131K produces (decode table).",
            13,
            fill=MUTED,
        ),
    ]
    b.append(rect(290, 70, 420, 38, INK, INK, rx=8))
    b.append(text(500, 94, "Tokens s ≠ t share a code, so they share a cut string c", 14, "middle", 700, "#ffffff"))
    b.append(line(500, 108, 500, 126, arrow=True))
    qbox(b, 500, 128, 340, "Is the shared cut string c empty?")
    qbox(b, 203, 222, 290, "Are both fed strings empty?")
    edge(b, 380, 168, 250, 220, "yes", 305, 190, "end")
    qbox(b, 625, 222, 270, "Are the raw bytes equal?")
    edge(b, 600, 168, 620, 220, "no", 622, 190, "start")
    qbox(b, 785, 306, 290, "Were the fed strings already equal?")
    edge(b, 690, 262, 760, 304, "no", 740, 280)
    leaf(b, 106, 400, "empty", ["two tokens with no bytes:", "both get the all-zero", "code"], "real: 0 groups")
    edge(b, 160, 262, 112, 398, "yes", 120, 330, "end")
    leaf(
        b,
        300,
        400,
        "cut_to_empty",
        ["a long token that is not", "valid UTF-8, cut to nothing", "(Lemma 2 (f))"],
        "real: 0 groups",
    )
    edge(b, 246, 262, 296, 398, "no", 282, 330, "start")
    leaf(
        b,
        494,
        400,
        "same_raw",
        ["the same raw bytes twice,", "e.g. <s> as a special", "token and as text"],
        "real: 0 groups",
    )
    edge(b, 590, 262, 510, 398, "yes", 548, 330, "end")
    leaf(
        b,
        688,
        400,
        "bad_bytes_in",
        ["different raw bytes fed the", "same, via U+FFFD; e.g. B, C", "(collisions.csv: decode_collapse)"],
        "real: 10 groups",
        hit=True,
    )
    edge(b, 745, 346, 700, 398, "yes", 712, 372, "end")
    leaf(
        b,
        882,
        400,
        "truncation",
        ["the fed strings differ but", "agree up to the cut; one", "is longer than P"],
        "real: 0 groups",
    )
    edge(b, 825, 346, 868, 398, "no", 858, 372, "start")
    b.append(
        text(
            24,
            552,
            "Each question splits what is left in two, so exactly one leaf applies. "
            "All five occur on the synthetic vocabulary of test_theorem1.py.",
            12,
            fill=MUTED,
        )
    )
    return svg(1000, 572, "Decision tree of the five causes of a shared code, as in pair_cause", b)


# ------------------------------------------------------------------ figure 8: Theorem 1 on B and C
def fig8(f) -> str:
    b = [
        text(24, 30, "Theorem 1 on tokens B and C at P = 32", 17, weight=700),
        text(
            24,
            52,
            "The same two tokens in two tables. The code follows the fed string: different strings give different codes, equal strings one code.",
            13,
            fill=MUTED,
        ),
    ]
    d0, d1 = f["diffBC"]
    lanes = [
        (
            76,
            "RAW BYTE TABLE",
            RAW_W,
            RAW,
            f["B"],
            f["C"],
            True,
            [
                "the lit cells differ only at position 0:",
                f"B lights cell {d1:,} (b8 × 32),",
                f"C lights cell {d0:,} (ae × 32)",
            ],
            [
                "the codes differ in 2 of 8,192",
                "numbers; reading them back gives",
                "two different strings: unique codes",
            ],
        ),
        (
            216,
            "DECODE TABLE",
            DEC_W,
            DEC,
            f["Bd"],
            f["Cd"],
            False,
            ["both fed strings are", "ef bf bd e0 af 8d, so the lit", "cells are identical"],
            [
                "one code (0 of 8,192 numbers differ);",
                "cause bad_bytes_in, called",
                "decode_collapse in collisions.csv",
            ],
        ),
    ]
    for y, name, fill, stroke, rb, rc, raw, mid, verdict in lanes:
        b.append(rect(24, y, 952, 124, PAPER, RULE, rx=10, sw=1))
        b.append(text(40, y + 24, name, 11.5, weight=700, fill=stroke))
        for k, (lab, row) in enumerate((("B", rb), ("C", rc))):
            yy = y + 38 + k * 38
            b.append(text(40, yy + 18, lab, 14, weight=700))
            for p, v in enumerate(row):
                hi = raw and p == 0
                b.append(
                    byte_box(
                        60 + p * 34,
                        yy,
                        f"{v:02x}",
                        MARK if hi else fill,
                        MARK_E if hi else stroke,
                        w=31,
                        sw=1.8 if hi else 1.2,
                    )
                )
        b.append(line(276, y + 64, 316, y + 64, arrow=True))
        b.append(lines(328, y + 46, mid, 13))
        b.append(line(566, y + 64, 604, y + 64, arrow=True))
        b.append(rect(614, y + 24, 346, 78, GOOD_W, GOOD, sw=1.4))
        b.append(lines(630, y + 47, verdict, 13, weight=600))
    return svg(1000, 360, "Theorem 1 applied to tokens B and C in the raw and in the decode table", b)


def main() -> int:
    f = facts()
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {
        "fig1_pipeline.svg": fig1,
        "fig2_proof_map.svg": fig2,
        "fig3_cut_rule.svg": fig3,
        "fig4_marks_grid.svg": fig4,
        "fig5_two_values.svg": fig5,
        "fig6_margins.svg": fig6,
        "fig7_cause_tree.svg": fig7,
        "fig8_theorem_BC.svg": fig8,
    }
    for name, draw in figs.items():
        (OUT / name).write_text(draw(f), encoding="utf-8", newline="\n")
        print(f"wrote docs/figures/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
