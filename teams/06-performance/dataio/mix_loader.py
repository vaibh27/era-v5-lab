"""Mixture-aware loader: fixed per-batch shares, full packed rows with a doc mask, one global stateless cursor.

Row k of global batch n is a pure function of (seed, weights, stages, shard set, n), so the state is one global
batch number plus a skip list, the same on every rank, and a restore at another world size gives the same batch
(Team 7 `ckpt/loader_contract.py`).

1. Quotas: Indic gets `indic.share`, split by Team 2's CSV; the other lanes share the rest by `lanes`; packs inside
   a lane or language share by token count. Largest-remainder rounding at each level, in every batch.
2. Streams: each pack is its docs in a Philox order per (seed, pack, epoch); batch n reads its next quota tokens.
3. Packing: pieces shuffled, then best-fit decreasing into full rows (Ding et al. 2024); only what fits nowhere is
   cut. seg_id counts segments in a row; pos_id restarts at each segment.
4. Ranks: data-parallel rank r gets global rows [r::dp_world] (pass the DP rank, not the global rank).
5. Caps per source (manifest source_id, else `sources_file`, else the pack id), counted on its most-read pack:
   max_source_passes in this run, max_dataset_epochs over all phases via prior_epochs. next() raises CapReached.

`next()` returns int64 [rows_local, 3, seq_len] (tokens, seg_id, pos_id); `targets` and `cu_seqlens` give masked
targets and fla varlen offsets.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dataio.loader import open_shard
from dataio.shards import EOS_ID, P1536_SHA256, doc_tables, manifest_sha256, read_shards

ALGO = "g6-mix-v1"                    # changes when the stream, permutation or packing rule changes
STATE_SCHEMA = "g6-mix-state-v1"
LOG_SCHEMA = "g6-composition-v1"
INDIC = "indic"
IGNORE = -100
MISSING = ("error", "renormalise")
CONFIG_KEYS = {"seed", "shards", "stages", "total_batches", "weights", "sources_file", "caps"}
CAP_KEYS = {"max_source_passes", "max_dataset_epochs", "prior_epochs"}


class CapReached(RuntimeError):
    """The next batch would read a pack or dataset past its cap."""


@dataclass(frozen=True)
class Stage:
    from_batch: int
    seq_len: int
    rows: int

    @property
    def tokens(self) -> int:
        return self.seq_len * self.rows


@dataclass
class Layout:
    """The global layout of one batch: its segments in row order, none crossing a row edge. `start` is each
    segment's offset in the batch's rows x seq_len tokens; `doc` indexes the pack's DocTable; `off` is the segment's
    first token inside the doc. `pieces`: stream pieces before packing; `edge_cuts`: docs cut at this batch's start or
    end (exact quotas); `row_cuts`: extra cuts made by packing, of which `forced_cuts` for docs longer than a row."""
    n: int
    stage: Stage
    pack: np.ndarray
    doc: np.ndarray
    off: np.ndarray
    length: np.ndarray
    start: np.ndarray
    pieces: int = 0
    edge_cuts: int = 0
    row_cuts: int = 0
    forced_cuts: int = 0


# ---- counter-based randomness ------------------------------------------------------------------------------------

def _key(seed: int, *parts) -> np.ndarray:
    """A Philox key (2 x uint64) from the seed and a purpose: no state, no dependence on other packs."""
    digest = hashlib.sha256(json.dumps([ALGO, int(seed), *parts]).encode()).digest()
    return np.frombuffer(digest[:16], dtype="<u8").copy()


def permutation(n: int, key: np.ndarray) -> np.ndarray:
    """A permutation of range(n) from raw Philox output (fixed by the Philox spec, not by numpy's samplers)."""
    return np.argsort(np.random.Philox(key=key).random_raw(n), kind="stable")


# ---- packing pieces into rows -------------------------------------------------------------------------------------

def best_fit_rows(length: np.ndarray, rows: int, L: int, seed: int, n: int):
    """Best-fit decreasing into `rows` rows of L tokens, which the pieces fill exactly. Pieces longer than L are cut
    into L-chunks; whole pieces go to the row whose free space fits them most tightly (ties in the given, shuffled,
    order); pieces that fit nowhere are cut into the largest gaps. Rows and the segments in each row are then put
    in Philox order. Returns (piece index, offset in piece, length, start) per segment, in row order."""
    items = []
    for j, x in enumerate(length.tolist()):
        w = 0
        while x - w > L:
            items.append((j, w, L))
            w += L
        items.append((j, w, x - w))
    free = [(L, r) for r in range(rows)]
    placed = [[] for _ in range(rows)]
    left = []
    for j, w, x in sorted(items, key=lambda t: -t[2]):
        i = bisect.bisect_left(free, (x, -1))
        if i == len(free):
            left.append((j, w, x))
            continue
        f, r = free.pop(i)
        placed[r].append((j, w, x))
        if f > x:
            bisect.insort(free, (f - x, r))
    for j, w, x in left:
        while x > 0:
            f, r = free.pop()
            take = min(f, x)
            placed[r].append((j, w, take))
            w, x = w + take, x - take
            if f > take:
                bisect.insort(free, (f - take, r))
    if free:                                          # the pieces sum to rows x L, so every row must be full
        raise AssertionError(f"batch {n}: best-fit left {free} free")
    out = []
    for pos, r in enumerate(permutation(rows, _key(seed, "rows", n))):
        segs = placed[int(r)]
        col = pos * L
        for t in permutation(len(segs), _key(seed, "row", n, int(r))):
            j, w, x = segs[int(t)]
            out.append((j, w, x, col))
            col += x
    i, w, x, s = (np.array(v, dtype=np.int64) for v in zip(*out))
    return i, w, x, s


# ---- weights to quotas -------------------------------------------------------------------------------------------

def apportion(total: int, weights: dict) -> dict:
    """Integers summing to `total`, in proportion to `weights` (largest remainder; ties by name)."""
    names = sorted(weights)
    w = np.array([float(weights[k]) for k in names])
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"weights must be >= 0 with a positive sum: {weights}")
    exact = [total * x / w.sum() for x in w]
    out = {k: math.floor(e) for k, e in zip(names, exact)}
    left = total - sum(out.values())
    order = sorted(range(len(names)), key=lambda i: (-(exact[i] - math.floor(exact[i])), names[i]))
    for i in order[:left]:
        out[names[i]] += 1
    return out


def read_indic_split(path: Path) -> list:
    """[(language, batch percent, [packs])] from Team 2's indic_share CSV; languages with 0% are left out."""
    rows = []
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            pct = float(r["share_of_batch_pct"])
            if pct > 0:
                rows.append((r["lang"], pct, r["packs"].split()))
    if not rows:
        raise ValueError(f"{path}: no language with a positive share")
    return rows


def mix_tree(weights: dict, tables: dict, base_dir: Path, missing: str = "error"):
    """The weight tree {name: {"w": weight, "children": {...}}} with packs as leaves, plus notes on what was left out.
    Root children: "indic" (languages, then packs) and one node per other lane (packs)."""
    if missing not in MISSING:
        raise ValueError(f"missing={missing!r}: one of {MISSING}")
    notes = []

    def have(pack, where):
        if pack in tables:
            return True
        if missing == "renormalise":
            notes.append(f"{where}: pack {pack} is not in the shard set; dropped and renormalised")
            return False
        raise ValueError(f"{where}: pack {pack} is not in the shard set (set shards.missing: renormalise to allow)")

    root = {}
    indic = weights.get("indic") or {}
    share = float(indic.get("share", 0.08))
    if not 0 <= share < 1:
        raise ValueError(f"indic.share={share}: need 0 <= share < 1")
    if share > 0:
        csv_path = Path(indic["split_csv"])
        langs = {}
        for lang, pct, packs in read_indic_split(csv_path if csv_path.is_absolute() else base_dir / csv_path):
            kept = {p: {"w": tables[p].tokens} for p in packs if have(p, f"indic {lang}")}
            for p in kept:
                if tables[p].lane != INDIC:
                    raise ValueError(f"indic {lang}: pack {p} has lane {tables[p].lane!r}, not {INDIC!r}")
            if kept:
                langs[lang] = {"w": pct, "children": kept}
            else:
                notes.append(f"indic {lang}: no pack in the shard set; language dropped and renormalised")
        if not langs:
            raise ValueError("indic share > 0 but no Indic pack of the split is in the shard set")
        root[INDIC] = {"w": share, "children": langs}

    by_lane: dict = {}
    for p, t in tables.items():
        if t.lane != INDIC:
            by_lane.setdefault(t.lane, {})[p] = {"w": t.tokens}
    lanes = weights.get("lanes")
    if not isinstance(lanes, dict):
        raise ValueError(f"weights.lanes={lanes!r}: a dict of lane weights")
    lane_w = {}
    for lane, w in lanes.items():
        if lane == INDIC:
            raise ValueError("lanes: indic is set by indic.share, not by lanes")
        if float(w) <= 0:
            continue
        if lane not in by_lane:
            if missing == "renormalise":
                notes.append(f"lane {lane}: no pack in the shard set; dropped and renormalised")
                continue
            raise ValueError(f"lane {lane}: no pack in the shard set (set shards.missing: renormalise to allow)")
        lane_w[lane] = float(w)
    if not lane_w:
        raise ValueError("no non-Indic lane has a weight")
    total = sum(lane_w.values())
    for lane, w in sorted(lane_w.items()):
        root[lane] = {"w": (1 - share) * w / total, "children": by_lane[lane]}

    used = set(_leaves(root))
    for p in sorted(set(tables) - used):
        notes.append(f"pack {p} (lane {tables[p].lane}) has no weight; not read")
    return root, notes


def _leaves(tree: dict) -> list:
    out = []
    for name, node in sorted(tree.items()):
        out += _leaves(node["children"]) if "children" in node else [name]
    return out


def quotas(tree: dict, total: int) -> dict:
    """pack -> tokens per batch, apportioned level by level."""
    out = {}

    def walk(children, n):
        parts = apportion(n, {k: v["w"] for k, v in children.items()})
        for k, v in children.items():
            if "children" in v:
                walk(v["children"], parts[k])
            elif parts[k] == 0:
                raise ValueError(f"pack {k} gets 0 tokens in a {total}-token batch; use a larger batch or drop it")
            else:
                out[k] = parts[k]

    walk(tree, total)
    return dict(sorted(out.items()))


def pack_weights(tree: dict, scale: float = 1.0) -> dict:
    """pack -> share of the batch before rounding."""
    out = {}
    s = sum(v["w"] for v in tree.values())
    for k, v in tree.items():
        w = scale * v["w"] / s
        if "children" in v:
            out.update(pack_weights(v["children"], w))
        else:
            out[k] = w
    return dict(sorted(out.items()))


# ---- the loader ---------------------------------------------------------------------------------------------------

def read_weights(weights: dict, base: Path):
    """Read weights from the config or a YAML/JSON file; returns (weights, base dir, file info)."""
    if "file" not in weights:
        return weights, base, None
    if set(weights) != {"file"}:
        raise ValueError(f"weights: give `file` alone or the weights inline, not both: {sorted(weights)}")
    path = Path(os.path.expandvars(str(weights["file"]))).expanduser()
    path = path if path.is_absolute() else base / path
    raw = path.read_bytes()
    if path.suffix == ".json":
        data = json.loads(raw)
    else:
        import yaml
        data = yaml.safe_load(raw)
    if not isinstance(data, dict) or not set(data) <= {"lanes", "indic"}:
        raise ValueError(f"{path}: a weights file holds {{lanes, indic}} only")
    return data, path.parent, {"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def read_sources_file(path, base: Path):
    """({pack: source}, {name, sha256} or None) from Team 1's approved_inputs.json (`packs`: [{pack_id, source}])."""
    if not path:
        return {}, None
    path = Path(os.path.expandvars(str(path))).expanduser()
    path = path if path.is_absolute() else base / path
    raw = path.read_bytes()
    rows = json.loads(raw).get("packs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: no `packs` list")
    out = {}
    for r in rows:
        pack, src = r.get("pack_id"), r.get("source")
        if not pack or not src:
            raise ValueError(f"{path}: a pack row without pack_id or source: {r.get('pack_id')}")
        if out.get(pack, src) != src:
            raise ValueError(f"{path}: pack {pack} has two sources: {out[pack]!r}, {src!r}")
        out[pack] = str(src)
    return out, {"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def pack_sources(tables: dict, packs: list, file_map: dict):
    """Source id per pack: manifest, then the sources file, else the pack id; returns (map, notes)."""
    out, fallback = {}, []
    for p in packs:
        man, fil = tables[p].source_id, file_map.get(p)
        if man and fil and man != fil:
            raise ValueError(f"sources: pack {p} is {fil!r} in the sources file but {man!r} in its manifest")
        out[p] = man or fil or p
        if not (man or fil):
            fallback.append(p)
    notes = [f"source id = pack id for {len(fallback)} pack(s) with no source_id in the manifest and no "
             f"sources_file row (assumption): {', '.join(fallback)}"] if fallback else []
    return out, notes


def load_config(src):
    """(config dict, base dir for its relative paths) from a YAML path or a dict."""
    if isinstance(src, dict):
        return src, Path.cwd()
    import yaml
    path = Path(src).expanduser()
    return yaml.safe_load(path.read_text()), path.parent


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def batch_sha256(batch) -> str:
    """sha256 of a batch tensor's bytes (the same bytes loader_contract hashes)."""
    return hashlib.sha256(batch.cpu().numpy().tobytes()).hexdigest()


class MixLoader:
    """See the module docstring. Config keys: seed, shards {roots, split, tokenizer_sha256, eos_id, verify, missing},
    stages [{from_batch, seq_len, rows}], weights {lanes, indic {share, split_csv}} or {file}, sources_file,
    caps {max_source_passes, max_dataset_epochs, prior_epochs {source: epochs}}, total_batches."""

    def __init__(self, config, dp_rank: int = 0, dp_world: int = 1, log_path=None):
        cfg, base = load_config(config)
        if not 0 <= dp_rank < dp_world:
            raise ValueError(f"dp_rank {dp_rank} outside dp_world {dp_world}")
        self.cfg, self.dp_rank, self.dp_world, self.seed = cfg, dp_rank, dp_world, int(cfg.get("seed", 0))
        unknown = sorted(set(cfg) - CONFIG_KEYS)
        if unknown:
            raise ValueError(f"config: unknown keys {unknown} (known: {sorted(CONFIG_KEYS)})")
        self.packing = "best_fit"                     # kept in the mix hash and the log header
        self.stages = self._stages(cfg.get("stages") or [])
        for st in self.stages:
            if st.rows % dp_world:                    # AssertionError: the contract skips such a world size
                raise AssertionError(f"rows={st.rows} (from batch {st.from_batch}) is not divisible by "
                                     f"dp_world={dp_world}")

        sc = cfg.get("shards") or {}
        roots = [Path(os.path.expandvars(str(r))).expanduser() for r in sc.get("roots") or []]
        roots = [r if r.is_absolute() else base / r for r in roots]
        shards = read_shards(roots, split=sc.get("split", "train"),
                             tokenizer_sha256=sc.get("tokenizer_sha256", P1536_SHA256),
                             eos_id=int(sc.get("eos_id", EOS_ID)), verify=sc.get("verify", "size"))
        self.manifest = manifest_sha256(shards)
        self.tables = doc_tables(shards)
        weights, wbase, self.weights_file = read_weights(cfg.get("weights") or {}, base)
        tree, self.notes = mix_tree(weights, self.tables, wbase, sc.get("missing", "error"))
        self.tree = tree
        self.weights = pack_weights(tree)
        self.packs = sorted(self.weights)
        self.pack_ix = {p: i for i, p in enumerate(self.packs)}
        self.supply = {p: self.tables[p].tokens for p in self.packs}
        self.quota = [quotas(tree, st.tokens) for st in self.stages]
        self.base = [dict.fromkeys(self.packs, 0)]
        for i in range(1, len(self.stages)):
            span = self.stages[i].from_batch - self.stages[i - 1].from_batch
            self.base.append({p: self.base[-1][p] + span * self.quota[i - 1][p] for p in self.packs})
        self._froms = [st.from_batch for st in self.stages]

        self.mix = hashlib.sha256(_canon({                 # stages are checked separately (load_state_dict)
            "algo": ALGO, "seed": self.seed, "packing": self.packing,
            "weights": {p: repr(w) for p, w in self.weights.items()}}).encode()).hexdigest()

        file_map, self.sources_file = read_sources_file(cfg.get("sources_file"), base)
        self.sources, notes = pack_sources(self.tables, self.packs, file_map)
        self.notes += notes
        self.source_packs: dict = {}
        for p in self.packs:
            self.source_packs.setdefault(self.sources[p], []).append(p)
        self.source_supply = {src: sum(self.supply[p] for p in ps) for src, ps in sorted(self.source_packs.items())}
        self.prior_epochs, self.caps = self._caps(cfg.get("caps") or {})
        self.batches_allowed, self.limit_reason = self._limit()
        total = cfg.get("total_batches")
        if total is not None and int(total) > self.batches_allowed:
            raise ValueError(f"total_batches={total} exceeds the caps: {self.batches_allowed} batches allowed "
                             f"({self.limit_reason})")

        self.position, self.skip = 0, []
        self._mmaps, self._epochs, self._peek = {}, {}, None
        self._log = CompositionLog(log_path, self.header()) if (log_path and dp_rank == 0) else None

    @staticmethod
    def _stages(raw) -> list:
        if not raw:
            raise ValueError("config needs stages: [{from_batch: 0, seq_len, rows}]")
        st = sorted((Stage(int(s.get("from_batch", 0)), int(s["seq_len"]), int(s["rows"])) for s in raw),
                    key=lambda s: s.from_batch)
        if st[0].from_batch != 0 or len({s.from_batch for s in st}) != len(st):
            raise ValueError("stages must start at from_batch 0, with distinct from_batch values")
        if any(s.seq_len < 1 or s.rows < 1 for s in st):
            raise ValueError("seq_len and rows must be positive")
        return st

    # ---- caps --------------------------------------------------------------------------------------------------

    def _caps(self, caps: dict):
        """Per-source cap rows (passes this run, epochs over all phases) and prior epochs."""
        unknown = sorted(set(caps) - CAP_KEYS)
        if unknown:
            raise ValueError(f"caps: unknown keys {unknown} (known: {sorted(CAP_KEYS)})")
        prior = {str(k): float(v) for k, v in (caps.get("prior_epochs") or {}).items()}
        unknown = sorted(set(prior) - set(self.source_packs))
        if unknown:
            raise ValueError(f"caps.prior_epochs: {unknown} not a source of this run ({sorted(self.source_packs)})")
        if any(v < 0 for v in prior.values()):
            raise ValueError(f"caps.prior_epochs must be >= 0: {prior}")
        prior = {src: prior.get(src, 0.0) for src in self.source_packs}
        out = []
        reads, ep = caps.get("max_source_passes"), caps.get("max_dataset_epochs")
        for src, packs in sorted(self.source_packs.items()):
            for p in packs:                        # a source's count is its most-read pack: one cap per pack
                S = self.supply[p]
                if reads is not None:
                    out.append((f"source {src} at {reads} passes (pack {p})", S, math.floor(float(reads) * S), [p]))
                if ep is not None:
                    left = max(0.0, float(ep) - prior[src])
                    out.append((f"source {src} at {ep} epochs, {prior[src]:g} before this run (pack {p})", S,
                                math.floor(left * S), [p]))
        return dict(sorted(prior.items())), out

    def _limit(self):
        """(batches the caps allow, which cap binds). Closed form over the piecewise-linear cursor."""
        best, why = math.inf, "no cap"
        for name, _, cap, packs in self.caps:
            n = math.inf
            for i, st in enumerate(self.stages):
                q = sum(self.quota[i][p] for p in packs)
                b = sum(self.base[i][p] for p in packs)
                end = self.stages[i + 1].from_batch if i + 1 < len(self.stages) else None
                if end is not None and b + (end - st.from_batch) * q <= cap:
                    continue
                n = st.from_batch + (cap - b) // q
                break
            if n < best:
                best, why = n, name
        return best, why

    def passes(self, n: int) -> dict:
        """pack -> passes over its docs after batches 0..n-1."""
        return {p: self.cursor(p, n) / self.supply[p] for p in self.packs}

    def source_passes(self, n: int) -> dict:
        """source -> passes of its most-read pack after batches 0..n-1, this run only (what the caps count)."""
        per = self.passes(n)
        return {src: max(per[p] for p in ps) for src, ps in sorted(self.source_packs.items())}

    def source_passes_mean(self, n: int) -> dict:
        """source -> tokens read from all its packs / its supply, after batches 0..n-1 (reported, not capped)."""
        return {src: sum(self.cursor(p, n) for p in ps) / self.source_supply[src]
                for src, ps in sorted(self.source_packs.items())}

    # ---- streams and layout -----------------------------------------------------------------------------------

    def stage(self, n: int) -> Stage:
        return self.stages[bisect.bisect_right(self._froms, n) - 1]

    def cursor(self, pack: str, n: int) -> int:
        """Tokens of `pack` read by batches 0..n-1."""
        i = bisect.bisect_right(self._froms, n) - 1
        return self.base[i][pack] + (n - self.stages[i].from_batch) * self.quota[i][pack]

    def _epoch(self, pack: str, e: int):
        """(doc order, prefix sums of doc lengths in that order) for epoch e of a pack; the last two are cached."""
        cache = self._epochs.setdefault(pack, OrderedDict())
        if e not in cache:
            t = self.tables[pack]
            perm = permutation(t.docs, _key(self.seed, "docs", pack, e))
            cum = np.zeros(t.docs + 1, dtype=np.int64)
            np.cumsum(t.length[perm], out=cum[1:])
            cache[e] = (perm, cum)
            while len(cache) > 2:
                cache.popitem(last=False)
        cache.move_to_end(e)
        return cache[e]

    def _pieces(self, pack: str, a: int, b: int):
        """(doc, offset in doc, length) of the pack's stream tokens [a, b), in stream order."""
        S = self.supply[pack]
        docs, offs, lens = [], [], []
        while a < b:
            e, o = divmod(a, S)
            oe = min(b - e * S, S)
            perm, cum = self._epoch(pack, e)
            j0 = int(np.searchsorted(cum, o, "right")) - 1
            j1 = int(np.searchsorted(cum, oe, "left"))
            k = np.arange(j0, j1)
            lo = np.maximum(o - cum[k], 0)
            hi = np.minimum(oe - cum[k], cum[k + 1] - cum[k])
            docs.append(perm[k])
            offs.append(lo)
            lens.append(hi - lo)
            a = e * S + oe
        return np.concatenate(docs), np.concatenate(offs), np.concatenate(lens)

    def layout(self, n: int) -> Layout:
        """The global packed layout of batch n (no tokens read)."""
        if n < 0:
            raise ValueError(f"batch {n} < 0")
        if n >= self.batches_allowed:
            raise CapReached(f"batch {n}: {self.batches_allowed} batches allowed ({self.limit_reason})")
        st, q = self.stage(n), self.quota[bisect.bisect_right(self._froms, n) - 1]
        parts = []
        for p in self.packs:
            a = self.cursor(p, n)
            d, o, ln = self._pieces(p, a, a + q[p])
            parts.append((np.full(len(d), self.pack_ix[p], dtype=np.int32), d, o, ln))
        pack, doc, off, length = (np.concatenate(x) for x in zip(*parts))
        order = permutation(len(doc), _key(self.seed, "pieces", n))
        pack, doc, off, length = pack[order], doc[order], off[order], length[order]
        if int(length.sum()) != st.tokens:
            raise AssertionError(f"batch {n}: {int(length.sum())} tokens laid out, expected {st.tokens}")
        doclen = np.empty(len(doc), dtype=np.int64)
        for i, p in enumerate(self.packs):
            m = pack == i
            doclen[m] = self.tables[p].length[doc[m]]
        edge = int((off > 0).sum() + (off + length < doclen).sum())
        L = st.seq_len
        j, w, ln, start = best_fit_rows(length, st.rows, L, self.seed, n)
        return Layout(n=n, stage=st, pack=pack[j], doc=doc[j], off=off[j] + w, length=ln, start=start,
                      pieces=len(length), edge_cuts=edge, row_cuts=len(ln) - len(length),
                      forced_cuts=int(((length - 1) // L).sum()))

    def _mmap(self, pack: str, shard: int):
        key = (pack, int(shard))
        if key not in self._mmaps:
            self._mmaps[key] = open_shard(self.tables[pack].shards[int(shard)].bin_path)
        return self._mmaps[key]

    def rows(self, lay: Layout, which) -> np.ndarray:
        """int64 [len(which), 3, seq_len] (tokens, seg_id, pos_id) for global rows `which` of a layout."""
        L = lay.stage.seq_len
        ends = lay.start + lay.length
        out = np.empty((len(which), 3, L), dtype=np.int64)
        for i, k in enumerate(which):
            lo, hi = k * L, (k + 1) * L
            col = 0
            for seg, j in enumerate(range(int(np.searchsorted(ends, lo, "right")),
                                          int(np.searchsorted(lay.start, hi, "left")))):
                s, e = max(int(lay.start[j]), lo), min(int(ends[j]), hi)
                m = e - s
                pack = self.packs[lay.pack[j]]
                t, d = self.tables[pack], int(lay.doc[j])
                src = int(t.start[d]) + int(lay.off[j]) + (s - int(lay.start[j]))
                out[i, 0, col:col + m] = self._mmap(pack, t.shard[d])[src:src + m]
                out[i, 1, col:col + m] = seg
                out[i, 2, col:col + m] = np.arange(m)
                col += m
            if col != L:
                raise AssertionError(f"batch {lay.n} row {k}: {col} tokens, expected {L}")
        return out

    def batch(self, n: int) -> torch.Tensor:
        """This rank's rows of global batch n, without touching the state."""
        lay = self.layout(n)
        return torch.from_numpy(self.rows(lay, range(self.dp_rank, lay.stage.rows, self.dp_world)))

    def global_batch(self, n: int) -> torch.Tensor:
        """All rows of global batch n."""
        lay = self.layout(n)
        return torch.from_numpy(self.rows(lay, range(lay.stage.rows)))

    # ---- contract: state, peek, next, skip --------------------------------------------------------------------

    def _resolve(self) -> int:
        skip, n = set(self.skip), self.position
        while n in skip:
            n += 1
        return n

    def _get(self, n: int) -> torch.Tensor:
        if self._peek is None or self._peek[0] != n:
            self._peek = (n, self.batch(n))
        return self._peek[1]

    def next_batch_id(self) -> dict:
        """{"position": global batch number, "sha256": hash of this rank's next batch}, without consuming it."""
        n = self._resolve()
        return {"position": n, "sha256": batch_sha256(self._get(n))}

    def next(self) -> torch.Tensor:
        n = self._resolve()
        b = self._get(n)
        self.position, self._peek = n + 1, None
        self.skip = [x for x in self.skip if x > n]
        if self._log is not None:
            full = b if self.dp_world == 1 else self.global_batch(n)
            self._log.write(self.composition(n, batch_sha256(full)))
        return b

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def add_skip(self, k: int) -> None:
        """Skip the next k global batch numbers that are not already skipped."""
        skip, n = set(self.skip), self.position
        while k > 0:
            if n not in skip:
                skip.add(n)
                k -= 1
            n += 1
        self.skip, self._peek = sorted(skip), None

    def state_dict(self) -> dict:
        """One global position, the same on every rank (no rank or world in it)."""
        return {"schema": STATE_SCHEMA, "position": self.position, "skip": list(self.skip),
                "manifest": self.manifest, "mix": self.mix, "stages": [asdict(s) for s in self.stages]}

    def load_state_dict(self, d: dict) -> None:
        """Raises ValueError on another shard list, another mix, or stages that differ before the saved position
        (stages from the saved position on may differ: that is how a later stage is added on resume)."""
        if d.get("manifest") != self.manifest:
            raise ValueError(f"state manifest {d.get('manifest')!r} is not this loader's shard list {self.manifest}")
        if d.get("mix") != self.mix:
            raise ValueError(f"state mix {d.get('mix')!r} is not this loader's mix {self.mix}")
        pos = int(d["position"])
        if "stages" not in d:
            raise ValueError("state has no stages")
        before = lambda stages: [dict(s) for s in stages if int(s["from_batch"]) < pos]
        if before(d["stages"]) != before(asdict(s) for s in self.stages):
            raise ValueError(f"stages before position {pos} differ from the saved run's: {d['stages']}")
        self.position, self.skip, self._peek = pos, sorted(int(x) for x in d.get("skip", [])), None

    # ---- composition log --------------------------------------------------------------------------------------

    def header(self) -> dict:
        return {"schema": LOG_SCHEMA, "algo": ALGO, "manifest": self.manifest, "mix": self.mix, "seed": self.seed,
                "packing": self.packing, "stages": [asdict(s) for s in self.stages], "weights": self.weights, "quotas": self.quota,
                "supply": self.supply, "weights_file": self.weights_file, "sources_file": self.sources_file,
                "sources": {src: {"packs": ps, "supply": self.source_supply[src],
                                  "prior_epochs": self.prior_epochs[src]} for src, ps in sorted(self.source_packs.items())},
                "caps": [{"cap": c[0], "supply": c[1], "cap_tokens": c[2]} for c in self.caps],
                "batches_allowed": None if self.batches_allowed == math.inf else self.batches_allowed,
                "limit": self.limit_reason, "notes": self.notes}

    def composition(self, n: int, sha=None) -> dict:
        """One composition-log line for global batch n (deterministic; any rank or an offline CLI can make it)."""
        lay = self.layout(n)
        L = lay.stage.seq_len
        per = np.bincount(lay.pack, weights=lay.length, minlength=len(self.packs)).astype(np.int64)
        by_pack = {p: int(per[i]) for i, p in enumerate(self.packs)}
        by_lane, by_lang = {}, {}
        for p, v in by_pack.items():
            t = self.tables[p]
            by_lane[t.lane] = by_lane.get(t.lane, 0) + v
            by_lang[t.language] = by_lang.get(t.language, 0) + v
        end = lay.start + lay.length
        segments = int(((end - 1) // L - lay.start // L + 1).sum())
        nonstart = int(lay.length[lay.off > 0].sum())
        return {"batch": n, "tokens": lay.stage.tokens, "rows": lay.stage.rows, "seq_len": L, "segments": segments,
                "pieces": lay.pieces, "doc_starts": int((lay.off == 0).sum()),
                "cuts": {"batch_edge": lay.edge_cuts, "row": lay.row_cuts, "row_forced": lay.forced_cuts},
                "nonstart_share": round(nonstart / lay.stage.tokens, 6), "by_pack": by_pack, "by_lane": dict(sorted(by_lane.items())),
                "by_language": dict(sorted(by_lang.items())),
                "passes": {p: round(v, 6) for p, v in self.passes(n + 1).items()},
                "source_passes": {k: round(v, 6) for k, v in self.source_passes(n + 1).items()},
                "source_passes_mean": {k: round(v, 6) for k, v in self.source_passes_mean(n + 1).items()},
                "source_epochs": {k: round(v + self.prior_epochs[k], 6) for k, v in self.source_passes(n + 1).items()},
                "batch_sha256": sha if sha is not None else batch_sha256(self.global_batch(n))}


class CompositionLog:
    """JSONL: a header line, then one line per batch. Appends to an existing log only if its header has the same
    manifest and mix."""

    def __init__(self, path, header: dict):
        self.path = Path(path)
        if self.path.exists() and self.path.stat().st_size:
            first = json.loads(self.path.open().readline())
            if (first.get("manifest"), first.get("mix")) != (header["manifest"], header["mix"]):
                raise ValueError(f"{self.path}: log was written for another manifest or mix")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.write(header)

    def write(self, line: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")


# ---- helpers for the trainer --------------------------------------------------------------------------------------

def targets(batch: torch.Tensor, shift: int = 1) -> torch.Tensor:
    """y[t] = x[t + shift] when t and t + shift are in the same segment, else -100 (row ends, doc ends: the EOS to
    next-doc target is masked). batch: [R, 3, L] from next()."""
    x, seg = batch[:, 0], batch[:, 1]
    y = torch.full_like(x, IGNORE)
    same = seg[:, shift:] == seg[:, :-shift]
    y[:, :-shift] = torch.where(same, x[:, shift:], torch.full_like(x[:, shift:], IGNORE))
    return y


def cu_seqlens(batch: torch.Tensor) -> torch.Tensor:
    """int32 segment offsets over the flattened rows ([1, R x L] input), for fla varlen kernels."""
    starts = (batch[:, 2] == 0).reshape(-1).nonzero().squeeze(1)
    total = torch.tensor([batch.shape[0] * batch.shape[2]], dtype=starts.dtype)
    return torch.cat([starts, total]).to(torch.int32)


# ---- contract factory --------------------------------------------------------------------------------------------

def contract_factory(rank: int, world: int) -> MixLoader:
    """factory(rank, world) for loader_contract.py (its rank and world are the data-parallel ones); the config path
    is in G6_MIX_CONFIG."""
    path = os.environ.get("G6_MIX_CONFIG")
    if not path:
        raise RuntimeError("set G6_MIX_CONFIG to a mix config YAML")
    return MixLoader(path, dp_rank=rank, dp_world=world)


if __name__ == "__main__":                                 # the command line is dataio/mix_cli.py
    from dataio.mix_cli import main
    sys.exit(main())
