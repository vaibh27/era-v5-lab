"""Mixture loader (dataio/mix_loader.py, dataio/shards.py) on synthetic shard-v1 data. CPU only, no download.

Every non-EOS token id in the fixture is unique, so each token traces back to its pack, doc and offset.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dataio import mix_loader as ml
from dataio.prepare_fineweb import write_shard
from dataio.shards import EOS_ID, P1536_SHA256, manifest_sha256, read_shard, read_shards

REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / "teams" / "07-infrastructure" / "ckpt" / "loader_contract.py"

# pack: (lane, language, docs per shard)
PACKS = {
    "code-go": ("code", "go", [40, 30]),
    "code-py": ("code", "python", [60]),
    "cadj-commits": ("code-adjacent", "code", [50]),
    "en-web": ("english", "en", [40, 40]),
    "math-web": ("math", "en", [60]),
    "sci-pdf": ("science", "en", [50]),
    "ind-hi-fw2": ("indic", "hi", [40]),
    "ind-hi-hplt": ("indic", "hi", [30]),
    "ind-ta-sangraha-v": ("indic", "ta", [30]),
    "ind-hi-sangraha-s": ("indic", "hi", [20]),          # translated: not in the Indic split, so never read
}
LANES = {"code": 0.4, "code-adjacent": 0.1, "english": 0.3, "math": 0.1, "science": 0.1}
SPLIT = [("hi", "ind-hi-fw2 ind-hi-hplt", 5.0), ("ta", "ind-ta-sangraha-v", 3.0), ("ml", "no shard", 0.0)]


class Corpus:
    """Synthetic shard-v1 files in a directory, plus the ground truth: each pack's docs in DocTable order."""

    def __init__(self, root: Path, packs=PACKS, seed=0, max_len=60, split_rows=SPLIT, extra_split=None, rng=None,
                 sources=None):
        """rng: any object with integers(lo, hi); default numpy default_rng(seed). sources: pack -> manifest source_id."""
        self.root, self.docs, self.sources = root, {}, sources or {}
        root.mkdir(parents=True, exist_ok=True)
        rng, nxt = rng or np.random.default_rng(seed), 1
        self.lut = np.full((EOS_ID, 3), -1, dtype=np.int64)          # id -> (pack index, doc, offset)
        self.names = sorted(packs)
        for pi, pack in enumerate(self.names):
            lane, lang, per_shard = packs[pack]
            self.docs[pack] = []
            for si, n_docs in enumerate(per_shard):
                toks, starts = [], [0]
                for _ in range(n_docs):
                    n = int(rng.integers(1, max_len))
                    ids = np.arange(nxt, nxt + n, dtype=np.int64)
                    nxt += n
                    assert nxt < EOS_ID
                    d = len(self.docs[pack])
                    self.lut[ids] = np.stack([np.full(n, pi), np.full(n, d), np.arange(n)], 1)
                    doc = np.concatenate([ids, [EOS_ID]])
                    self.docs[pack].append(doc)
                    toks.append(doc)
                    starts.append(starts[-1] + len(doc))
                self._write(pack, lane, lang, si, np.concatenate(toks), np.array(starts, dtype=np.int64), "train")
            if extra_split == pack:
                self._write(pack, lane, lang, 0, np.array([5, EOS_ID]), np.array([0, 2], dtype=np.int64),
                            "validation")
        self.csv = root / "indic_share.csv"
        lines = ["lang,packs,native_tokens,share_of_native_tokens,share_of_batch_pct"]
        lines += [f"{lang},{p},0,0%,{pct}" for lang, p, pct in split_rows]
        self.csv.write_text("\n".join(lines) + "\n")

    def _write(self, pack, lane, lang, num, tokens, idx, split):
        sid = f"{pack}-{split}-{num:06d}"
        write_shard(self.root / f"{sid}.bin", tokens)
        np.save(self.root / f"{sid}.idx.npy", idx)
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        m = {"schema_version": "shard-v1", "dataset_version": "test", "shard_id": sid,
             "format": "brahmic_u32_v2+doc_index", "token_dtype": "uint32", "payload_uri": f"{sid}.bin",
             "payload_sha256": sha(self.root / f"{sid}.bin"), "payload_bytes": 1024 + 4 * len(tokens),
             "index_uri": f"{sid}.idx.npy", "index_sha256": sha(self.root / f"{sid}.idx.npy"),
             "doc_count": len(idx) - 1, "token_count": len(tokens), "eos_token_id": EOS_ID,
             "tokenizer_sha256": P1536_SHA256, "pack_id": pack, "language": lang, "script": "Latn",
             "domain": lane, "capability_lane": lane, "split": split}
        if pack in self.sources:
            m["source_id"] = self.sources[pack]
        (self.root / f"{sid}.json").write_text(json.dumps(m, indent=1))

    def config(self, rows=8, seq_len=64, stages=None, caps=None, lanes=LANES, missing="error", total=None, seed=7):
        return {"seed": seed,
                "shards": {"roots": [str(self.root)], "missing": missing},
                "stages": stages or [{"from_batch": 0, "seq_len": seq_len, "rows": rows}],
                "total_batches": total,
                "weights": {"lanes": lanes, "indic": {"share": 0.08, "split_csv": str(self.csv)}},
                "caps": caps if caps is not None else {}}


@pytest.fixture
def corpus(tmp_path):
    return Corpus(tmp_path)


def segments(row):
    """[(start, stop)] of each segment of one row [3, L], from seg_id."""
    seg = row[1].numpy()
    cuts = [0] + [i for i in range(1, len(seg)) if seg[i] != seg[i - 1]] + [len(seg)]
    return list(zip(cuts[:-1], cuts[1:]))


def check_content(c: Corpus, batch, loader=None, n=None):
    """Every segment is a contiguous run of one source doc; seg_id counts from 0; pos_id is the segment offset.
    A lone-EOS segment (a doc cut just before its EOS) has no traceable id: with loader and n given, the layout
    must place it at that doc's last token, and it is counted for its pack."""
    per_pack = np.zeros(len(c.names), dtype=np.int64)
    lay = loader.layout(n) if loader is not None else None
    for r, row in enumerate(batch):
        tok = row[0].numpy()
        for k, (a, b) in enumerate(segments(row)):
            assert (row[1, a:b] == k).all()
            assert (row[2, a:b].numpy() == np.arange(b - a)).all()
            real = tok[a:b][tok[a:b] != EOS_ID]
            if not len(real):
                assert b - a == 1
                if lay is not None:
                    g = r * batch.shape[2] + a
                    j = int(np.searchsorted(lay.start, g, "right")) - 1
                    pack = loader.packs[lay.pack[j]]
                    doc = int(lay.doc[j])
                    assert int(lay.off[j]) + g - int(lay.start[j]) == len(c.docs[pack][doc]) - 1
                    per_pack[c.names.index(pack)] += 1
                continue
            pi, d, off = c.lut[real[0]]
            first = int(np.nonzero(tok[a:b] != EOS_ID)[0][0])
            assert first == 0, "an EOS may only end a segment"
            doc = c.docs[c.names[pi]][d]
            assert (tok[a:b] == doc[off:off + b - a]).all()
            per_pack[pi] += b - a
    return {c.names[i]: int(v) for i, v in enumerate(per_pack) if v}


def ranks_to_global(loaders, n):
    """Global batch n rebuilt from ranks: global row k is rank k % world, local row k // world."""
    world = len(loaders)
    parts = [x.batch(n) for x in loaders]
    rows = [parts[k % world][k // world] for k in range(sum(len(p) for p in parts))]
    return torch.stack(rows)


# ---- shares and content --------------------------------------------------------------------------------------------

def test_quotas_are_exact_in_every_batch(corpus):
    L = ml.MixLoader(corpus.config())
    T = 8 * 64
    q = L.quota[0]
    assert sum(q.values()) == T
    assert "ind-hi-sangraha-s" not in q and any("ind-hi-sangraha-s" in n for n in L.notes)
    indic = sum(v for p, v in q.items() if p.startswith("ind-"))
    assert abs(indic / T - 0.08) <= 1 / T
    hi = q["ind-hi-fw2"] + q["ind-hi-hplt"]
    assert abs(hi - round(indic * 5 / 8)) <= 1
    for n in range(12):
        comp = L.composition(n)
        assert comp["by_pack"] == q
        assert comp["by_lane"]["indic"] == indic
        assert check_content(corpus, L.global_batch(n), L, n) == q


def test_rows_are_full_and_traceable(corpus):
    L = ml.MixLoader(corpus.config(rows=4, seq_len=100))
    for n in range(8):
        b = L.global_batch(n)
        assert b.shape == (4, 3, 100) and b.dtype == torch.int64
        check_content(corpus, b)
        comp = L.composition(n)
        assert comp["segments"] == sum(len(segments(r)) for r in b)


def test_each_epoch_reads_every_doc_once_in_a_new_order(corpus):
    L = ml.MixLoader(corpus.config())
    for pack in ("code-go", "math-web"):
        S, t = L.supply[pack], L.tables[pack]
        d, off, ln = L._pieces(pack, 0, 2 * S)
        assert (off == 0).all() and (ln == t.length[d]).all()
        e0, e1 = d[: t.docs], d[t.docs:]
        assert sorted(e0) == list(range(t.docs)) and sorted(e1) == list(range(t.docs))
        assert (e0 != e1).any()
        a, b = 37, S + 91                                          # a range that crosses the epoch boundary
        d2, off2, ln2 = L._pieces(pack, a, b)
        assert int(ln2.sum()) == b - a


def test_adding_a_pack_keeps_other_packs_order(tmp_path):
    a = Corpus(tmp_path / "a")
    packs = dict(PACKS, **{"code-rs": ("code", "rust", [30])})
    b = Corpus(tmp_path / "b", packs=packs)
    La, Lb = ml.MixLoader(a.config()), ml.MixLoader(b.config())
    assert (La._epoch("en-web", 3)[0] == Lb._epoch("en-web", 3)[0]).all()


# ---- packing helpers -----------------------------------------------------------------------------------------------

def test_targets_mask_segment_edges_and_eos(corpus):
    L = ml.MixLoader(corpus.config())
    b = L.global_batch(3)
    x, seg, pos = b[:, 0], b[:, 1], b[:, 2]
    for shift, y in ((1, ml.targets(b)), (2, ml.targets(b, 2))):
        keep = y != ml.IGNORE
        assert (y[:, -shift:] == ml.IGNORE).all()
        r, t = keep.nonzero(as_tuple=True)
        assert (seg[r, t] == seg[r, t + shift]).all() and (y[r, t] == x[r, t + shift]).all()
        r, t = (~keep[:, :-shift]).nonzero(as_tuple=True)
        assert (seg[r, t] != seg[r, t + shift]).all()
    y = ml.targets(b)
    assert (y[x == EOS_ID] == ml.IGNORE).all()                      # EOS to next doc is never a target
    cu = ml.cu_seqlens(b)
    assert cu.dtype == torch.int32 and cu[0] == 0 and cu[-1] == b.shape[0] * b.shape[2]
    lens = torch.diff(cu).tolist()
    assert lens == [bb - aa for row in b for aa, bb in segments(row)]
    assert (pos.reshape(-1)[cu[:-1].long()] == 0).all()


# ---- statelessness, ranks, state -----------------------------------------------------------------------------------

def test_jump_to_n_equals_run_from_zero(corpus):
    run = ml.MixLoader(corpus.config())
    straight = [run.next() for _ in range(15)]
    jump = ml.MixLoader(corpus.config())
    assert torch.equal(jump.batch(14), straight[14])
    st = dict(jump.state_dict(), position=9)
    jump.load_state_dict(st)
    assert torch.equal(jump.next(), straight[9])


@pytest.mark.parametrize("world", [2, 4, 8])
def test_same_global_batch_at_any_world_size(corpus, world):
    one = ml.MixLoader(corpus.config())
    many = [ml.MixLoader(corpus.config(), dp_rank=r, dp_world=world) for r in range(world)]
    for n in (0, 5, 17):
        assert torch.equal(ranks_to_global(many, n), one.global_batch(n))
        assert ml.batch_sha256(ranks_to_global(many, n)) == one.composition(n)["batch_sha256"]


def test_world_must_divide_rows(corpus):
    with pytest.raises(AssertionError):
        ml.MixLoader(corpus.config(rows=6), dp_rank=0, dp_world=4)


def test_state_peek_skip_and_mismatch(corpus):
    L = ml.MixLoader(corpus.config())
    ref = [L.batch(n) for n in range(10)]
    for _ in range(3):
        L.next()
    peek = L.next_batch_id()
    assert L.next_batch_id() == peek and peek["position"] == 3
    assert peek["sha256"] == ml.batch_sha256(ref[3])
    st = json.loads(json.dumps(L.state_dict()))
    assert set(st) == {"schema", "position", "skip", "manifest", "mix", "stages"}
    L.add_skip(2)
    assert L.next_batch_id()["position"] == 5 and torch.equal(L.next(), ref[5])
    assert L.state_dict()["skip"] == [] and L.state_dict()["position"] == 6
    L.add_skip(1)
    L2 = ml.MixLoader(corpus.config())
    L2.load_state_dict(json.loads(json.dumps(L.state_dict())))
    assert torch.equal(L2.next(), ref[7])
    for bad in (dict(st, manifest="0" * 64), dict(st, mix="0" * 64)):
        with pytest.raises(ValueError):
            ml.MixLoader(corpus.config()).load_state_dict(bad)
    other_seed = ml.MixLoader(corpus.config(seed=8))
    with pytest.raises(ValueError):
        other_seed.load_state_dict(st)


# ---- epochs and caps -----------------------------------------------------------------------------------------------

def test_tiny_pack_spans_several_epochs_per_batch(tmp_path):
    packs = dict(PACKS, **{"math-web": ("math", "en", [3])})
    c = Corpus(tmp_path, packs=packs, max_len=5)
    L = ml.MixLoader(c.config())
    S, q = L.supply["math-web"], L.quota[0]["math-web"]
    assert q > 2 * S
    for n in range(6):
        got = check_content(c, L.global_batch(n), L, n)
        assert got == L.quota[0]
    assert L.passes(6)["math-web"] == pytest.approx(6 * q / S)


def test_caps_stop_at_closed_form_limit(corpus):
    caps = {"max_source_passes": 2, "max_dataset_epochs": 4}
    L = ml.MixLoader(corpus.config(caps=caps))
    q, S = L.quota[0], L.supply
    want = min((2 * S[p]) // q[p] for p in L.packs)
    assert L.batches_allowed == want
    L.load_state_dict(dict(L.state_dict(), position=want - 1))
    L.next()
    with pytest.raises(ml.CapReached):
        L.next()
    assert max(L.passes(want).values()) <= 2
    assert ml.MixLoader(corpus.config(caps=caps, total=want)).batches_allowed == want
    with pytest.raises(ValueError, match="exceeds the caps"):
        ml.MixLoader(corpus.config(caps=caps, total=want + 1))


@pytest.mark.parametrize("key", ["max_source_passes", "max_dataset_epochs"])
def test_source_cap_binds_on_its_most_read_pack(tmp_path, key):
    """Two packs of one source at different rates: the cap holds on each pack."""
    c = Corpus(tmp_path / "c", sources={"en-web": "hub/web", "math-web": "hub/web"})
    L = ml.MixLoader(c.config(caps={key: 1}))
    assert L.sources["en-web"] == L.sources["math-web"] == "hub/web"
    q, S = L.quota[0], L.supply
    assert L.batches_allowed == min(S[p] // q[p] for p in L.packs)
    n = L.batches_allowed
    per = L.passes(n)
    assert L.source_passes(n)["hub/web"] == max(per["en-web"], per["math-web"]) <= 1
    assert L.source_passes_mean(n)["hub/web"] < L.source_passes(n)["hub/web"]          # rates differ
    pair = ("en-web", "math-web")
    by_mean = (S["en-web"] + S["math-web"]) // (q["en-web"] + q["math-web"])
    assert by_mean > min(S[x] // q[x] for x in pair)              # a mean rule would over-read the faster pack
    line = L.composition(n - 1)
    assert set(line["source_passes"]) == set(line["source_passes_mean"]) == set(L.source_packs)
    assert not any("source id = pack id" in x and "en-web" in x for x in L.notes)
    assert any("source id = pack id" in x and "code-go" in x for x in L.notes)       # the assumption is written down


def _sources_file(tmp_path, rows):
    path = tmp_path / "approved_inputs.json"
    path.write_text(json.dumps({"name": "test", "packs": rows}))
    return path


def test_sources_file(tmp_path):
    c = Corpus(tmp_path / "c", sources={"math-web": "hub/math"})
    rows = [{"pack_id": "ind-hi-fw2", "source": "HuggingFaceFW/fineweb-2"},
            {"pack_id": "ind-ta-sangraha-v", "source": "HuggingFaceFW/fineweb-2"},
            {"pack_id": "math-web", "source": "hub/math"},
            {"pack_id": "not-in-this-shard-set", "source": "x"}]
    path = _sources_file(tmp_path, rows)
    L = ml.MixLoader(dict(c.config(), sources_file=str(path)))
    assert L.sources["ind-hi-fw2"] == L.sources["ind-ta-sangraha-v"] == "HuggingFaceFW/fineweb-2"
    assert L.sources["math-web"] == "hub/math" and L.sources["en-web"] == "en-web"
    assert L.header()["sources_file"] == {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    clash = _sources_file(tmp_path, rows[:2] + [{"pack_id": "math-web", "source": "hub/other"}])
    with pytest.raises(ValueError, match="in its manifest"):
        ml.MixLoader(dict(c.config(), sources_file=str(clash)))
    twice = _sources_file(tmp_path, rows + [{"pack_id": "ind-hi-fw2", "source": "y"}])
    with pytest.raises(ValueError, match="two sources"):
        ml.MixLoader(dict(c.config(), sources_file=str(twice)))


def test_repo_sources_file():
    cfg = yaml.safe_load((Path(ml.__file__).resolve().parents[1] / "configs" / "mix" / "stage1_placeholder.yaml")
                         .read_text())
    base = Path(ml.__file__).resolve().parents[1] / "configs" / "mix"
    m, info = ml.read_sources_file(cfg["sources_file"], base)
    assert len(m) == 38 and len(set(m.values())) == 14
    assert m["ind-hi-fw2"] == m["ind-ta-fw2"] == "HuggingFaceFW/fineweb-2"
    assert info["name"] == "approved_inputs.json"


def test_source_id_from_the_manifest(tmp_path):
    c = Corpus(tmp_path / "c", sources={"en-web": "hub/web", "math-web": "hub/web"})
    L = ml.MixLoader(c.config(caps={"max_source_passes": 3}))
    assert L.sources["en-web"] == L.sources["math-web"] == "hub/web"
    assert L.header()["sources"]["hub/web"]["packs"] == ["en-web", "math-web"]
    second = c.root / "en-web-train-000001.json"                 # one shard of the pack names another source
    second.write_text(json.dumps(json.loads(second.read_text()) | {"source_id": "hub/elsewhere"}))
    with pytest.raises(ValueError, match="disagree on source_id"):
        ml.MixLoader(c.config())


def test_prior_epochs_count_against_the_epoch_cap(corpus):
    L0 = ml.MixLoader(corpus.config())
    S, q = L0.supply["math-web"], L0.quota[0]["math-web"]
    L = ml.MixLoader(corpus.config(caps={"max_dataset_epochs": 4, "prior_epochs": {"math-web": 3.5}}))
    assert L.batches_allowed == math.floor(0.5 * S) // q
    assert "math-web" in L.limit_reason and "3.5 before" in L.limit_reason
    lines = [L.composition(n) for n in (0, L.batches_allowed - 1)]
    assert lines[0]["source_epochs"]["math-web"] == pytest.approx(3.5 + q / S, abs=1e-6)
    assert lines[1]["source_epochs"]["math-web"] <= 4
    done = ml.MixLoader(corpus.config(caps={"max_dataset_epochs": 4, "prior_epochs": {"math-web": 4}}))
    assert done.batches_allowed == 0
    with pytest.raises(ml.CapReached):
        done.next()
    with pytest.raises(ValueError, match="not a source"):
        ml.MixLoader(corpus.config(caps={"max_dataset_epochs": 4, "prior_epochs": {"typo": 1}}))
    with pytest.raises(ValueError, match="caps: unknown keys"):
        ml.MixLoader(corpus.config(caps={"datasets": {}}))


# ---- weights file and realised shares ------------------------------------------------------------------------------

WEIGHT_SETS = {
    "equal": {"code": 1, "code-adjacent": 1, "english": 1, "math": 1, "science": 1},
    "skewed": {"code": 0.60, "code-adjacent": 0.05, "english": 0.20, "math": 0.10, "science": 0.05},
}


def _weights_file(c, tmp_path, name, fmt):
    path = tmp_path / "w" / f"{name}.{fmt}"
    path.parent.mkdir(exist_ok=True)
    data = {"lanes": WEIGHT_SETS[name], "indic": {"share": 0.08, "split_csv": f"../c/{c.csv.name}"}}
    path.write_text(json.dumps(data) if fmt == "json" else yaml.safe_dump(data))
    cfg = c.config()
    cfg["weights"] = {"file": str(path)}
    return cfg, path


@pytest.mark.parametrize("name", sorted(WEIGHT_SETS))
def test_realised_lane_shares_match_the_weights_file(tmp_path, name):
    """Each lane and the Indic total within 1 token per batch, each pack within 3."""
    c = Corpus(tmp_path / "c")
    cfg, path = _weights_file(c, tmp_path, name, "yaml")
    L = ml.MixLoader(cfg)
    w = WEIGHT_SETS[name]
    target = {lane: 0.92 * v / sum(w.values()) for lane, v in w.items()} | {"indic": 0.08}
    T, n_batches = L.stages[0].tokens, 20
    total = dict.fromkeys(target, 0)
    for n in range(n_batches):
        by_pack = check_content(c, L.global_batch(n), L, n)
        by_lane = {}
        for p, v in by_pack.items():
            by_lane[L.tables[p].lane] = by_lane.get(L.tables[p].lane, 0) + v
        assert sum(by_lane.values()) == T
        assert by_lane == L.composition(n)["by_lane"]
        for lane, share in target.items():
            assert abs(by_lane[lane] - share * T) < 1, (n, lane)
            total[lane] += by_lane[lane]
        for pack, v in by_pack.items():                          # one rounding per level: Indic share, language, pack
            assert abs(v - L.weights[pack] * T) < 3, (n, pack)
    for lane, share in target.items():
        assert abs(total[lane] - share * T * n_batches) < n_batches
    assert L.header()["weights_file"] == {"name": path.name,
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_weights_file_formats_and_errors(tmp_path):
    c = Corpus(tmp_path / "c")
    a = ml.MixLoader(_weights_file(c, tmp_path, "skewed", "yaml")[0])
    b = ml.MixLoader(_weights_file(c, tmp_path, "skewed", "json")[0])
    inline = ml.MixLoader(c.config(lanes=WEIGHT_SETS["skewed"]))
    assert a.mix == b.mix == inline.mix                          # same weights, same mix, whatever the source
    assert a.mix != ml.MixLoader(_weights_file(c, tmp_path, "equal", "yaml")[0]).mix
    cfg = c.config()
    cfg["weights"] = {"file": "x.yaml", "lanes": {}}
    with pytest.raises(ValueError, match="not both"):
        ml.MixLoader(cfg)
    bad = tmp_path / "bad.yaml"
    bad.write_text("lanes: supply\nseed: 3\n")
    cfg["weights"] = {"file": str(bad)}
    with pytest.raises(ValueError, match="lanes, indic"):
        ml.MixLoader(cfg)


def test_repo_weight_files_parse():
    accepted = {"code", "code-adjacent", "english", "math", "science"}
    for name in ("weights_equal.yaml", "weights_skewed.yaml"):
        path = Path(ml.__file__).resolve().parents[1] / "configs" / "mix" / name
        data = yaml.safe_load(path.read_text())
        assert set(data["lanes"]) == accepted and all(v > 0 for v in data["lanes"].values())
        assert data["indic"]["share"] == 0.08
        assert (path.parent / data["indic"]["split_csv"]).is_file()


# ---- stages, missing packs, shards, log ----------------------------------------------------------------------------

def test_stage_change_keeps_earlier_batches(corpus):
    one = ml.MixLoader(corpus.config())
    two = ml.MixLoader(corpus.config(stages=[{"from_batch": 0, "seq_len": 64, "rows": 8},
                                             {"from_batch": 5, "seq_len": 128, "rows": 2}]))
    for n in range(5):
        assert torch.equal(one.batch(n), two.batch(n))
    b = two.global_batch(5)
    assert b.shape == (2, 3, 128)
    assert check_content(corpus, b, two, 5) == two.quota[1]
    assert one.mix == two.mix                                       # stages are checked by position instead
    for p in two.packs:
        assert two.cursor(p, 7) == 5 * two.quota[0][p] + 2 * two.quota[1][p]


def test_missing_pack_is_an_error_unless_renormalise(tmp_path):
    c = Corpus(tmp_path / "bn", split_rows=SPLIT + [("bn", "ind-bn-fw2", 1.0)])
    with pytest.raises(ValueError, match="ind-bn-fw2"):
        ml.MixLoader(c.config())
    L = ml.MixLoader(c.config(missing="renormalise"))
    assert any("ind-bn-fw2" in n for n in L.notes)
    assert abs(sum(v for p, v in L.quota[0].items() if p.startswith("ind-")) / 512 - 0.08) <= 1 / 512
    plain = Corpus(tmp_path / "plain")
    with pytest.raises(ValueError, match="lane poetry"):
        ml.MixLoader(plain.config(lanes=dict(LANES, poetry=0.1)))


def test_lanes_must_be_a_dict(corpus):
    with pytest.raises(ValueError, match="a dict of lane weights"):
        ml.MixLoader(corpus.config(lanes="supply"))


def test_shard_checks(tmp_path):
    c = Corpus(tmp_path, extra_split="en-web")
    shards = read_shards(tmp_path)
    assert all(s.split == "train" for s in shards)
    sha = manifest_sha256(shards)
    path = tmp_path / "en-web-train-000000.json"
    m = json.loads(path.read_text())
    path.write_text(json.dumps(dict(m, payload_bytes=m["payload_bytes"] + 4)))
    with pytest.raises(ValueError, match="payload_bytes"):
        read_shard(path)
    path.write_text(json.dumps(dict(m, tokenizer_sha256="0" * 64)))
    with pytest.raises(ValueError, match="tokenizer"):
        read_shard(path)
    path.write_text(json.dumps(dict(m, payload_sha256="f" * 64)))
    assert manifest_sha256(read_shards(tmp_path)) != sha
    with pytest.raises(ValueError, match="payload_sha256"):
        read_shard(path, verify="full")
    path.write_text(json.dumps(m))
    idx = np.load(tmp_path / "en-web-train-000000.idx.npy")
    bad = idx.copy()
    bad[1] -= 1                                                     # a doc boundary off by one: no EOS there
    np.save(tmp_path / "en-web-train-000000.idx.npy", bad)
    path.write_text(json.dumps(dict(m, index_sha256=hashlib.sha256(
        (tmp_path / "en-web-train-000000.idx.npy").read_bytes()).hexdigest())))
    with pytest.raises(ValueError, match="EOS"):
        read_shard(path)


def test_composition_log(corpus, tmp_path):
    log = tmp_path / "comp.jsonl"
    L = ml.MixLoader(corpus.config(), log_path=log)
    batches = [L.next() for _ in range(4)]
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert lines[0]["schema"] == ml.LOG_SCHEMA and lines[0]["manifest"] == L.manifest
    assert [x["batch"] for x in lines[1:]] == [0, 1, 2, 3]
    assert [x["batch_sha256"] for x in lines[1:]] == [ml.batch_sha256(b) for b in batches]
    assert lines[1]["by_pack"] == L.quota[0]
    with pytest.raises(ValueError, match="another manifest or mix"):
        ml.MixLoader(corpus.config(seed=99), log_path=log)


# ---- Team 7 contract -----------------------------------------------------------------------------------------------

def _contract():
    spec = importlib.util.spec_from_file_location("loader_contract", CONTRACT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("world", [1, 2, 4])
def test_loader_contract(corpus, world):
    cfg = corpus.config()
    problems = _contract().check(lambda r, w: ml.MixLoader(cfg, dp_rank=r, dp_world=w), world=world)
    assert problems == []


def test_contract_factory_reads_env(corpus, tmp_path, monkeypatch):
    import yaml
    path = tmp_path / "mix.yaml"
    path.write_text(yaml.safe_dump(corpus.config()))
    monkeypatch.setenv("G6_MIX_CONFIG", str(path))
    assert ml.contract_factory(1, 2).state_dict() == ml.MixLoader(corpus.config(), 0, 2).state_dict()


def test_stage_added_on_resume_only_after_the_saved_position(corpus):
    one = ml.MixLoader(corpus.config())
    two = ml.MixLoader(corpus.config(stages=[{"from_batch": 0, "seq_len": 64, "rows": 8},
                                             {"from_batch": 5, "seq_len": 128, "rows": 2}]))
    for pos in (0, 3, 5):                                           # nothing before the new stage differs
        two.load_state_dict(dict(one.state_dict(), position=pos))
        assert two.next_batch_id()["position"] == pos
    for pos in (6, 9):                                              # batch 5 was already read with the old shape
        with pytest.raises(ValueError, match="stages before position"):
            two.load_state_dict(dict(one.state_dict(), position=pos))
        with pytest.raises(ValueError, match="stages before position"):
            one.load_state_dict(dict(two.state_dict(), position=pos))
    two.load_state_dict(dict(one.state_dict(), position=4))
    assert torch.equal(two.next(), one.batch(4))
    assert two.next().shape == (2, 3, 128)


def test_unknown_config_keys_are_refused(corpus):
    for key, value in (("packing", "concat"), ("sources", {}), ("total_batch", 3)):
        with pytest.raises(ValueError, match="config: unknown keys"):
            ml.MixLoader(dict(corpus.config(), **{key: value}))


def test_best_fit_counts_its_cuts(corpus):
    """Segments never cross a row; cuts in the log add up."""
    L = ml.MixLoader(corpus.config(rows=4, seq_len=100))
    for n in range(20):
        lay, comp = L.layout(n), L.composition(n)
        assert (lay.start // 100 == (lay.start + lay.length - 1) // 100).all()
        assert comp["segments"] == comp["pieces"] + comp["cuts"]["row"] == len(lay.length)
        assert comp["cuts"]["row"] >= comp["cuts"]["row_forced"]
        assert comp["doc_starts"] == int((lay.off == 0).sum())


def test_best_fit_rows_fill_exactly():
    rng = np.random.default_rng(3)
    for trial in range(200):
        rows, L = int(rng.integers(1, 9)), int(rng.integers(1, 300))
        cuts = np.sort(rng.choice(np.arange(1, rows * L), size=min(rows * L - 1, int(rng.integers(0, 40))),
                                  replace=False))
        length = np.diff(np.concatenate([[0], cuts, [rows * L]])).astype(np.int64)
        j, w, x, s = ml.best_fit_rows(length, rows, L, seed=1, n=trial)
        assert (x > 0).all() and int(x.sum()) == rows * L
        assert (s // L == (s + x - 1) // L).all()                     # no segment crosses a row
        assert (np.diff(s) == x[:-1]).all() and s[0] == 0             # rows laid out back to back
        for i in range(len(length)):                                  # each piece covered exactly once, in order
            m = np.sort(w[j == i])
            assert int(x[j == i].sum()) == length[i] and m[0] == 0
