"""Shard-v1 reader for the mixture loader: manifests, document indexes, checks and the shard-list sha256.

Format: teams/01-data-a/shards/shard_format_v1.md. One shard is `<shard_id>.json` (manifest), `<shard_id>.bin`
(1,024-byte header, then uint32 tokens; the same layout dataio.loader.open_shard reads) and `<shard_id>.idx.npy`
(int64, doc_count + 1 start offsets, the last one = token_count). Every document ends with one EOS.

The shard list is one `<payload_sha256>  <shard_id>` line per shard, sorted by shard_id; its sha256 is the loader
state's `manifest` (shard format "Dataset level"; ADR 0001 section 7), so a resume on another shard set is refused.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dataio.loader import HEADER_BYTES, open_shard

SCHEMA = "shard-v1"
P1536_SHA256 = "e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b"
EOS_ID = 130822
SHARD_NAME = re.compile(r"^(?P<pack>.+)-(?P<split>train|validation)-(?P<num>\d{6})\.json$")
VERIFY = ("size", "full")            # size: header, sizes and index checks; full: also the sha256 of every .bin


@dataclass(frozen=True)
class Shard:
    shard_id: str
    pack_id: str
    lane: str
    language: str
    split: str
    token_count: int
    doc_count: int
    payload_sha256: str
    bin_path: Path
    idx_path: Path
    manifest: dict = field(repr=False, compare=False, hash=False)
    source_id: str | None = None       # manifest `source_id` (one Hub ID or URL); not in shard-v1 yet


@dataclass
class DocTable:
    """The documents of one pack, over its shards in shard_id order. `shard` indexes the pack's `shards` list;
    `start` is the document's first token inside that shard; `length` includes the EOS."""
    pack_id: str
    lane: str
    language: str
    shards: list
    shard: np.ndarray
    start: np.ndarray
    length: np.ndarray
    source_id: str | None = None

    @property
    def tokens(self) -> int:
        return int(self.length.sum())

    @property
    def docs(self) -> int:
        return len(self.length)


def _sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _local(manifest_path: Path, uri: str, fallback: str) -> Path:
    """A manifest's payload/index URI as a file next to the manifest (shards are written flat for now)."""
    p = manifest_path.parent / Path(uri).name if uri else None
    return p if p is not None and p.exists() else manifest_path.parent / fallback


def read_shard(path: Path, tokenizer_sha256: str = P1536_SHA256, eos_id: int = EOS_ID, verify: str = "size") -> Shard:
    """One shard-v1 manifest and its files, checked. Raises ValueError naming the shard and the check that failed."""
    if verify not in VERIFY:
        raise ValueError(f"verify={verify!r}: one of {VERIFY}")
    m = json.loads(Path(path).read_text())
    sid = m.get("shard_id", Path(path).stem)

    def bad(why):
        return ValueError(f"shard {sid} ({path}): {why}")

    if m.get("schema_version") != SCHEMA:
        raise bad(f"schema_version {m.get('schema_version')!r}, expected {SCHEMA!r}")
    if m.get("tokenizer_sha256") != tokenizer_sha256:
        raise bad(f"tokenizer_sha256 {m.get('tokenizer_sha256')!r}, expected {tokenizer_sha256}")
    if m.get("eos_token_id") != eos_id:
        raise bad(f"eos_token_id {m.get('eos_token_id')!r}, expected {eos_id}")
    if m.get("token_dtype") != "uint32":
        raise bad(f"token_dtype {m.get('token_dtype')!r}, expected uint32")
    n_tok, n_doc = int(m["token_count"]), int(m["doc_count"])
    if n_tok <= 0 or n_doc <= 0:
        raise bad("empty shard")
    if int(m["payload_bytes"]) != HEADER_BYTES + 4 * n_tok:
        raise bad(f"payload_bytes {m['payload_bytes']} != {HEADER_BYTES} + 4 x token_count {n_tok}")
    bin_path = _local(Path(path), m.get("payload_uri", ""), f"{sid}.bin")
    idx_path = _local(Path(path), m.get("index_uri", ""), f"{sid}.idx.npy")
    for p in (bin_path, idx_path):
        if not p.exists():
            raise bad(f"missing {p.name}")
    if bin_path.stat().st_size != int(m["payload_bytes"]):
        raise bad(f"{bin_path.name} is {bin_path.stat().st_size} bytes, manifest says {m['payload_bytes']}")
    if _sha256_file(idx_path) != m.get("index_sha256"):
        raise bad(f"{idx_path.name} sha256 differs from index_sha256")
    if verify == "full" and _sha256_file(bin_path) != m.get("payload_sha256"):
        raise bad(f"{bin_path.name} sha256 differs from payload_sha256")
    tokens = open_shard(bin_path)                       # header magic/version/count and file size
    if len(tokens) != n_tok:
        raise bad(f"header token count {len(tokens)} != token_count {n_tok}")
    idx = np.load(idx_path, allow_pickle=False)
    if idx.dtype != np.int64 or idx.ndim != 1 or len(idx) != n_doc + 1:
        raise bad(f"index must be int64 [doc_count + 1], got {idx.dtype} {idx.shape}")
    if idx[0] != 0 or idx[-1] != n_tok or (np.diff(idx) <= 0).any():
        raise bad("index must start at 0, end at token_count and increase strictly")
    if (np.asarray(tokens[idx[1:] - 1]) != eos_id).any():
        raise bad("a document does not end with EOS")
    for key in ("pack_id", "capability_lane", "language", "split", "payload_sha256"):
        if not m.get(key):
            raise bad(f"no {key}")
    return Shard(shard_id=sid, pack_id=m["pack_id"], lane=m["capability_lane"], language=m["language"],
                 split=m["split"], token_count=n_tok, doc_count=n_doc, payload_sha256=m["payload_sha256"],
                 bin_path=bin_path, idx_path=idx_path, manifest=m, source_id=m.get("source_id") or None)


def find_manifests(roots) -> list:
    """Every `<pack>-<split>-<6 digits>.json` under the roots (quality reports and summaries are not matched)."""
    out = []
    for root in [roots] if isinstance(roots, (str, Path)) else roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"shard root {root} is not a directory")
        out += [p for p in root.rglob("*.json") if SHARD_NAME.match(p.name)]
    return sorted(out)


def read_shards(roots, split: str = "train", tokenizer_sha256: str = P1536_SHA256, eos_id: int = EOS_ID,
                verify: str = "size") -> list:
    """The checked shards of one split under the roots, sorted by shard_id. Other splits are left out."""
    shards = [read_shard(p, tokenizer_sha256, eos_id, verify) for p in find_manifests(roots)]
    shards = sorted((s for s in shards if s.split == split), key=lambda s: s.shard_id)
    if not shards:
        raise FileNotFoundError(f"no {split} shard-v1 manifests under {roots}")
    ids = [s.shard_id for s in shards]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"shard_id found more than once: {dup}")
    return shards


def shard_list(shards) -> str:
    """`<payload_sha256>  <shard_id>` per shard, sorted by shard_id, newline-terminated."""
    return "".join(f"{s.payload_sha256}  {s.shard_id}\n" for s in sorted(shards, key=lambda s: s.shard_id))


def manifest_sha256(shards) -> str:
    """The loader state's `manifest`: sha256 of the shard list."""
    return hashlib.sha256(shard_list(shards).encode()).hexdigest()


def doc_tables(shards) -> dict:
    """pack_id -> DocTable. A pack's shards must agree on lane, language and source_id."""
    by_pack: dict = {}
    for s in sorted(shards, key=lambda s: s.shard_id):
        by_pack.setdefault(s.pack_id, []).append(s)
    tables = {}
    for pack, group in sorted(by_pack.items()):
        lanes, langs = {s.lane for s in group}, {s.language for s in group}
        if len(lanes) > 1 or len(langs) > 1:
            raise ValueError(f"pack {pack}: shards disagree on lane {sorted(lanes)} or language {sorted(langs)}")
        srcs = {s.source_id for s in group}
        if len(srcs) > 1:
            raise ValueError(f"pack {pack}: shards disagree on source_id {sorted(map(str, srcs))}")
        shard_ix, starts, lengths = [], [], []
        for i, s in enumerate(group):
            idx = np.load(s.idx_path, allow_pickle=False)
            starts.append(idx[:-1])
            lengths.append(np.diff(idx))
            shard_ix.append(np.full(len(idx) - 1, i, dtype=np.int32))
        tables[pack] = DocTable(pack_id=pack, lane=group[0].lane, language=group[0].language, shards=group,
                                shard=np.concatenate(shard_ix), start=np.concatenate(starts).astype(np.int64),
                                length=np.concatenate(lengths).astype(np.int64), source_id=group[0].source_id)
    return tables
