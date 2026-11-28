"""Reproduce the Week 4 document-length table for sample-v0 and the frozen G5 evaluation set."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer

from week2.artifacts import P1536_SHA256, sha256
from week2.scoring import INDIC, source_slices


TEAM = Path(__file__).resolve().parents[1]
DATA_A = TEAM.parent / "01-data-a" / "audit"
# This public inventory pins the manifest for each of the 41 approved sample-v0 shards.
INVENTORY = DATA_A / "shard_decode_patilvijay23" / "summary.json"
WINDOWS = (1024, 2048, 4096, 8192, 16384)


def training_lengths(directory):
    inventory = json.loads(INVENTORY.read_bytes())["inventory"]
    shard_ids = {row["shard_id"] for row in inventory}
    if len(inventory) != len(shard_ids) or len(shard_ids) != 41:
        raise ValueError("Expected the full 41-shard inventory")

    groups = defaultdict(list)
    for row in inventory:
        shard = row["shard_id"]
        path = directory / f"{shard}.json"
        if sha256(path) != row["manifest_sha256"]:
            raise ValueError(f"Manifest hash mismatch for {shard}")
        manifest = json.loads(path.read_bytes())
        if (manifest["tokenizer_sha256"] != P1536_SHA256
                or manifest["eos_token_id"] != 130822 or manifest["split"] != "train"):
            raise ValueError(f"Unexpected tokenizer, EOS or split for {shard}")

        path = directory / f"{shard}.idx.npy"
        if sha256(path) != manifest["index_sha256"]:
            raise ValueError(f"Index hash mismatch for {shard}")
        offsets = np.load(path, allow_pickle=False)
        if (offsets.dtype != np.dtype("<i8") or offsets.shape != (manifest["doc_count"] + 1,)
                or offsets[0] != 0 or offsets[-1] != manifest["token_count"]
                or np.any(offsets[1:] <= offsets[:-1])):
            raise ValueError(f"Invalid document offsets in {shard}")
        lengths = np.diff(offsets) - 1  # Each stored document ends in one EOS.
        groups["all"].append(lengths)
        domain = manifest["domain"]
        groups[domain].append(lengths)
        if domain == "indic":
            groups[f"indic/{manifest['language']}"].append(lengths)
    return {name: np.concatenate(arrays) for name, arrays in groups.items()}


def evaluation_lengths(tokenizer_path):
    root = TEAM / "heldout_v1"
    docs = root / "data" / "heldout_v1.jsonl"
    receipt = json.loads((root / "heldout_report.json").read_bytes())
    if sha256(docs) != receipt["heldout_sha256"] or sha256(tokenizer_path) != P1536_SHA256:
        raise ValueError("Evaluation corpus or tokenizer hash differs from its frozen release")
    rows = [json.loads(line) for line in docs.read_bytes().splitlines() if line]
    slices = source_slices(root / "data" / "heldout_v1_sources.jsonl", rows)
    if len(rows) != receipt["docs_checked"] or set(slices) != {"en", "code", "math", "sci", *INDIC}:
        raise ValueError("Frozen heldout document count or slice inventory changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    lengths = defaultdict(list)
    for row, name in zip(rows, slices, strict=True):
        length = len(tokenizer.encode(row["text"], add_special_tokens=False).ids)
        lengths[name].append(length)
    return lengths


def write_counts(writer, name, lengths):
    lengths = np.asarray(lengths, dtype=np.int64)
    docs, total = lengths.size, lengths.sum()
    for window in WINDOWS:
        reaching = np.count_nonzero(lengths >= window)
        tail = np.maximum(lengths - window, 0).sum()
        complete, remainder = np.divmod(lengths, window)
        padded = (complete + (remainder > 0)).sum() * window
        # Each full window contributes its second half. A final partial window may contribute less.
        second_half = (complete * (window // 2) + np.maximum(remainder - window // 2, 0)).sum()
        writer.writerow((name, window, docs, total, reaching, tail, padded, second_half))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path, help="Frozen p1536 tokenizer.json")
    parser.add_argument("--sample-indexes", type=Path, help="Staged 41 sample-v0 .json and .idx.npy pairs in one directory")
    args = parser.parse_args()
    slices = evaluation_lengths(args.tokenizer)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("source", "window", "documents", "content_tokens", "docs_reaching_window",
                     "tokens_past_window", "padded_positions_one_pass", "targets_in_second_half_of_windows"))
    if args.sample_indexes:
        for name, lengths in sorted(training_lengths(args.sample_indexes).items()):
            write_counts(writer, f"sample-v0 {name}", lengths)
    write_counts(writer, "heldout all", [n for group in slices.values() for n in group])
    write_counts(writer, "heldout Indic", [n for name in INDIC for n in slices[name]])
    for name in sorted(slices):
        write_counts(writer, f"heldout {name}", slices[name])


if __name__ == "__main__":
    main()
