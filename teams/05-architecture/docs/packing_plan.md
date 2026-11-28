# Packing the training rows for the main run

**Team 5, Architecture | follow-up to #825 (context length) | 26 November 2026**

## Summary

Today the trainer puts one window of one document in each row and pads the rest (`harness/week2/data.py`, `batch()`). On the 41 sample-v0 shards that fills 35.22% of positions at 2,048 and 19.48% at 4,096 (#825); dry run 1's log shows 36.0%. Most of every step's compute goes to padding.

Measured here on two real sample-v0 shards, packing the rows would need **0.43x today's positions at 2,048 and 0.25x at 4,096** for the same documents. **Recommendation for the main run: pack documents into full rows with a document mask**, so documents still do not attend to each other, and reset positions at each document. Keep length-bucketing as the fallback that needs no model change. Dry run 2 does not change.

## What was measured

`harness/packing_fill.py` reads only shard `.idx.npy` document offsets (no token ids, no text) and counts, for one pass, the positions each option needs. Every count below is the script's output on these two index files. Packed: each document plus its EOS, end to end (113,426,354 + 106,665 = 113,533,019 tokens), cut into rows of L, the last row padded: 55,437 rows of 2,048 = 113,534,976 positions, and 27,719 rows of 4,096 = 113,537,024. Two full sample-v0 shards (TSAI's bundle of 16 Nov, the files used in the Week 3 Data A rotation check), 106,665 documents, 113,426,354 content tokens, median length 577:

| Window | Option | Positions per pass | Fill | vs today |
|---:|---|---:|---:|---:|
| 2,048 | today (one document per row, padded to L) | 263,260,160 | 43.1% | 1.00x |
| 2,048 | length-bucketed (pad to the longest window in the batch) | 114,568,268 | 99.0% | 0.44x |
| 2,048 | packed (documents end to end, EOS between) | 113,534,976 | 99.9% | 0.43x |
| 4,096 | today | 462,909,440 | 24.5% | 1.00x |
| 4,096 | length-bucketed | 115,149,630 | 98.5% | 0.25x |
| 4,096 | packed | 113,537,024 | 99.9% | 0.25x |

| Shard | Documents | Content tokens | Today at 2,048 | Packed at 2,048 |
|---|---:|---:|---:|---:|
| `code-python-train-000000` | 53,221 | 60,484,111 | 44.2% | 99.9% |
| `en-fineweb-edu-train-000000` | 53,444 | 52,942,243 | 41.8% | 99.9% |

These two shards fill a little better today (43.1%) than all 41 (35.22%, #825). Packing fills about 100% either way, so the saving across all 41 should be larger, not smaller. For the full set, run the same script on the 41 `.idx.npy` files:

```sh
python teams/05-architecture/harness/packing_fill.py --index /path/to/sample-v0-indexes/*.idx.npy --windows 2048 4096
```

From #825's totals (1,008,187,260 content tokens and 1,200,725 documents), packing needs about 1.01B positions per pass at either window, against 2.86B (2,048) and 5.18B (4,096) today.

## The three options

| | Length-bucketed | Packed, document mask (recommended) | Packed, plain |
|---|---|---|---|
| Row content | One window of one document, as today | Several documents per row | Several documents per row |
| Fill (measured) | ~99% | ~100% | ~100% |
| Attention | Unchanged | Block-diagonal causal mask: no attention across documents | Documents attend to the ones before them in the row |
| Positions (RoPE) | Unchanged | Reset to 0 at each document | Continue across documents unless reset |
| Model change | None | Mask and position ids in attention | Position ids only, if reset |
| Tokens per step | **Varies**: batches of short windows are short | Fixed | Fixed |
| Matches today's "no cross-document attention" rule | Yes | Yes | **No** |

**Length-bucketing** keeps the model as it is. Its catch is the step: a batch of short windows has fewer positions, so tokens per optimizer step vary. The WSD schedule (#713) is indexed by step and the loss is normalized per update, so the schedule would need to move to a token index, or batches sized by tokens instead of rows. It also changes tensor shapes every step.

**Packing with a document mask** keeps the rule the loader was written for (no cross-document attention) and keeps a fixed shape and fixed tokens per step, so the WSD schedule is unchanged. What it needs:

- a document id per position, so attention can be masked to the same document;
- position ids that restart at each document, so RoPE sees the same positions it sees today;
- in the eager path, the mask is one more term in the existing `masked_fill` (`harness/week2/model.py`). The dense score matrix is still computed per row, so attention work per row is unchanged, but there are 0.43x as many rows at 2,048;
- in the SDPA path (#804, opt-in), a boolean mask stops the Flash kernel from being used. Keeping the speed needs a variable-length kernel or FlexAttention with a document mask. This is the one real cost, and it bears on the SDPA A/B (#805).

**Plain packing** is the simplest, but documents see the unrelated text before them. The current code avoids that on purpose, and the held-out comparison would change.

## What it changes for 2,048 vs 4,096 (#825)

#825's cost comparison holds for today's loader: 4,096 needs 1.81x the padded positions and 3.62x the dense attention pairs of 2,048. With packing, both windows need about the same positions (0.43x and 0.25x of today's at each window), so 4,096's extra cost is the attention per row, not padding. The 2,048 starting window still stands; the packing decision should be made before the 2,048 vs 4,096 timing in #825's step 2 is priced.

## Proposed next steps

1. **Full numbers.** Run `packing_fill.py` on the 41 shard indexes (whoever has them staged; it reads offsets only).
2. **Code, behind a switch.** In `week2/data.py`, a packed loader (`data.pack: true`) that writes document ids and position ids with each row; in `week2/model.py`, the document mask in the eager path and position ids in RoPE. Off by default; dry run 2 and its arm unchanged.
3. **Check.** With `data.pack: false` the trainer stays bit-identical; with packing on, a single-document row gives the same loss as today, and a short T2 A/B (packed vs today, same seed and token budget) compares held-out bpb per scored token.
4. **Then decide** with the main-run settings: context length (#825), model size (#821) and SDPA (#805).

## Files

- `harness/packing_fill.py`: the measurement (offsets only).
- `tests/test_packing_fill.py`: its tests.
