# Tokenizer v0: reserved special slots

v0 is BrahmicTokenizer-131K with its 250 generic reserved slots split into named blocks.
Nothing else changes: vocab 131,072, merges 301,398 (same rank order), 356 specials,
every token id the same, and plain text encodes to the same ids as the baseline.

The vocab is already full at 131,072, so v0 does not add new ids. It renames
`<|reserved_0|>` .. `<|reserved_127|>` in place. G4 can size the embedding now (131,072)
and it will not change when G10 sends the final list on 31 Oct.

## Slot map

| Ids | Count | Tokens | Owner / use |
|---|---|---|---|
| 0 .. 130,715 | 130,716 | normal BPE tokens | unchanged (id 36 is `<\|end_of_text\|>`) |
| 130,725 .. 130,821 | 97 | named specials (`<\|begin_of_text\|>`, `<\|im_start\|>`, `<think>`, FIM, tool, vision ...) | unchanged from baseline |
| 130,822 .. 130,885 | 64 | `<\|g10_slot_00\|>` .. `<\|g10_slot_63\|>` | G10 special-token list (31 Oct) |
| 130,886 .. 130,917 | 32 | `<\|lang_slot_00\|>` .. `<\|lang_slot_31\|>` | language / script tags (Brahmic scripts, English, code) |
| 130,918 .. 130,949 | 32 | `<\|ctrl_slot_00\|>` .. `<\|ctrl_slot_31\|>` | data, eval and training control |
| 130,950 .. 131,071 | 122 | `<\|reserved_128\|>` .. `<\|reserved_249\|>` | unassigned, kept for later |

Other named specials sit outside the main block at ids 36, 128512, 128601, 129279, 129411,
129994, 130173, 130174 and 130226 (baseline layout, unchanged).

Rules for filling a slot (v1): rename the string only, keep the id, keep `special: true`.
Never reuse an id that a model has been trained with.

## v0.1: G10 ChatML v0 markers (31 Oct 2026, #132)

G10's list (`teams/10-post-training/special-tokens/special_tokens_v0.md`, PR #14) has 9 markers.
G10's numbers (000, 001, ...) are positions in their list, not tokenizer ids.

| G10 marker | Tokenizer id | How |
|---|---|---|
| `<\|im_start\|>` | 129,994 | already a baseline special, unchanged |
| `<\|im_end\|>` | 130,816 | already a baseline special, unchanged |
| `<\|tool_call\|>` | 130,773 | already a baseline special, unchanged |
| `<\|tool_result\|>` | 130,774 | already a baseline special, unchanged |
| `<\|endoftext\|>` | 130,822 | renamed from `<\|g10_slot_00\|>` |
| `<\|think\|>` | 130,823 | renamed from `<\|g10_slot_01\|>` |
| `<\|/think\|>` | 130,824 | renamed from `<\|g10_slot_02\|>` |
| `<\|/tool_call\|>` | 130,825 | renamed from `<\|g10_slot_03\|>` |
| `<\|/tool_result\|>` | 130,826 | renamed from `<\|g10_slot_04\|>` |

Rename only: ids, `special: true`, vocab size and merges are unchanged. `<\|g10_slot_05\|>` ..
`<\|g10_slot_63\|>` stay reserved. `<\|endoftext\|>` (130,822) is G10's end-of-document marker; the
baseline's `<\|end_of_text\|>` (id 36) is a different token and stays as it is.

```bash
python /path/to/era-v5-lab/teams/03-tokenizer/v0/build_v0_1.py v0_tokenizer.json v0_1_tokenizer.json
sha256sum v0_1_tokenizer.json   # a33c9907ec16876bc57b011714e8380879919fba1327d84bfb0d60407fe6e578
```

| Check | v0.1 |
|---|---|
| 3 gating verify scripts, identity (self) | PASS |
| v0 vs v0.1 identity | BPE table identical; only the 5 renamed special strings differ |
| plain text encodes the same as v0 (held-out En, code, 11 Indic) | yes |
| each of the 9 markers encodes as one token at its id | yes |

## Build and check

```bash
git clone https://github.com/theschoolofai/BrahmicTokenizer-131K   # commit 7dfcf398172c825c6689ad578e8b484ca5f57bc9
cd BrahmicTokenizer-131K
python /path/to/era-v5-lab/teams/03-tokenizer/v0/build_v0.py tokenizer.json v0_tokenizer.json
sha256sum tokenizer.json v0_tokenizer.json   # compare with v0/SHA256SUMS
V=v0_tokenizer.json
python verify_no_cross_script_merges.py        $V && \
python verify_max_byte_length.py               $V && \
python verify_kronecker_constraints_unified.py $V && \
python verify_sarvam_m_structural_identity.py  $V $V
python verify_sarvam_m_structural_identity.py tokenizer.json $V   # base vs v0
```

The tokenizer.json is 7.7 MB and the repo limit is 5 MB, so it is not committed here.
Build it with the script; the sha256 in `SHA256SUMS` proves it is the same file.

## Results (27 Oct 2026)

| Check | Baseline | v0 |
|---|---|---|
| verify_no_cross_script_merges | PASS (301,398 merges, 0 cross-script) | PASS |
| verify_max_byte_length | PASS (130,716 normal, 356 special, 0 > 32 B) | PASS |
| verify_kronecker_constraints_unified | PASS | PASS |
| verify_sarvam_m_structural_identity (self) | PASS | PASS |
| base vs v0 identity | - | BPE table identical; only 128 special strings differ |
| encode equal on Hindi, Odia, Bengali, Tamil, Telugu, English, code | - | yes |
| each new slot encodes as one token at its id | - | yes |
