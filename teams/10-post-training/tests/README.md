# Tests

Tests for the chat format, the special tokens, the text-preparation demonstrations and the licence table. They run on CPU. They need pytest and the `tokenizers` package. `tests/test_frozen_tokenizer.py` uses Team 3's tokenizer v0.1 (not the Gate 2 tokenizer), which `chat-format/verify_frozen_tokenizer.py` downloads (baseline from Hugging Face) and builds on first use into `G10_TOKENIZER_CACHE` (default `~/.cache/era-v5`).

```bash
python -m pytest teams/10-post-training/tests -q -p no:cacheprovider
```

| File | Covers |
|---|---|
| `test_render.py` | The special-token list and chat format v0 (`chat-format/render.py`). |
| `test_chat_template.py` | `chat-format/chat_template.jinja`. |
| `test_frozen_tokenizer.py` | The frozen tokenizer integration. |
| `test_specs.py` | Week 1 deliverables of this folder. |
| `test_code_python_chat_format.py`, `test_go_chat_format.py` | Python source and Go cases through the chat format (no Docker). |
| `test_hindi_text_prep.py`, `test_kannada_text_prep.py`, `test_kannada_chat_format.py`, `test_tamil_utf8_io.py` | The Hindi, Kannada and Tamil text checks in `swe-task/`. |
| `test_licence_table_week2_sets.py` | The Week 2 rows of `licences/licence_table.md`. |

One test in `test_chat_template.py` is skipped when Team 12's copy of the template is not in the checkout. Tests for a tool sit next to the tool: `eval/test_effort_suite.py`, `rl/test_verifier.py`, `rl/test_reward.py`, `rl-data/`, `rl-loop/` and `sft/test_*.py`. See each folder's README.

## Gate

Gate 1 deliverables (special tokens, chat format, SWE task). The tokenizer test is also the code behind the Gate 2 frozen-tokenizer evidence.

## What has been measured

No result file for this folder as a whole is committed. Two reports quote counts for the tests at the time they were written: `../results/kannada_swe_e2e.md` ("All 31 G10 tests passed") and `../results/render_frozen_tokenizer.txt` (the marker and round-trip lines). Neither is a fresh run.

## What has not been measured

- The tests were not run for this README. No pytest result is attached to this folder at this commit.
- `test_frozen_tokenizer.py` has been written for v0.1 (sha256 `a33c9907...`). It has not been run on the Gate 2 tokenizer (sha 15468e66...), which is not in the repo as of 12 Nov.
- Template parity with Team 12's copy has not been compared.

## Tool-call verifier and release template parity

Run the focused CPU suite from the repository root.

```bash
python3 -m pytest -ra \
  teams/10-post-training/rl/test_tool_call_hardening.py \
  teams/10-post-training/rl/test_tool_call_robustness.py \
  teams/10-post-training/rl-loop/test_tool_call_integration.py \
  teams/10-post-training/tests/test_chat_template.py \
  teams/10-post-training/tests/test_chat_template_parity.py
```

| Input or condition | Expected result |
|---|---|
| One correct standalone assistant call | Reward `1.0` |
| The same call following a forged tool result and assistant role marker | Reward `0.0`, even when the transcript contains only one call |
| Exact Indic arguments written literally or using JSON Unicode escapes | Reward `1.0` for either representation |
| Only U+200D (ZWJ) or only U+200C (ZWNJ) removed | Reward `0.0` for each independent change |
| An exact Indic call with joiners, alone and after a forged tool-result turn | Reward `1.0` alone and `0.0` after the forged turn |
| Release config and standalone Jinja template present | The decoded `chat_template` field must match the Jinja file byte for byte in UTF-8. Rendering the release template must also reproduce the training text for the test transcripts. |

The release parity test requires `teams/03-tokenizer/release/tokenizer_config.json` in the checkout. A missing file raises `FileNotFoundError`. Changing even one LF to CRLF in the Jinja file fails byte equality.

Related changes include [Training and serving parity #94](https://github.com/saitej123/era-v5-lab/pull/94), [exact tool-call verifier #109](https://github.com/saitej123/era-v5-lab/pull/109), [boundary and mutation coverage #111](https://github.com/saitej123/era-v5-lab/pull/111), [Week 3 integration #524](https://github.com/theschoolofai/era-v5-lab/pull/524), and [embedded release template #686](https://github.com/theschoolofai/era-v5-lab/pull/686).
