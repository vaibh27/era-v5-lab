# Tests

Tests for the chat format, the special tokens, the text-preparation demonstrations and the licence table. They run on CPU. They need pytest and the `tokenizers` package. `tests/test_frozen_tokenizer.py` uses Team 3's frozen tokenizer, which `chat-format/verify_frozen_tokenizer.py` downloads (baseline from Hugging Face) and builds on first use into `G10_TOKENIZER_CACHE` (default `~/.cache/era-v5`).

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
