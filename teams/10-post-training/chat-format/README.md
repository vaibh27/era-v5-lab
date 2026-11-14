# Chat format v0

The ChatML-style chat format G10 uses, and the code that renders and parses it. The rule behind it (instructor, 27 Oct): special tokens only for structure the model must never confuse with normal text (role-turn start and end, tool call start and end, tool result start and end, thinking start and end). Everything inside that structure, such as tool names, arguments and the effort level, is plain text.

| File | What it is |
|---|---|
| `chat_format_v0.md` | The specification: the core rule, the context privilege planes, the turn structure, an example, and the applications to Tamil, Hindi, Go, Kannada and Python data. |
| `render.py` | Renders a conversation to text and parses it back. Token IDs come from Team 3's tokenizer (`teams/03-tokenizer/v0/`), not from the old 000..255 slot numbering. |
| `chat_template.jinja` | The same format as a Jinja chat template, handed to Team 12 (Quantization and Serving). |
| `team12_chat_template_handoff.md` | The hand-off note to Team 12 (Quantization and Serving) about the template. |
| `verify_frozen_tokenizer.py` | Checks Team 3's frozen tokenizer v0.1: its SHA-256, that each of the 9 active G10 markers is exactly one atomic token with its assigned ID, and that plain-text lookalikes such as `<think>` never resolve to them. On first use it downloads the baseline tokenizer from Hugging Face and builds v0.1 with Team 3's scripts into a cache folder (`G10_TOKENIZER_CACHE`, default `~/.cache/era-v5`). |

Tests are in `tests/` (`tests/test_render.py`, `tests/test_chat_template.py`, `tests/test_frozen_tokenizer.py` and the per-language format tests). The tokens themselves are listed in `special-tokens/`. `sft/convert.py` sends every candidate row through `render.py` unchanged.
