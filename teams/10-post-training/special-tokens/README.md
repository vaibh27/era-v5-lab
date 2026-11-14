# Special tokens

The special-token lists G10 prepared for the Tokenizer team (Team 3 / G3), with v1 also prepared for Team 4 and Team 9. The lead owns the final list, and TSAI reviews it at the Gate.

| File | What it is |
|---|---|
| `special_tokens_v0.md` | The first list: the 27 Oct ruling, the 256 reserved IDs (`0..255`), what is assigned and what stays reserved, and the token-smuggling defences. |
| `special_tokens_v1.md` | The consolidated allocation: the marker IDs, a machine-readable table for Team 3's tokenizer training and a pre-training sanity section. It replaces the 000 to 255 numbering of v0 with the real Team 3 token IDs, and records that reasoning effort is a plain-text system line (`Reasoning effort: {1k|4k|16k|64k}`) and not a token. |

Use `special_tokens_v1.md` for current IDs. Keep v0 for the reasoning behind each marker. The format that uses these tokens is in `chat-format/`.
