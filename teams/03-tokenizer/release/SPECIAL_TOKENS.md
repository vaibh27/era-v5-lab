# Special tokens of the frozen tokenizer (p1536)

Applies to `tokenizer.json` sha256 `e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b` (p1536, frozen by TSAI on 16 Nov). It lists all 356 special tokens: id, string, meaning, status and who uses it.

- **Machine-readable copy:** [`special_slots.json`](special_slots.json), made from `tokenizer.json` by [`special_slots.py`](special_slots.py). It is pinned in `SHA256SUMS`, so `verify_release.py` fails if it drifts.
- **Source of truth:** `added_tokens` with `special: true` in `tokenizer.json`. The 356 entries are byte-for-byte the same as in C3 (`15468e66…`) and v0.1 (`a33c9907…`), so **ids never moved**.
- **Never reuse an id a model was trained with.** A slot is filled by renaming its string, and a rename changes this sha, so it needs a new freeze by TSAI.
- **Users:** G1/G2 data, G5 trainer, G9 eval, G10 post-training, G12 export and serving. G4's KE2 table has a byte row for every id (no special is longer than 32 bytes), so G4 reads the whole list.

## Roles (read by `verify_release.py` and `special_tokens_map.json`)

| Role | Token | Id | Why |
|---|---|---|---|
| bos | `<\|begin_of_text\|>` | 130725 | baseline bos, named for loaders; never added (`add_bos_token` false) |
| eos | `<\|endoftext\|>` | 130822 | end of document: Data A's shard writer ends every doc with it (#373) |
| pad | `<\|pad\|>` | 130726 | padding, never in shards |
| unk | `<\|unk\|>` | 130727 | byte-level BPE never emits it; named for loaders that need one |

Generation stops at `<|im_end|>` 130816 or `<|endoftext|>` 130822 (`generation_config.json`). `add_bos_token` and `add_eos_token` are false: the shard writer adds the eos itself.

## Tokens in use

| Id | Token | Name | Meaning | Used by |
|---:|---|---|---|---|
| 130822 | `<\|endoftext\|>` | eos | End of document. Data A's shard writer ends every document with it; generation stops on it. | G1, G2, G5, G9, G10, G12 |
| 130726 | `<\|pad\|>` | pad | Padding. Never in shards; trainers mask it out of the loss. | G5, G9, G12 |
| 130725 | `<\|begin_of_text\|>` | bos | Begin of text, named for loaders. Not added: add_bos_token is false and shards have no bos. | G5, G12 |
| 130727 | `<\|unk\|>` | unk | Unknown, named for loaders. Byte-level BPE never emits it. | G12 |
| 36 | `<\|end_of_text\|>` | legacy_eos | The baseline's old end of text. Not eos: do not use. | - |
| 129994 | `<\|im_start\|>` | im_start | Start of a chat turn; the role (system, user, assistant, tool) follows as plain text. | G10, G12, G9 |
| 130816 | `<\|im_end\|>` | im_end | End of a chat turn. Generation also stops on it (generation_config.json). | G10, G12, G9 |
| 130823 | `<\|think\|>` | think | Start of a reasoning trace. | G10, G12, G9 |
| 130824 | `<\|/think\|>` | think_end | End of a reasoning trace. | G10, G12, G9 |
| 130773 | `<\|tool_call\|>` | tool_call | Start of a JSON tool call. | G10, G12, G9 |
| 130825 | `<\|/tool_call\|>` | tool_call_end | End of a JSON tool call. | G10, G12, G9 |
| 130774 | `<\|tool_result\|>` | tool_result | Start of a tool's output (untrusted). | G10, G12, G9 |
| 130826 | `<\|/tool_result\|>` | tool_result_end | End of a tool's output. | G10, G12, G9 |
| 130801 | `<\|fim_prefix\|>` | fim_prefix | Fill-in-the-middle: the code before the gap. | G10, G9 |
| 130803 | `<\|fim_suffix\|>` | fim_suffix | Fill-in-the-middle: the code after the gap. | G10, G9 |
| 130802 | `<\|fim_middle\|>` | fim_middle | Fill-in-the-middle: the gap to write. | G10, G9 |
| 130804 | `<\|fim_pad\|>` | fim_pad | Fill-in-the-middle padding for batched infilling. | G10, G9 |
| 130817 | `<\|file_sep\|>` | file_sep | Boundary between files of a repository. | G10 |
| 130818 | `<\|repo_name\|>` | repo_name | Repository name anchor. | G10 |

The chat format (G10 ChatML v0), as rendered by `teams/10-post-training/chat-format/render.py`. The `chat_template.jinja` in `tokenizer_config.json` renders it byte for byte the same:

```
<|im_start|>system
Reasoning effort: 4k<|im_end|>
<|im_start|>user
Weather in Pune?<|im_end|>
<|im_start|>assistant
<|think|>
Need the tool.
<|/think|>
<|tool_call|>
{"arguments":{"city":"Pune"},"name":"weather"}
<|/tool_call|>
<|im_end|>
<|im_start|>tool
<|tool_result|>
{"c": 24}
<|/tool_result|><|im_end|>
<|im_start|>assistant
24 °C.<|im_end|>
<|endoftext|>
```

- **Roles:** system, user, assistant and tool are **plain text** after `<|im_start|>` (G10's 27 Oct ruling: structure is special tokens, values are text). The base tokenizer's `<|system|>`, `<|user|>`, `<|assistant|>` and `<|tool|>` are not used.
- **Fill-in-the-middle:** `<|fim_prefix|>` prefix `<|fim_suffix|>` suffix `<|fim_middle|>` middle.

## Reserved for G10, string not renamed

| Id | Frozen string | G10 name | Note |
|---:|---|---|---|
| 130827 | `<\|g10_slot_05\|>` | subagent_call | G10's sub-agent call marker (special_tokens_v1.md). The frozen string is still <|g10_slot_05|>; a rename would change the frozen sha, so G10 uses the id with this string. |
| 130828 | `<\|g10_slot_06\|>` | subagent_return | G10's sub-agent return marker. The frozen string is still <|g10_slot_06|>. |

G10's `special_tokens_v1.md` lists these two as "rename pending in G3". They were not renamed before the freeze, so the frozen strings are `<|g10_slot_05|>` and `<|g10_slot_06|>`:
- G10 can use them now **by id**, or by the slot string.
- **Decided by G10 on 28 Nov (#873): keep the frozen sha.** G10 uses ids 130827 / 130828 with the slot strings, does not emit them in SFT v1 or RL v1, and records the slot strings in its `special_tokens_v1.md`. A rename to `<|subagent_call|>` / `<|subagent_return|>` would change the sha and needs a new freeze by TSAI.

## Free slots (for later)

| Ids | Count | Strings | Owner |
|---|---:|---|---|
| 130,829 .. 130,885 | 57 | `<\|g10_slot_07\|>` .. `<\|g10_slot_63\|>` | G10 |
| 130,886 .. 130,917 | 32 | `<\|lang_slot_00\|>` .. `<\|lang_slot_31\|>` | G3 (language and script tags) |
| 130,918 .. 130,949 | 32 | `<\|ctrl_slot_00\|>` .. `<\|ctrl_slot_31\|>` | G3 (data, eval and training control) |
| 130,950 .. 131,071 | 122 | `<\|reserved_128\|>` .. `<\|reserved_249\|>` | TSAI (unassigned) |

## Base tokens kept but unused (92)

These are named specials from the BrahmicTokenizer-131K base. No format emits them, so **do not emit them**: they are kept only because ids never move. Data cleaning strips the plain-text ghost tags (`<think>`, `[INST]`, …) from pretraining text (G10 v1 §4).

128512 `<think>`, 128601 `</think>`, 129279 `<\|response\|>`, 129411 `<\|context\|>`, 130173 `<\|newline\|>`, 130174 `<tool_call>`, 130226 `</tool_call>`, 130728 `<\|sep\|>`, 130729 `<\|cls\|>`, 130730 `<\|mask\|>`, 130731 `<\|paragraph\|>`, 130732 `<\|document\|>`, 130733 `<\|system\|>`, 130734 `<\|user\|>`, 130735 `<\|assistant\|>`, 130736 `<\|tool\|>`, 130737 `<\|function\|>`, 130738 `<\|instruction\|>`, 130739 `<\|turn\|>`, 130740 `<\|end_turn\|>`, 130741 `<\|code_begin\|>`, 130742 `<\|code_end\|>`, 130743 `<\|output_begin\|>`, 130744 `<\|output_end\|>`, 130745 `<\|error\|>`, 130746 `<\|stdin\|>`, 130747 `<\|stdout\|>`, 130748 `<\|stderr\|>`, 130749 `<\|file_begin\|>`, 130750 `<\|file_end\|>`, 130751 `<\|lang:python\|>`, 130752 `<\|lang:javascript\|>`, 130753 `<\|lang:typescript\|>`, 130754 `<\|lang:java\|>`, 130755 `<\|lang:cpp\|>`, 130756 `<\|lang:c\|>`, 130757 `<\|lang:rust\|>`, 130758 `<\|lang:go\|>`, 130759 `<\|lang:ruby\|>`, 130760 `<\|lang:php\|>`, 130761 `<\|lang:swift\|>`, 130762 `<\|lang:kotlin\|>`, 130763 `<\|lang:scala\|>`, 130764 `<\|lang:r\|>`, 130765 `<\|lang:julia\|>`, 130766 `<\|lang:sql\|>`, 130767 `<\|lang:html\|>`, 130768 `<\|lang:css\|>`, 130769 `<\|lang:bash\|>`, 130770 `<\|lang:shell\|>`, 130771 `<\|json_begin\|>`, 130772 `<\|json_end\|>`, 130775 `<\|function_call\|>`, 130776 `<\|function_result\|>`, 130777 `<\|api_request\|>`, 130778 `<\|api_response\|>`, 130779 `<\|schema\|>`, 130780 `<\|arguments\|>`, 130781 `<\|source:wikipedia\|>`, 130782 `<\|source:github\|>`, 130783 `<\|source:arxiv\|>`, 130784 `<\|source:web\|>`, 130785 `<\|source:book\|>`, 130786 `<\|source:code\|>`, 130787 `<\|source:docs\|>`, 130788 `<\|source:news\|>`, 130789 `<\|source:social\|>`, 130790 `<\|source:other\|>`, 130791 `<\|think_begin\|>`, 130792 `<\|think_end\|>`, 130793 `<\|step\|>`, 130794 `<\|plan\|>`, 130795 `<\|reflect\|>`, 130796 `<\|verify\|>`, 130797 `<\|conclude\|>`, 130798 `<\|reason\|>`, 130799 `<\|analyze\|>`, 130800 `<\|synthesize\|>`, 130805 `<\|vision_start\|>`, 130806 `<\|vision_end\|>`, 130807 `<\|vision_pad\|>`, 130808 `<\|image_pad\|>`, 130809 `<\|video_pad\|>`, 130810 `<\|object_ref_start\|>`, 130811 `<\|object_ref_end\|>`, 130812 `<\|box_start\|>`, 130813 `<\|box_end\|>`, 130814 `<\|quad_start\|>`, 130815 `<\|quad_end\|>`, 130819 `<tool_response>`, 130820 `</tool_response>`, 130821 `<\|EOT\|>`

## Test (`special_slots.py`, result in `../results/special_tokens_test.md`)

| Check | p1536 |
|---|---|
| Each of the 356 special tokens encodes to one id, its own (alone, and between words) | 356 / 356 |
| Plain text makes no special id: held-out dev text in 11 Indic languages, English and code | 0 in 4,270,468 tokens |
| The same on p1536's training text (`../counts/`, #719) | 0 in 144,239,593 tokens |
| Look-alikes make no special id (fullwidth bars, spaces, case, zero-width characters, cut-off markers) | 0 of 16 |
| A literal marker string in raw text becomes its special id | yes, as expected |
| G10's `special_tokens_v1.md` markers: special at their id, with G10's string | 15 / 17 (subagent_call 130827 and subagent_return 130828 keep their slot strings) |

The last row is how `tokenizers` treats added tokens. The two guards are elsewhere:
- Data A's `shard_writer.py` leaves out any document with a special id inside, so the only special id in a shard is the eos at the end of each document.
- G10's `render.py` rejects literal markers and look-alikes in message content.

```
python teams/03-tokenizer/release/special_slots.py release/tokenizer.json --text heldout/dev \
    --out-json teams/03-tokenizer/release/special_slots.json --report teams/03-tokenizer/results/special_tokens_test
```

## History

- **v0:** the base's 250 generic reserved slots split into named blocks (`../v0/special_slots.md`).
- **v0.1 (#132, #173):** G10's five markers renamed into `g10_slot_00..04`.
- **#407:** the roles (bos, eos, pad, unk).
- **C3 and then p1536:** these keep the v0.1 specials unchanged (`train/p1536_s0_manifest.json`: `special_layout` v0, `v0_tokenizer_sha256` `a33c9907…`).
