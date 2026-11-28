# ERA V5 Lab: model card

**Date:** [date: waiting for TSAI]

## Overview

The ERA V5 lab of The School of AI (TSAI) builds and trains its own language model. Students of the lab work in
teams, and each team owns one part of the work. The work is in a public repository, so anyone can follow it.

The model is a mixture-of-experts language model. A token is a piece of text that the model reads or predicts, such
as part of a word or a punctuation mark. The model has many small parts, called experts, and uses only a few of them
for each token. Each token goes to two of its 16 routed experts, and to one shared expert. The last part of the model
is the KE2 output head, which scores each token as the next token. KE2 is the lab team that tests its embedding
method on the lab's tokenizer. Architecture proposes the exact size of the model once the lab knows the real token
count of its first full corpus. TSAI decides it.

## Intended use

[text: waiting for Post-training and Release]

## Languages

[text: waiting for Data A, Data B and Tokenizer]

## Training data

### The general text and code

Data A (the team that cleans training text, removes duplicates and prepares training files) checked English, code,
maths and science samples.

These samples came from public collections:

- English web text: FineWeb-Edu, the DCLM baseline and Nemotron-CC. The Nemotron-CC samples include web text and text
  rewritten by a model.
- Code: StarCoderData, covering C++, Go, Java, JavaScript and TypeScript, Python and Rust.
- Maths: FineMath and the web part of MegaMath.
- Science: FinePDFs, text taken from PDF files.

Readers labelled documents with a shared list of data problems. Later audits compared raw and cleaned samples to find
missed problems and damage caused by cleaning.

The readers found these problems:

- Web text keeps menus, share and follow prompts, comment counters, tags and footers.
- Some documents end early, stop at a teaser, or join unrelated pages together.
- Some pages contain keyword lists, links, spam or filler written for search engines.
- Maths text loses formulas and figures, or flattens formulas into plain text. Some pages contain text written by a
  model.
- Science text has broken words, damaged letter pairs, merged table cells and incorrect reading order after extraction
  from PDFs.
- Rewritten web text sometimes keeps the rewriting model's own comments instead of only the source content.
- Code includes repeated licence headers, generated or minified files, and large blocks of embedded data.
- Some documents contain personal contact details or secrets, such as access keys and passwords.

The audits also found cleaning mistakes, including removal of useful text and technical numbers mistaken for phone
numbers. Data A added scoped fixes and recorded the remaining findings.

The current corpus-v1 job selects FineWeb-Edu, StarCoderData, the web part of MegaMath and FinePDFs from these
sources. The DCLM, Nemotron-CC and FineMath packs were audited but are excluded from this job's current input
selection.

### The Indian-language and code-adjacent text

Data B (the team for Indian-language and code-adjacent data) checked the Indian-language part of the training data.
It also checked the code-adjacent part: commit messages, notebooks and question-and-answer posts.

The Indian-language text comes from public collections: FineWeb-2, Sangraha, HPLT and IndicCorp. It covers Bengali,
Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Sanskrit, Tamil and Telugu. Sangraha has a verified part and a
synthetic part, which is machine-translated text.

The code-adjacent text comes from CommitPackFT, a public set of Kaggle notebooks, and Stack Exchange posts from the
StarCoder2 extra data.

Native readers read samples of each source. They labelled each document with a shared list of data problems. For each
language, the documents that both of their readers marked clean form a gold set. Data B uses the gold sets to test that
a cleaning rule does not harm good text.

The readers found these problems:

- The share of clean documents depends strongly on the source. The machine-translated part of Sangraha has the fewest
  clean documents of all the sources.
- Telugu has the least clean text of the languages that a second reader also checked.
- Native readers rated the machine-translated text. No language reached the quality bar that Data B proposed, so the
  native Indian-language share of the mix leaves translated text out.
- Romanised Indian-language text, an Indian language written in Latin letters, is rare in the samples. The mix caps it
  at 1% of the tokens of each language.
- Malayalam and Odia had no second reader, so Data B could not measure their clean share.

In the code-adjacent text, the readers found these problems:

- Commit messages are mostly clean. Their main problem is personal data: the e-mail addresses of the people who
  wrote the commits, in author and sign-off lines. Cleaning replaces these addresses with a placeholder. Long licence
  headers also repeat in many commits.
- Many notebooks have text around the code that is not part of the work, such as headers and links, and leftover
  template text.
- Every question-and-answer post has format markers left in its text. Some posts also contain HTML tags, carry a
  thread from another question, or repeat in other posts.
- Only the commit messages have a gold set, so no cleaning rule was tested on the notebooks and the posts.
  [text: waiting for TSAI, whether the notebooks and the posts are cleaned before training]

Indian scripts break easily. Data B looked for invisible characters, text in the wrong script, words cut before a vowel
sign, and lost joiner characters (ZWJ and ZWNJ). In the cleaned training shards, no joiner was lost from the text that
cleaning kept.

Text in Indian languages, in their own scripts, is 8% of every training batch.

## Data cleaning

Data A built and rehearsed a pipeline that cleans text, removes selected duplicates and writes tokenised training
files. It checks the text against Evaluation's benchmark fingerprints before tokenisation.

Data A optimised cleaning rules and added parallel processing across input files. Checks confirmed that parallel and
sequential cleaning produced identical outputs on the tested inputs.

Cleaning rules apply to named sources and kinds of text. They repair selected encoding and extraction problems, remove
recognised boilerplate, and drop documents that meet the rules for exclusion. Other rules mask detected personal
details and secrets.

Data A tested the rules individually and together on available gold sets: documents that readers had marked clean. The
tests record changes and drops, including failures and gaps in gold coverage. Passing a harm test does not mean that
every document is correct or that cleaning found every problem. Cleaning cannot recover formulas or passages already
missing from the input.

The duplicate check compares exact and near copies, with different rules for prose, maths and code. It protects the
held-out evaluation split by removing matching training copies. Cross-language duplicate detection remains report-only
and does not remove documents.

The benchmark check, called F2, uses Evaluation's fixed set of text fingerprints. It drops documents that match the
check and records the matches. This does not establish that every possible benchmark copy has been found.

The pipeline uses the frozen tokenizer p1536 and writes token IDs into shards, the files that training reads. Each
shard preserves whole documents and marks where each document ends. A manifest and quality report record its contents
and processing history. Checks compare decoded tokens with the normalised input and reconcile the shards with the
pipeline's document records.

The completed end-to-end rehearsal used sample data. Verified corpus-v1 production shards and their final
training-token count remain pending.

[text: waiting for Data B]

## Tokenizer

[text: waiting for Tokenizer]

## KE2

The model uses the KE2 output head, the last part of the model, which scores each token as the next token. TSAI
chose this head on 21 Nov 2026, in place of the dense untied head, the standard output head. Architecture added the
head to the training code on 24 Nov 2026.

## Architecture

[text: waiting for Architecture]

## Training

[text: waiting for Performance and Architecture]

## Training infrastructure

[text: waiting for Infrastructure]

## Compute

[text: waiting for Infrastructure, after TSAI approves the numbers]

## Monitoring

[text: waiting for Monitoring]

## Post-training

[text: waiting for Post-training]

## Alignment

[text: waiting for Post-training]

## Quantized versions

[text: waiting for Quantization]

## Serving

[text: waiting for Quantization]

## Evaluation

Evaluation tests the model on public benchmarks, and keeps benchmark items out of the training text. During
training, Evaluation scores each checkpoint, a saved copy of the model, on a small set of benchmark questions. These
scores do not come from the full benchmarks.

During training, Evaluation uses the benchmarks MMLU and MILU. MMLU has questions on school and professional subjects,
such as mathematics, history and law, in English. MILU has questions from Indian exams on many subjects, including
Indian history, arts and laws. Evaluation uses its questions in Hindi, Telugu, Tamil and Bengali. Each question has
answer choices, and the answer of the model is the choice that it rates most likely.

| Benchmark | What it tests | Score | Score of a random guess | Training step |
|---|---|---|---|---|
| MMLU | School and professional subjects, in English | [number: waiting for TSAI] | [number: waiting for TSAI] | [number: waiting for TSAI] |
| MILU (Hindi) | Subjects from Indian exams, in Hindi | [number: waiting for TSAI] | [number: waiting for TSAI] | [number: waiting for TSAI] |
| MILU (Telugu) | Subjects from Indian exams, in Telugu | [number: waiting for TSAI] | [number: waiting for TSAI] | [number: waiting for TSAI] |
| MILU (Tamil) | Subjects from Indian exams, in Tamil | [number: waiting for TSAI] | [number: waiting for TSAI] | [number: waiting for TSAI] |
| MILU (Bengali) | Subjects from Indian exams, in Bengali | [number: waiting for TSAI] | [number: waiting for TSAI] | [number: waiting for TSAI] |

Each score is in points out of 100. The check of the training text covers both benchmarks. Every file of training
text is checked against their questions before training reads it.

Each language has a small set of questions, so a small gap between the scores of two languages can come from chance.

## Limitations and risks

[text: waiting for Evaluation and Release]

## What this card does not claim

[text: waiting for Evaluation and Release]

## Licence and attribution

Post-training uses only datasets with permissive commercial licences (MIT, Apache 2.0, CC BY 4.0, CC BY-SA, ODC-By,
or per-file permissive licences). Datasets from closed models, Llama models, use-restricted licences such as
OpenRAIL-M, or unknown generators are excluded.

Post-training datasets admitted so far and their credits:

- `nvidia/OpenCodeReasoning-2` (CC BY 4.0): supervised fine-tuning code and reasoning traces.
- `deepmind/code_contests` (CC BY 4.0): reinforcement learning (RL) code training pool (train split, decontaminated).
- `nebius/SWE-rebench` (CC BY 4.0, arXiv 2505.20411) and Post-training's `swe-envs/` repositories (MIT or
  BSD-3-Clause): human issues, patches, and tests.

The full dataset ledger, generator licences, and attribution list stay in
`teams/10-post-training/licences/licence_table.md` (alongside the Data A, Data B, and MILU CC BY 4.0 credits).

[text: waiting for Data A and Data B]

Evaluation uses the MILU benchmark of AI4Bharat, which is under the CC BY 4.0 licence.

## Contributors

These teams of the ERA V5 lab do the work. TSAI approves each decision before the lab acts on it.

- **Data A** cleans the training text, removes duplicates and packs the text into files for training.
- **Data B** reads Indian-language and code-adjacent text with native readers, and proposes how much of each kind
  goes into training.
- **Tokenizer** builds the tokenizer, which splits text into the pieces that the model reads.
- **KE2** tests its embedding method on the lab's tokenizer.
- **Architecture** sets the shape of the model and its training settings.
- **Performance** makes each training step faster, and tests that each fast part matches a high-precision
  reference.
- **Infrastructure** runs the training jobs, checks the health of the machines, and saves the work so that a run
  can restart after a crash.
- **Monitoring** watches the training run, raises alarms and builds the public dashboard.
- **Evaluation** tests the model on public benchmarks and keeps benchmark items out of the training text.
- **Post-training** tunes the model after its main training, for example to chat in the lab's chat format.
- **Quantization** makes smaller versions of the model for serving, and checks them against the full model.
- **Release** writes the public text of the lab: the weekly updates, the dashboard text, the model card and the
  run report.

## How to find more

The public lab repository holds the work of every team: https://github.com/theschoolofai/era-v5-lab

## What is released

[text: waiting for Release, after TSAI decides what is released]

## Sources

[text: waiting for Release]

## How to cite

[text: waiting for Release, after TSAI names the model]

---

## Review record (remove this section before publishing)

This record is public, like the rest of the pull request. Put only public facts in it: yes or no, and links to
public pull requests. Never link to the CAP chat. Chat links are internal.

This is the first draft of the model card. The sections follow `model_card_outline.md`, which names the owner team of
each section.

How to fill it:

1. The owner team writes the text of its section, in its own folder or in a pull request comment.
2. Release pastes the text into its section and edits it only for style. Release sends the edit back to the writer.
3. The lead of each owner team gives an OK on the text in a pull request comment (`OK @handle`).
4. Write `[number: waiting for TSAI]` for any number that TSAI has not approved yet.
5. Run `python teams/11-release/style/check_style.py <this file> --numbers` and clear every error.
6. Update the status of the section in `model_card_outline.md`.
7. TSAI approves this pull request before the G11 lead merges it.

| Section | Owner team | Text in this draft | Source | Lead OK |
|---|---|---|---|---|
| Overview | Release | who made the model: yes. What the model is: yes, from the Gate 2 and Gate 3 updates and the launch announcement. The short forms KE2 and token: yes, from the Gate 3 update and the dashboard glossary. The size sentence describes work that is not done (style rule 6), so TSAI approves its wording in this pull request. Before publishing, it changes to the size that TSAI approves | `launch/launch_announcement_v0.md` ([#600](https://github.com/theschoolofai/era-v5-lab/pull/600), [#603](https://github.com/theschoolofai/era-v5-lab/pull/603), [#753](https://github.com/theschoolofai/era-v5-lab/pull/753)). What the model is: `updates/gate2.md` (first model configuration), `updates/gate3.md` (attention and experts, model layout and size, the output head, short forms), `dashboard/glossary.md` (token) | Release: yes (the Release lead wrote this draft); Architecture: yes, and Architecture sent the sentence on the size ([comment on #791](https://github.com/theschoolofai/era-v5-lab/pull/791#issuecomment-7135447264)) |
| Training data: the Indian-language and code-adjacent text | Data B | yes, with no edit, and the problems in the code-adjacent text, with no edit. One open point waits for TSAI: whether the notebooks and the posts are cleaned before training | `teams/02-data-b/model_card/training_data.md` ([#525](https://github.com/theschoolofai/era-v5-lab/pull/525); the code-adjacent problems: [#782](https://github.com/theschoolofai/era-v5-lab/pull/782), from the Data B lead). Its source table gives the TSAI approval of each number and the source of each problem | Data B: yes for the first part ([comment on #759](https://github.com/theschoolofai/era-v5-lab/pull/759#issuecomment-7108567909)); the code-adjacent problems: no, waiting for the OK |
| Training data: the general text and code | Data A | yes, from the Data A lead, with no edit. Open: "corpus-v1" needs a short explanation at its first use (style rule 10). The edit goes back to Data A | Reading findings: [`teams/01-data-a/spec_v1_1/results/reading/summary.md`](https://github.com/theschoolofai/era-v5-lab/blob/main/teams/01-data-a/spec_v1_1/results/reading/summary.md). Current input selection: [`staged_packs_33.json`](https://github.com/theschoolofai/era-v5-lab/blob/main/teams/01-data-a/jobs/2026-11-19-first-training/staged_packs_33.json). Pipeline and cleaner updates: [#767](https://github.com/theschoolofai/era-v5-lab/pull/767) | Data A: yes ([comment on #811](https://github.com/theschoolofai/era-v5-lab/pull/811#issuecomment-7169583801)) |
| Data cleaning | Data A, Data B | the Data A part: yes, from the Data A lead, with no edit. The Data B part: no | Cleaner tests and their limits: [`harm_tests.md`](https://github.com/theschoolofai/era-v5-lab/blob/main/teams/01-data-a/spec_v1_1/results/cleaner/harm_tests.md). Pipeline and cleaner updates: [#767](https://github.com/theschoolofai/era-v5-lab/pull/767). Sample rehearsal: [`README.md`](https://github.com/theschoolofai/era-v5-lab/blob/main/teams/01-data-a/jobs/logs/2026-11-19-first-training-rehearsal-38packs/README.md). Duplicate check: [`dedup_design_v1.md`](https://github.com/theschoolofai/era-v5-lab/blob/main/teams/01-data-a/dedup/dedup_design_v1.md). Benchmark check: [`f2/README.md`](https://github.com/theschoolofai/era-v5-lab/blob/main/teams/01-data-a/results/spec_v1/f2/README.md). Shard format: [`shard_format_v1.md`](https://github.com/theschoolofai/era-v5-lab/blob/main/teams/01-data-a/shards/shard_format_v1.md) | Data A: yes ([comment on #811](https://github.com/theschoolofai/era-v5-lab/pull/811#issuecomment-7169583801)); Data B: no |
| Licence and attribution | Post-training, Data A, Data B | the Post-training text: yes, from the Post-training lead, with two style edits (style rule 10): "RL" is spelled out, and "G10" is "Post-training's". The Data A and Data B text: no. The MILU credit: yes, from the Evaluation lead's note. It names AI4Bharat, the group that made MILU, and no person (style rule 7). TSAI decides whether the credit also cites the MILU paper and links its public page. Open: the Post-training text says that `licence_table.md` holds the MILU credit, but the table has no MILU row yet. Post-training adds the row, or changes the sentence. Open for TSAI: whether the card cites papers outside the lab repository, both the MILU paper and the SWE-rebench arXiv number | the Post-training text: [comment on #811](https://github.com/theschoolofai/era-v5-lab/pull/811#issuecomment-7160236414). Facts so far: `teams/10-post-training/licences/licence_table.md`. MILU is under CC BY 4.0, so this section credits its authors ([comment on #770](https://github.com/theschoolofai/era-v5-lab/pull/770#issuecomment-7123142597)) | Post-training: yes for the text ([comment on #811](https://github.com/theschoolofai/era-v5-lab/pull/811#issuecomment-7160236414)) and for the style edits ([comment on #811](https://github.com/theschoolofai/era-v5-lab/pull/811#issuecomment-7162658478)). Data A: no; Data B: no; Evaluation, for the MILU credit: yes ([comment on #811](https://github.com/theschoolofai/era-v5-lab/pull/811#issuecomment-7182428058)) |
| Evaluation | Evaluation | what Evaluation does: yes, from text that TSAI approved. The benchmarks, the table rows and the check of the training text: yes, from the Evaluation lead, with no edit. The scores: no, TSAI approves each score. Before a score from an early step goes in, check with Evaluation, because early scores can be above a random guess without any learning. Open: whether MMLU, under the MIT licence in `teams/09-evaluation/benchmarks_v1.md`, needs a credit line too. Evaluation decides | the Evaluation line of `launch/launch_announcement_v0.md` ([#753](https://github.com/theschoolofai/era-v5-lab/pull/753)); the Eval-Mini panel text in `dashboard/dashboard_text_new_panels.md` and the format in `dashboard/score_text_format.md` ([#760](https://github.com/theschoolofai/era-v5-lab/pull/760)). Benchmarks and table rows: [comment on #770](https://github.com/theschoolofai/era-v5-lab/pull/770#issuecomment-7123142597). Plan: `teams/09-evaluation/eval_schedule.md` | Evaluation: yes ([comment on #770](https://github.com/theschoolofai/era-v5-lab/pull/770#issuecomment-7121407240)); the Evaluation lead sent the benchmark text and confirmed the paste ([comment on #784](https://github.com/theschoolofai/era-v5-lab/pull/784#issuecomment-7129730898)). The sentence on the size of the question sets: from the Evaluation lead, with no edit ([comment on #811](https://github.com/theschoolofai/era-v5-lab/pull/811#issuecomment-7182428058)) |
| Contributors | Release | yes | `launch/launch_announcement_v0.md` ([#600](https://github.com/theschoolofai/era-v5-lab/pull/600), [#603](https://github.com/theschoolofai/era-v5-lab/pull/603)) | yes (the Release lead wrote this draft) |
| How to find more | Release | yes | the public lab repository | yes (the Release lead wrote this draft) |
| KE2 | KE2 | the output head: yes, from text that TSAI approved. The dates now have the year, and the text no longer says Gate 3 (style rules 10 and 11). The placeholder on how the lab tested its embedding method is gone: it was wider than the scope in `model_card_outline.md` | `updates/gate3.md` (the output head, short forms; [#701](https://github.com/theschoolofai/era-v5-lab/pull/701)), the KE2 line of `launch/launch_announcement_v0.md` ([#753](https://github.com/theschoolofai/era-v5-lab/pull/753)), the training code: [#717](https://github.com/theschoolofai/era-v5-lab/pull/717), merged 24 Nov | KE2: yes for the text before the date edit ([comment on #791](https://github.com/theschoolofai/era-v5-lab/pull/791#issuecomment-7142627011)); the date edit: no, waiting for the OK |
| Every other section | the owner team in `model_card_outline.md` | no | waiting for the owner team | no |

- [ ] `check_style.py` gives 0 errors
- [ ] Every number in the text has TSAI approval
- [ ] The text names teams only, and no person
- [ ] Nothing from the internal list in `public_internal.md`
- [ ] No `waiting for` placeholder left in the text
- [ ] The size sentence in the Overview gives the size that TSAI approved
- [ ] A team member commented on this pull request
- [ ] TSAI approved on: [DD Mon YYYY]
- [ ] Merged after TSAI approved
