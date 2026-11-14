# Licence table

`licence_table.md` is the one shared G10 file for the SFT and RL datasets we considered. It has three stages: foundation data (Stage 0), supervised fine-tuning (Stage 1) and preference optimization and RLVR (Stage 2). Each set has one summary row (set, source, language, licence and decision, plus the generator model in Stages 1 and 2). Most sets also have a detail entry with the licence read on a stated date, the generator model, the decision and an opinion.

Decisions used in the table:

| Decision | Meaning |
|---|---|
| `TSAI reserves` | The final decision is TSAI's. It is the default for a set that is not excluded, and many entries note which TSAI rules the set passes. |
| `allowed` | TSAI ruled it in (for example CC-BY-SA with attribution, or the SWE environments). |
| `excluded` | Out of every training mix, with the reason in the entry (non-commercial licence, text made by closed OpenAI or Anthropic models, a Llama or custom licence, unstated licence or generator, or sources we cannot trace). |
| `needs TSAI decision` | A question only TSAI can answer, written in the entry. |

A Hugging Face tag is not legal clearance. `sft/convert.py` reads this table and refuses any set that is `excluded` or missing. `tests/test_licence_table_week2_sets.py` checks five Week 2 rows and that set ids are unique.

To add a set, add a summary row in the right stage and a detail entry in the same format, and open a pull request to the lead's branch.
