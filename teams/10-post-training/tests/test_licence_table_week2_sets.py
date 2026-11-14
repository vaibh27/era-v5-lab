"""Consistency checks for the Week 2 tool-use / Indic / safety rows of licences/licence_table.md.

CPU only, no network. They pin what the data licence rule needs from each of the five sets added in the
"licence check of 5 tool-use and Indic sets" PR, plus two table-wide invariants (unique set ids, summary
decision agrees with the detail entry). They do not re-read the Hub cards: those facts were checked by hand
against the card revisions quoted in each entry.
"""

from __future__ import annotations

import re
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
TABLE = (TEAM / "licences" / "licence_table.md").read_text(encoding="utf-8")

DECISIONS = ("TSAI reserves", "excluded", "needs TSAI decision", "allowed")

# set id -> (decision label, Hub id, card revision quoted in the detail entry)
WEEK2_SETS = {
    "sft-agentic-toucan": ("TSAI reserves", "Agent-Ark/Toucan-1.5M", "0df3cf37f2"),
    "sft-tool-when2call": ("needs TSAI decision", "nvidia/When2Call", "0582f7749d"),
    "sft-indic-indic-instruct-hi": ("TSAI reserves", "ai4bharat/indic-instruct-data-v0.1", "82cd1437a0"),
    "sft-indic-updesh": ("excluded", "microsoft/Updesh_beta", "73ed1f8e08"),
    "rl-safety-aya-redteaming": ("TSAI reserves", "CohereLabs/aya_redteaming", "5a16fee03c"),
}


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split(" | ")]


def _summary_rows() -> dict[str, list[list[str]]]:
    rows: dict[str, list[list[str]]] = {}
    for line in TABLE.splitlines():
        if line.startswith("| ") and not line.startswith("| ---"):
            cells = _cells(line)
            if re.fullmatch(r"[a-z][a-z0-9.-]*", cells[0]):
                rows.setdefault(cells[0], []).append(cells)
    return rows


def _detail_sections() -> dict[str, str]:
    sections: dict[str, str] = {}
    for part in re.split(r"^### ", TABLE, flags=re.M)[1:]:
        match = re.search(r"^- \*\*Set:\*\* `([^`]+)`", part, flags=re.M)
        if match:
            sections[match.group(1)] = part
    return sections


def _decision_of(section: str) -> str:
    match = re.search(r"^- \*\*Decision:\*\* `([^`]+)`", section, flags=re.M)
    assert match, "detail entry has no **Decision:** line in backticks"
    return match.group(1)


def test_set_ids_are_unique_in_the_summary_tables():
    dup = [sid for sid, rows in _summary_rows().items() if len(rows) > 1]
    assert not dup, f"set ids listed more than once: {dup}"


def test_week2_sets_have_one_summary_row_with_a_valid_decision():
    for sid, (label, hub_id, _) in sorted(WEEK2_SETS.items()):
        rows = _summary_rows().get(sid)
        assert rows and len(rows) == 1, f"{sid}: expected exactly one summary row"
        cells = rows[0]
        assert len(cells) == 6, f"{sid}: summary row has {len(cells)} cells, expected 6"
        assert hub_id in cells[1], f"{sid}: source cell does not link {hub_id}"
        assert cells[3] and cells[4], f"{sid}: generator-model and licence cells must not be empty"
        assert cells[5] == label or cells[5].startswith(label), f"{sid}: decision {cells[5]!r}, expected {label!r}"
        assert cells[5] in DECISIONS or cells[5].startswith("allowed")


def test_week2_set_detail_entries_match_the_summary():
    for sid, (label, hub_id, revision) in sorted(WEEK2_SETS.items()):
        section = _detail_sections().get(sid)
        assert section, f"{sid}: no ### detail entry with '- **Set:** `{sid}`'"
        assert _decision_of(section) == label, sid
        assert f"https://huggingface.co/datasets/{hub_id}" in section, sid
        assert revision in section, f"{sid}: card revision {revision} not recorded"
        for field in ("Stage", "Language", "Source", "Licence", "Generator model", "Decision", "Added by"):
            assert re.search(rf"^- \*\*{field}:\*\*", section, flags=re.M), f"{sid}: missing **{field}:**"
        assert _summary_rows()[sid][0][5].startswith(_decision_of(section)), sid


def test_updesh_is_excluded_for_its_licence_not_only_for_llama():
    section = _detail_sections()["sft-indic-updesh"]
    assert _decision_of(section) == "excluded"
    assert "non-commercial" in section.lower()
    assert "Llama" in section  # second, independent reason is recorded


def test_when2call_does_not_pass_the_test_split_into_training():
    section = _detail_sections()["sft-tool-when2call"]
    assert "train_sft" in section and "train_pref" in section
    assert "BFCL" in section, "the test split is built from BFCL v2 Live; the entry must say so"


def test_indic_instruct_entry_keeps_only_the_human_written_subsets():
    section = _detail_sections()["sft-indic-indic-instruct-hi"]
    assert "wikihow" in section and "nmt-seed" in section
    for out in ("anudesh", "lm_sys", "hh-rlhf"):
        assert out in section and "excluded" in section, f"{out} must be recorded as left out"


def test_aya_redteaming_is_hindi_prompts_only():
    section = _detail_sections()["rl-safety-aya-redteaming"]
    assert "915" in section and "aya_hin" in section
    assert "Hindi" in section and "Team 9" in section  # eval-overlap hand-off recorded
