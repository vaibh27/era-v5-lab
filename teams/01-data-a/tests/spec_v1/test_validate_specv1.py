"""Unit tests for spec_v1/validate/validate_specv1.py: the paths it writes into results/spec_v1/cleaner/specv1_validation.json and
specv1.md never name a local folder (TSAI review of spec v1, 2 Nov). The validation run itself needs all 13 packs, so it is
not run here.

  python -m pytest tests/spec_v1
"""
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2] / "spec_v1"
TEAM = SPEC.parent
sys.path.insert(0, str(SPEC / "validate"))
import validate_specv1 as v


def test_the_gold_path_starts_at_the_repository_root(monkeypatch):
    assert v.ROOT / "teams" / "01-data-a" == TEAM
    monkeypatch.chdir(TEAM)  # run as the README runs it, with --gold gold
    assert v.report_paths(Path("gold") / "code-python_gold.csv", Path("/any/packs/folder/code-python.parquet")) == {
        "gold_path": "teams/01-data-a/gold/code-python_gold.csv", "pack_path": "code-python.parquet"}


def test_folders_outside_the_repository_are_not_written(tmp_path):
    paths = v.report_paths(tmp_path / "gold" / "code-python_gold.csv", tmp_path / "packs" / "code-python.parquet")
    assert paths == {"gold_path": "code-python_gold.csv", "pack_path": "code-python.parquet"}


def test_the_script_holds_no_local_folder():
    src = (SPEC / "validate" / "validate_specv1.py").read_text(encoding="utf-8")
    assert "/home/" not in src and "/Users/" not in src


def test_harm_table_refresh_preserves_indic_policy_and_scope_evidence():
    integration = '<!-- INDIC INTEGRATION -->\nown-pack gold policy\n<!-- END INDIC INTEGRATION -->\n\n'
    siblings = '<!-- VALIDATED SIBLING SCOPES -->\nshared F1 scope\n<!-- END VALIDATED SIBLING SCOPES -->\n\n'
    card = '\n## Medicine 1 — code / A1 / `DA-A1-001`\ncard body'
    original = '# Spec\n<!-- VALIDATION -->\nold harm table\n' + integration + siblings + card
    refreshed = v.replace_validation_summary(original, ['fresh harm table'])
    assert 'old harm table' not in refreshed
    assert integration in refreshed and siblings in refreshed
    assert refreshed.endswith(card)
    assert v.replace_validation_summary(refreshed, ['fresh harm table']) == refreshed
