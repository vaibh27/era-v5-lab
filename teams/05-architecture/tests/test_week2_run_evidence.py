import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from validate_week2_run import main, validate
from test_week2_model import fixture
import train_week2


class RunEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.run_dir = Path(temp.name)
        self.write_run()

    def write_json(self, name, value):
        (self.run_dir / name).write_text(json.dumps(value), encoding="utf8")

    def write_run(self, *, smoke=False, completed=3, status="COMPLETED", timed_out=False,
                  extra_identity=None):
        identity = {
            "config": {"run": {"steps": 3}},
            "artifacts": {},
            "train_sha256": "a" * 64,
            "heldout_sha256": "b" * 64,
            "heldout_sources_sha256": None,
            "heldout_checksum_sha256": None,
            "sources": {},
            "device": "cpu",
            "torch": "test",
            "smoke": smoke,
        }
        identity.update(extra_identity or {})
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.write_json("manifest.json", {
            **identity,
            "identity_sha256": self.identity,
            "status": "SMOKE_NOT_A_QUALITY_RESULT" if smoke else "AGREED_BASELINE_RUN",
        })
        self.write_json("summary.json", {
            "status": status, "steps": completed, "timed_out": timed_out,
            "eligible_for_quality_comparison": status == "COMPLETED",
            "final_model_sha256": "c" * 64,
        })
        self.write_metrics(range(completed + 1))
        for name in ("losses.txt", "total_losses.txt"):
            (self.run_dir / name).write_text("\n".join(["0x1p+0"] * completed) + "\n", encoding="utf8")
        checkpoint = self.run_dir / "checkpoint.pt"
        if smoke:
            checkpoint.unlink(missing_ok=True)
        else:
            torch.save({"identity_sha256": self.identity, "step": completed}, checkpoint)

    def write_metrics(self, steps):
        (self.run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps({"step": step}) + "\n" for step in steps), encoding="utf8"
        )

    def test_complete_run_and_json_cli(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["validator", str(self.run_dir), "--json", "--trusted-checkpoint"]):
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(), 0)
        result = json.loads(output.getvalue())
        self.assertTrue(result["quality_ready"])
        self.assertEqual(result["row_counts"], {
            "metrics.jsonl": 4, "losses.txt": 3, "total_losses.txt": 3,
        })
        self.assertEqual(len(result["sha256"]), 6)
        self.assertTrue(result["checkpoint_checked"])
        self.assertEqual(result["verdict"], "PASS")
        self.assertIn("internally consistent evidence", result["evidence_scope"])
        self.assertIn("BPB result", result["evidence_scope"])

    def test_quality_requires_trust_before_loading_checkpoint(self):
        output = io.StringIO()
        with (
            patch.object(torch, "load") as load,
            patch.object(sys, "argv", ["validator", str(self.run_dir)]),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(main(), 1)
        load.assert_not_called()
        self.assertIn("FAIL: quality run", output.getvalue())
        self.assertIn("Use --trusted-checkpoint", output.getvalue())
        self.assertIn("winning configuration", output.getvalue())

    def test_smoke_and_incomplete_are_ineligible(self):
        for status, completed, smoke, classification in (
            ("SMOKE", 3, True, "smoke"),
            ("INCOMPLETE_NOT_A_RESULT", 0, False, "incomplete"),
        ):
            with self.subTest(status=status):
                self.write_run(smoke=smoke, status=status, completed=completed, timed_out=not smoke)
                result = validate(self.run_dir)
                self.assertEqual(result["classification"], classification)
                self.assertTrue(result["valid"], result["errors"])
                self.assertFalse(result["quality_ready"])

    def test_missing_and_duplicate_metrics_steps_and_truncated_losses(self):
        self.write_metrics([0, 1, 1, 3])
        (self.run_dir / "losses.txt").write_text("0x1p+0\n", encoding="utf8")
        errors = "\n".join(validate(self.run_dir)["errors"])
        self.assertIn("duplicate steps [1]", errors)
        self.assertIn("1 missing steps, first [2]", errors)
        self.assertIn("losses.txt: expected 3 rows, found 1", errors)

    def test_wrong_budget_and_missing_required_checkpoint(self):
        (self.run_dir / "checkpoint.pt").unlink()
        errors = "\n".join(validate(self.run_dir, expected_steps=4)["errors"])
        self.assertIn("configured 3 steps, expected 4", errors)
        self.assertIn("completed 3 steps against a budget of 4", errors)
        self.assertIn("checkpoint.pt: missing", errors)

    def test_optional_trainer_identity_fields(self):
        bpb = "original UTF-8 bytes, with p1536 tokenizer normalization"
        kas = {"codec_fixture_sha256": "d" * 64, "prior": {"mode": "zeros"}}
        for extra in (
            {"bpb_policy": bpb},
            {"consolidate_step_checks": False},
            {"consolidate_step_checks": True},
            {"kas_v2": kas},
            {"bpb_policy": bpb, "consolidate_step_checks": True, "kas_v2": kas},
        ):
            with self.subTest(fields=extra):
                self.write_run(extra_identity=extra)
                result = validate(self.run_dir, trusted_checkpoint=True)
                self.assertTrue(result["quality_ready"], result["errors"])

    def test_optional_identity_changes_and_removals_are_rejected(self):
        self.write_run(extra_identity={
            "consolidate_step_checks": True,
            "kas_v2": {"codec_fixture_sha256": "d" * 64, "prior": {"mode": "zeros"}},
        })
        original = (self.run_dir / "manifest.json").read_text(encoding="utf8")
        for field in ("consolidate_step_checks", "kas_v2"):
            for remove in (False, True):
                with self.subTest(field=field, remove=remove):
                    manifest = json.loads(original)
                    if remove:
                        del manifest[field]
                    elif field == "consolidate_step_checks":
                        manifest[field] = False
                    else:
                        manifest[field]["codec_fixture_sha256"] = "e" * 64
                    self.write_json("manifest.json", manifest)
                    result = validate(self.run_dir, trusted_checkpoint=True)
                    self.assertFalse(result["quality_ready"])
                    self.assertIn(
                        "manifest.json: identity_sha256 differs from recorded inputs",
                        result["errors"],
                    )

    def test_actual_trainer_smoke_has_valid_evidence_but_is_not_quality(self):
        cfg, byte_rows, _ = fixture()
        cfg["model"]["ke_max_pos"] = byte_rows.shape[1]
        artifacts = SimpleNamespace(
            bytes=byte_rows, lengths=np.ones(len(byte_rows), dtype=np.int16),
            pos_dim=byte_rows.shape[1], bos_id=0, pad_id=1, normalize_input=False,
            encode=lambda text: [ord(char) - ord("a") + 2 for char in text],
            identity=lambda: {},
        )
        train, heldout = self.run_dir / "train.jsonl", self.run_dir / "heldout.jsonl"
        train.write_text('{"text":"aaaa"}\n', encoding="utf8")
        heldout.write_text('{"text":"bbbb"}\n', encoding="utf8")
        out = self.run_dir / "native"
        args = [
            "train_week2.py", "--config", "fixture.yaml",
            "--tokenizer", "fixture.json", "--byte-table", "fixture.npz",
            "--train-data", str(train), "--heldout-data", str(heldout),
            "--out", str(out), "--smoke", "--smoke-steps", "2",
            "--smoke-batch-size", "1", "--device", "cpu", "--watchdog-s", "0",
        ]
        with (
            patch.object(sys, "argv", args),
            patch.object(train_week2, "load_runtime", return_value=cfg),
            patch.object(train_week2, "Artifacts", return_value=artifacts),
        ):
            train_week2.main()
        result = validate(out, trusted_checkpoint=True)
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["checkpoint_checked"])
        self.assertEqual(result["classification"], "smoke")
        self.assertFalse(result["quality_ready"])
        self.assertEqual(result["verdict"], "FAIL")

    def test_manifest_identity_is_recomputed(self):
        manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf8"))
        manifest["device"] = "cuda"
        self.write_json("manifest.json", manifest)
        self.assertIn(
            "manifest.json: identity_sha256 differs from recorded inputs",
            validate(self.run_dir)["errors"],
        )

    def test_explicit_checkpoint_check(self):
        checkpoint = self.run_dir / "checkpoint.pt"
        state = {"identity_sha256": self.identity, "step": 3}
        torch.save(state, checkpoint)
        result = validate(self.run_dir, trusted_checkpoint=True)
        self.assertTrue(result["checkpoint_checked"])
        self.assertTrue(result["quality_ready"], result["errors"])
        state.update(identity_sha256="wrong run", step=2)
        torch.save(state, checkpoint)
        result = validate(self.run_dir, trusted_checkpoint=True)
        self.assertIn("checkpoint.pt: identity_sha256 differs from manifest", result["errors"])
        self.assertIn("checkpoint.pt: step differs from summary", result["errors"])
        self.assertFalse(result["quality_ready"])

    def test_unreadable_checkpoint_cannot_pass(self):
        (self.run_dir / "checkpoint.pt").write_bytes(b"invalid checkpoint")
        result = validate(self.run_dir, trusted_checkpoint=True)
        self.assertFalse(result["checkpoint_checked"])
        self.assertEqual(result["verdict"], "FAIL")

    def test_each_quality_artifact_is_required(self):
        for name in (
            "manifest.json", "summary.json", "metrics.jsonl",
            "losses.txt", "total_losses.txt", "checkpoint.pt",
        ):
            with self.subTest(name=name):
                self.write_run()
                (self.run_dir / name).unlink()
                result = validate(self.run_dir, trusted_checkpoint=True)
                self.assertEqual(result["verdict"], "FAIL")
                self.assertIn(f"{name}: missing or empty evidence file", result["errors"])

    def test_inventory_detects_changed_bytes(self):
        self.write_json("inventory.json", validate(self.run_dir))
        reference = self.run_dir / "inventory.json"
        self.assertTrue(validate(
            self.run_dir, trusted_checkpoint=True, verify_inventory=reference
        )["quality_ready"])
        with (self.run_dir / "checkpoint.pt").open("ab") as stream:
            stream.write(b"changed")
        result = validate(self.run_dir, verify_inventory=reference)
        self.assertIn("checkpoint.pt: SHA-256 inventory mismatch", result["errors"])


if __name__ == "__main__":
    unittest.main()
