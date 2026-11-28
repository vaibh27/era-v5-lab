"""Held-out bits per byte for the base and the kept SFT arms, in the dtype Team 9 scores in, as forgetting.py reads it.

In simple words: forgetting.py compares a base report with an after-SFT report. Both need held-out bits per byte and
Team 9's benchmark rows, scored in the same dtype. sft_g5.py measures bits per byte in float32 with other key names, and
sft/bench/evaluate.py loads Hugging Face folders only. This script loads Team 5's model through the same adapter config
as the SFT job, loads each kept Team 7 run folder (keep_weights.py) into it, and runs Team 5's own held-out evaluate()
under the watcher's dtype rule (bfloat16 autocast on a native-bf16 GPU, float32 elsewhere; weights are never cast).

    python capstone/forget_bpb_kept.py --adapter-config adapter.yaml --heldout <heldout.jsonl> \\
        --heldout-sources <sources.jsonl> --kept keep_t7/g10-dr2-full --kept keep_t7/g10-dr2-lora \\
        --kept keep_t7/g10-dr2-dt --device cuda --dtype auto --out-dir out/forget

It writes forget_base.json and one forget_<run id>.json per kept folder. Then, per arm, with Team 9's rows:

    python capstone/watcher_rows_to_report.py --watcher-doc <base rows> --report out/forget/forget_base.json --out base.json
    python capstone/watcher_rows_to_report.py --watcher-doc <arm rows> --report out/forget/forget_<id>.json --out arm.json
    python sft/tests/forgetting.py --before base.json --after arm.json --out forgetting_<arm>.md

forgetting.py exits 0 for no forgetting, 1 for a forgetting verdict and 2 when it refuses the inputs.

Each report lists the `model_revisions` it measured, so watcher_rows_to_report.py (with --report) refuses rows that
belong to another checkpoint. A kept folder: the sha256 of its step's MANIFEST.json, which is what Team 9 records. The
base: Team 5's model hash (what Team 9 records with --team5-run), plus the Team 7 store manifest sha256 when a
G10_FROM_STORE.json says so: found beside a run folder rebuilt from the store, or passed as --base-store-record. If the
job used Team 5's own checkpoint.pt and Team 9 scored the base from the store, copy the delivered files without
checkpoint.pt into a new folder, run run_folder_from_store.py on it, and pass that record. A record whose model hash is
not the base's is refused.

Not checked: that a kept folder was trained from this base (only its model config, keys and frozen tables are). Compare
`kept.manifest_sha256` with `kept_weights.manifest_sha256` in the arm's sft_report.json.
It reads no benchmark items and no benchmark scores.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEAM10 = HERE.parent
sys.path.insert(0, str(TEAM10 / "rl-loop"))
sys.path.insert(0, str(HERE))

SCHEMA = "g10-bench-eval/1"  # sft/tests/forgetting.py


class KeptError(ValueError):
    pass


def _bench_eval():
    """sft/bench/evaluate.py, for its dtype rule (one definition for both scripts)."""
    name = "g10_bench_evaluate"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, TEAM10 / "sft" / "bench" / "evaluate.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise KeptError(f"{path}: cannot be read as JSON: {e}") from e
    if not isinstance(d, dict):
        raise KeptError(f"{path}: not a JSON object")
    return d


def read_pointer(run_dir: Path) -> tuple[dict, bytes]:
    """The step latest.json points at: (its MANIFEST.json, those bytes). Never a directory listing."""
    latest = Path(run_dir) / "latest.json"
    if not latest.is_file():
        raise KeptError(f"{latest} not found: no committed checkpoint in this folder")
    path = read_json(latest).get("path")
    if not path:
        raise KeptError(f"{latest} has no path: nothing is committed")
    manifest = Path(run_dir) / path / "MANIFEST.json"
    if not manifest.is_file():
        raise KeptError(f"{manifest} not found: the commit is incomplete")
    raw = manifest.read_bytes()
    record = read_json(manifest)
    if "step" not in record:
        raise KeptError(f"{manifest} names no step: the file is damaged")
    return record, raw


def check_kept(bundle, run_dir: Path) -> tuple[dict, bytes]:
    """The cheap checks, made on every folder before anything is measured: model config and a committed step."""
    import keep_weights as kw  # noqa: PLC0415

    try:
        kw.check_reader_config(run_dir, bundle.spec)
    except ValueError as e:
        raise KeptError(str(e)) from e
    return read_pointer(run_dir)


def read_kept(bundle, run_dir: Path) -> tuple[dict, dict]:
    """Every check on one keep_weights run folder, without touching bundle.model: (what it holds, its state dict).

    Refuses a folder saved with another model config, a step that is not the one latest.json names, anything Team 7's
    export refuses (damaged or missing files, a key set that differs from the model's, other probe bits on the saved
    device class) and frozen tables (KE2 byte table, codec) that differ from those in bundle.model, which must hold the
    base or an arm already checked against it.
    """
    import keep_weights as kw  # noqa: PLC0415  also puts Team 7 on sys.path
    from ckpt.ckpt import CheckpointError, device_class, export_safetensors  # noqa: PLC0415  Team 7, unmodified
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415
    import torch  # noqa: PLC0415

    run_dir = Path(run_dir).resolve()
    manifest, raw = check_kept(bundle, run_dir)
    with tempfile.TemporaryDirectory(prefix="g10_forget_") as tmp:
        out = Path(tmp) / "export.safetensors"
        try:
            info = export_safetensors(str(run_dir), lambda: copy.deepcopy(bundle.model), kw.team5_probe, str(out),
                                      step=manifest["step"])
        # IndexError: the step is not in commits.jsonl. OSError, KeyError, ValueError: a missing or garbled file
        except (CheckpointError, IndexError, KeyError, OSError, ValueError) as e:
            raise KeptError(f"{run_dir}: Team 7 export refused it: {type(e).__name__}: {e}") from e
        with safe_open(str(out), framework="pt") as f:
            digest = (f.metadata() or {}).get("state_digest")
        if digest != manifest.get("state_digest"):
            raise KeptError(f"{run_dir}: exported state_digest {digest} differs from MANIFEST.json "
                            f"{manifest.get('state_digest')}")
        state = {k: v.clone() for k, v in load_file(str(out)).items()}  # own memory: the export file is removed
    current = bundle.model.state_dict()
    moved = [k for k in kw.TEAM5_FROZEN_KEYS if k in current and not torch.equal(state[k].cpu(), current[k].cpu())]
    if moved:
        raise KeptError(f"{run_dir}: frozen tables differ from the base's: {moved}; it was not trained from this base")
    # Team 7 compares the probe-loss bits only when the step saved bits for the device class the export ran on
    saved, here = sorted(manifest.get("probe_loss") or {}), device_class(bundle.model)
    return {"run_dir": str(run_dir), "run_id": run_dir.name, "step": manifest["step"],
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "state_digest": digest,
            "probe": {"result": "compared" if here in saved else "skipped", "bits": info.get("probe_bits"),
                      "device_class": here, "saved_device_classes": saved}}, state


def load_kept(bundle, run_dir: Path) -> dict:
    """read_kept, then load the weights into bundle.model (strict keys). Returns what was loaded."""
    info, state = read_kept(bundle, run_dir)
    bundle.model.load_state_dict(state, strict=True)
    return info


def heldout_windows(bundle, heldout: Path, sources: Path | None):
    """Team 5's DocumentWindows, as sft_g5.forgetting_bpb builds them."""
    data = importlib.import_module("week2.data")
    try:
        return data.DocumentWindows(heldout, bundle.artifacts, bundle.spec["block_size"], require_ids=False,
                                    sources_path=sources)
    except (OSError, KeyError, ValueError) as e:
        raise KeptError(f"held-out files refused: {type(e).__name__}: {e}") from e


def heldout_bpb(bundle, heldout: Path, sources: Path | None, device: str, use_bf16: bool) -> dict:
    """Team 5's evaluate() on Team 5's DocumentWindows (as sft_g5.forgetting_bpb), under the given autocast, with the
    key names forgetting.py reads."""
    evaluation = importlib.import_module("week2.evaluation")
    be = _bench_eval()
    if device == "mps":
        import g5_launch  # noqa: PLC0415
        g5_launch.install_mps_double_patch()
    window = bundle.spec["block_size"]
    val = heldout_windows(bundle, heldout, sources)
    res = evaluation.evaluate(bundle.model, val, 1, __import__("torch").device(device),
                              lambda: be._autocast_ctx(device, use_bf16),
                              max_batch_tokens=evaluation.EVAL_BATCH_TOKENS)
    return {"heldout_sha256": sha256_file(heldout),
            "heldout_sources_sha256": sha256_file(sources) if sources else None,
            "documents": val.document_count, "tokens": res["val/scored_tokens"], "windows": len(val.windows),
            "window": window, "bpb_all": res["val/bpb"],
            "groups": {k: v for k, v in res.items() if k.startswith("val/groups/") or k == "val/three_group_bpb"},
            "slices": {k.split("/")[2]: v for k, v in res.items()
                       if k.startswith("val/slices/") and k.endswith("/bpb")}}


def base_revisions(bundle, adapter_config: Path | None, store_record: Path | None = None) -> list[str]:
    """Names Team 9 may record for the base: Team 5's model hash, and the Team 7 manifest sha256 of every
    G10_FROM_STORE.json (run_folder_from_store.py) given or found beside the adapter's run folder. Each record must name
    the base's own model hash."""
    own = bundle.manifest.get("model_hash_g5")
    records = []
    if adapter_config is not None:
        import yaml  # noqa: PLC0415
        cfg = yaml.safe_load(Path(adapter_config).read_text(encoding="utf-8"))
        folder = (cfg.get("run_folder") or {}).get("path")
        if folder:
            found = Path(adapter_config).resolve().parent / folder / "G10_FROM_STORE.json"
            if found.is_file():
                records.append(found)
    if store_record is not None:
        if not Path(store_record).is_file():
            raise KeptError(f"{store_record} not found")
        records.append(Path(store_record))
    revs = [own]
    for path in records:
        rec = read_json(path)
        if rec.get("model_hash_g5") != own:
            raise KeptError(f"{path}: model_hash_g5 {rec.get('model_hash_g5')} is not the base's {own}")
        revs.append(rec.get("manifest_sha256"))
    return list(dict.fromkeys(r for r in revs if r))


def run(bundle, heldout: Path, sources: Path | None, kept: list[Path], device: str, dtype: str, out_dir: Path,
        adapter_config: Path | None = None, base_store_record: Path | None = None) -> dict[str, Path]:
    """Check every input, then measure the base (the bundle as built) and every kept folder.
    Returns {name: report path}. Reports of the same names from an earlier run are removed before measuring."""
    import torch  # noqa: PLC0415

    try:
        compute_dtype, use_bf16 = _bench_eval().resolve_compute_dtype(device, dtype)
    except (RuntimeError, ValueError) as e:
        raise KeptError(str(e)) from e
    if use_bf16 and not device.startswith(("cuda", "cpu")):  # the autocast would be a CPU one: fp32 labelled bf16
        raise KeptError(f"bfloat16 is not applied on device {device}: use --dtype float32")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    names = ["base"] + [Path(k).resolve().name for k in kept]
    if len(set(names)) != len(names):
        raise KeptError(f"two reports would share a name: {names}")
    for folder in kept:  # the cheap checks on all of them first, so a mistyped path is reported at once
        check_kept(bundle, Path(folder).resolve())
    base_revs = base_revisions(bundle, adapter_config, base_store_record)
    heldout_windows(bundle, heldout, sources)
    checked = [read_kept(bundle, folder) for folder in kept]  # every check, against the base, before any measurement
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"schema": SCHEMA, "device": device, "torch": torch.__version__, "compute_dtype": compute_dtype,
              "tokenizer_sha256": getattr(bundle.artifacts, "tokenizer_sha", None),
              "model_spec_sha256": bundle.manifest.get("model_spec_sha256"),
              # of the base: a rehearsal (stand-in) run folder says so here, so its numbers are not read as dry run 2's
              "base_adapter_status": bundle.manifest.get("status"),
              "base_adapter_origin": bundle.manifest.get("origin")}
    written: dict[str, Path] = {}
    for name in names:  # a run that stops half way must not leave an older report beside its new ones
        (out_dir / f"forget_{name}.json").unlink(missing_ok=True)

    def write(name: str, extra: dict) -> None:
        report = {**common, **extra,
                  "forgetting": heldout_bpb(bundle, heldout, sources, device, use_bf16)}
        path = out_dir / f"forget_{name}.json"
        path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        written[name] = path
        print(f"{name}: bpb_all {report['forgetting']['bpb_all']:.6f} ({compute_dtype}) -> {path}", flush=True)

    write("base", {"model_revisions": base_revs})
    for (info, state), name in zip(checked, names[1:]):
        bundle.model.load_state_dict(state, strict=True)
        write(name, {"model_revisions": [info["manifest_sha256"]], "kept": info})
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter-config", required=True, type=Path, help="the SFT job's team5_adapter config (the base)")
    ap.add_argument("--heldout", required=True, type=Path)
    ap.add_argument("--heldout-sources", type=Path)
    ap.add_argument("--base-store-record", type=Path,
                    help="a G10_FROM_STORE.json naming the base in the Team 7 store, when the run folder has none")
    ap.add_argument("--kept", action="append", default=[], type=Path, help="a keep_weights run folder; repeat per arm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto", choices=("float32", "bfloat16", "auto"))
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    import team5_adapter as ta  # noqa: PLC0415

    try:
        result = ta.build(a.adapter_config, device=a.device)
        if result.bundle is None:
            raise KeptError(f"the adapter built no model: {result.status} ({'; '.join(result.reasons)})")
        run(result.bundle, a.heldout, a.heldout_sources, a.kept, a.device, a.dtype, a.out_dir, a.adapter_config,
            a.base_store_record)
    except (KeptError, ta.ArtifactError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
