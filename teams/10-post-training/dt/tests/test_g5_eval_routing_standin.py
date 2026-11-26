"""DT calibration's expert choice against Team 5's own forward in EVAL mode, on a real G5 stand-in, with no copy of the rule.

Calibration (dt/mask.py wanda_scores) runs the model in eval mode and picks each routed expert's rows with
`g5_expert_choice`. The reference here is read from the model itself: in the last block, perturbing one expert's down weight
changes exactly the positions Team 5's forward routes to that expert (nothing after the last block mixes positions, so the
head is skipped). The full [valid position x expert] routing matrix read this way must equal the one `g5_expert_choice`
gives on the same MoE inputs: the matrix, not per-expert counts, so two positions with their choices swapped are caught.
Unlike Team 5's `expert_load` counter, which counts only in train mode, this sees an eval-mode rule change (for example
a bias used only while training). Checked once with the stand-in's own balancer setting and once with bias_gamma 0.

Scope: the LAST block only (nothing after it mixes positions, so a perturbation there is read cleanly). Earlier blocks
choose through the same `g5_expert_choice` call, but this test does not run them; do not read it as a per-layer check.

Where it runs: CI skips it (no stand-in there). G10 runs it with the stand-in variables set before merging any change to
`dt/mask.py` routing or to Team 5's MoE routing, and it can run on the job machine against the delivered run folder.

Needs a CPU run folder trained with this tree's Team 5 code (job ADP's KAS stand-in has a non-zero router_bias):

    G10_STANDIN_RUN_FOLDER=<run folder> G10_STANDIN_ARTIFACTS=<artifacts> python -m pytest dt/tests/test_g5_eval_routing_standin.py -q
"""
import importlib.util
import os
import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
T10 = HERE.parents[1]
spec = importlib.util.spec_from_file_location("g10_dt_mask_eval_routing", HERE.parent / "mask.py")
dtmask = importlib.util.module_from_spec(spec)
sys.modules["g10_dt_mask_eval_routing"] = dtmask
spec.loader.exec_module(dtmask)


class _Stop(Exception):
    pass


def _standin():
    run, art = os.environ.get("G10_STANDIN_RUN_FOLDER"), os.environ.get("G10_STANDIN_ARTIFACTS")
    if not run or not art:
        pytest.skip("set G10_STANDIN_RUN_FOLDER and G10_STANDIN_ARTIFACTS (a CPU run folder trained with this tree's Team 5 code)")
    # set but wrong is a mistake, not a reason to skip: fail with the missing path instead of deep inside the adapter
    for f in (Path(run) / "checkpoint.pt", Path(art) / "tokenizer_p1536.json"):
        assert f.is_file(), f"stand-in variable set but {f} does not exist"
    sys.path[:0] = [str(T10 / "capstone"), str(T10 / "rl-loop")]
    import make_adapter_config as mac  # noqa: PLC0415
    import sft_g5  # noqa: PLC0415
    import team5_adapter as ta  # noqa: PLC0415
    return Path(run), Path(art), mac, sft_g5, ta


def test_calibration_expert_choice_equals_team5s_eval_forward_on_the_standin(tmp_path):
    run, art, mac, sft_g5, ta = _standin()
    table = T10.parent / "04-ke2" / "results" / "tables" / "p1536-s0" / "raw_table_P32.npz"
    cfg = mac.write_config(tmp_path / "adapter.yaml", T10.parents[1], run, art / "tokenizer_p1536.json", table, T10,
                           "g10_rehearsal", False)
    bundle = ta.build(cfg, device="cpu").require_bundle()
    lm = bundle.model.eval()
    wrapper = sft_g5.G5ForBench(lm)
    moe = lm.blocks[-1].moe
    # 4 rows of token ids, drawn from the checkpoint's own vocabulary range (fixtures, not training data)
    gen = random.Random(10_001)
    lengths = [37, 64, 21, 50]
    rows = [{"id": f"t/{i}", "input_ids": [gen.randrange(1000, bundle.spec["vocab_size"]) for _ in range(n)],
             "loss_mask": [1] * n, "split": "train"} for i, n in enumerate(lengths)]
    batch = sft_g5.bench.collate(rows, 0)
    valid = batch.attention_mask.bool()
    seen = {}

    def pre(mod, a, kw):
        seen["x"], seen["valid"] = a[0].detach().clone(), a[1].detach().clone()

    def stop(mod, a, out):
        seen["out"] = (out if torch.is_tensor(out) else out[0]).detach().clone()
        raise _Stop

    hooks = [moe.register_forward_pre_hook(pre, with_kwargs=True), lm.blocks[-1].register_forward_hook(stop)]

    def forward():
        with torch.no_grad():
            try:
                wrapper(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            except _Stop:
                pass
        return seen["out"]

    def model_routes():
        """[B*T, E] bool: position p is routed to expert e when perturbing e changes p's output (valid positions only)."""
        base = forward()
        cols = []
        for expert in moe.experts:
            w = expert.down.weight
            keep = w.detach().clone()
            try:
                with torch.no_grad():
                    w.add_(1.0)
                # Exact `!=`: a routed position changes by gate * sum(hidden) in every output channel. If that were exactly 0,
                # or absorbed by float32, the position would be missed; a miss can only LOWER a count (nothing unrouted can
                # change), and the total check below then fails, so the test cannot pass wrongly. On the ADP stand-in the
                # smallest change at a routed position was 0.0040, about 28,000 float32 ulps of that output.
                cols.append(((forward() != base).any(-1) & valid).reshape(-1))
            finally:
                with torch.no_grad():
                    w.copy_(keep)
        forward()  # leave `seen` on the unperturbed inputs
        return torch.stack(cols, -1)

    def our_routes():
        flat = seen["x"].reshape(-1, seen["x"].shape[-1])
        active = seen["valid"].reshape(-1).bool()
        scores = F.linear(flat.float(), moe.router.weight.float()).softmax(-1)
        choice = dtmask.g5_expert_choice(moe, scores)
        sel = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, choice, True) & active[:, None]
        return sel

    def same(ours, ref, msg):
        bad = (ours != ref).any(-1)
        assert not bad.any(), f"{msg}: {int(bad.sum())} positions differ, first at flat index {int(bad.nonzero()[0])}"

    gammas = [b.moe.balance_gamma for b in lm.blocks]
    try:
        assert not any(m.training for m in lm.modules())
        biased = model_routes()
        assert int(biased.sum()) == int(valid.sum()) * moe.topk, (
            "a routed position did not change under the perturbation (exact != missed it): the reference undercounts")
        same(our_routes(), biased, "calibration's expert choice differs from Team 5's eval-mode forward")
        has_bias = bool(moe.balance_gamma) and bool(moe.router_bias.ne(0).any())
        for b in lm.blocks:
            b.moe.balance_gamma = 0.0  # gamma 0: Team 5 ignores the buffer; so must we
        plain = model_routes()
        same(our_routes(), plain, "with bias_gamma 0 calibration must use the plain top-k, as Team 5 does")
        if has_bias:
            assert not torch.equal(plain, biased), (
                "the stand-in's router_bias no longer changes any choice: this test has lost its power")
    finally:
        for b, g in zip(lm.blocks, gammas):
            b.moe.balance_gamma = g
        for h in hooks:
            h.remove()
