"""P1.11 tests: the training diagnostics moved from the P1.10 audit.

``Diagnostics`` now uses optimiser step hooks instead of replacing ``optimizer.step``; on a tiny
CPU fit its records equal those of an inline copy of the audit's wrapper (``scripts/eff_overfit.py``
at ``f8884ec``) and the learning-rate sequence is unchanged. The state statistics cover dense and
routed models, and ``trained_state`` adds the edge-logit spread (exactly 0 while the dense cells
stay symmetric).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.data.dataset import ByteDataset
from hsn.device import seed_everything
from hsn.experiments import diagnostics as dg
from hsn.model.hsn import HSN
from hsn.runner import build_model, make_trainer
from p1_11_support import (
    cpu_ctx,
    docs,
    restore_determinism,  # noqa: F401  (fixture)
    tiny_gru_config,
    tiny_hsn_config,
)

pytestmark = pytest.mark.usefixtures("restore_determinism")


class _AuditWrapper:
    """The audit's ``Diagnostics.wrap`` (f8884ec), verbatim apart from names: it replaces
    ``optimizer.step`` and counts its calls."""

    def __init__(self, model: Any, cfg: Any, ctx: Any, docs_: ByteDataset, every: int) -> None:
        self.inner = dg.Diagnostics(model, cfg, ctx, docs_, every)
        self.every = every
        self.calls = 0
        self.records: list[dict[str, Any]] = []

    def wrap(self, optimizer: torch.optim.Optimizer) -> None:
        original = optimizer.step
        groups = self.inner.groups

        def step(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls % self.every != 0:
                return original(*args, **kwargs)
            with torch.no_grad():
                before = {n: [p.detach().clone() for p in ps] for n, ps in groups.items()}
                grads = {
                    n: [p.grad.detach() for p in ps if p.grad is not None]
                    for n, ps in groups.items()
                }
            out = original(*args, **kwargs)
            record: dict[str, Any] = {"step": self.calls}
            with torch.no_grad():
                g_total = dg._norm([g for gs in grads.values() for g in gs])
                for n, ps in groups.items():
                    delta = dg._norm([p.detach() - b for p, b in zip(ps, before[n], strict=True)])
                    weight = dg._norm([p.detach() for p in ps])
                    record[f"update_ratio/{n}"] = delta / weight if weight > 0 else math.nan
                    record[f"grad_share/{n}"] = dg._norm(grads[n]) / g_total if g_total > 0 else 0.0
            record.update(self.inner.state_stats())
            self.records.append(record)
            return out

        optimizer.step = step  # type: ignore[method-assign]


def _run(family: str, tmp: Path, mode: str) -> tuple[list[dict[str, Any]], list[float]]:
    cfg = tiny_hsn_config(steps=8) if family == "hsn" else tiny_gru_config(steps=8)
    seed_everything(cfg.seed)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, cpu_ctx(), run_dir=tmp / mode)
    dev = ByteDataset(docs("delayed", "dev", 8))
    train = ByteDataset(docs("delayed", "train", 32))
    if mode == "hooks":
        diag = dg.Diagnostics(model, cfg, cpu_ctx(), dev, 3)
        diag.attach(trainer)
        records = diag.records
    else:
        wrapper = _AuditWrapper(model, cfg, cpu_ctx(), dev, 3)
        wrapper.wrap(trainer.optimizer)
        records = wrapper.records
    history = trainer.fit(train, dev, 8)
    if mode == "hooks":
        assert diag.optimizer_calls == 8 and diag.seconds > 0.0
        for r in records:
            assert r.pop("optimizer_calls") == r["step"]
    return records, [s.lr for s in history.steps]


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return bool(a == b)


@pytest.mark.parametrize("family", ["hsn", "gru"])
def test_p1_11_diagnostics_hooks_match_wrapper(family: str, tmp_path: Path) -> None:
    hooks, lr_hooks = _run(family, tmp_path, "hooks")
    wrapped, lr_wrapped = _run(family, tmp_path, "wrapper")
    assert lr_hooks == lr_wrapped, "the learning-rate schedule is unchanged"
    assert [r["step"] for r in hooks] == [3, 6]
    assert len(hooks) == len(wrapped)
    for a, b in zip(hooks, wrapped, strict=True):
        assert a.keys() == b.keys()
        assert all(_same(a[k], b[k]) for k in a), (a, b)
    shares = [v for k, v in hooks[0].items() if k.startswith("grad_share/")]
    assert sum(s * s for s in shares) == pytest.approx(1.0, rel=1e-5), "plain norm ratios"
    if family == "hsn":
        assert "rank_pr/answer" in hooks[0] and "gate/p50" in hooks[0]
    else:
        assert not any("/answer" in k for k in hooks[0])


def test_p1_11_diagnostics_step_mismatch_raises(tmp_path: Path) -> None:
    cfg = tiny_hsn_config(steps=4)
    seed_everything(0)
    model = build_model(cfg)
    trainer = make_trainer(model, cfg, cpu_ctx(), run_dir=tmp_path)
    diag = dg.Diagnostics(model, cfg, cpu_ctx(), ByteDataset(docs("kv", "dev", 4)), 2)
    diag.attach(trainer)
    with pytest.raises(RuntimeError, match="already attached"):
        diag.attach(trainer)
    trainer.step = 1  # a trainer whose step no longer matches the optimiser calls
    with pytest.raises(RuntimeError, match="optimiser call"):
        trainer.fit(ByteDataset(docs("kv", "train", 8)), ByteDataset(docs("kv", "dev", 4)), 4)
    diag.detach()
    with pytest.raises(ValueError, match="every"):
        dg.Diagnostics(model, cfg, cpu_ctx(), ByteDataset(docs("kv", "dev", 4)), 0)


@pytest.mark.parametrize("routing", ["dense", "topk"])
def test_p1_11_window_state_stats_routed(routing: str) -> None:
    cfg = tiny_hsn_config(f"model.routing={routing}")
    seed_everything(0)
    model = build_model(cfg)
    assert isinstance(model, HSN)
    model.train()
    dev = ByteDataset(docs("delayed", "dev", 6))
    out = dg.trained_state(model, cfg, cpu_ctx(), dev, 4)
    assert model.training, "the model's mode is restored"
    N = cfg.model.n_cells
    for tag in ("answer", "valid"):
        assert 1.0 - 1e-4 <= out[f"rank_pr/{tag}"] <= N
        assert 0.0 <= out[f"live_cells/{tag}"] <= N
    assert 0.0 <= out["gate/p10"] <= out["gate/p50"] <= out["gate/p90"] <= 1.0
    assert out["edge_logit/absmax"] == 0.0 and out["edge_logit/spread"] == 0.0  # init
    if routing == "dense":
        # the zero-state symmetry (finding 1): every cell live and parallel, bitwise identical
        assert out["live_cells/valid"] == N
        assert out["rank_pr/valid"] == pytest.approx(1.0, abs=1e-4)
        assert out["cell_spread/valid"] == 0.0
        assert "write_share/max" not in out
    else:
        assert out["live_cells/valid"] < N
        assert 0.0 <= out["write_share/min"] <= out["write_share/max"] <= 1.0
    gru = build_model(tiny_gru_config())
    assert dg.trained_state(gru, tiny_gru_config(), cpu_ctx(), dev, 4) == {}


def test_trained_state_without_edges() -> None:
    """The no-graph ablation (``model.in_degree = 0``) has no edge logit: the end-of-run
    measurement reports 0 for both edge statistics instead of failing on an empty tensor."""
    cfg = tiny_hsn_config("model.routing=topk", "model.in_degree=0")
    seed_everything(0)
    model = build_model(cfg)
    assert isinstance(model, HSN)
    assert model.cell.graph.edge_logit.numel() == 0
    out = dg.trained_state(model, cfg, cpu_ctx(), ByteDataset(docs("delayed", "dev", 6)), 4)
    assert out["edge_logit/absmax"] == 0.0 and out["edge_logit/spread"] == 0.0
    assert 0.0 <= out["write_share/min"] <= out["write_share/max"] <= 1.0


def test_participation_ratio_equals_the_singular_value_form() -> None:
    """The Gram form is the SVD form's number (no decomposition): random, rank-1 and zero."""
    g = torch.Generator().manual_seed(0)
    hs = torch.randn(40, 6, 9, generator=g, dtype=torch.float64)
    hs[1] = torch.randn(1, 9, generator=g, dtype=torch.float64).expand(6, 9)  # rank 1
    hs[2] = 0.0
    sv2 = torch.linalg.svdvals(hs).square()
    expected = sv2.sum(-1).square() / sv2.square().sum(-1).clamp_min(1e-24)
    actual = dg.participation_ratio(hs)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert float(actual[1]) == pytest.approx(1.0, abs=1e-9) and float(actual[2]) == 0.0
    single = dg.participation_ratio(hs.float())
    torch.testing.assert_close(single, expected.float(), rtol=1e-4, atol=1e-6)


def test_diagnostics_state_window_runs_the_trainers_compiled_window(tmp_path: Path) -> None:
    """A compiled trainer's state window goes through its compiled diagnostic variant (called
    like the eager forward with ``capture=True``); an eager trainer keeps the eager forward."""
    cfg = tiny_hsn_config("model.routing=topk", "train.gate_mode=ste", steps=4)
    seed_everything(0)
    model = build_model(cfg)
    assert isinstance(model, HSN)
    trainer = make_trainer(model, cfg, cpu_ctx(), run_dir=tmp_path)
    dev = ByteDataset(docs("delayed", "dev", 4))
    diag = dg.Diagnostics(model, cfg, cpu_ctx(), dev, 2)
    eager = diag.state_stats()
    calls: list[dict[str, Any]] = []

    def window(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        assert not torch.is_grad_enabled() and not model.training
        return model(*args, **kwargs)

    trainer._compiled = window  # what TBPTTTrainer.compiled_window returns
    rng = torch.get_rng_state()
    before = {k: v.clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
    diag.attach(trainer)  # compiles the variant now: two dropped passes
    assert calls == [{"capture": True}] * 2 and diag.warm_seconds > 0.0
    assert torch.equal(torch.get_rng_state(), rng), "the warm-up draws nothing"
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())
    assert diag.state_stats() == eager
    assert len(calls) == 3
    diag.detach()
    assert diag.state_stats() == eager and len(calls) == 3  # detached: eager again
