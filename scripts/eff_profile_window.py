#!/usr/bin/env python
"""P1.10 efficiency audit, part B (time per step): where the launches and the time of one HSN
training step go, module by module.

    python scripts/eff_profile_window.py --recipe overfit|g1 --mode count|sync|cprofile|timing \
        --out evidence/EFF/t2 [--warm 3] [--timing-steps 20] [--override KEY=VALUE ...]

Recipes: ``overfit`` is the ``eff_overfit.py`` recipe (fp32, batch 4, the first 64 delayed/in_range
training documents, lr 1e-3); ``g1`` is ``configs/experiments/g1_ref_a1.yaml`` as configured (bf16
autocast, batch 16, the frozen G1 task set's in-range training documents). Steps run through the
real :class:`~hsn.train.tbptt.TBPTTTrainer` (its evaluation and checkpoint writes are stubbed out),
after ``--warm`` unmeasured steps.

Modes:

* ``count``: two training steps under ``torch.profiler`` (CPU + CUDA activity): a plain step
  and a stats step (``train.stats_every``; the last step of a fit always is one). Device events
  (kernels, memcpys, memsets: the P0.6 definition of a launch) are attributed through the op that
  launched them, using the P0.6 link (``linked_correlation_id`` to the frontend op):

  - forward ops to the innermost labelled range around them (``MOD::<module>`` from forward
    pre/post hooks, ``FN::<function>`` from wrappers of the step's functions);
  - backward ops through the autograd node that ran them, whose ``sequence_nr`` names the forward
    op that recorded it, to that op's forward range (``bwd:<label>``); the leaf accumulation
    (``AccumulateGrad``) and the engine's input-buffer sums (ops under ``evaluate_function`` but
    outside the node itself) get the rows ``bwd:grad_accumulation(leaf)`` and
    ``bwd:grad_accumulation(buffer)``;
  - trainer phases to ``TRN::<phase>`` (loss assembly, per-term gradient norms, group norms,
    clipping, the optimiser's own ``Optimizer.step`` range).

  Reports launches per step and per position (``/ T``) for every row, host self-CPU time per row,
  the step wall time inside the profiler (inflated by it), and the GPU busy fraction (the union of
  device-event intervals over the step's wall interval).
* ``sync``: one step with ``torch.cuda.set_sync_debug_mode("warn")``; every synchronising call is
  recorded with its Python call site and counted.
* ``cprofile``: ``--timing-steps`` steps under :mod:`cProfile`; total time per function, with the
  share spent in ``python/hsn`` Python code, ``torch.nn`` module machinery and torch built-ins.
* ``timing``: ``--timing-steps`` steps without a profiler; the trainer's per-step seconds (median
  over plain steps, stats steps apart) and a split into forward, backward and optimiser step with a
  device synchronisation on each boundary (three per step).

Every output records the GPU compute processes seen before and after; a timing taken with another
process of >= 200 MiB on the card is marked ``shared`` (contaminated: each process is launch-bound
on its own CPU core, and the card time-slices between contexts).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import cProfile
import functools
import json
import os
import pstats
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eff_overfit import REPO, card_status, gpu_processes, overfit_docs, recipe_overrides  # noqa: E402
from hsn.config import RunConfig, load_config  # noqa: E402
from hsn.data.dataset import ByteDataset  # noqa: E402
from hsn.device import resolve, seed_everything, synchronize  # noqa: E402
from hsn.model import cell as cell_mod  # noqa: E402
from hsn.model import graph as graph_mod  # noqa: E402
from hsn.model import hsn as hsn_mod  # noqa: E402
from hsn.model import router as router_mod  # noqa: E402
from hsn.model import state as state_mod  # noqa: E402
from hsn.model import units as units_mod  # noqa: E402
from hsn.model.hsn import HSN  # noqa: E402
from hsn.runner import build_model, make_trainer  # noqa: E402
from hsn.train import tbptt as tbptt_mod  # noqa: E402
from torch.autograd import DeviceType  # type: ignore[attr-defined]  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

LAUNCH_ACTIVITIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
EVAL_PREFIX = "autograd::engine::evaluate_function: "


# --------------------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------------------


def build(recipe: str, extra: list[str], horizon: int) -> tuple[RunConfig, Any, Any, ByteDataset]:
    if recipe == "overfit":
        overrides = recipe_overrides(
            profile="rtx3060",
            lr=1e-3,
            steps=horizon,
            eval_every=10 * horizon,
            seed=0,
            run_id="eff_profile_overfit",
            extra=extra,
        )
        cfg = load_config(REPO / "configs" / "experiments" / "g1_ref_a1.yaml", overrides=overrides)
        docs = overfit_docs(cfg, 64)
    elif recipe == "g1":
        overrides = [
            "profile=rtx3060",
            f"train.steps={horizon}",
            f"train.eval_every={10 * horizon}",
            f"train.ckpt_every={10 * horizon}",
            "run_id=eff_profile_g1",
            *extra,
        ]
        cfg = load_config(REPO / "configs" / "experiments" / "g1_ref_a1.yaml", overrides=overrides)
        docs = ByteDataset.load(
            Path(str(cfg.data.manifest)),
            "train",
            list(cfg.data.tasks),
            list(cfg.data.variants),
            expected_unroll=cfg.train.unroll,
        )
    else:
        raise ValueError(f"unknown recipe {recipe!r}")
    ctx = resolve(cfg.hardware)
    if ctx.device.type != "cuda":
        raise RuntimeError(f"eff_profile_window.py profiles CUDA; resolved {ctx.device}")
    seed_everything(cfg.seed)
    model = build_model(cfg)
    if not isinstance(model, HSN):
        raise TypeError("eff_profile_window.py profiles the HSN model")
    trainer = make_trainer(model, cfg, ctx, run_dir=Path("/nonexistent/never-written"))
    trainer.evaluate = lambda *a, **k: {}  # type: ignore[method-assign]
    trainer.save = lambda *a, **k: Path("/dev/null")  # type: ignore[method-assign]
    return cfg, model, trainer, docs


# --------------------------------------------------------------------------------------------------
# Labelled ranges
# --------------------------------------------------------------------------------------------------


def _wrap(owner: Any, name: str, label: str) -> None:
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with record_function(label):
            return original(*args, **kwargs)

    setattr(owner, name, wrapped)


def install_labels(model: HSN, trainer: Any) -> None:
    """Forward hooks on the modules, wrappers on the step's functions and the trainer's phases."""
    stack: list[Any] = []

    def pre(label: str) -> Callable[..., None]:
        def hook(module: Any, inputs: Any) -> None:
            rf = record_function(label)
            rf.__enter__()
            stack.append(rf)

        return hook

    def post(module: Any, inputs: Any, output: Any) -> None:
        stack.pop().__exit__(None, None, None)

    modules = {
        "byte_emb": model.byte_emb,
        "ctx_emb": model.ctx_emb,
        "encoder": model.encoder,
        "predictor": model.predictor,
        "shared": model.shared,
        "scorer.net": model.router.scorer,
        "cell.V": model.cell.V,
        "cell.norm": model.cell.norm,
        "cell.W_out": model.cell.W_out,
        "cell.gate": model.cell.gate,
        "decoder": model.decoder,
    }
    for label, module in modules.items():
        module.register_forward_pre_hook(pre(f"MOD::{label}"))
        module.register_forward_hook(post)
    _wrap(hsn_mod.HSN, "forward", "FN::hsn.forward(window glue)")
    _wrap(hsn_mod.HSN, "step", "FN::hsn.step(glue)")
    _wrap(hsn_mod.HSN, "_shared_path", "FN::hsn._shared_path(glue)")
    _wrap(hsn_mod.HSN, "_write_dense", "FN::hsn._write_dense")
    _wrap(cell_mod.PDGCell, "step_dense", "FN::cell.step_dense(glue)")
    _wrap(units_mod.UnitBank, "forward_all", "FN::units.forward_all")
    _wrap(graph_mod.CellGraph, "messages_dense", "FN::graph.messages_dense")
    _wrap(graph_mod.CellGraph, "_padded_weights", "FN::graph.edge_weights")
    _wrap(router_mod.Scorer, "features", "FN::scorer.features")
    _wrap(state_mod.CellGraphState, "reset", "FN::state.reset")
    _wrap(state_mod.CellGraphState, "with_dense", "FN::state.with_dense")
    _wrap(state_mod.CellGraphState, "tick", "FN::state.tick")
    _wrap(state_mod.CellGraphState, "detach", "TRN::state.detach")
    # trainer phases (the names tbptt.py resolves at call time)
    _wrap(tbptt_mod, "assemble_terms", "TRN::assemble_terms")
    _wrap(tbptt_mod, "auxiliary_terms", "TRN::auxiliary_terms")
    _wrap(tbptt_mod, "total_and_parts", "TRN::total_and_parts")
    _wrap(tbptt_mod, "grad_norm_report", "TRN::grad_norm_report")
    _wrap(torch.nn.utils, "clip_grad_norm_", "TRN::clip_grad_norm_")
    _wrap(trainer, "_group_grad_norms", "TRN::group_grad_norms")
    _wrap(trainer, "_to_device", "TRN::to_device")
    _wrap(trainer, "_window_start", "TRN::window_start")


@contextlib.contextmanager
def step_ranges(trainer: Any) -> Iterator[list[Any]]:
    """A ``STEP::<k>`` range from each ``optimizer.zero_grad`` (the first call of a step) to the
    next one, and the last one closed on exit."""
    opened: list[Any] = []
    original = trainer.optimizer.zero_grad
    count = [0]

    def zero_grad(*args: Any, **kwargs: Any) -> Any:
        if opened:
            opened[-1].__exit__(None, None, None)
        count[0] += 1
        rf = record_function(f"STEP::{count[0]}")
        rf.__enter__()
        opened.append(rf)
        return original(*args, **kwargs)

    trainer.optimizer.zero_grad = zero_grad
    try:
        yield opened
    finally:
        synchronize(trainer.ctx.device)
        if opened:
            opened[-1].__exit__(None, None, None)
        trainer.optimizer.zero_grad = original


# --------------------------------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------------------------------


def _is_launch(event: Any) -> bool:
    return event.device_type == DeviceType.CUDA and event.activity_type in LAUNCH_ACTIVITIES


def _chain(event: Any) -> list[Any]:
    out = []
    parent = event.cpu_parent
    while parent is not None:
        out.append(parent)
        parent = parent.cpu_parent
    return out


def _range_label(chain: list[Any]) -> str | None:
    for anc in chain:  # innermost first
        if anc.name.startswith(("MOD::", "FN::", "TRN::")):
            return str(anc.name)
        if anc.name.startswith("Optimizer.step#"):
            return "TRN::optimizer.step"
    return None


def classify(host: Any, seq2label: dict[int, str]) -> str:
    """The row of a launch whose frontend op is ``host`` (see the module docstring)."""
    chain = [host, *_chain(host)]
    for i, anc in enumerate(chain):
        if anc.name.startswith(EVAL_PREFIX):
            node_name = anc.name[len(EVAL_PREFIX) :]
            node = next((c for c in chain[:i] if c.name == node_name), None)
            if node_name.endswith("AccumulateGrad"):
                return "bwd:grad_accumulation(leaf)"
            if node is None:
                return "bwd:grad_accumulation(buffer)"
            label = seq2label.get(int(node.sequence_nr))
            return f"bwd:{label}" if label else f"bwd:(unlabelled) {node_name}"
    label = _range_label(chain)
    return label if label else "(main thread, unlabelled)"


def attribute(events: list[Any], T: int) -> dict[str, Any]:
    cpu = [e for e in events if e.device_type == DeviceType.CPU]
    frontend = {e.id: e for e in cpu if e.linked_correlation_id == 0 and e.external_id != 0}
    steps = sorted(
        (e for e in cpu if e.is_user_annotation and e.name.startswith("STEP::")),
        key=lambda e: e.time_range.start,
    )
    seq2label: dict[int, str] = {}
    for e in cpu:
        if getattr(e, "sequence_nr", -1) is not None and e.sequence_nr >= 0:
            chain = [e, *_chain(e)]
            if any(a.name.startswith(EVAL_PREFIX) for a in chain):
                continue  # a backward op
            label = _range_label(chain)
            if label is not None:
                seq2label.setdefault(int(e.sequence_nr), label)
    per_step: list[dict[str, Any]] = []
    for s in steps:
        per_step.append(
            {
                "step_range": s.name,
                "wall_ms_profiled": (s.time_range.end - s.time_range.start) / 1000.0,
                "launches": collections.Counter(),
                "device_intervals": [],
            }
        )
    unattributed = 0
    for ev in events:
        if not _is_launch(ev):
            continue
        host = frontend.get(ev.linked_correlation_id)
        start = host.time_range.start if host is not None else ev.time_range.start
        idx = next(
            (i for i, s in enumerate(steps) if s.time_range.start <= start <= s.time_range.end),
            None,
        )
        if idx is None:
            unattributed += 1
            continue
        row = classify(host, seq2label) if host is not None else "(no host op)"
        per_step[idx]["launches"][row] += 1
        per_step[idx]["device_intervals"].append((ev.time_range.start, ev.time_range.end))
    # host self-CPU time per row (frontend ops only, by the same classification)
    cpu_us: list[collections.Counter[str]] = [collections.Counter() for _ in steps]
    for e in frontend.values():
        start = e.time_range.start
        idx = next(
            (i for i, s in enumerate(steps) if s.time_range.start <= start <= s.time_range.end),
            None,
        )
        if idx is not None:
            cpu_us[idx][classify(e, seq2label)] += e.self_cpu_time_total
    report: dict[str, Any] = {"T": T, "unattributed_launches": unattributed, "steps": []}
    for rec, cpu_row in zip(per_step, cpu_us, strict=True):
        total = sum(rec["launches"].values())
        wall_us = rec["wall_ms_profiled"] * 1000.0
        busy = _union_us(rec["device_intervals"])
        rows = sorted(rec["launches"].items(), key=lambda kv: -kv[1])
        report["steps"].append(
            {
                "step_range": rec["step_range"],
                "launches_total": total,
                "launches_per_position": round(total / T, 2),
                "wall_ms_profiled": round(rec["wall_ms_profiled"], 1),
                "gpu_busy_fraction": round(busy / wall_us, 4) if wall_us > 0 else None,
                "gpu_busy_ms": round(busy / 1000.0, 1),
                "rows": [
                    {
                        "row": name,
                        "launches": n,
                        "per_position": round(n / T, 2),
                        "host_self_cpu_ms": round(cpu_row.get(name, 0.0) / 1000.0, 2),
                    }
                    for name, n in rows
                ],
                "host_self_cpu_ms_by_row_without_launches": {
                    k: round(v / 1000.0, 2) for k, v in cpu_row.items() if k not in rec["launches"]
                },
            }
        )
    return report


def _union_us(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -1.0
    for a, b in sorted(intervals):
        if a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


# --------------------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------------------


def mode_count(cfg: RunConfig, model: HSN, trainer: Any, docs: ByteDataset, warm: int) -> Any:
    trainer.fit(docs, docs, warm)
    install_labels(model, trainer)
    synchronize(trainer.ctx.device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with step_ranges(trainer):
            trainer.fit(docs, docs, warm + 2)
    report = attribute(list(prof.events()), cfg.train.unroll)
    report["note"] = (
        "steps[0] is a plain step, steps[1] a stats step (the last step of a fit); wall times are "
        "inflated by the profiler"
    )
    return report


def mode_sync(cfg: RunConfig, model: HSN, trainer: Any, docs: ByteDataset, warm: int) -> Any:
    trainer.fit(docs, docs, warm)
    synchronize(trainer.ctx.device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            trainer.fit(docs, docs, warm + 1)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    sites: collections.Counter[str] = collections.Counter()
    for w in caught:
        if "synchroniz" not in str(w.message).lower():
            continue
        sites[f"{os.path.relpath(w.filename, REPO)}:{w.lineno}"] += 1
    return {
        "note": "one stats step (the last step of a fit); site = the Python line of the call",
        "syncs_total": sum(sites.values()),
        "sites": dict(sites.most_common()),
    }


def mode_cprofile(
    cfg: RunConfig, model: HSN, trainer: Any, docs: ByteDataset, warm: int, n: int
) -> Any:
    trainer.fit(docs, docs, warm)
    synchronize(trainer.ctx.device)
    pr = cProfile.Profile()
    started = time.perf_counter()
    pr.enable()
    trainer.fit(docs, docs, warm + n)
    synchronize(trainer.ctx.device)
    pr.disable()
    wall = time.perf_counter() - started
    stats = pstats.Stats(pr)
    groups: collections.Counter[str] = collections.Counter()
    top: list[tuple[float, str, int]] = []
    for (filename, lineno, func), (_cc, nc, tt, _ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        if "python/hsn" in filename:
            group = "hsn python"
        elif "torch/nn/modules" in filename:
            group = "torch.nn module machinery"
        elif filename == "~" or "torch" in func or "Tensor" in func:
            group = "torch built-ins and C calls"
        elif "torch" in filename:
            group = "torch python"
        else:
            group = "other python"
        groups[group] += tt
        top.append(
            (
                tt,
                f"{os.path.relpath(filename, REPO) if filename != '~' else '~'}:{lineno}({func})",
                nc,
            )
        )
    top.sort(reverse=True)
    total = sum(groups.values())
    return {
        "steps": n,
        "wall_s_under_cprofile": round(wall, 3),
        "tottime_share": {k: round(v / total, 4) for k, v in groups.most_common()},
        "top": [{"tottime_s": round(t, 4), "calls": c, "function": f} for t, f, c in top[:40]],
    }


def mode_timing(
    cfg: RunConfig, model: HSN, trainer: Any, docs: ByteDataset, warm: int, n: int
) -> Any:
    trainer.fit(docs, docs, warm)
    device = trainer.ctx.device
    split: dict[str, list[float]] = {"forward": [], "backward": [], "optimizer": []}

    def timed(kind: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            synchronize(device)
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            synchronize(device)
            split[kind].append(time.perf_counter() - t0)
            return out

        return inner

    plain = trainer.fit(docs, docs, warm + n)  # unsynchronised: the trainer's own seconds
    stats_every = cfg.train.stats_every
    records = plain.steps
    plain_s = sorted(r.seconds for r in records if r.step % stats_every != 0 and r.step != warm + n)
    stats_s = sorted(r.seconds for r in records if r.step % stats_every == 0 or r.step == warm + n)
    model.forward = timed("forward", model.forward)  # type: ignore[method-assign]
    original_backward = torch.Tensor.backward
    torch.Tensor.backward = timed("backward", original_backward)  # type: ignore[method-assign]
    trainer.optimizer.step = timed("optimizer", trainer.optimizer.step)
    try:
        synced = trainer.fit(docs, docs, warm + 2 * n)
    finally:
        torch.Tensor.backward = original_backward  # type: ignore[method-assign]
    synced_s = sorted(r.seconds for r in synced.steps)

    def med(xs: list[float]) -> float | None:
        return round(xs[len(xs) // 2], 4) if xs else None

    return {
        "steps": n,
        "step_s_p50_plain": med(plain_s),
        "step_s_p50_stats_steps": med(stats_s),
        "n_plain": len(plain_s),
        "n_stats": len(stats_s),
        "synchronised_pass": {
            "step_s_p50": med(synced_s),
            "forward_s_p50": med(sorted(split["forward"])),
            "backward_s_p50": med(sorted(split["backward"])),
            "optimizer_s_p50": med(sorted(split["optimizer"])),
            "note": "backward includes the extra per-term backwards of stats steps",
        },
        "tokens_per_step": cfg.train.batch * cfg.train.unroll,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recipe", choices=("overfit", "g1"), default="overfit")
    ap.add_argument("--mode", choices=("count", "sync", "cprofile", "timing"), default="count")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--timing-steps", type=int, default=20)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    seen = [gpu_processes()]
    horizon = args.warm + 2 * args.timing_steps + 5
    cfg, model, trainer, docs = build(args.recipe, args.override, horizon)
    started = time.perf_counter()
    if args.mode == "count":
        result = mode_count(cfg, model, trainer, docs, args.warm)
    elif args.mode == "sync":
        result = mode_sync(cfg, model, trainer, docs, args.warm)
    elif args.mode == "cprofile":
        result = mode_cprofile(cfg, model, trainer, docs, args.warm, args.timing_steps)
    else:
        result = mode_timing(cfg, model, trainer, docs, args.warm, args.timing_steps)
    seen.append(gpu_processes())
    result = {
        "recipe": args.recipe,
        "mode": args.mode,
        "overrides": args.override,
        "batch": cfg.train.batch,
        "unroll": cfg.train.unroll,
        "precision": cfg.train.precision,
        "card": card_status(os.getpid(), seen),
        "gpu_processes": seen,
        "wall_s": round(time.perf_counter() - started, 1),
        **result,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "_".join([args.recipe, args.mode, *[o.replace("=", "-") for o in args.override]])
    path = out / f"{tag}.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    print(path)
    print(json.dumps(result, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
