"""P1.13 test support: golden runs of today's defaults, captured before the levers existed.

P1.13 adds three opt-in levers (``train.schedule=wsd``, the cell parameter group,
``model.cand_norm``) whose defaults must reproduce the code of ``8eb2f01`` bit for bit. This
module trains a few small runs with the trainers exactly as ``hsn run`` builds them
(:func:`hsn.runner.build_model`, :func:`hsn.runner.make_trainer`) and records, per optimiser step,
the loss, the pre-clip gradient norm and the learning rate (``float.hex``) and SHA-256 digests of
every gradient and every parameter; after the run, digests of the final checkpoint's tensors
(weights, optimiser state, the carried recurrent state) and of the carried state the trainer
keeps. Only APIs that exist at ``8eb2f01`` are used, so the same code captures the fixture on the
base commit and replays it on the branch (``tests/test_p1_13_defaults.py``).

Capture (once, with the base commit's library: the branch before its first library change, or
``PYTHONPATH=<an export of 8eb2f01>/python``; the fixtures are committed)::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python tests/p1_13_support.py capture \\
        --device cpu --out tests/fixtures/p1_13/defaults_cpu.json --repeats 2
    python tests/p1_13_support.py capture --device cuda \\
        --out tests/fixtures/p1_13/defaults_cuda.json --repeats 2

A fixture is replayed bit for bit only on the platform it was captured on (:func:`fingerprint`:
torch version, CPU capability, device name); elsewhere (another CPU, CI) the floats are compared to
a tolerance and the digests are skipped, because BLAS kernels differ by instruction set.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from hsn.config import RunConfig, load_config
from hsn.data.dataset import ByteDataset
from hsn.data.synthetic import Example, TaskSpec, generate, load_dataset_spec
from hsn.device import DeviceContext, resolve, seed_everything
from hsn.runner import build_model, make_trainer
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "p1_13"
REF_A1 = REPO / "configs" / "ref_a1.yaml"
GRU_2M = REPO / "configs" / "baselines" / "gru_2m.yaml"
UNROLL = 32
STEPS = 6  # the whole schedule: warm-up (2 steps), then the cosine decay to its last step
CUDA_MAX_BLOCK = 8  # P2.8 compiled blocks of at most 8 positions (a short compile)

#: The common overrides: a short cosine schedule with warm-up, so the learning rate changes at
#: every step, evaluation and checkpoints only at the end.
_SCHEDULE = [
    f"train.unroll={UNROLL}",
    f"train.steps={STEPS}",
    "train.warmup_steps=2",
    "train.schedule=cosine",
    "train.lr=0.003",
    "train.weight_decay=0.01",
    f"train.eval_every={STEPS}",
    f"train.ckpt_every={STEPS}",
    "train.stats_every=1",
]
_SMALL_HSN = [
    "model.d=8",
    "model.context_dim=8",
    "model.n_cells=4",
    "model.in_degree=2",
    "model.n_units=2",
    "model.k_select=2",
    "model.hidden=16",
    "model.scorer_hidden=8",
    "train.batch=4",
    "train.micro_batch=4",
]

#: name -> (config file, overrides, device kind). CPU cases run fp32 with deterministic kernels.
CASES: dict[str, tuple[Path, list[str], str]] = {
    # dense HSN, AdamW: step_dense, the P1.10a window, the P0.5 groups and the schedule
    "hsn_dense": (
        REF_A1,
        [*_SMALL_HSN, *_SCHEDULE, "profile=cpu", "train.deterministic=true"],
        "cpu",
    ),
    # top-K HSN with straight-through gates: step_sparse and the P2.6 lazy AdamW unit groups
    "hsn_topk": (
        REF_A1,
        [
            *_SMALL_HSN,
            *_SCHEDULE,
            "profile=cpu",
            "train.deterministic=true",
            "model.routing=topk",
            "train.gate_mode=ste",
        ],
        "cpu",
    ),
    # the GRU baseline on the generic P0.5 trainer
    "gru": (
        GRU_2M,
        [
            *_SCHEDULE,
            "profile=cpu",
            "train.deterministic=true",
            "train.batch=4",
            "train.micro_batch=4",
            "baseline.gru.d_model=16",
            "baseline.gru.n_layers=1",
        ],
        "cpu",
    ),
    # dense HSN in P2.8's compiled window on CUDA under bf16 autocast (the experiments' path)
    "hsn_dense_compiled_cuda": (
        REF_A1,
        [
            "model.d=16",
            "model.context_dim=16",
            "model.n_cells=8",
            "model.in_degree=2",
            "model.n_units=2",
            "model.hidden=32",
            "train.batch=4",
            "train.micro_batch=4",
            *_SCHEDULE,
            "hardware.compile_mode=reduce-overhead",
        ],
        "cuda",
    ),
}


# --------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------


def _task() -> TaskSpec:
    """The committed kv in-range task with documents that fit the 32-position window."""
    spec = load_dataset_spec(REPO / "configs" / "data" / "synthetic_v1.yaml")
    task = spec.task("kv", "in_range")
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "n_train": 64,
            "n_dev": 16,
            "n_test": 16,
            "length_range": [16, UNROLL],
            "dep_len_range": [4, 20],
            "unroll": UNROLL,
        }
    )


def datasets() -> tuple[ByteDataset, ByteDataset]:
    """64 training and 16 dev documents, generated (no data directory needed)."""
    task = _task()
    train: list[Example] = [generate(task, "train", i) for i in range(64)]
    dev: list[Example] = [generate(task, "dev", i) for i in range(16)]
    return ByteDataset(train), ByteDataset(dev)


def case_config(name: str, profile: str | None = None, extra: list[str] | None = None) -> RunConfig:
    """The configuration of case ``name``; a CUDA case takes the test card's ``profile``."""
    path, overrides, kind = CASES[name]
    head = [] if kind == "cpu" else [f"profile={profile or 'rtx3060'}"]
    return load_config(path, overrides=[*head, *overrides, *(extra or [])])


# --------------------------------------------------------------------------------------------------
# Digests
# --------------------------------------------------------------------------------------------------


def _tensor_bytes(t: Tensor) -> bytes:
    flat = t.detach().to("cpu").contiguous().reshape(-1)
    return bytes(flat.view(torch.uint8).numpy().tobytes())


def _walk(obj: Any, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
    """Every tensor in nested dicts, lists and tuples, with a path, in a deterministic order."""
    if isinstance(obj, Tensor):
        yield prefix, obj
    elif isinstance(obj, Mapping):
        for key in sorted(obj, key=str):
            yield from _walk(obj[key], f"{prefix}/{key}")
    elif isinstance(obj, list | tuple):
        for i, value in enumerate(obj):
            yield from _walk(value, f"{prefix}/{i}")
    elif hasattr(obj, "_fields"):  # a NamedTuple-like state
        yield from _walk(obj._asdict(), prefix)
    elif hasattr(obj, "__dataclass_fields__"):  # CellGraphState and friends
        yield from _walk({k: getattr(obj, k) for k in obj.__dataclass_fields__}, prefix)


def digest(obj: Any) -> str:
    """SHA-256 over every tensor of ``obj`` (path, dtype, shape and bytes)."""
    h = hashlib.sha256()
    for path, t in _walk(obj):
        h.update(f"{path}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(_tensor_bytes(t))
    return h.hexdigest()


def _grads(model: nn.Module) -> dict[str, Any]:
    return {
        name: (p.grad if p.grad is not None else torch.zeros(0))
        for name, p in model.named_parameters()
    }


def _params(model: nn.Module) -> dict[str, Tensor]:
    return {name: p.detach() for name, p in model.named_parameters()}


def fingerprint(device: str) -> dict[str, str]:
    """What a bitwise comparison depends on: torch, the CPU kernels' ISA, the card."""
    out = {
        "torch": torch.__version__,
        "cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
        "machine": platform.machine(),
        "device": device,
    }
    if device == "cuda":
        out["cuda"] = str(torch.version.cuda)
        out["card"] = torch.cuda.get_device_name(0)
    return out


def state_dict_layout(model: nn.Module) -> list[list[Any]]:
    """``[key, shape, dtype]`` of every state-dict entry, in order (``[key, None, <type>]`` for
    an extra state that is not a tensor)."""
    return [
        [k, list(v.shape), str(v.dtype)] if isinstance(v, Tensor) else [k, None, type(v).__name__]
        for k, v in model.state_dict().items()
    ]


# --------------------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _max_block(n: int | None) -> Iterator[None]:
    if n is None:
        yield
        return
    from hsn.train import compile as compile_mod

    old = compile_mod.MAX_BLOCK
    compile_mod.MAX_BLOCK = n
    try:
        yield
    finally:
        compile_mod.MAX_BLOCK = old


def run_case(
    name: str, profile: str | None = None, extra: list[str] | None = None
) -> dict[str, Any]:
    """Train case ``name`` for :data:`STEPS` steps and return its record (module docstring)."""
    cfg = case_config(name, profile, extra)
    kind = CASES[name][2]
    ctx: DeviceContext = resolve(cfg.hardware)
    if ctx.device.type != kind:
        raise RuntimeError(f"case {name} needs a {kind} device, the profile resolved {ctx.device}")
    train_ds, dev_ds = datasets()
    seed_everything(cfg.seed)
    model = build_model(cfg)
    steps: list[dict[str, Any]] = []

    def record(trainer: Any) -> None:
        last = trainer_history[-1]
        steps.append(
            {
                "step": int(last.step),
                "loss": float(last.loss).hex(),
                "grad_norm": float(last.grad_norm).hex(),
                "lr": float(last.lr).hex(),
                "grads": digest(_grads(trainer.model)),
                "params": digest(_params(trainer.model)),
            }
        )

    trainer_history: list[Any] = []

    def callback(trainer: Any, history: Any) -> bool:
        trainer_history[:] = history.steps
        record(trainer)
        return False

    with (
        tempfile.TemporaryDirectory() as tmp,
        _max_block(CUDA_MAX_BLOCK if kind == "cuda" else None),
    ):
        trainer = make_trainer(model, cfg, ctx, run_dir=Path(tmp) / "run")
        history = trainer.fit(train_ds, dev_ds, STEPS, callback=callback)
        trainer_history[:] = history.steps  # the last step: the callback is not called there
        record(trainer)
        checkpoint = torch.load(history.checkpoints[-1], map_location="cpu", weights_only=True)
    carried = getattr(trainer, "_carried_state", None)
    return {
        "steps": steps,
        "model": digest(checkpoint["model"]),
        "optimizer": digest(checkpoint["optimizer"]),
        "carried_checkpoint": digest(checkpoint["extra"].get("carried_state")),
        "carried": digest(carried) if carried is not None else None,
        "last_lr": [float(x).hex() for x in checkpoint["scheduler"]["_last_lr"]],
        "groups": [
            {k: v for k, v in g.items() if k != "params"} | {"n_params": len(g["params"])}
            for g in checkpoint["optimizer"]["param_groups"]
        ],
    }


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def capture(device: str, repeats: int, profile: str | None) -> dict[str, Any]:
    import hsn

    names = [n for n, (_, _, kind) in CASES.items() if kind == device]
    runs = [{name: run_case(name, profile) for name in names} for _ in range(repeats)]
    out: dict[str, Any] = {
        "about": "P1.13 golden runs of the defaults (tests/p1_13_support.py)",
        "hsn_file": str(Path(hsn.__file__).resolve()),
        "git_rev_of_capture_tree": _git_rev(),
        "fingerprint": fingerprint(device),
        "steps": STEPS,
        "cuda_max_block": CUDA_MAX_BLOCK if device == "cuda" else None,
        "cases": runs[0],
        "reproducible": all(r == runs[0] for r in runs[1:]),
        "repeats": repeats,
    }
    if device == "cpu":
        layouts: dict[str, Any] = {}
        for label, overrides in (
            ("ref_a1_dense", []),
            ("ref_a1_topk", ["model.routing=topk", "train.gate_mode=ste"]),
        ):
            cfg = load_config(REF_A1, overrides=overrides)
            seed_everything(cfg.seed)
            layouts[label] = state_dict_layout(build_model(cfg))
        out["state_dict_layouts"] = layouts
    if repeats > 1 and not out["reproducible"]:
        out["all_runs"] = runs
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("capture")
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--profile", default=None, help="CUDA profile (default: tests/cards.py)")
    args = ap.parse_args(argv)
    profile = args.profile
    if args.device == "cuda" and profile is None:
        from cards import cuda_profile

        profile = cuda_profile()
    result = capture(args.device, args.repeats, profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: reproducible={result['reproducible']} from {result['hsn_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
