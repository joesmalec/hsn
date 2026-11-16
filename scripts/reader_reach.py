#!/usr/bin/env python3
"""Score a trained stateless model by window position and say how far its attention reads.

For the text-gap question of 15 Nov 2026 (``docs/hsn_t.md``, "The reader's positions"): with
documents aligned to windows the position in the window is the number of bytes a stateless
model can see, so the loss by position says how well it uses near and far bytes. For each run
directory (a ``config.resolved.yaml`` and a ``step_*.pt`` checkpoint of a Transformer baseline,
a reader alone or an HSN-T without its memory) the script prints the dev loss in bits per byte
by window position and, with ``--attention``, the share of each attention head's weight that
lies ``--far`` or more positions back, for the queries at window positions ``--from`` and
later. CPU, float32, no gradient; nothing is written. A reader with ``model.reader_carry``
carries its cache from window to window of a stream for the loss (its attention's reach is
not measured).

Usage::

    python scripts/reader_reach.py RUN_DIR [RUN_DIR ...] \\
        --manifest data/tinystories_v1/manifest.json [--docs 300] [--attention]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from hsn.baselines.reader_lm import ReaderLM
from hsn.baselines.transformer import TransformerBaseline
from hsn.config import RunConfig, load_config
from hsn.data.dataset import ByteDataset
from hsn.model.hsn_t import HSNTNoMemory
from hsn.model.reader import _rotate
from hsn.runner import build_model
from hsn.train.checkpoint import latest_checkpoint
from hsn.train.loop import StreamLoader
from torch import Tensor, nn

Batch = dict[str, Tensor]
#: Window positions ``[lo, hi)`` of the printed rows, clipped to the window.
BUCKETS: tuple[tuple[int, int], ...] = ((0, 8), (8, 32), (32, 64), (64, 128), (128, 256))
STATELESS = (TransformerBaseline, ReaderLM, HSNTNoMemory)


def attention_blocks(model: nn.Module) -> list[nn.Module]:
    """The attention blocks of a supported model, first to last."""
    if isinstance(model, TransformerBaseline):
        return list(model.blocks)
    if isinstance(model, ReaderLM | HSNTNoMemory):
        return list(model.reader.blocks)
    raise TypeError(
        f"{type(model).__name__} is not a stateless model this script reads "
        f"({', '.join(c.__name__ for c in STATELESS)})"
    )


def position_loss(model: nn.Module, windows: Iterable[Batch]) -> tuple[Tensor, Tensor]:
    """``(nll, count)``, both ``[T]`` float64: the summed next-byte loss in nats and the number
    of loss positions at every window position, over ``windows``."""
    attention_blocks(model)  # refuses a model with state
    nll: Tensor | None = None
    count: Tensor | None = None
    state = None  # model.reader_carry: the reader's cache, carried as the trainer carries it
    with torch.no_grad():
        for i, batch in enumerate(windows):
            if i == 0:
                state = model.init_state(int(batch["tokens"].shape[0]), "cpu")  # type: ignore[operator]
            logits, state, _ = model(batch["tokens"], batch["context"], state, batch["reset"])
            mask = batch["loss_mask"]
            targets = batch["targets"].clamp(0, logits.shape[-1] - 1)  # PAD where no loss counts
            loss = -F.log_softmax(logits.float(), dim=-1).gather(-1, targets[..., None])[..., 0]
            if nll is None or count is None:
                nll = torch.zeros(logits.shape[1], dtype=torch.float64)
                count = torch.zeros(logits.shape[1], dtype=torch.float64)
            nll += (loss * mask).sum(0).double()
            count += mask.sum(0).double()
    if nll is None or count is None:
        raise ValueError("position_loss needs at least one window")
    return nll, count


def attention_reach(
    model: nn.Module, windows: Iterable[Batch], *, far: int = 32, first: int = 160
) -> Tensor:
    """``[blocks, heads]``: the share of each head's attention on keys ``far`` or more positions
    back, averaged over the loss positions at window position ``first`` or later.

    The blocks' own inputs are read by forward pre-hooks and the attention weights recomputed
    from them (the blocks call ``scaled_dot_product_attention``, which returns none)."""
    blocks = attention_blocks(model)
    if isinstance(model, ReaderLM | HSNTNoMemory) and model.reader.carry:
        raise ValueError(
            "model.reader_carry: the attention also reads the cache of the windows before, "
            "which this measure does not recompute; the loss by position is carried"
        )
    seen: list[tuple[Tensor, ...]] = []
    hooks = [
        block.register_forward_pre_hook(lambda _m, args: seen.append(args)) for block in blocks
    ]
    total: Tensor | None = None
    queries = 0
    try:
        with torch.no_grad():
            for batch in windows:
                seen.clear()
                model(batch["tokens"], batch["context"], None, batch["reset"])
                B, T = batch["tokens"].shape
                index = torch.arange(T)
                distance = index[:, None] - index[None, :]  # query position − key position
                chosen = batch["loss_mask"] & (index[None, :] >= first)
                shares = []
                for block, args in zip(blocks, seen, strict=True):
                    weights = _attention_weights(block, *args)  # [B, H, T, T]
                    beyond = (weights * (distance >= far)).sum(-1)  # [B, H, T]
                    shares.append(beyond.permute(0, 2, 1)[chosen].sum(0).double())
                step = torch.stack(shares)
                total = step if total is None else total + step
                queries += int(chosen.sum())
    finally:
        for hook in hooks:
            hook.remove()
    if total is None or queries == 0:
        raise ValueError(f"no loss position at window position {first} or later")
    return total / queries


def _attention_weights(
    block: nn.Module, x: Tensor, mask: Tensor, cos: Tensor | None = None, sin: Tensor | None = None
) -> Tensor:
    """The attention weights ``[B, H, T, T]`` a block computes from its input (the Transformer
    baseline's block takes ``(x, mask)``, the reader's ``(x, mask, cos, sin)``)."""
    B, T, w = x.shape
    heads = block.heads if hasattr(block, "heads") else block.n_heads
    q, k, _ = block.qkv(block.norm_attn(x)).split(w, dim=-1)
    q = q.view(B, T, heads, -1).transpose(1, 2)
    k = k.view(B, T, heads, -1).transpose(1, 2)
    if cos is not None and sin is not None:
        q, k = (
            _rotate(q, cos.to(q.dtype), sin.to(q.dtype)),
            _rotate(k, cos.to(k.dtype), sin.to(k.dtype)),
        )
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return scores.masked_fill(~mask, -math.inf).softmax(dim=-1)


def load_run(run_dir: Path) -> tuple[nn.Module, RunConfig, Path]:
    """The model of a run directory with its latest checkpoint's weights, on the CPU."""
    cfg = load_config(run_dir / "config.resolved.yaml")
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"{run_dir}: no step_*.pt checkpoint")
    model = build_model(cfg)
    attention_blocks(model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval(), cfg, checkpoint


def dev_windows(cfg: RunConfig, manifest: Path, docs: int, batch: int) -> list[Batch]:
    """The first ``docs`` dev documents of the run's tasks and variants, in the run's windows."""
    data = ByteDataset.load(manifest, "dev", cfg.data.tasks, cfg.data.variants, verify=False)
    if docs and docs < len(data):
        data = ByteDataset(data.examples[:docs])
    loader = StreamLoader(
        data,
        batch=batch,
        unroll=cfg.train.unroll,
        seed=0,
        shuffle=False,
        align_docs=cfg.data.align_docs,
        epochs=1,
    )
    return list(loader)


def report(name: str, model: nn.Module, windows: Sequence[Batch], args: argparse.Namespace) -> str:
    nll, count = position_loss(model, windows)
    T = int(nll.shape[0])
    bits = math.log(2.0)
    total = float(nll.sum() / count.sum()) / bits
    lines = [f"{name}: {total:.4f} bits per byte over {int(count.sum()):,} bytes"]
    for lo, hi in BUCKETS:
        lo, hi = min(lo, T), min(hi, T)
        if hi > lo and float(count[lo:hi].sum()) > 0:
            share = float(count[lo:hi].sum() / count.sum())
            value = float(nll[lo:hi].sum() / count[lo:hi].sum()) / bits
            lines.append(
                f"  positions {lo:3d} to {hi - 1:3d}  {share:6.1%} of the bytes  {value:.4f}"
            )
    if args.attention:
        reach = attention_reach(model, windows, far=args.far, first=args.first)
        lines.append(
            f"  share of attention {args.far} or more positions back, queries at positions "
            f"{args.first} and later (one row per block, one value per head):"
        )
        lines += [
            f"    block {i + 1}: " + "  ".join(f"{float(v):.3f}" for v in row)
            for i, row in enumerate(reach)
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--manifest", type=Path, required=True, help="the data set's manifest.json")
    parser.add_argument("--docs", type=int, default=300, help="dev documents read (0: all)")
    parser.add_argument("--batch", type=int, default=64, help="rows of a window")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--attention", action="store_true", help="also print the attention's reach")
    parser.add_argument("--far", type=int, default=32, help="the distance that counts as far")
    parser.add_argument("--from", dest="first", type=int, default=160, help="first query position")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    for run_dir in args.runs:
        model, cfg, checkpoint = load_run(run_dir)
        windows = dev_windows(cfg, args.manifest, args.docs, args.batch)
        print(report(f"{cfg.run_id} ({checkpoint.name})", model, windows, args), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
