"""A5 divergence dumps retain the byte from the start of the failing step."""

from pathlib import Path

import torch
from hsn.train.divergence import load_dump
from hsn.train.loop import History, Trainer
from p1_15_support import datasets, hsn_config, hsn_trainer


def test_a5_divergence_keeps_step_start_previous_byte(tmp_path: Path) -> None:
    cfg = hsn_config(
        3,
        "model.routing=topk",
        "train.gate_mode=ste",
        "model.key_address=true",
        "model.selected_readout=true",
        "train.unroll=8",
    )
    trainer = hsn_trainer(cfg, tmp_path)
    expected: torch.Tensor | None = None

    def remember(t: Trainer, history: History) -> bool:
        del t, history
        nonlocal expected
        if trainer.step == 1:
            assert trainer._prev_tokens is not None
            expected = trainer._prev_tokens.clone()
        return False

    def fail_second_step(t: Trainer) -> None:
        if t.step == 1:
            for p in t.model.parameters():
                if p.grad is not None:
                    p.grad.fill_(float("inf"))

    trainer.grad_hook = fail_second_step
    trainer.fit(*datasets(), 3, callback=remember)
    assert trainer.divergence is not None
    saved = load_dump(trainer.divergence["path"])
    assert saved["step"] == 1
    assert expected is not None
    assert torch.equal(saved["extra"]["prev_tokens"], expected)
    resumed = hsn_trainer(cfg, tmp_path / "resume")
    resumed.resume(trainer.divergence["path"])
    assert torch.equal(resumed._prev_tokens, expected)
