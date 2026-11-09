"""A5 counterfactual probes preserve the previous-byte address across window boundaries."""

from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import load_config
from hsn.device import seed_everything
from hsn.model.hsn import HSN
from hsn.train.probes import probe_window


def _case() -> tuple[HSN, dict[str, Any]]:
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs/ref_a1.yaml",
        overrides=[
            "model.d=8",
            "model.context_dim=8",
            "model.n_cells=4",
            "model.n_units=2",
            "model.in_degree=2",
            "model.k_select=2",
            "model.hidden=16",
            "model.routing=topk",
            "train.gate_mode=ste",
            "model.key_address=true",
            "model.selected_readout=true",
        ],
    )
    seed_everything(3)
    model = HSN.from_config(cfg).eval()
    rng = torch.Generator().manual_seed(17)
    reset = torch.zeros(2, 12, dtype=torch.bool)
    reset[:, 0] = True
    return model, {
        "tokens": torch.randint(0, 256, (2, 12), generator=rng),
        "context": torch.zeros(2, 12, dtype=torch.int64),
        "targets": torch.randint(0, 256, (2, 12), generator=rng),
        "loss_mask": torch.ones(2, 12, dtype=torch.bool),
        "reset": reset,
        "split": "train",
    }


def test_a5_probe_matches_across_window_boundary() -> None:
    model, window = _case()
    state = model.init_state(2)
    full = probe_window(
        model, state, window, 7, n_probe=1, horizon=3, generator=torch.Generator().manual_seed(19)
    )
    cut = 4
    with torch.no_grad():
        _, carried, _ = model(
            window["tokens"][:, :cut],
            window["context"][:, :cut],
            state,
            window["reset"][:, :cut],
        )
    tail = {k: v[:, cut:] if isinstance(v, torch.Tensor) else v for k, v in window.items()}
    tail["prev_tokens"] = window["tokens"][:, cut - 1]
    split = probe_window(
        model,
        carried,
        tail,
        7 - cut,
        n_probe=1,
        horizon=3,
        generator=torch.Generator().manual_seed(19),
    )
    torch.testing.assert_close(full.features, split.features, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full.delta, split.delta, rtol=1e-5, atol=1e-6)
    assert torch.equal(full.candidates, split.candidates)
    assert torch.equal(full.valid, split.valid)


def test_a5_probe_reset_uses_bos_instead_of_previous_document() -> None:
    model, window = _case()
    results = []
    for previous in (torch.tensor([64, 65]), torch.tensor([112, 113])):
        results.append(
            probe_window(
                model,
                model.init_state(2),
                {**window, "prev_tokens": previous},
                0,
                n_probe=1,
                horizon=3,
                generator=torch.Generator().manual_seed(19),
            )
        )
    assert torch.equal(results[0].features, results[1].features)
    assert torch.equal(results[0].delta, results[1].delta)


def test_a5_probe_refuses_invalid_previous_byte_shape() -> None:
    model, window = _case()
    with pytest.raises(ValueError, match="prev_tokens"):
        probe_window(
            model,
            model.init_state(2),
            {**window, "prev_tokens": torch.tensor([64])},
            0,
            n_probe=1,
            horizon=3,
            generator=torch.Generator().manual_seed(19),
        )
