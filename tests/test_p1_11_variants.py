"""P1.11 tests: the variant registry moved from the P1.10 audit (``scripts/eff_overfit.py``).

Every variant installs on a tiny HSN, keeps ``param_report``'s honesty check balanced, runs
``forward`` with and without ``capture``; the cell-identity variants break the dense cell symmetry;
the GRU-cell variants report a NaN gate (no gate quantiles); retired variants, unknown names and
variants on a baseline raise.
"""

from __future__ import annotations

import pytest
import torch
from hsn.data.dataset import ByteDataset
from hsn.device import seed_everything
from hsn.experiments import variants as vr
from hsn.experiments.diagnostics import trained_state
from hsn.model.hsn import HSN
from hsn.runner import build_model
from p1_11_support import cpu_ctx, docs, tiny_gru_config, tiny_hsn_config

HSN_VARIANTS = [v for v in vr.VARIANTS if v != "none"]


def _hsn() -> HSN:
    seed_everything(0)
    model = build_model(tiny_hsn_config())
    assert isinstance(model, HSN)
    return model


@pytest.mark.parametrize("variant", HSN_VARIANTS)
def test_p1_11_variants_install(variant: str) -> None:
    model = _hsn()
    base = model.param_report()["total_trainable"]
    record = vr.install_variant(model, variant, seed=0)
    assert record["variant"] == variant
    report = model.param_report()  # raises when a parameter is unaccounted for
    if variant in ("pdg", "decoder_fc1_fan_in", "cell_offset"):
        assert report["total_trainable"] == base
    d, N = model.cell.d, model.cell.n_cells
    if variant in ("cell_embed", "cell_embed_linear_head"):
        assert record["extra_params"] == N * 4 * d and record["offset_std"] == 0.5
    if variant.startswith("gru_cell"):
        assert record["extra_params"] == sum(p.numel() for p in model.cell.gru_update.parameters())
    B, T = 2, 6
    cfg = model.cfg
    tokens = torch.randint(0, cfg.vocab_size, (B, T))
    ctx = torch.randint(0, cfg.n_context, (B, T))
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    logits, final, _ = model(tokens, ctx, model.init_state(B), reset)
    assert logits.shape == (B, T, cfg.vocab_size) and torch.isfinite(logits).all()
    spread = float((final.h - final.h[:, :1]).detach().abs().amax())
    if variant in vr.OFFSET_VARIANTS:
        assert spread > 0.0, "a per-cell input breaks the zero-state symmetry"
    elif variant in ("pdg", "linear_head", "decoder_fc1_fan_in"):
        assert spread == 0.0
    stats = trained_state(
        model, tiny_hsn_config(), cpu_ctx(), ByteDataset(docs("delayed", "dev", 4)), 4
    )
    assert ("gate/p50" in stats) == (not variant.startswith("gru_cell"))


def test_p1_11_variant_registry_rejects() -> None:
    with pytest.raises(NotImplementedError, match="retired"):
        vr.install_variant(_hsn(), "gru_cell_direct_input", seed=0)
    with pytest.raises(ValueError, match="unknown variant"):
        vr.variant_params("no_such_variant")
    with pytest.raises(ValueError, match="takes no parameters"):
        vr.variant_params("pdg", {"offset_std": 0.5})
    with pytest.raises(ValueError, match="positive"):
        vr.variant_params("cell_offset", {"offset_std": 0.0})
    assert vr.variant_params("cell_embed") == {"offset_std": 0.5}
    assert vr.variant_params("cell_offset", {"offset_std": 1}) == {"offset_std": 1.0}
    gru = build_model(tiny_gru_config())
    assert vr.install_variant(gru, "none", seed=0) == {"variant": "none"}
    with pytest.raises(TypeError, match="needs an HSN"):
        vr.install_variant(gru, "cell_offset", seed=0)
    with pytest.raises(TypeError, match="needs an HSN"):
        vr.install_variant(gru, "pdg", seed=0)
    with pytest.raises(ValueError, match="for the baselines"):
        vr.install_variant(_hsn(), "none", seed=0)


def test_p1_11_offset_variants_are_seeded() -> None:
    a, b, c = _hsn(), _hsn(), _hsn()
    vr.install_variant(a, "cell_offset", seed=3)
    vr.install_variant(b, "cell_offset", seed=3)
    vr.install_variant(c, "cell_offset", seed=4)
    assert isinstance(a.cell.norm, vr.CellIdentity)
    assert torch.equal(a.cell.norm.offset, b.cell.norm.offset)
    assert not torch.equal(a.cell.norm.offset, c.cell.norm.offset)
