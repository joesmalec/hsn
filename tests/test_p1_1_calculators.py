"""P1.1 tests: ref_a1 / tiny_gradcheck configs and the Appendix C closed-form calculators."""

import subprocess
import sys
from pathlib import Path

import pytest
from hsn.config import ModelConfig, load_config
from hsn.train.memory_plan import (
    SCORER_SCALAR_FEATURES,
    TRAIN_STATE_BYTES_PER_PARAM,
    MemoryPlan,
    activations_per_token,
    expected_param_count,
    format_bytes,
    macs_per_token,
    memory_plan,
    value_bytes_for,
)

REPO = Path(__file__).resolve().parents[1]
REF_A1 = REPO / "configs" / "ref_a1.yaml"
TINY = REPO / "configs" / "tiny_gradcheck.yaml"
SCRIPT = REPO / "scripts" / "count_params.py"

MIB = 1024**2
KIB = 1024

# Appendix C.1, row by row.
APPENDIX_C1 = {
    "byte_emb": 16_384,
    "ctx_emb": 1_024,
    "encoder": 66_112,
    "predictor": 66_112,
    "shared": 131_648,
    "units": 1_315_328,
    "w_out": 4_096,
    "v_msg": 4_096,
    "gate": 16_448,
    "cell_norm": 256,  # P1.5a (amendment A1): the RMSNorm gain over the 4d cell input
    "cand_norm": 0,  # P1.13 (amendment A4): the candidate's RMSNorm gain, off by default
    "scorer": 1_608,
    "halting": 129,
    "decoder": 197_376,
    "total": 1_820_617,  # 1,820,361 before amendment A1
}


# --------------------------------------------------------------------------------------------------
# Tests named by the chunk
# --------------------------------------------------------------------------------------------------


def test_p1_1_ref_a1_total_is_1_820_617() -> None:
    """1,820,361 Appendix C parameters plus the 256 of amendment A1's ``cell_norm`` row."""
    cfg = load_config(REF_A1)
    assert cfg.profile == "cpu" and cfg.run_id == "ref_a1"
    # The schema defaults are the Appendix C configuration; the YAML spells it out and agrees.
    assert cfg.model == ModelConfig()
    assert 3 * cfg.model.d + SCORER_SCALAR_FEATURES == 200  # scorer input width (Appendix H.2)

    counts = expected_param_count(cfg.model)
    assert list(counts) == list(APPENDIX_C1)  # table order, "total" last
    for component, expected in APPENDIX_C1.items():
        assert counts[component] == expected, component
    assert counts["total"] == 1_820_617
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total")
    assert counts["units"] == 8 * 164_416  # Appendix E.2 unit size


def test_p1_1_macs_match_appendix_c2() -> None:
    cfg = load_config(REF_A1)
    macs = macs_per_token(cfg.model)
    assert macs == {
        "shared": 461_248,
        "per_cell": 201_344,
        "selected": 805_376,
        "scorer": 51_456,
        "sparse_total": 1_318_080,
        "dense_total": 6_955_712,
    }
    assert macs["per_cell"] == 164_416 + 4_096 + 4 * 4_096 + 16_448  # the C.2 breakdown
    assert macs["sparse_total"] == macs["shared"] + 4 * macs["per_cell"] + macs["scorer"]
    assert macs["dense_total"] == macs["shared"] + 32 * macs["per_cell"] + macs["scorer"]

    # Two-layer scorer, 128 hidden, over 32 cells: 32 × 26,760 = 856,320 (through the loader).
    mlp = load_config(REF_A1, overrides=["model.scorer=mlp"])
    assert expected_param_count(mlp.model)["scorer"] == 26_760
    mlp_macs = macs_per_token(mlp.model)
    assert mlp_macs["scorer"] == 856_320
    assert mlp_macs["shared"] == macs["shared"] and mlp_macs["per_cell"] == macs["per_cell"]
    assert mlp_macs["sparse_total"] == 461_248 + 805_376 + 856_320
    assert round(100 * mlp_macs["sparse_total"] / macs["dense_total"]) == 31  # "31 % of dense"


def test_p1_1_memory_matches_appendix_c3() -> None:
    cfg = load_config(REF_A1)
    assert (cfg.train.batch, cfg.train.unroll, cfg.train.precision) == (4, 256, "bf16")
    value_bytes = value_bytes_for(cfg.train.precision)
    assert value_bytes == 2

    plan = memory_plan(cfg.model, cfg.train.batch, cfg.train.unroll, value_bytes)
    assert isinstance(plan, MemoryPlan)
    assert plan.train_state_bytes == 1_820_617 * 16 == 29_129_872  # A1: 1,820,361 + 256
    assert format_bytes(plan.train_state_bytes) == "27.8 MiB"
    assert plan.rec_state_bytes == 4 * 32 * 64 * 2 == 16 * KIB
    assert format_bytes(plan.rec_state_bytes) == "16 KiB"
    assert plan.saved_sparse_bytes == 4 * 256 * (2_560 + 4 * 960) * 2 == 12.5 * MIB
    assert format_bytes(plan.saved_sparse_bytes) == "12.5 MiB"
    assert plan.saved_dense_bytes == 4 * 256 * (2_560 + 32 * 960) * 2 == 65 * MIB
    assert format_bytes(plan.saved_dense_bytes) == "65 MiB"
    assert plan.overhead_bytes == 0  # the per-profile estimate arrives in P4.2
    assert plan.total_sparse == plan.train_state_bytes + plan.rec_state_bytes + 12.5 * MIB
    assert plan.total_dense == plan.train_state_bytes + plan.rec_state_bytes + 65 * MIB
    assert activations_per_token(cfg.model) == {"shared": 2_560, "per_cell": 960}  # not 2,816

    # "Recurrent state, scaled example": 16 × 64 × 256 × 2 = 512 KiB (N=64, d=256, batch 16).
    scaled = ModelConfig(n_cells=64, d=256, context_dim=256)
    assert memory_plan(scaled, 16, 256, 2).rec_state_bytes == 512 * KIB
    assert format_bytes(512 * KIB) == "512 KiB"

    # value_bytes scales every value-sized row but not the training state.
    fp32 = memory_plan(cfg.model, 4, 256, value_bytes=4)
    assert fp32.train_state_bytes == plan.train_state_bytes
    assert fp32.rec_state_bytes == 2 * plan.rec_state_bytes
    assert fp32.saved_sparse_bytes == 2 * plan.saved_sparse_bytes
    assert fp32.saved_dense_bytes == 2 * plan.saved_dense_bytes
    assert value_bytes_for("fp64") == 8 and value_bytes_for("fp16") == 2


# --------------------------------------------------------------------------------------------------
# Extras: the tiny config, generalisation to Appendix E.2, the script, and rejections
# --------------------------------------------------------------------------------------------------


def test_p1_1_tiny_gradcheck_loads() -> None:
    cfg = load_config(TINY)
    m = cfg.model
    assert (m.vocab_size, m.d, m.n_cells, m.n_units, m.k_select) == (16, 4, 3, 2, 2)
    assert (m.r_max, m.in_degree, m.hidden, m.n_context, m.context_dim) == (1, 2, 8, 4, 4)
    assert cfg.train.precision == "fp64" and cfg.profile == "cpu"
    counts = expected_param_count(m)
    # By hand: 64 + 16 + 76 + 76 + 140 + 2 × 172 + 16 + 16 + 68 + 16 (A1 cell_norm, 4d) + 42 + 9
    # + 216 = 1,099 (1,083 before amendment A1).
    assert counts == {
        "byte_emb": 64,
        "ctx_emb": 16,
        "encoder": 76,
        "predictor": 76,
        "shared": 140,
        "units": 344,
        "w_out": 16,
        "v_msg": 16,
        "gate": 68,
        "cell_norm": 16,
        "cand_norm": 0,  # P1.13: off by default
        "scorer": 42,
        "halting": 9,
        "decoder": 216,
        "total": 1_099,
    }
    assert activations_per_token(m) == {"shared": 64, "per_cell": 36}
    macs = macs_per_token(m)
    assert macs["per_cell"] == 172 + 16 + 2 * 16 + 68 == 288
    assert macs["sparse_total"] == (76 + 76 + 140 + 216) + 2 * 288 + 3 * 42
    plan = memory_plan(m, cfg.train.batch, cfg.train.unroll, value_bytes_for("fp64"))
    assert plan.train_state_bytes == 1_099 * TRAIN_STATE_BYTES_PER_PARAM
    assert plan.rec_state_bytes == 2 * 3 * 4 * 8
    assert plan.saved_sparse_bytes == 2 * 8 * (64 + 2 * 36) * 8


def test_p1_1_closed_forms_generalise_to_appendix_e2() -> None:
    # scale_20m: d=128, N=48, M=28, K=6, R=2, H=1024 -> 20,104,445 (Appendix E.2 breakdown).
    m = ModelConfig(
        d=128, n_cells=48, n_units=28, k_select=6, r_max=2, hidden=1024, context_dim=128
    )
    counts = expected_param_count(m)
    assert counts["byte_emb"] == 32_768 and counts["ctx_emb"] == 2_048
    assert counts["encoder"] == counts["predictor"] == 263_296
    assert counts["shared"] == 525_440
    assert counts["units"] == 28 * 656_512 == 18_382_336
    assert counts["w_out"] == counts["v_msg"] == 16_384
    assert counts["gate"] == 65_664
    assert counts["scorer"] == 11_004  # 392 -> 28
    assert counts["halting"] == 257
    assert counts["decoder"] == 525_568
    assert counts["cell_norm"] == 512  # amendment A1: 4d
    assert counts["total"] == 20_104_957  # Appendix E.2's 20,104,445 + 512
    # Appendix E.3: per-token activations 4,864 / 1,920; saved sparse 109.0 MiB at B=8, T=256.
    assert activations_per_token(m) == {"shared": 4_864, "per_cell": 1_920}
    plan = memory_plan(m, 8, 256, 2)
    assert plan.saved_sparse_bytes == 8 * 256 * (4_864 + 2 * 6 * 1_920) * 2
    assert format_bytes(plan.saved_sparse_bytes) == "109.0 MiB".replace(".0", "")
    assert plan.rec_state_bytes == 96 * KIB and format_bytes(plan.rec_state_bytes) == "96 KiB"
    # Appendix E.3 prints the training state as 306.7 MiB; 20,104,445 × 16 B = 306.77 MiB, which
    # the C.3 convention (27.776 -> "27.8 MiB") rounds to 306.8. Left to P4.2 to settle.
    assert plan.train_state_bytes == 20_104_957 * 16
    assert plan.saved_dense_bytes == 8 * 256 * (4_864 + 2 * 48 * 1_920) * 2  # 739.0 MiB
    assert format_bytes(plan.saved_dense_bytes) == "739 MiB"

    # scale_100m: d=256, N=64, M=36, K=8, R=2, H=2048 -> 100,722,757; B=16, T=512 (Appendix E.3).
    big = ModelConfig(
        d=256, n_cells=64, n_units=36, k_select=8, r_max=2, hidden=2048, context_dim=256
    )
    assert expected_param_count(big)["total"] == 100_723_781  # Appendix E's 100,722,757 + 1,024
    assert expected_param_count(big)["units"] == 36 * 2_623_744
    assert activations_per_token(big) == {"shared": 9_472, "per_cell": 3_840}
    big_plan = memory_plan(big, 16, 512, 2)
    assert format_bytes(big_plan.train_state_bytes) == "1.50 GiB"
    assert format_bytes(big_plan.rec_state_bytes) == "512 KiB"
    assert format_bytes(big_plan.saved_sparse_bytes) == "1.08 GiB"
    assert format_bytes(big_plan.saved_dense_bytes) == "7.64 GiB"


def test_p1_1_count_params_script_prints_appendix_c() -> None:
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", "configs/ref_a1.yaml"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    out = run.stdout
    for figure in (
        "| Cell input RMSNorm gain (amendment A1) | 256 | 256 |",
        "| **Total** |  | **1,820,617** |",
        "| Router scorer, linear | 200 → 8 | 1,608 |",
        "| Shared per-token work E + P + S + D | 461,248 | 35 % |",
        "| 201,344 | 15 % |",
        "| 805,376 | 61 % |",
        "| Linear scorer over all 32 cells (32 × 1,608) | 51,456 | 4 % |",
        "| **Sparse total, K = 4** | **1,318,080** | 100 % |",
        "| 6,955,712 | 528 % |",
        "| Two-layer scorer, 128 hidden, over 32 cells (32 × 26,760) | 856,320 | would lift sparse "
        "to 31 % of dense |",
        "| 1,820,617 × 16 | 27.8 MiB |",
        "| 4 × 32 × 64 × 2 | 16 KiB |",
        "| 16 × 64 × 256 × 2 | 512 KiB |",
        "| 4 × 256 × (2,560 + 4 × 960) × 2 | 12.5 MiB |",
        "| 4 × 256 × (2,560 + 32 × 960) × 2 | 65 MiB |",
        "Per-cell saved activations 960 = v (256) + unit hidden (512) + c, b, h (192).",
        "Shared activations 2,560 = z, ẑ, ε, u (256) + 4 hidden layers (2,048) + logits (256).",
    ):
        assert figure in out, figure
    for row in APPENDIX_C1.values():
        assert f"{row:,}" in out
    assert "2,816" not in out  # the paper's typo never appears

    # Dotted overrides reach the loader: the mlp scorer swaps the C.1 row and the C.2 comparison.
    mlp = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", "configs/ref_a1.yaml", "model.scorer=mlp"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert mlp.returncode == 0, mlp.stderr
    assert "| Router scorer, mlp | 200 → 128 → 8 | 26,760 |" in mlp.stdout
    assert "| Linear scorer over 32 cells (32 × 1,608) | 51,456 |" in mlp.stdout

    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", "configs/does_not_exist.yaml"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2 and "does_not_exist.yaml" in missing.stderr


def test_p1_1_counts_the_shared_u_ablation() -> None:
    """``model.ablate_shared_u`` (P2.13's "remove shared path": remove ``u`` from ``v``): the cell
    input is ``[ε; h_i; m_i]`` of width ``3d``, and the closed forms count the three rows that
    read it (units, gate, cell_norm) at that width; everything else, the scorer's ``3d + 8``
    features and the shared activations (``u`` is still computed) are unchanged."""
    base, ablated = ModelConfig(), ModelConfig(ablate_shared_u=True)
    assert base.cell_input_width == 256 and ablated.cell_input_width == 192
    assert "cell_input_width" not in ablated.model_dump()  # derived, never dumped or hashed
    full, cut = expected_param_count(base), expected_param_count(ablated)
    assert list(cut) == list(full)  # the Appendix C.1 rows, in order
    assert {row for row in full if full[row] != cut[row]} == {
        "units",
        "gate",
        "cell_norm",
        "total",
    }
    assert cut["units"] == 8 * (192 * 512 + 512 + 512 * 64 + 64) == 1_053_184
    assert cut["gate"] == 192 * 64 + 64 == 12_352
    assert cut["cell_norm"] == 192
    # one d less of input: M·d·H unit weights, d·d gate weights and d gains fewer
    assert full["total"] - cut["total"] == 8 * 64 * 512 + 64 * 64 + 64 == 266_304
    assert cut["total"] == 1_554_313
    no_norm = ModelConfig(ablate_shared_u=True, cell_norm="none")
    assert expected_param_count(no_norm)["cell_norm"] == 0

    macs = macs_per_token(ablated)
    unit = cut["units"] // 8
    assert unit == 131_648
    assert macs["per_cell"] == unit + 4_096 + 4 * 4_096 + 12_352 == 164_480
    assert macs["shared"] == macs_per_token(base)["shared"] == 461_248  # S still runs
    assert macs["scorer"] == macs_per_token(base)["scorer"] == 51_456  # 3d + 8 features, with u
    assert macs["sparse_total"] == 461_248 + 4 * 164_480 + 51_456 == 1_170_624
    assert macs["dense_total"] == 461_248 + 32 * 164_480 + 51_456 == 5_776_064

    acts = activations_per_token(ablated)
    assert acts == {"shared": 2_560, "per_cell": 192 + 512 + 192}  # v is 3d wide; u is saved
    plan, ref = memory_plan(ablated, 4, 256), memory_plan(base, 4, 256)
    assert plan.train_state_bytes == TRAIN_STATE_BYTES_PER_PARAM * 1_554_313
    assert plan.rec_state_bytes == ref.rec_state_bytes  # the state is unchanged
    assert plan.saved_sparse_bytes == 4 * 256 * (2_560 + 4 * 896) * 2
    assert plan.saved_dense_bytes == 4 * 256 * (2_560 + 32 * 896) * 2
    assert ref.saved_sparse_bytes - plan.saved_sparse_bytes == 4 * 256 * 4 * 64 * 2

    # the tiny gradcheck configuration and the script's tables
    tiny = load_config(TINY, overrides=["model.ablate_shared_u=true"]).model
    d, h, m_units = tiny.d, tiny.hidden, tiny.n_units
    tiny_full = expected_param_count(load_config(TINY).model)
    assert tiny_full["total"] - expected_param_count(tiny)["total"] == m_units * d * h + d * d + d
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            "configs/ref_a1.yaml",
            "model.ablate_shared_u=true",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    for figure in (
        "Ablation model.ablate_shared_u: the cell input is [ε; h_i; m_i] of width 192",
        "| Capability units F(1)–F(8) | 8 × (192 → 512 → 64) | 1,053,184 |",
        "| Gate B | 192 → 64 | 12,352 |",
        "| Cell input RMSNorm gain (amendment A1) | 192 | 192 |",
        "| Router scorer, linear | 200 → 8 | 1,608 |",
        "| **Total** |  | **1,554,313** |",
        "| **Sparse total, K = 4** | **1,170,624** | 100 % |",
        "| 4 × 256 × (2,560 + 4 × 896) × 2 |",
        "Per-cell saved activations 896 = v (192) + unit hidden (512) + c, b, h (192).",
        "Shared activations 2,560 = z, ẑ, ε, u (256) + 4 hidden layers (2,048) + logits (256).",
    ):
        assert figure in run.stdout, figure


def test_p1_1_rejects_unsupported() -> None:
    m = ModelConfig()
    for batch, unroll, value_bytes in ((0, 256, 2), (4, 0, 2), (4, 256, 0)):
        with pytest.raises(ValueError, match=">= 1"):
            memory_plan(m, batch, unroll, value_bytes)
    with pytest.raises(ValueError, match="unknown precision"):
        value_bytes_for("int8")
    with pytest.raises(ValueError, match=">= 0"):
        format_bytes(-1)

    assert format_bytes(0) == "0 B" and format_bytes(192) == "192 B"
    assert format_bytes(16 * KIB) == "16 KiB"
    assert format_bytes(1_500) == "1.5 KiB"
    assert format_bytes(29_125_776) == "27.8 MiB"
    assert format_bytes(65 * MIB) == "65 MiB"
    assert format_bytes(round(1.5 * 1024**3)) == "1.50 GiB"
    assert format_bytes(1024**3 - 1) == "1024.0 MiB".replace(".0", "")

    # in_degree=0 (propagation ablation) needs no special case: zero message MACs, V still counted.
    no_edges = ModelConfig(in_degree=0)
    assert expected_param_count(no_edges)["v_msg"] == 4_096
    assert macs_per_token(no_edges)["per_cell"] == 201_344 - 4 * 4_096
