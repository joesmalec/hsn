"""Closed-form parameter, MAC and memory calculators for an HSN configuration (P1.1).

The three functions reproduce the Appendix C tables of ``HSN_Implementation_Plan.md`` from a
:class:`~hsn.config.ModelConfig` alone, with integer arithmetic and no tensors: C.1 parameters
(:func:`expected_param_count`), C.2 multiply-accumulates per input position (:func:`macs_per_token`)
and C.3 memory (:func:`memory_plan`). P1.2 and P1.5 check their *measured* parameter counts against
:func:`expected_param_count`; P4.2 extends :func:`memory_plan` with checkpointing modes and a
per-profile overhead estimate. ``scripts/count_params.py`` prints the tables.

Conventions (plan P1.1, Appendix C, Appendix H):

* Every linear layer costs ``in × out + out`` parameters (bias included) except ``W_out`` and the
  message projection ``V``, which have no bias. Biases count as MACs (Appendix H.3), so a linear
  layer's MACs per application equal its parameter count.
* The router scorer reads ``3d + 8`` features per cell: ``u``, ``ε`` and ``h_i`` (``d`` each) plus
  the eight scalars of the P1.5 layout (:data:`SCORER_SCALAR_FEATURES`, Appendix H.2).
  ``model.scorer_features.exclude`` zeroes features without removing them, so it changes no count.
* The per-token shared activation count is ``4d + 4H + V`` = 2,560 for the reference configuration,
  not the paper's 2,816 (Appendix H.1); MiB means 2^20 bytes.
* The closed forms fix the cell input ``v = [u; ε; h_i; m_i]`` at width ``4d``, so a configuration
  with ``model.ablate_shared_u`` raises instead of being counted with a guessed width (errors over
  silence); P1.5 defines that ablation's shapes.
* Learned edge weights ``a_ij`` are not in Appendix C and are not counted here (P1.4).
"""

from dataclasses import dataclass

from hsn.config import ModelConfig

__all__ = [
    "SCORER_SCALAR_FEATURES",
    "TRAIN_STATE_BYTES_PER_PARAM",
    "VALUE_BYTES",
    "MemoryPlan",
    "activations_per_token",
    "expected_param_count",
    "format_bytes",
    "macs_per_token",
    "memory_plan",
    "value_bytes_for",
]

#: Scalar router features appended to ``[u; ε; h_i]`` in the P1.5 scorer layout (Appendix H.2):
#: age, budget, round, context, ema_utility, ema_selected, position, constant 1. Scorer input
#: width is ``3 * d + SCORER_SCALAR_FEATURES`` (200 for the reference configuration).
SCORER_SCALAR_FEATURES: int = 8

#: Training state per trainable parameter (plan §2.10): fp16 weights 2 + grads 2 + fp32 master 4
#: + Adam moments 8. Independent of the activation value size.
TRAIN_STATE_BYTES_PER_PARAM: int = 16

#: Bytes per activation / state value for each ``train.precision`` (Appendix A.3).
VALUE_BYTES: dict[str, int] = {"bf16": 2, "fp16": 2, "fp32": 4, "fp64": 8}

_KIB = 1024
_MIB = 1024**2
_GIB = 1024**3


@dataclass(frozen=True)
class MemoryPlan:
    """Appendix C.3 / E.3 memory estimate in bytes for one training configuration.

    ``overhead_bytes`` is the device context, allocator and library-workspace estimate that P4.2
    takes from the hardware profile; P1.1 leaves it at 0, so the totals are training state +
    recurrent state + saved activations. Planner numbers are estimates, never measurements.
    """

    train_state_bytes: int  # TRAIN_STATE_BYTES_PER_PARAM * parameters
    rec_state_bytes: int  # B * N * d * value_bytes
    saved_sparse_bytes: int  # B * T * (act_shared + R * K * act_cell) * value_bytes
    saved_dense_bytes: int  # B * T * (act_shared + R * N * act_cell) * value_bytes
    overhead_bytes: int  # 0 until P4.2 adds the per-profile estimate
    total_sparse: int  # train_state + rec_state + saved_sparse + overhead
    total_dense: int  # train_state + rec_state + saved_dense + overhead


def _check_supported(m: ModelConfig) -> None:
    """Raise for configurations the closed forms do not model."""
    if m.ablate_shared_u:
        raise ValueError(
            "model.ablate_shared_u=true is not covered by the P1.1 closed forms, which fix the "
            "cell input v = [u; eps; h_i; m_i] at width 4d; P1.5 defines the ablation's shapes"
        )


def _mlp(d_in: int, hidden: int, d_out: int) -> int:
    """Parameters of a two-layer MLP ``d_in -> hidden -> d_out`` with both biases."""
    return d_in * hidden + hidden + hidden * d_out + d_out


def _scorer_params(m: ModelConfig) -> int:
    """Parameters of one scorer application: ``(3d + 8) -> M``, or via ``scorer_hidden`` for mlp."""
    d_in = 3 * m.d + SCORER_SCALAR_FEATURES
    if m.scorer == "linear":
        return d_in * m.n_units + m.n_units
    return _mlp(d_in, m.scorer_hidden, m.n_units)


def expected_param_count(m: ModelConfig) -> dict[str, int]:
    """Appendix C.1: parameters per component, in table order, plus ``"total"``.

    Keys: ``byte_emb`` (V × d), ``ctx_emb`` (n_context × context_dim), ``encoder`` E and
    ``predictor`` P (d → H → d), ``shared`` S (2d + context_dim → H → d), ``units`` (M × (4d → H →
    d)), ``w_out`` (d × d, no bias), ``v_msg`` (d × d, no bias), ``gate`` B (4d → d), ``scorer``
    (3d + 8 → M, or → scorer_hidden → M when ``scorer=mlp``), ``halting`` (2d → 1), ``decoder`` D
    (2d → H → V), ``cell_norm`` (the P1.5a RMSNorm gain over the 4d cell input, 0 when
    ``cell_norm=none``), ``cand_norm`` (the P1.13 RMSNorm gain of width d before the candidate's
    tanh, amendment A4; 0 unless ``cand_norm=rms``), ``total``. The reference configuration gives
    1,820,617 (1,820,361 before amendment A1, which added the 256-parameter ``cell_norm`` row;
    1,820,681 with ``cand_norm=rms``).
    """
    _check_supported(m)
    d, h, v = m.d, m.hidden, m.vocab_size
    rows = {
        "byte_emb": v * d,
        "ctx_emb": m.n_context * m.context_dim,
        "encoder": _mlp(d, h, d),
        "predictor": _mlp(d, h, d),
        "shared": _mlp(2 * d + m.context_dim, h, d),
        "units": m.n_units * _mlp(4 * d, h, d),
        "w_out": d * d,
        "v_msg": d * d,
        "gate": 4 * d * d + d,
        "cell_norm": 4 * d if m.cell_norm == "rms" else 0,  # P1.5a: the RMSNorm gain
        "cand_norm": d if m.cand_norm == "rms" else 0,  # P1.13 (amendment A4)
        "scorer": _scorer_params(m),
        "halting": 2 * d + 1,
        "decoder": _mlp(2 * d, h, v),
    }
    rows["total"] = sum(rows.values())
    return rows


def macs_per_token(m: ModelConfig) -> dict[str, int]:
    """Appendix C.2: multiply-accumulates per input position for one round, biases counted.

    Keys: ``shared`` (E + P + S + D), ``per_cell`` (one unit + W_out + D_g messages through V +
    gate B), ``selected`` (K × per_cell), ``scorer`` (N × one scorer application; linear or mlp per
    ``model.scorer``), ``sparse_total`` (shared + selected + scorer) and ``dense_total`` (shared +
    N × per_cell + scorer). 1 MAC = 2 FLOPs. The table counts the linear layers (biases included,
    Appendix H.3); the activations, the tanh, the sigmoid, the P1.5a RMSNorm on the cell input and
    the P1.13 RMSNorm on the candidate are not linear layers and stay outside it, so amendments A1
    and A4 leave C.2 unchanged.
    """
    params = expected_param_count(m)
    unit = params["units"] // m.n_units
    shared = params["encoder"] + params["predictor"] + params["shared"] + params["decoder"]
    per_cell = unit + params["w_out"] + m.in_degree * params["v_msg"] + params["gate"]
    scorer = m.n_cells * params["scorer"]
    return {
        "shared": shared,
        "per_cell": per_cell,
        "selected": m.k_select * per_cell,
        "scorer": scorer,
        "sparse_total": shared + m.k_select * per_cell + scorer,
        "dense_total": shared + m.n_cells * per_cell + scorer,
    }


def activations_per_token(m: ModelConfig) -> dict[str, int]:
    """Values saved for backward per input position (Appendix C.3 / E.3).

    ``shared`` = z, ẑ, ε, u (4d) + the four MLP hidden layers of E, P, S, D (4H) + logits (V):
    2,560 for the reference configuration (Appendix H.1). ``per_cell`` = v (4d) + unit hidden (H)
    + c, b, h (3d): 960.
    """
    _check_supported(m)
    return {
        "shared": 4 * m.d + 4 * m.hidden + m.vocab_size,
        "per_cell": 4 * m.d + m.hidden + 3 * m.d,
    }


def memory_plan(m: ModelConfig, batch: int, unroll: int, value_bytes: int = 2) -> MemoryPlan:
    """Appendix C.3 memory estimate for ``batch`` rows and a TBPTT window of ``unroll`` positions.

    ``value_bytes`` is the size of one activation or state value (2 for bf16/fp16; see
    :func:`value_bytes_for`); the training state uses :data:`TRAIN_STATE_BYTES_PER_PARAM`
    regardless. Saved activations assume every round's values are kept (no checkpointing; P4.2
    adds the modes).
    """
    if batch < 1 or unroll < 1 or value_bytes < 1:
        raise ValueError(
            f"batch={batch}, unroll={unroll} and value_bytes={value_bytes} must all be >= 1"
        )
    acts = activations_per_token(m)
    params = expected_param_count(m)["total"]
    train_state = TRAIN_STATE_BYTES_PER_PARAM * params
    rec_state = batch * m.n_cells * m.d * value_bytes
    per_round_sparse = m.r_max * m.k_select * acts["per_cell"]
    per_round_dense = m.r_max * m.n_cells * acts["per_cell"]
    saved_sparse = batch * unroll * (acts["shared"] + per_round_sparse) * value_bytes
    saved_dense = batch * unroll * (acts["shared"] + per_round_dense) * value_bytes
    overhead = 0  # P4.2: per-profile estimate (context, allocator, library workspace)
    return MemoryPlan(
        train_state_bytes=train_state,
        rec_state_bytes=rec_state,
        saved_sparse_bytes=saved_sparse,
        saved_dense_bytes=saved_dense,
        overhead_bytes=overhead,
        total_sparse=train_state + rec_state + saved_sparse + overhead,
        total_dense=train_state + rec_state + saved_dense + overhead,
    )


def value_bytes_for(precision: str) -> int:
    """Bytes per value for a ``train.precision`` name; unknown names raise."""
    try:
        return VALUE_BYTES[precision]
    except KeyError:
        raise ValueError(f"unknown precision {precision!r}; known: {sorted(VALUE_BYTES)}") from None


def format_bytes(n: int) -> str:
    """Bytes as Appendix C/E print them: KiB below 1 MiB, MiB with one decimal, GiB with two.

    A trailing ``.0`` is dropped (``16 KiB``, ``65 MiB``, ``27.8 MiB``, ``12.5 MiB``, ``1.50 GiB``);
    counts below 1 KiB print as plain bytes.
    """
    if n < 0:
        raise ValueError(f"byte count {n} must be >= 0")
    if n >= _GIB:
        return f"{n / _GIB:.2f} GiB"
    if n >= _MIB:
        return f"{_strip_zero(f'{n / _MIB:.1f}')} MiB"
    if n >= _KIB:
        return f"{_strip_zero(f'{n / _KIB:.1f}')} KiB"
    return f"{n} B"


def _strip_zero(text: str) -> str:
    return text[:-2] if text.endswith(".0") else text
