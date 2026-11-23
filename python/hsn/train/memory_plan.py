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
* The cell input ``v = [u; ε; h_i; m_i]`` has width ``4d``. With ``model.ablate_shared_u`` (the
  "remove shared path" ablation of the P2.13 matrix, :mod:`hsn.model.cell`) it is ``[ε; h_i; m_i]``
  of width ``3d`` (:attr:`hsn.config.ModelConfig.cell_input_width`), and the three rows that read
  it are counted at that width: the units (``M × (3d → H → d)``), the gate ``B`` (``3d → d``) and
  the ``cell_norm`` gain (``3d``); a cell then saves ``3d`` values for ``v``. ``u`` is still
  computed and still read by the scorer and the decoder, so every other row, the scorer's ``3d +
  8`` features and the shared activations are unchanged.
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
    "memory_off_param_count",
    "memory_plan",
    "reader_param_count",
    "cell_read_param_count",
    "word_cells_param_count",
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


def _mlp(d_in: int, hidden: int, d_out: int) -> int:
    """Parameters of a two-layer MLP ``d_in -> hidden -> d_out`` with both biases."""
    return d_in * hidden + hidden + hidden * d_out + d_out


def _scorer_params(m: ModelConfig) -> int:
    """Parameters of one scorer application: ``(3d + 8) -> M``, or via ``scorer_hidden`` for mlp."""
    # A5: a_cur, a_prev; A9: two more per extra n-gram length
    d_in = 3 * m.d + SCORER_SCALAR_FEATURES + (2 * m.address_ngram if m.key_address else 0)
    if m.scorer == "linear":
        # A12: the address gate, d -> n_units * (address features), no bias
        gate = m.d * m.n_units * 2 * m.address_ngram if m.address_gate else 0
        # A10: the write output is a second linear layer of the same shape
        return (2 if m.write_gate else 1) * (d_in * m.n_units + m.n_units) + gate
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
    1,820,681 with ``cand_norm=rms``). With ``model.ablate_shared_u`` the cell input is ``3d``
    wide, so ``units``, ``gate`` and ``cell_norm`` read ``3d`` in place of ``4d`` (1,554,313 for
    the reference configuration: 262,144 + 4,096 + 64 fewer).
    """
    d, h, v = m.d, m.hidden, m.vocab_size
    v_in = m.cell_input_width  # 4d: [u; eps; h_i; m_i]; 3d with model.ablate_shared_u
    c = m.reader_width if m.reader != "none" else 0  # HSN-T: S and D read the reader's output
    a = m.d if m.cell_attention_heads else 0  # HSN-T: D reads the attention read of the cells
    a += m.d + 2 if m.word_cells != "off" else 0  # HSN-T: D reads the keyed word cells' p_t
    rows = {
        "byte_emb": v * d,
        "ctx_emb": m.n_context * m.context_dim,
        "encoder": _mlp(d, h, d),
        "predictor": _mlp(d, h, d),
        # A8: + f; HSN-T: + c
        "shared": _mlp(2 * d + m.context_dim + (d if m.fast_readout else 0) + c, h, d),
        "units": m.n_units * _mlp(v_in, h, d),
        "w_out": d * d,
        "v_msg": d * d,
        "gate": v_in * d + d,
        "cell_norm": v_in if m.cell_norm == "rms" else 0,  # P1.5a: the RMSNorm gain
        "cand_norm": d if m.cand_norm == "rms" else 0,  # P1.13 (amendment A4)
        "scorer": _scorer_params(m),
        "halting": 2 * d + 1,
        "decoder": _mlp((2 + int(m.selected_readout) + int(m.fast_readout)) * d + c + a, h, v),
    }
    rows["total"] = sum(rows.values())
    return rows


def reader_param_count(m: ModelConfig) -> int:
    """HSN-T: the parameters of the window reader (:class:`hsn.model.reader.Reader`),
    outside Appendix C like the edge logits; 0 without ``model.reader``. The byte embedding
    ``V w``, per block ``12 w² + 13 w`` (two LayerNorms ``4 w``, the attention's ``w → 3w`` and
    ``w → w`` maps, the MLP ``w → 4w → w``) and the final LayerNorm ``2 w``: 116,480 for the
    default reader (``w`` 64, two blocks) at ``V`` 256. With ``model.reader_position =
    learned`` the position table adds ``model.reader_positions × w``; the rotary positions
    have no parameter. ``model.reader_qk_norm`` adds ``2 w / heads`` in every block it enables
    (the queries' and the keys' RMSNorm weights, shared by the heads)."""
    if m.reader == "none":
        return 0
    w = m.reader_width
    table = m.reader_positions * w if m.reader_position == "learned" else 0
    qk = m.reader_qk_norm
    normed = sum(qk) if isinstance(qk, list) else m.reader_layers * int(qk)
    return (
        m.vocab_size * w
        + m.reader_layers * (12 * w * w + 13 * w)
        + 2 * w
        + table
        + normed * 2 * (w // m.reader_heads)
    )


def cell_read_param_count(m: ModelConfig) -> int:
    """HSN-T with ``model.cell_attention_heads``: the parameters of the attention read of the
    cells (:class:`hsn.model.cell_read.CellAttentionRead`), outside Appendix C like the reader;
    0 without the key. The query map ``w → d``, the keys' and values' map of the state
    ``d → 2d``, the keys' map of the address ``d → d`` without a bias and the output map
    ``d → d``: ``d (w + 4d + 4)``, 20,736 at ``d`` 64 with the default reader (``w`` 64). The
    ``d`` decoder columns that read ``m_t`` are counted in the ``decoder`` row."""
    if not m.cell_attention_heads:
        return 0
    return m.d * (m.reader_width + 4 * m.d + 4)


def word_cells_param_count(m: ModelConfig) -> int:
    """HSN-T with ``model.word_cells``: the parameters of the keyed word cells
    (:class:`hsn.model.word_cells.WordCells`), outside Appendix C like the reader; 0 without
    the key. The read maps ``U_j``, one ``d x d`` map for each of the 16 offsets: 65,536 at
    ``d`` 64. The ``d + 2`` decoder columns that read ``p_t`` are counted in the ``decoder``
    row; the cells themselves are state, not parameters. The HSN's keyed write
    (``model.word_cells_value = word | cell``) adds the logits of the keyed cells' in-edges
    (``word_cells_n * in_degree``), the gate bias ``beta_K`` (``d``) and, for ``word``, ``V_s``
    (``d * d``): 4,160 and 8,256 at the recipe's sizes."""
    if m.word_cells == "off":
        return 0
    count = 16 * m.d * m.d
    if m.word_cells_value != "bytes":
        count += m.word_cells_n * m.in_degree + m.d
        count += m.d * m.d if m.word_cells_value == "word" else 0
    return count


def memory_off_param_count(m: ModelConfig) -> dict[str, int]:
    """HSN-T without its memory (``model.memory = false``, ``docs/hsn_t_no_memory.md``,
    :class:`hsn.model.hsn_t.HSNTNoMemory`): parameters per module in the model's order, plus
    ``"total"``. The five Appendix C.1 rows that are not the memory (``byte_emb``, ``ctx_emb``,
    ``encoder``, ``shared``, ``decoder``) at HSN-T's widths, so ``S`` reads ``2d + context_dim
    + w`` and ``D`` reads ``3d + w`` with ``w`` the reader's width, and the reader
    (:func:`reader_param_count`). 627,328 for the default sizes (``d`` 64, ``H`` 512, ``V``
    256, 16 context symbols, the reader ``w`` 64 with two blocks); in general, with ``L``
    blocks and those other sizes, ``445,312 + 1,282 w + L (12 w² + 13 w)``."""
    if m.reader == "none" or not m.selected_readout or m.fast_readout:
        raise ValueError(
            "HSN-T without memory has a reader, A5's decoder columns and no fast read: "
            f"got reader={m.reader!r}, selected_readout={m.selected_readout}, "
            f"fast_readout={m.fast_readout}"
        )
    full = expected_param_count(m)
    rows = {name: full[name] for name in ("byte_emb", "ctx_emb", "encoder", "shared", "decoder")}
    rows["reader"] = reader_param_count(m)
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
    # A5's fixed code/address dot products are scorer work (global rule 5).
    scorer = m.n_cells * params["scorer"] + (
        2 * m.address_ngram * m.n_cells * m.d if m.key_address else 0
    )
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
    + c, b, h (3d): 960. With ``model.ablate_shared_u`` ``v`` is ``3d`` wide (896); ``u`` stays
    in ``shared`` (the scorer and the decoder read it).
    """
    return {
        "shared": (4 + int(m.selected_readout) + int(m.fast_readout)) * m.d
        + 4 * m.hidden
        + m.vocab_size,
        "per_cell": m.cell_input_width + m.hidden + 3 * m.d,
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
