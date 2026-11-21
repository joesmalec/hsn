"""HSN-T (``model_name: hsn_t``, ``model.reader = transformer``, ``docs/hsn_t.md``): the
window reader.

A small causal Transformer over the byte ids of one window, all positions at once. Its output at
position ``t``, ``c_t`` (width ``model.reader_width``), takes the place of the fast cells' read of
amendment A8 in the two places the model reads it: ``u_t = S([z_t ; q_{t−1} ; ctx_t ; c_t])`` and
``D([q_t ; u_t ; r_t ; c_t])``. The cells and their carried state, the fixed addresses, the
router, the units, the convex tanh gate, the graph and the summary ``q`` are unchanged: the memory
is the HSN's, the reading of the current window is the reader's. This is the one module of the
model with sequence attention, which A5's readout text excludes; the key is off by default.

What ``c_t`` may depend on (tested in ``tests/test_hsn_t.py``):

* the bytes ``x_s`` with ``s ≤ t`` of the same window and the same document (``reset`` splits a
  row into documents, as it resets the cells; the attention mask is block-diagonal over them);
* with ``model.reader_span = n`` only the last ``n`` of them (``t − n < s ≤ t``): the
  short-sighted reader, which has to leave everything further back to the memory;
* nothing of an earlier window, no cell state, no context symbol.

``train.reach_dropout`` (``docs/reach_dropout.md``; opt-in, training only): the trainer hands
the model, per training window, the batch rows that read as with ``model.reader_span = S``
(``short`` of :meth:`Reader.forward`); the other rows, and every call without ``short``, read
as configured. With ``model.reader_carry`` (:meth:`Reader.read`, below) ``short`` narrows the
cached positions too, so a short row reads as the reader with ``model.reader_carry = S − 1``.

Layout: byte embedding → ``model.reader_layers`` pre-norm blocks (LayerNorm → causal multi-head
self-attention → residual; LayerNorm → GELU MLP of width ``4 w`` → residual) → LayerNorm. The
position enters as a rotation of the attention's queries and keys by the position in the window
(rotary embedding), so only the distance ``t − s`` is read and there is no position parameter:
with a span the output is a function of the last ``n`` bytes alone, wherever they sit in the
window.

The rotation's cosines and sines are read from a table of the first ``_ROTARY_TABLE`` window
positions, extended through position ``C`` with a carry of ``C``
(``tests/test_reader_rotary_table.py``, ``tests/test_reader_carry_speed.py``).
It holds what the forward pass computed on
every call until 15 Nov 2026: the float64 angles, computed on the device of the parameters and
cast to their dtype. It is made when the reader is built and made again, not moved or cast,
whenever the module is (``.to()``, ``.cuda()``, ``.double()``), so the outputs are the same bit
for bit. Under ``torch.compile`` the computation in the forward pass was inlined into the
kernels that rotate the queries and keys, as float64 trigonometry for each of their elements
(78 ms of a 111 ms training step on an RTX 3060 at width 224). The table is a buffer outside
the state dict, so checkpoints are unchanged. The forward pass still computes the angles
itself for a window longer than the table, on MPS (which has no float64; its float32 angles
are kept as they were) and for parameters that were moved or cast without the module.

The parameters are drawn from the module's own generator (``seed``) inside a forked RNG, so
building a reader neither changes another parameter's initialisation nor consumes the global
random stream.

Four opt-in keys change how the position enters (``docs/hsn_t.md``, "The reader's positions";
15 Nov 2026). At their defaults the reader is the one above: the same parameters, state dict and
outputs, bit for bit (``tests/test_reader_positions.py``).

* ``model.reader_rotary_pairs = n``: only the first ``n`` pairs of each head (the ``n`` highest
  frequencies of the same ladder) are rotated. The other pairs get the angle zero, so their part
  of a query is compared with a key as it is, at every distance. Still distances only. A list
  gives ``n`` for each block: the Transformer baseline adds its positions once, before the first
  block, and the reader rotates in every block; ``[all, …, 0]`` leaves the last block's attention
  without a rotation: it compares queries and keys as they are (what it reads of the order is
  what the blocks before it wrote into their outputs).
* ``model.reader_rotary_dtype = parameters``: the rotation is computed in the dtype of the
  parameters (float32 master weights) and the rotated queries and keys are cast back to the
  working dtype for the attention. Under bf16 autocast the default rotates in bf16: the cos /
  sin and every product are rounded to 8 bits.
* ``model.reader_rotary_base = b`` (default 10000): the base of the frequency ladder, so pair
  ``i`` of a head turns by ``t · b^(−i / half)`` at distance ``t``, in the table and in the
  angles the forward pass computes itself alike. A larger base gives the slower pairs longer
  wavelengths; pair 0 keeps one radian a position. Still distances only, no parameter; it acts
  on the pairs ``model.reader_rotary_pairs`` rotates, and ``model.reader_position = learned``
  refuses it.
* ``model.reader_position = learned``: no rotation; a table ``pos`` of
  ``model.reader_positions`` rows is added to the byte embedding by window position, as the
  Transformer baseline's. The reader then reads the position in the window: its output is no
  longer a function of distances alone, and with a span not of the last ``n`` bytes alone. The
  table is drawn after the other parameters, which therefore start as the rotary reader's.

Reading past the window (``docs/hsn_t_no_memory.md``; opt-in, 15 Nov 2026, untested
candidates). ``model.reader_carry = C``: every block keeps the un-rotated keys and the values of
the last ``C`` positions of the windows before, detached, in a cache ``(keys, values, valid)``
with ``keys`` / ``values [L, B, H, C, w / H]`` (block, row, head, slot, channel; slot ``j`` is
the position ``j − C`` before the window) and ``valid [B, C]`` (bool: the slot holds a position
of the document going on). :meth:`Reader.read` takes the cache and returns the next one; the
models that hold the reader carry it as their state (``init_state``), so the trainer carries it
from window to window and detaches it as it does the GRU's. Window position ``t`` reads the
cached and window positions ``s`` of its document with ``0 ≤ t − s ≤ C``: a sliding span, the
same at every window and document length, so no distance beyond ``C`` is ever read. The cached
keys are rotated by their positions ``−C … −1`` (the table's rows ``C … 1`` with the sine's
sign turned) and the window's by ``0 … T − 1``, as without the cache; a block that rotates
nothing needs nothing. ``reset`` acts as before: a position reads nothing of a document before
its own, cached or not. The shapes do not depend on the data; at ``C = 0`` (the default)
:meth:`Reader.forward` is the reader above and there is no cache.

``model.reader_top_words = N`` (needs ``C > 0`` and the top block unrotated): the cache also
holds the top block's keys and values at the last ``N`` word ends of the document before the
window (:func:`word_ends`: a byte that is not a letter or digit right after one; bytes of 128
and above count as letters), ``(word_keys, word_values [B, H, N, w / H], word_valid [B, N],
word_pos [B, N], last_letter [B])``: the slots the most recent first, ``word_pos`` their
positions from the window's start (int64), ``last_letter`` whether the window's last byte was a
letter (a word end may be the next window's first byte). The top block reads them as more keys
(it rotates nothing, so they need no position), and also the window's own word ends beyond the
span when the window is longer than ``C + 1``: of these two together the last ``N``. An entry
inside the span (``t − s ≤ C``) is read by the span only, never twice; ``reset`` acts as on
the cache. With ``N = 256`` the top block reaches about 256 words back at a fixed cost.

The causal mask (``docs/reader_causal_flash.md``; opt-in, 17 Nov 2026). ``model.reader_mask =
causal``: every block calls ``scaled_dot_product_attention(q, k, v, is_causal=True)`` with no
mask, which the flash kernel takes on CUDA (the additive document mask of the default sends
it to the memory-efficient kernel). In a window whose resets are at column 0 only, every
position of a row is of one document (the one that starts at column 0, or the one going on
from the window before), so the document mask is the causal mask and the reader is the same
function. The configuration allows the key only where that holds: no ``reader_span``, no
``reader_carry`` and ``data.align_docs`` (the loader starts a document at a window boundary
only). :meth:`Reader.forward` also asserts it on every window with ``torch._assert_async``:
on the device, with no host read (no synchronisation; under ``torch.compile`` a reduction
fused into the graph, CUDA graphs included); a reset after column 0 stops the run (on CUDA a
device-side assert, which ends the process). At ``document`` (the default) nothing of this
runs and the reader is the one above, bit for bit.

Normalised queries and keys (``docs/amendments/reader_qk_norm.md``; opt-in, 21 Nov 2026).
``model.reader_qk_norm`` (true for every block, or a list with one entry for each): in an
enabled block the queries and the keys are normalised per head after the projection and before
the rotation, ``q ← g_q ⊙ q / sqrt(mean(q²) + eps)`` over a head's ``w / H`` channels, and so
for ``k`` with ``g_k`` (:func:`_head_norm`). ``g_q`` and ``g_k`` are parameters of width
``w / H`` shared by the heads, initialised to 1, made without a random draw (every other
parameter starts as it would without the key); ``eps`` is that of the block's LayerNorms. The
values are not normalised and the softmax scale stays ``1 / sqrt(w / H)``, so a logit is at most
``max|g_q| · max|g_k| · sqrt(w / H)`` (the rotation keeps a pair's length), whatever the qkv
and LayerNorm weights. With ``model.reader_carry`` the cache holds the normalised keys, the
keys the same positions have in a pass without the cache. At false (the default) a block has
no such parameter and runs the code it ran, bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hsn.config import ModelConfig

__all__ = ["Cache", "Reader", "word_bytes", "word_ends"]

#: The default of ``model.reader_rotary_base``: the base of the rotary frequency ladder.
_ROTARY_BASE = 10_000.0
#: Window positions in the rotary table: four windows of the default ``train.unroll`` (the
#: reader is built from ``ModelConfig``, which has no window length). 2 · 1024 · half values.
_ROTARY_TABLE = 1024
_INIT_STD = 0.02


def _rotary_angles(
    T: int,
    half: int,
    device: torch.device,
    pairs: int | None = None,
    base: float = _ROTARY_BASE,
) -> Tensor:
    """The rotary angles ``[T, half]``: position ``t`` times ``base^(−i / half)``
    (``model.reader_rotary_base``, 10000 by default). In float64 where the device has it
    (float32 on MPS), so that a rotation by ``t`` against one by ``s`` is the rotation by
    ``t − s`` to the working precision at every window position. With ``pairs``
    (``model.reader_rotary_pairs``) the pairs ``i ≥ pairs`` have the angle zero: their cos is 1
    and their sin 0, and the rotation returns them as they are, bit for bit."""
    wide = torch.float32 if device.type == "mps" else torch.float64
    exponent = torch.arange(half, device=device, dtype=wide) / half
    angles = torch.arange(T, device=device, dtype=wide)[:, None] * (base**-exponent)
    if pairs is not None:
        angles[:, pairs:] = 0.0
    return angles


#: ``model.reader_top_words``: the position of an empty word slot (far below any real one, and
#: far enough from the int64 limit that shifting it by a window never wraps)
_NO_POSITION = -(2**62)


def word_bytes(tokens: Tensor) -> Tensor:
    """Bool, the shape of ``tokens``: the byte is a letter or digit for the word ends of
    ``model.reader_top_words``: ASCII ``0-9``, ``A-Z``, ``a-z``, and every byte of 128 and
    above (the bytes of a UTF-8 letter)."""
    return (
        ((tokens >= 48) & (tokens <= 57))
        | ((tokens >= 65) & (tokens <= 90))
        | ((tokens >= 97) & (tokens <= 122))
        | (tokens >= 128)
    )


def word_ends(tokens: Tensor, reset: Tensor, last_letter: Tensor) -> Tensor:
    """Bool ``[B, T]``: position ``t`` ends a word. Its byte is not a letter or digit
    (:func:`word_bytes`: a space, punctuation, the newline and tab markers 16 and 17, a control
    symbol) and the byte before it, of the same document, is one; for ``t = 0`` that byte is
    the last of the window before (``last_letter [B]``). A document's first position ends
    nothing. Causal: at a word end the whole word has been read."""
    letters = word_bytes(tokens)
    before = torch.cat([last_letter[:, None].to(torch.bool), letters[:, :-1]], dim=1)
    return ~letters & before & ~reset


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the pairs ``(x[..., i], x[..., i + half])`` of ``x [B, H, T, 2·half]`` by the
    position's angles ``cos`` / ``sin [T, half]``."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


def _head_norm(x: Tensor, norm: nn.RMSNorm) -> Tensor:
    """``model.reader_qk_norm``: ``x [B, H, T, w / H]`` (queries or keys) normalised over each
    head's channels with ``norm``'s weight ``[w / H]`` (shared by the heads) and eps. Computed
    in float32 at least (under bf16 autocast the projection's output is bf16) and returned in
    ``x``'s dtype, the attention's working dtype."""
    wide = x if x.dtype in (torch.float32, torch.float64) else x.float()
    normed = F.rms_norm(wide, norm.normalized_shape, norm.weight.to(wide.dtype), norm.eps)
    return normed.to(x.dtype)


def _attention_bias(mask: Tensor, x: Tensor) -> Tensor:
    """Share SDPA's bool-to-additive conversion across the blocks of one window.

    CPU and CUDA SDPA convert True/False to 0/-inf in the queries' dtype. The
    queries come from an autocast linear, while ``x`` comes from the embedding.
    Float64 linears do not autocast. Keep other backends' existing bool path.
    """
    device = x.device.type
    if device not in ("cpu", "cuda"):
        return mask
    dtype = x.dtype
    if dtype != torch.float64 and torch.is_autocast_enabled(device):
        dtype = torch.get_autocast_dtype(device)
    return torch.where(mask, 0.0, torch.full((), -torch.inf, device=x.device, dtype=dtype))


def _keep_tail(old: Tensor, new: Tensor, count: int, dim: int) -> Tensor:
    """Last ``count`` entries of cat(old, new), without copying discarded entries.

    The caller stacks/contiguates the result to keep the public cache layout.
    Detach before copying: these entries are next-window state, never a loss input.
    """
    new = new.detach()
    length = new.shape[dim]
    if length >= count:
        return new.narrow(dim, length - count, count).to(old.dtype)
    return torch.cat([old.detach().narrow(dim, length, count - length), new.to(old.dtype)], dim)


def _check_short(short: tuple[Tensor, int], B: int, T: int | None = None) -> None:
    """``short = (rows, n)`` of ``train.reach_dropout``: ``rows`` a bool ``[B]`` tensor, or
    with ``T`` (:meth:`Reader.read`) a bool ``[B, T]`` one, and ``n`` an int of 1 or more."""
    rows, n = short
    shapes = [(B,)] if T is None else [(B,), (B, T)]
    if rows.dtype is not torch.bool or tuple(rows.shape) not in shapes:
        names = "[B]" if T is None else f"[B] or [B, T] (T = {T})"
        raise ValueError(
            f"short rows must be a bool {names} tensor (B = {B}), got "
            f"{rows.dtype} {tuple(rows.shape)}"
        )
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"the short span must be an int >= 1, got {n!r}")


class _ReaderBlock(nn.Module):
    """One pre-norm block: causal multi-head self-attention with rotary positions and a GELU
    MLP, both residual."""

    #: ``model.reader_qk_norm``: the queries' and the keys' RMSNorm over a head's channels
    #: (:func:`_head_norm`); None in a block without it (no parameter, no state-dict entry)
    q_norm: nn.RMSNorm | None
    k_norm: nn.RMSNorm | None

    def __init__(
        self, width: int, heads: int, *, rotate_in: str = "working", qk_norm: bool = False
    ) -> None:
        super().__init__()
        self.heads = heads
        #: ``model.reader_rotary_dtype``: "working" rotates in the dtype of the queries and keys,
        #: "parameters" in the dtype of the cos / sin it is given (the parameters')
        self.rotate_in = rotate_in
        self.norm_attn = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.fc_in = nn.Linear(width, 4 * width)
        self.fc_out = nn.Linear(4 * width, width)
        if qk_norm:  # made last and without a draw: the other parameters start as without it
            per_head = width // heads
            self.q_norm = nn.RMSNorm(per_head, eps=self.norm_attn.eps)
            self.k_norm = nn.RMSNorm(per_head, eps=self.norm_attn.eps)
        else:
            self.q_norm = self.k_norm = None

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The queries, keys and values ``[B, H, T, w / H]`` of ``x [B, T, w]``, not rotated;
        with ``model.reader_qk_norm`` the queries and keys normalised per head."""
        B, T, w = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).split(w, dim=-1)
        q = q.view(B, T, self.heads, -1).transpose(1, 2)  # [B, H, T, w / H]
        k = k.view(B, T, self.heads, -1).transpose(1, 2)
        v = v.view(B, T, self.heads, -1).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = _head_norm(q, self.q_norm), _head_norm(k, self.k_norm)
        return q, k, v

    def forward(
        self, x: Tensor, mask: Tensor | None, cos: Tensor | None, sin: Tensor | None
    ) -> Tensor:
        """``cos`` / ``sin [T, half]``: the rotary angles' in the dtype of the parameters, or
        ``None`` for no rotation (``model.reader_position = learned``). ``mask`` None: the
        causal mask (``model.reader_mask = causal``), SDPA's ``is_causal``."""
        B, T, w = x.shape
        q, k, v = self._project(x)
        if cos is not None and sin is not None:
            if self.rotate_in == "parameters":  # in the cos / sin's dtype, then cast back
                working = q.dtype
                q = _rotate(q.to(cos.dtype), cos, sin).to(working)
                k = _rotate(k.to(cos.dtype), cos, sin).to(working)
            else:
                cos, sin = cos.to(q.dtype), sin.to(q.dtype)
                q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        if mask is None:  # model.reader_mask = causal: no mask, the flash kernel on CUDA
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(attn.transpose(1, 2).reshape(B, T, w))
        out: Tensor = x + self.fc_out(F.gelu(self.fc_in(self.norm_mlp(x))))
        return out

    def forward_carry(
        self,
        x: Tensor,
        mask: Tensor,
        cos: Tensor | None,
        sin: Tensor | None,
        keys: Tensor,
        values: Tensor,
        word_keys: Tensor | None = None,
        word_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """:meth:`forward` over the carried keys and values as well (``model.reader_carry``,
        :meth:`Reader.read`). ``keys`` / ``values [B, H, C, w / H]``: the un-rotated keys and
        the values of the ``C`` positions before the window; ``cos`` / ``sin [C + T, half]``:
        the angles of the positions ``−C … T − 1`` (None: no rotation); ``mask
        [B, 1, T, C + T]``, the cached columns first. ``word_keys`` / ``word_values
        [B, H, N, w / H]`` (``model.reader_top_words``, the top block, which rotates nothing):
        ``N`` more columns after them, the mask ``[B, 1, T, C + T + N]``. Returns the output
        and the window's un-rotated keys and its values ``[B, H, T, w / H]`` (the next
        cache's; with ``model.reader_qk_norm`` the keys normalised, as the cached ones are)."""
        B, T, w = x.shape
        C = keys.shape[2]
        q, k, v = self._project(x)
        all_k = torch.cat([keys.to(k.dtype), k], dim=2)  # [B, H, C + T, w / H]
        all_v = torch.cat([values.to(v.dtype), v], dim=2)
        if cos is not None and sin is not None:
            if self.rotate_in == "parameters":  # in the cos / sin's dtype, then cast back
                working = q.dtype
                q = _rotate(q.to(cos.dtype), cos[C:], sin[C:]).to(working)
                all_k = _rotate(all_k.to(cos.dtype), cos, sin).to(working)
            else:
                cos, sin = cos.to(q.dtype), sin.to(q.dtype)
                q, all_k = _rotate(q, cos[C:], sin[C:]), _rotate(all_k, cos, sin)
        if word_keys is not None and word_values is not None:  # position-free entries
            all_k = torch.cat([all_k, word_keys.to(all_k.dtype)], dim=2)
            all_v = torch.cat([all_v, word_values.to(all_v.dtype)], dim=2)
        attn = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=mask)
        x = x + self.proj(attn.transpose(1, 2).reshape(B, T, w))
        out: Tensor = x + self.fc_out(F.gelu(self.fc_in(self.norm_mlp(x))))
        return out, k, v


#: ``model.reader_carry``: the cache of :meth:`Reader.read`, ``(keys, values, valid)`` (the
#: module docstring has the shapes). A plain tuple: the trainer walks it as it walks a state.
Cache = tuple[Tensor, ...]


class Reader(nn.Module):
    """``c = Reader(tokens, reset)``: ``[B, T] → [B, T, model.reader_width]`` (module docstring)."""

    #: The rotary table (module docstring): cos / sin ``[_ROTARY_TABLE, half]`` in the dtype
    #: and on the device of the parameters; None where the forward pass computes them (MPS)
    #: and without a rotation (``model.reader_position = learned``).
    _rot_cos: Tensor | None
    _rot_sin: Tensor | None
    #: ``model.reader_position = learned``: the position table ``[model.reader_positions, w]``;
    #: None with rotary positions (no entry in the state dict).
    pos: nn.Parameter | None

    def __init__(self, cfg: ModelConfig, *, seed: int = 0) -> None:
        super().__init__()
        if cfg.reader != "transformer":
            raise ValueError(f"Reader needs model.reader=transformer, got {cfg.reader!r}")
        width, heads = cfg.reader_width, cfg.reader_heads
        self.width = width
        self.heads = heads
        self.span: int | None = cfg.reader_span
        self.position = cfg.reader_position
        #: ``model.reader_rotary_base``: the base of the frequency ladder of the table and of
        #: the angles the forward pass computes itself
        self.rotary_base: float = float(cfg.reader_rotary_base)
        #: ``model.reader_rotary_pairs`` as configured: one count, one for each block, or None
        self.rotary_pairs: int | list[int] | None = cfg.reader_rotary_pairs
        #: ``model.reader_carry``: the positions of the windows before that every block keeps
        #: (0: none, every window read alone; :meth:`read` reads with the cache)
        self.carry: int = cfg.reader_carry
        if self.carry and self.position != "rotary":
            raise ValueError(
                "model.reader_carry needs the rotary positions: a learned position is the "
                "position in the window"
            )
        if self.carry and self.span is not None:
            raise ValueError("model.reader_carry is a sliding span itself: leave reader_span null")
        #: ``model.reader_top_words``: the word ends the top block also reads (0: none)
        self.top_words: int = cfg.reader_top_words
        if self.top_words and not self.carry:
            raise ValueError("model.reader_top_words needs model.reader_carry > 0")
        #: ``model.reader_mask = causal``: SDPA's causal mask in place of the document mask
        #: (module docstring, "The causal mask"); False: the document mask (the default)
        self.causal: bool = cfg.reader_mask == "causal"
        if self.causal and (self.span is not None or self.carry):
            raise ValueError(
                "model.reader_mask=causal is the document mask only without model.reader_span "
                "and model.reader_carry"
            )
        qk_norm = cfg.reader_qk_norm
        #: ``model.reader_qk_norm`` for each block, first to last (module docstring)
        self.qk_norm: list[bool] = (
            list(qk_norm) if isinstance(qk_norm, list) else [qk_norm] * cfg.reader_layers
        )
        if len(self.qk_norm) != cfg.reader_layers:
            raise ValueError(
                f"model.reader_qk_norm={qk_norm} names {len(self.qk_norm)} blocks and the reader "
                f"has {cfg.reader_layers}"
            )
        half = width // heads // 2
        counts = (
            list(self.rotary_pairs)
            if isinstance(self.rotary_pairs, list)
            else [self.rotary_pairs] * cfg.reader_layers
        )
        #: The rotated pairs of each block's heads; None: all of them (the default)
        self._block_pairs: list[int | None] = [
            None if n is None or n >= half else n for n in counts
        ]
        #: The pairs the rotary table itself keeps: the blocks' count when they share one (the
        #: table is then handed to every block as it is), None when it holds every pair
        self._table_pairs: int | None = (
            self._block_pairs[0] if len(set(self._block_pairs)) == 1 else None
        )
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed((seed + 0xA14EAD) % (2**63 - 1))
            torch.set_rng_state(generator.get_state())
            self.emb = nn.Embedding(cfg.vocab_size, width)
            self.blocks = nn.ModuleList(
                _ReaderBlock(width, heads, rotate_in=cfg.reader_rotary_dtype, qk_norm=on)
                for on in self.qk_norm
            )
            self.norm = nn.LayerNorm(width)
            for module in self.modules():
                if isinstance(module, nn.Linear | nn.Embedding):
                    nn.init.normal_(module.weight, std=_INIT_STD)
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
            if cfg.reader_position == "learned":  # drawn last: the other draws are unchanged
                self.pos = nn.Parameter(torch.empty(cfg.reader_positions, width))
                nn.init.normal_(self.pos, std=_INIT_STD)
            else:
                self.register_parameter("pos", None)
        if self.top_words and self._block_pairs[-1] != 0:
            raise ValueError("model.reader_top_words needs the top block unrotated (pairs 0)")
        self.register_buffer("_rot_cos", None, persistent=False)
        self.register_buffer("_rot_sin", None, persistent=False)
        self._fill_rotary_table()

    def _fill_rotary_table(self) -> None:
        """Compute the rotary table where the parameters are: the cos / sin the forward pass
        would compute there, including position ``C`` for the negative carry positions.

        Carry 1024 needs row 1024 (1025 rows): missing it used to put float64
        trigonometry back into every forward, including compiled forwards.
        """
        weight = self.emb.weight
        if weight.device.type == "mps" or self.position != "rotary":
            self._rot_cos = self._rot_sin = None
            return
        angles = _rotary_angles(
            max(_ROTARY_TABLE, self.carry + 1),
            self.width // self.heads // 2,
            weight.device,
            self._table_pairs,
            self.rotary_base,
        )
        self._rot_cos = angles.cos().to(weight.dtype)
        self._rot_sin = angles.sin().to(weight.dtype)

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        """``.to()`` / ``.cuda()`` / ``.double()``: the table is computed again for the new
        device and dtype. Moved, a CPU table could differ from the card's own in a rounding;
        cast, a float32 table would not be the float64 angles."""
        self._rot_cos = self._rot_sin = None
        super()._apply(fn, recurse)  # type: ignore[no-untyped-call]  # torch has no types here
        self._fill_rotary_table()
        return self

    def _rotation(self, T: int, x: Tensor) -> tuple[Tensor, Tensor]:
        """``cos`` / ``sin [T, half]`` of the rotary angles in the dtype of ``x``: read from
        the table when it holds them for ``x``, computed otherwise (module docstring)."""
        cos, sin = self._rot_cos, self._rot_sin
        if (
            cos is not None
            and sin is not None
            and T <= cos.shape[0]
            and cos.dtype == x.dtype
            and cos.device == x.device
        ):
            return cos[:T], sin[:T]
        half = self.width // self.heads // 2
        angles = _rotary_angles(T, half, x.device, self._table_pairs, self.rotary_base)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)

    def _block_rotation(
        self, pairs: int | None, cos: Tensor | None, sin: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None]:
        """The cos / sin one block is handed: the table's when it already holds the block's
        pairs (the default, and one count for all blocks), none for a block that rotates
        nothing, and otherwise the table's with the angle zero in the pairs ``i ≥ pairs``."""
        if cos is None or sin is None or pairs == 0:
            return None, None
        if pairs is None or pairs == self._table_pairs:
            return cos, sin
        keep = torch.arange(cos.shape[-1], device=cos.device) < pairs
        return torch.where(keep, cos, 1.0), torch.where(keep, sin, 0.0)

    def attention_mask(self, reset: Tensor, short: tuple[Tensor, int] | None = None) -> Tensor:
        """Bool ``[B, 1, T, T]``, True where position ``t`` (row) may read position ``s``
        (column): ``s ≤ t``, the same document of the row (``reset [B, T]`` starts a document at
        its True columns), and with a span ``t − s < span``. Every row holds its own position.

        ``short = (rows, n)`` (``train.reach_dropout``, ``docs/reach_dropout.md``; a reader
        without a span): the batch rows where ``rows`` (bool ``[B]``, on the device) is True
        also need ``t − s < n``, the mask of ``model.reader_span = n``; the other rows keep the
        whole window, bit for bit. Tensor operations only: no host read, one shape."""
        T = int(reset.shape[1])
        segment = reset.to(torch.int64).cumsum(dim=1)  # the document index of every position
        same = segment[:, :, None] == segment[:, None, :]
        index = torch.arange(T, device=reset.device)
        distance = index[:, None] - index[None, :]  # t − s
        near = distance >= 0
        if self.span is not None:
            near = near & (distance < self.span)
        mask = same & near
        if short is not None:
            rows, n = short
            mask = mask & ((distance < n) | ~rows[:, None, None])
        return mask[:, None]

    def forward(
        self, tokens: Tensor, reset: Tensor, short: tuple[Tensor, int] | None = None
    ) -> Tensor:
        """``c [B, T, w]``; ``short``: :meth:`attention_mask`'s, the rows that read with a
        short span (``train.reach_dropout``; None, the default: every row as configured)."""
        if tokens.ndim != 2 or reset.shape != tokens.shape or reset.dtype is not torch.bool:
            raise ValueError(
                f"Reader needs tokens [B, T] and a bool reset of the same shape, got "
                f"{tuple(tokens.shape)} and {reset.dtype} {tuple(reset.shape)}"
            )
        if self.carry:
            raise ValueError(
                f"a reader with model.reader_carry={self.carry} reads with "
                "read(tokens, reset, cache), the cache from init_cache"
            )
        if short is not None:
            if self.span is not None:
                raise ValueError(
                    f"short rows (train.reach_dropout) shorten a whole-window reader; this one "
                    f"has model.reader_span={self.span}"
                )
            if self.causal:
                # model.reader_mask=causal reads with SDPA's is_causal and no mask, so the short
                # rows' span mask could not act (docs/reader_causal_flash.md)
                raise ValueError(
                    "short rows (train.reach_dropout) need the document mask they narrow; this "
                    "reader has model.reader_mask=causal (no mask)"
                )
            _check_short(short, int(tokens.shape[0]))
        x = self.emb(tokens)
        T = int(tokens.shape[1])
        cos: Tensor | None
        sin: Tensor | None
        if self.pos is not None:  # model.reader_position = learned: the table, no rotation
            if T > self.pos.shape[0]:
                raise ValueError(
                    f"a window of {T} positions exceeds the reader's position table "
                    f"(model.reader_positions={self.pos.shape[0]})"
                )
            x = x + self.pos[:T]
            cos = sin = None
        else:
            cos, sin = self._rotation(T, x)
        mask: Tensor | None
        if self.causal:  # model.reader_mask = causal (module docstring, "The causal mask")
            # on the device, no host read: the causal mask is the document mask only when no
            # row starts a document after column 0
            torch._assert_async(
                reset[:, 1:].any().logical_not(),
                "model.reader_mask=causal: a window has a reset after column 0, where the "
                "causal mask is not the document mask (needs data.align_docs windows)",
            )
            mask = None
        else:
            mask = _attention_bias(self.attention_mask(reset, short), x)
        for i, block in enumerate(self.blocks):
            block_cos, block_sin = self._block_rotation(self._block_pairs[i], cos, sin)
            x = block(x, mask, block_cos, block_sin)
        out: Tensor = self.norm(x)
        return out

    # model.reader_carry ----------------------------------------------------------------------

    def init_cache(self, B: int, device: torch.device | str) -> Cache | None:
        """The cache of ``B`` fresh streams (module docstring, "Reading past the window"),
        nothing valid; None without ``model.reader_carry``. In the dtype of the parameters:
        keys and values are stored in it whatever the working dtype (a bf16 value is exact in
        float32), so the cache's dtype and shapes are the same at every window."""
        if B < 1:
            raise ValueError(f"B must be >= 1, got {B}")
        if not self.carry:
            return None
        weight = self.emb.weight
        shape = (len(self.blocks), B, self.heads, self.carry, self.width // self.heads)
        keys = torch.zeros(shape, device=device, dtype=weight.dtype)
        valid = torch.zeros(B, self.carry, device=device, dtype=torch.bool)
        cache: Cache = (keys, torch.zeros_like(keys), valid)
        if self.top_words:  # the top block's word slots, all empty
            N = self.top_words
            words = (B, self.heads, N, self.width // self.heads)
            word_keys = torch.zeros(words, device=device, dtype=weight.dtype)
            cache += (
                word_keys,
                torch.zeros_like(word_keys),
                torch.zeros(B, N, device=device, dtype=torch.bool),
                torch.full((B, N), _NO_POSITION, device=device, dtype=torch.int64),
                torch.zeros(B, device=device, dtype=torch.bool),
            )
        return cache

    def _cache_layout(self, B: int) -> list[tuple[tuple[int, ...], str]]:
        """The shape and kind (``float`` | ``bool`` | ``int``) of every tensor of the cache."""
        d = self.width // self.heads
        kv = (len(self.blocks), B, self.heads, self.carry, d)
        layout = [(kv, "float"), (kv, "float"), ((B, self.carry), "bool")]
        if self.top_words:
            N = self.top_words
            words = (B, self.heads, N, d)
            layout += [(words, "float"), (words, "float"), ((B, N), "bool"), ((B, N), "int")]
            layout.append(((B,), "bool"))
        return layout

    def check_cache(self, cache: object, B: int) -> Cache:
        """``cache`` as :meth:`init_cache` makes it for ``B`` streams, or a ``ValueError``."""
        if not self.carry:
            raise ValueError("a reader without model.reader_carry has no cache")
        layout = self._cache_layout(B)

        def kind(t: Tensor) -> str:
            return "bool" if t.dtype is torch.bool else "float" if t.is_floating_point() else "int"

        got: object = type(cache).__name__
        if isinstance(cache, tuple) and all(isinstance(t, Tensor) for t in cache):
            got = [(tuple(t.shape), kind(t)) for t in cache]
            if got == layout:
                return cache
        raise ValueError(
            f"model.reader_carry={self.carry}: the cache must be {layout} (init_cache), got {got}"
        )

    def reset_cache(self, cache: Cache, mask: Tensor) -> Cache:
        """``cache`` with nothing valid in the rows where ``mask`` (``[B]`` bool) is True."""
        if mask.dtype is not torch.bool or mask.dim() != 1:
            raise ValueError(f"mask must be a bool [B] tensor, got {tuple(mask.shape)}")
        parts = list(self.check_cache(cache, int(mask.shape[0])))
        rows = mask.to(parts[2].device)
        parts[2] = parts[2] & ~rows[:, None]
        if self.top_words:  # the word slots, and the byte before the next window
            parts[5] = parts[5] & ~rows[:, None]
            parts[7] = parts[7] & ~rows
        return tuple(parts)

    def carry_mask(
        self, reset: Tensor, valid: Tensor, short: tuple[Tensor, int] | None = None
    ) -> Tensor:
        """Bool ``[B, 1, T, C + T]``: True where window position ``t`` (row) may read a
        cached position (the first ``C`` columns, slot ``j`` at the position ``j − C``) or a
        window position ``s`` (the last ``T``). ``t − s ≤ C`` and ``s ≤ t`` for both; a cached
        position needs ``valid`` (``[B, C]``, its document is the one going on) and no reset at
        or before ``t``; a window position needs the document of ``t``. Every row holds its own
        position.

        ``short = (rows, n)`` (``train.reach_dropout`` on the carried reader,
        ``docs/reach_dropout.md``): where ``rows`` is True, a batch row (bool ``[B]``) or a
        position of one (bool ``[B, T]``, by ``t``) also needs ``t − s < n``, over the cached
        and the window positions alike: the mask of ``model.reader_carry = n − 1``. The others
        keep the carry's span, bit for bit. Tensor operations only: no host read, one shape."""
        T = int(reset.shape[1])
        C = self.carry
        segment = reset.to(torch.int64).cumsum(dim=1)  # the document index of every position
        same = segment[:, :, None] == segment[:, None, :]
        index = torch.arange(T, device=reset.device)
        distance = index[:, None] - index[None, :]  # t − s
        near = (distance >= 0) & (distance <= C)
        going_on = segment == 0  # [B, T]: no reset at or before t, the cached document goes on
        slot = torch.arange(C, device=reset.device)
        reach = slot[None, :] >= index[:, None]  # [T, C]: t − (j − C) ≤ C
        cached = valid[:, None, :] & going_on[:, :, None] & reach
        mask = torch.cat([cached, same & near], dim=-1)
        if short is not None:
            rows, n = short
            # t − s of every column: the cached slot j stands at the position j − C
            apart = torch.cat([index[:, None] + (C - slot)[None, :], distance], dim=-1)
            whole = ~rows[:, None, None] if rows.dim() == 1 else ~rows[:, :, None]
            mask = mask & ((apart < n) | whole)
        return mask[:, None]

    def word_masks(
        self, reset: Tensor, ends: Tensor, word_valid: Tensor, word_pos: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``model.reader_top_words``: what the top block reads of the word ends besides
        :meth:`carry_mask`, as ``(window [B, T, T], cached [B, T, N])``. ``ends [B, T]``:
        :func:`word_ends` of the window. ``window``: window position ``s`` for ``t``, a word end
        of ``t``'s document beyond the span (``t − s > C``), one of the last ``N`` such ends
        of the window. ``cached``: word slot ``j`` (the most recent first; ``word_pos`` is its
        position from the window's start, negative) for ``t``: valid, no reset at or before
        ``t``, beyond the span (``word_pos < t − C``), and one of the last ``N`` of the
        window's word ends before ``t − C`` and the slots together. An entry inside the span
        is read by the span (:meth:`carry_mask`) and is False in both."""
        T = int(reset.shape[1])
        C, N = self.carry, self.top_words
        segment = reset.to(torch.int64).cumsum(dim=1)
        same = segment[:, :, None] == segment[:, None, :]
        index = torch.arange(T, device=reset.device)
        distance = index[:, None] - index[None, :]  # t − s
        count = ends.to(torch.int64).cumsum(dim=1)  # the window's word ends at or before s
        # the window's word ends before t − C: count[t − C − 1], 0 for t ≤ C
        start = torch.zeros_like(count[:, :1]).expand(-1, C + 1)
        before = torch.cat([start, count], dim=1)[:, :T]
        later = before[:, :, None] - count[:, None, :]  # word ends in (s, t − C − 1]
        window = same & (distance > C) & ends[:, None, :] & (later < N)
        going_on = segment == 0
        slot = torch.arange(N, device=reset.device)
        cached = (
            word_valid[:, None, :]
            & going_on[:, :, None]
            & (word_pos[:, None, :] < (index - C)[None, :, None])
            & (slot[None, None, :] < N - before[:, :, None])
        )
        return window, cached

    def read(
        self,
        tokens: Tensor,
        reset: Tensor,
        cache: Cache,
        short: tuple[Tensor, int] | None = None,
    ) -> tuple[Tensor, Cache]:
        """``model.reader_carry``: ``(c, cache)`` for the window ``tokens`` after the windows
        ``cache`` holds (module docstring, "Reading past the window"). ``c [B, T, w]`` is the
        reader's output, as :meth:`forward`'s; the cache returned holds the last ``C``
        positions of the stream after this window, and with ``model.reader_top_words`` its
        last ``N`` word ends, detached.

        ``short``: :meth:`carry_mask`'s, the rows or positions that read with a short span
        (``train.reach_dropout``; None, the default: every row as configured). What they leave
        in the cache was computed under the short span, so the caller keeps a document short
        or whole from its start to its end (the trainer holds a row's draw until its next
        document start, ``docs/reach_dropout.md``): such a document is read, cached positions
        included, exactly as the reader with ``model.reader_carry = n − 1`` reads it."""
        if tokens.ndim != 2 or reset.shape != tokens.shape or reset.dtype is not torch.bool:
            raise ValueError(
                f"Reader needs tokens [B, T] and a bool reset of the same shape, got "
                f"{tuple(tokens.shape)} and {reset.dtype} {tuple(reset.shape)}"
            )
        if not self.carry:
            raise ValueError("read() is the reader with model.reader_carry; call the reader")
        B, T = (int(n) for n in tokens.shape)
        cache = self.check_cache(cache, B)
        keys, values, valid = cache[:3]
        C, N = self.carry, self.top_words
        if short is not None:
            if N:
                raise ValueError(
                    "short rows (train.reach_dropout) narrow the carried span; the top block's "
                    f"word slots (model.reader_top_words={N}) reach beyond any span"
                )
            _check_short(short, B, T)
        x = self.emb(tokens)
        # the angles of the positions −C … T − 1: the window's as forward's, the cached ones
        # those of 1 … C with the sine's sign turned (cos is even, sin odd)
        cos_all, sin_all = self._rotation(max(T, C + 1), x)
        cos = torch.cat([cos_all[1 : C + 1].flip(0), cos_all[:T]])
        sin = torch.cat([-sin_all[1 : C + 1].flip(0), sin_all[:T]])
        mask = self.carry_mask(reset, valid, short)
        word_keys = word_values = None
        top_mask = mask
        if N:  # the top block also reads word ends beyond the span: the window's and the slots
            word_keys, word_values, word_valid, word_pos, last_letter = cache[3:]
            ends = word_ends(tokens, reset, last_letter)
            window_words, cached_words = self.word_masks(reset, ends, word_valid, word_pos)
            spans = mask[:, 0]
            window_reads = spans[:, :, C:] | window_words
            top_mask = torch.cat([spans[:, :, :C], window_reads, cached_words], dim=-1)[:, None]
        mask = _attention_bias(mask, x)
        top_mask = _attention_bias(top_mask, x) if N else mask
        kept_keys: list[Tensor] = []
        kept_values: list[Tensor] = []
        top = len(self.blocks) - 1
        for i, module in enumerate(self.blocks):
            block = cast(_ReaderBlock, module)
            block_cos, block_sin = self._block_rotation(self._block_pairs[i], cos, sin)
            words = N > 0 and i == top
            x, k, v = block.forward_carry(
                x,
                top_mask if words else mask,
                block_cos,
                block_sin,
                keys[i],
                values[i],
                word_keys if words else None,
                word_values if words else None,
            )
            kept_keys.append(_keep_tail(keys[i], k, C, dim=2))
            kept_values.append(_keep_tail(values[i], v, C, dim=2))
        segment = reset.to(torch.int64).cumsum(dim=1)
        last = segment[:, -1:]  # the document the stream is in after the window
        still = valid & (last == 0)  # cached positions of a document that goes on
        # contiguous: the cache keeps one memory layout from window to window (a compiled
        # window is specialised to its inputs' strides)
        valid_next = _keep_tail(still, segment == last, C, dim=1).contiguous()
        out: Tensor = self.norm(x)
        kept: Cache = (
            torch.stack(kept_keys).detach(),
            torch.stack(kept_values).detach(),
            valid_next,
        )
        if N:  # the next word slots: the last N word ends of the document going on
            assert word_keys is not None and word_values is not None
            index = torch.arange(T, device=tokens.device)
            position = torch.cat([word_pos, index.expand(B, T)], dim=1)  # [B, N + T]
            ok = torch.cat([word_valid & (last == 0), ends & (segment == last)], dim=1)
            pick = torch.where(ok, position, _NO_POSITION).topk(N, dim=1).indices
            ok_next = ok.gather(1, pick)  # the most recent first, the empty slots last
            pos_next = torch.where(ok_next, position.gather(1, pick) - T, _NO_POSITION)
            rows = pick[:, None, :, None].expand(-1, self.heads, -1, word_keys.shape[-1])
            all_keys = torch.cat([word_keys, k.to(word_keys.dtype)], dim=2)  # k: the top's
            all_values = torch.cat([word_values, v.to(word_values.dtype)], dim=2)
            kept += (
                all_keys.gather(2, rows).detach(),
                all_values.gather(2, rows).detach(),
                ok_next,
                pos_next,
                word_bytes(tokens[:, -1]),
            )
        return out, kept

    def extra_repr(self) -> str:
        text = f"width={self.width}, heads={self.heads}, span={self.span}"
        if self.position != "rotary":
            text += f", position={self.position}"
        if self.rotary_pairs is not None:
            text += f", rotary_pairs={self.rotary_pairs}"
        if self.rotary_base != _ROTARY_BASE:
            text += f", rotary_base={self.rotary_base:g}"
        if self.carry:
            text += f", carry={self.carry}"
        if self.top_words:
            text += f", top_words={self.top_words}"
        if self.causal:
            text += ", mask=causal"
        if any(self.qk_norm):
            text += f", qk_norm={True if all(self.qk_norm) else self.qk_norm}"
        return text
