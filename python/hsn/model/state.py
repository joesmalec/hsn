"""Cell graph state (P1.3): the persistent recurrent state of the HSN cell graph, its incrementally
maintained summary, age counters, resets, and versioned snapshot/restore.

:class:`CellGraphState` holds, for a batch of ``B`` sessions over ``N`` cells of width ``d``:

* ``h [B, N, d]`` the cell states (``h[i, t, r]`` in the plan's notation);
* ``q [B, d]`` the summary ``q_t = (1/N) Σ_i h_i`` (plan §2.4), maintained *incrementally*: after a
  sparse update of the cells ``idx`` it becomes ``q + Σ_idx (h_new − h[idx]) / N``;
* ``age [B, N]`` rounds since each cell was last updated (a router feature, plan §2.4);
* ``pos [B]`` the input position each session has reached.

Two update paths that never mix (plan §1, chunk P1.3):

* **Functional** methods (:meth:`~CellGraphState.reset`, :meth:`~CellGraphState.with_delta`,
  :meth:`~CellGraphState.with_dense`, :meth:`~CellGraphState.tick`) return a new state and keep
  autograd intact, so training (TBPTT, P1.7) can backpropagate through a window of updates and carry
  :meth:`~CellGraphState.detach` between windows. The state is a frozen dataclass; a periodic resync
  of the summary is ``dataclasses.replace(state, q=state.recompute_q())``.
* **In-place** :meth:`~CellGraphState.apply_delta_` is the serving path. It refuses to run while
  grad mode is enabled or while any state tensor carries autograd history, because an in-place
  write would corrupt the recorded graph (the chunk's pitfall).

Snapshots (paper §14, plan P1.3 and P5.3): :meth:`~CellGraphState.snapshot` copies the four
tensors to the CPU and binds them to a :class:`StateMeta` (model version, state-schema version,
tenant, expiry) under a blake2b-256 integrity digest. :meth:`~CellGraphState.restore` accepts a
:class:`StateSnapshot` only and raises :class:`TenantMismatch`, :class:`VersionMismatch`,
:class:`SchemaMismatch`, :class:`IntegrityError` or :class:`StateExpired` (all
:class:`StateRestoreError`) before any tensor reaches a device. Shape agreement never suffices, and
no state is serialised without its meta.

Digest layout (``hashlib.blake2b``, 32-byte digest): the domain tag ``hsn.model.state/1``; then
for each of ``h``, ``q``, ``age``, ``pos`` an ASCII header line ``<name>:<dtype>:<shape>:<bytes>``
followed by the tensor's raw bytes in host byte order (every supported target is little-endian);
then ``meta:<bytes>`` followed by the meta as canonical JSON (sorted keys, compact separators,
UTF-8). The headers make the same bytes under another dtype or shape fail verification.
"""

from __future__ import annotations

import hmac
import json
import math
import time
from dataclasses import asdict, dataclass
from hashlib import blake2b

import torch
from torch import Tensor

__all__ = [
    "CellGraphState",
    "IntegrityError",
    "SchemaMismatch",
    "StateExpired",
    "StateMeta",
    "StateRestoreError",
    "StateSnapshot",
    "TenantMismatch",
    "VersionMismatch",
]

_HASH_DOMAIN = b"hsn.model.state/1"
_DIGEST_BYTES = 32
_TENSOR_FIELDS = ("h", "q", "age", "pos")
_INDEX_DTYPE = torch.int64


# --------------------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------------------


class StateRestoreError(Exception):
    """Base class of every reason :meth:`CellGraphState.restore` refuses a snapshot."""


class VersionMismatch(StateRestoreError):
    """The snapshot was written by another model version."""


class SchemaMismatch(StateRestoreError):
    """The snapshot uses another state-schema version."""


class TenantMismatch(StateRestoreError):
    """The snapshot belongs to another tenant."""


class IntegrityError(StateRestoreError):
    """The snapshot's digest does not match its tensors and meta."""


class StateExpired(StateRestoreError):
    """The snapshot's ``expires_at`` lies in the past."""


# --------------------------------------------------------------------------------------------------
# Meta and snapshot
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StateMeta:
    """What a snapshot is bound to: the model and state-schema versions that produced the state, the
    tenant it belongs to, and the POSIX time (seconds) after which it must not be restored (``None``
    never expires)."""

    model_version: str
    state_schema_version: int
    tenant_id: str
    expires_at: float | None

    def __post_init__(self) -> None:
        for name in ("model_version", "tenant_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"StateMeta.{name} must be a non-empty str, got {value!r}")
        schema = self.state_schema_version
        if isinstance(schema, bool) or not isinstance(schema, int) or schema < 1:
            raise ValueError(f"StateMeta.state_schema_version must be an int >= 1, got {schema!r}")
        expires = self.expires_at
        if expires is not None:
            if (
                isinstance(expires, bool)
                or not isinstance(expires, int | float)
                or not math.isfinite(expires)
            ):
                raise ValueError(
                    f"StateMeta.expires_at must be None or a finite POSIX time, got {expires!r}"
                )
            object.__setattr__(self, "expires_at", float(expires))

    def to_bytes(self) -> bytes:
        """Canonical serialisation (sorted keys, compact JSON, UTF-8); the form that is hashed."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, eq=False)
class StateSnapshot:
    """CPU copies of the four state tensors, the :class:`StateMeta` they are bound to and the
    blake2b-256 digest over both. Build one with :meth:`CellGraphState.snapshot`; load one only
    through :meth:`CellGraphState.restore`, which verifies the digest."""

    h: Tensor
    q: Tensor
    age: Tensor
    pos: Tensor
    meta: StateMeta
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.meta, StateMeta):
            raise TypeError(
                f"StateSnapshot.meta must be a StateMeta, got {type(self.meta).__name__}"
            )
        if not isinstance(self.digest, str) or len(self.digest) != 2 * _DIGEST_BYTES:
            raise ValueError(
                f"StateSnapshot.digest must be {2 * _DIGEST_BYTES} hex characters, got "
                f"{self.digest!r}"
            )
        try:
            bytes.fromhex(self.digest)
        except ValueError as exc:
            raise ValueError(f"StateSnapshot.digest is not hexadecimal: {self.digest!r}") from exc
        _check_layout(self.h, self.q, self.age, self.pos, where="StateSnapshot")
        for name in _TENSOR_FIELDS:
            tensor: Tensor = getattr(self, name)
            if tensor.device.type != "cpu":
                raise ValueError(f"StateSnapshot.{name} must live on the CPU, got {tensor.device}")
            if tensor.requires_grad:
                raise ValueError(f"StateSnapshot.{name} must not require grad")


def _digest(h: Tensor, q: Tensor, age: Tensor, pos: Tensor, meta: StateMeta) -> str:
    """blake2b-256 over the domain tag, each tensor's header and raw bytes, and the meta."""
    hasher = blake2b(digest_size=_DIGEST_BYTES)
    hasher.update(_HASH_DOMAIN)
    for name, tensor in zip(_TENSOR_FIELDS, (h, q, age, pos), strict=True):
        flat = tensor.detach().to("cpu").contiguous().reshape(-1)
        n_bytes = flat.numel() * flat.element_size()
        shape = "x".join(str(n) for n in tensor.shape)
        hasher.update(f"\n{name}:{tensor.dtype}:{shape}:{n_bytes}\n".encode("ascii"))
        # A uint8 view reinterprets the storage bit for bit (bf16/fp16 have no NumPy dtype).
        hasher.update(flat.view(torch.uint8).numpy())
    meta_bytes = meta.to_bytes()
    hasher.update(f"\nmeta:{len(meta_bytes)}\n".encode("ascii"))
    hasher.update(meta_bytes)
    return hasher.hexdigest()


# --------------------------------------------------------------------------------------------------
# The state
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CellGraphState:
    """Per-cell states ``h [B, N, d]``, summary ``q [B, d]``, ``age [B, N]`` and ``pos [B]`` on one
    device. Every tensor is referenced, not copied: do not mutate a tensor after handing it to a
    state. See the module docstring for the functional and in-place update paths."""

    h: Tensor
    q: Tensor
    age: Tensor
    pos: Tensor

    def __post_init__(self) -> None:
        _check_layout(self.h, self.q, self.age, self.pos, where="CellGraphState")

    @classmethod
    def _make(cls, h: Tensor, q: Tensor, age: Tensor, pos: Tensor) -> CellGraphState:
        """A state built without :func:`_check_layout` (P1.10e): only for internal updates whose
        result has a valid layout by construction — the functional updates below after their own
        argument checks, and the model's window path. The public constructor still validates."""
        state = object.__new__(cls)
        object.__setattr__(state, "h", h)
        object.__setattr__(state, "q", q)
        object.__setattr__(state, "age", age)
        object.__setattr__(state, "pos", pos)
        return state

    # -- shape and placement ----------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return int(self.h.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.h.shape[1])

    @property
    def width(self) -> int:
        return int(self.h.shape[2])

    @property
    def device(self) -> torch.device:
        return self.h.device

    @property
    def dtype(self) -> torch.dtype:
        return self.h.dtype

    # -- construction -----------------------------------------------------------------------------

    @classmethod
    def zeros(
        cls, B: int, N: int, d: int, device: torch.device | str, dtype: torch.dtype
    ) -> CellGraphState:
        """A fresh all-zero state (the state at a session start and after a reset)."""
        for name, value in (("B", B), ("N", N), ("d", d)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"CellGraphState.zeros: {name} must be an int >= 1, got {value!r}")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"CellGraphState.zeros: dtype must be a floating dtype, got {dtype!r}")
        dev = torch.device(device)
        return cls(
            h=torch.zeros(B, N, d, device=dev, dtype=dtype),
            q=torch.zeros(B, d, device=dev, dtype=dtype),
            age=torch.zeros(B, N, device=dev, dtype=_INDEX_DTYPE),
            pos=torch.zeros(B, device=dev, dtype=_INDEX_DTYPE),
        )

    # -- functional updates (autograd intact) -----------------------------------------------------

    def reset(self, mask: Tensor | None = None) -> CellGraphState:
        """Zero every session (``mask=None``) or only the sessions where ``mask [B]`` is true.

        Rows that are kept keep their values and their autograd history. ``masked_fill`` rather than
        a multiplication, so a NaN or inf in a row being reset cannot leak into the zeros.
        """
        if mask is None:
            return CellGraphState.zeros(
                self.batch_size, self.n_cells, self.width, self.device, self.dtype
            )
        self._check_mask(mask, (self.batch_size,), "mask")
        return CellGraphState._make(
            h=self.h.masked_fill(mask[:, None, None], 0),
            q=self.q.masked_fill(mask[:, None], 0),
            age=self.age.masked_fill(mask[:, None], 0),
            pos=self.pos.masked_fill(mask, 0),
        )

    def with_delta(self, cell_idx: Tensor, h_new: Tensor) -> CellGraphState:
        """Replace the cells ``cell_idx [B, K]`` by ``h_new [B, K, d]``; ``q`` moves by
        ``Σ (h_new − h[idx]) / N`` and the updated cells' ``age`` becomes 0.

        ``cell_idx`` must hold distinct indices in ``[0, N)`` within each row: a duplicate would be
        counted twice in ``q`` and make the scatter order-dependent. The check synchronises with the
        device once per call (P2.8 may gate it behind a debug flag if it shows in launch profiles).
        Differentiable with respect to ``h``, ``q`` and ``h_new``.
        """
        index = self._check_delta(cell_idx, h_new)
        h_old = self.h.gather(1, index)
        # torch's out-of-place scatter: a clone of h with h_new written at the selected cells.
        h = self.h.scatter(1, index, h_new)
        q = self.q + (h_new - h_old).sum(dim=1) / self.n_cells
        age = self.age.scatter(1, cell_idx, 0)
        return CellGraphState._make(h=h, q=q, age=age, pos=self.pos)

    def with_dense(self, h_new: Tensor) -> CellGraphState:
        """Replace every cell by ``h_new [B, N, d]``: ``q`` is the exact mean (every cell changed,
        so the incremental formula reduces to it without drift) and every ``age`` is 0."""
        if not isinstance(h_new, Tensor):
            raise TypeError(f"h_new must be a torch.Tensor, got {type(h_new).__name__}")
        if h_new.shape != self.h.shape:
            raise ValueError(
                f"h_new must have shape {tuple(self.h.shape)}, got {tuple(h_new.shape)}"
            )
        self._check_like_h(h_new, "h_new")
        return CellGraphState._make(
            h=h_new, q=h_new.mean(dim=1), age=torch.zeros_like(self.age), pos=self.pos
        )

    def tick(self, selected_mask: Tensor) -> CellGraphState:
        """End of a round: ``age += 1`` where ``selected_mask [B, N]`` is false; ``pos += 1``."""
        self._check_mask(selected_mask, (self.batch_size, self.n_cells), "selected_mask")
        return CellGraphState._make(
            h=self.h,
            q=self.q,
            age=self.age + (~selected_mask).to(_INDEX_DTYPE),
            pos=self.pos + 1,
        )

    # -- in-place update (serving) ----------------------------------------------------------------

    def apply_delta_(self, cell_idx: Tensor, h_new: Tensor) -> None:
        """:meth:`with_delta` written into this state's tensors. Serving only: raises
        ``RuntimeError`` unless grad mode is off and no state tensor carries autograd history."""
        if torch.is_grad_enabled():
            raise RuntimeError(
                "apply_delta_ is the serving path and must run under torch.no_grad(); grad mode is "
                "enabled. Training uses the functional with_delta."
            )
        tracked = [name for name in _TENSOR_FIELDS if getattr(self, name).requires_grad]
        if tracked:
            raise RuntimeError(
                "apply_delta_ refuses state tensors with autograd history "
                f"({', '.join(tracked)} require grad); detach() the state first"
            )
        index = self._check_delta(cell_idx, h_new)
        h_old = self.h.gather(1, index)
        self.q.add_((h_new - h_old).sum(dim=1) / self.n_cells)
        self.h.scatter_(1, index, h_new)
        self.age.scatter_(1, cell_idx, 0)

    # -- summary, carry, snapshot -----------------------------------------------------------------

    def recompute_q(self) -> Tensor:
        """The exact mean over cells: for tests and the periodic resync (``train.q_resync_every``).

        Resync functionally with ``dataclasses.replace(state, q=state.recompute_q())``.
        """
        return self.h.mean(dim=1)

    def detach(self) -> CellGraphState:
        """The same values without autograd history (the state carried between TBPTT windows)."""
        return CellGraphState._make(
            h=self.h.detach(), q=self.q.detach(), age=self.age.detach(), pos=self.pos.detach()
        )

    def snapshot(self, meta: StateMeta) -> StateSnapshot:
        """CPU copies of the tensors bound to ``meta`` under an integrity digest. Always a copy, so
        later updates of this state (in place or not) cannot alter the snapshot."""
        if not isinstance(meta, StateMeta):
            raise TypeError(f"snapshot needs a StateMeta, got {type(meta).__name__}")
        copies = {
            name: getattr(self, name).detach().contiguous().to(device="cpu", copy=True)
            for name in _TENSOR_FIELDS
        }
        digest = _digest(copies["h"], copies["q"], copies["age"], copies["pos"], meta)
        return StateSnapshot(**copies, meta=meta, digest=digest)

    @staticmethod
    def restore(
        snap: StateSnapshot, expected: StateMeta, device: torch.device | str
    ) -> CellGraphState:
        """Load ``snap`` onto ``device`` after verifying it against ``expected``.

        Checks, in order: tenant, model version and schema version (cheap scope checks), then the
        digest (which proves the meta and tensors are the ones written), then expiry of the verified
        meta against ``time.time()``. ``expected.expires_at`` is the caller's own value and is not
        compared. Anything that is not a :class:`StateSnapshot` is refused outright: tensors are
        never loaded because their shapes fit.
        """
        if not isinstance(snap, StateSnapshot):
            raise TypeError(
                f"restore accepts a StateSnapshot, got {type(snap).__name__}; bare tensors are "
                "never loaded, whatever their shape"
            )
        if not isinstance(expected, StateMeta):
            raise TypeError(f"restore needs the expected StateMeta, got {type(expected).__name__}")
        found = snap.meta
        if found.tenant_id != expected.tenant_id:
            raise TenantMismatch(
                f"snapshot belongs to tenant {found.tenant_id!r}, expected {expected.tenant_id!r}"
            )
        if found.model_version != expected.model_version:
            raise VersionMismatch(
                f"snapshot was written by model version {found.model_version!r}, expected "
                f"{expected.model_version!r}"
            )
        if found.state_schema_version != expected.state_schema_version:
            raise SchemaMismatch(
                f"snapshot uses state schema {found.state_schema_version}, expected "
                f"{expected.state_schema_version}"
            )
        actual = _digest(snap.h, snap.q, snap.age, snap.pos, found)
        if not hmac.compare_digest(actual, snap.digest):
            raise IntegrityError(
                f"snapshot digest {snap.digest[:16]}... does not match its tensors and meta "
                f"({actual[:16]}...)"
            )
        if found.expires_at is not None:
            now = time.time()
            if now >= found.expires_at:
                raise StateExpired(
                    f"snapshot expired at {found.expires_at:.3f} (now {now:.3f}, "
                    f"{now - found.expires_at:.3f} s ago)"
                )
        dev = torch.device(device)
        return CellGraphState(
            h=snap.h.to(device=dev, copy=True),
            q=snap.q.to(device=dev, copy=True),
            age=snap.age.to(device=dev, copy=True),
            pos=snap.pos.to(device=dev, copy=True),
        )

    # -- validation -------------------------------------------------------------------------------

    def _check_like_h(self, tensor: Tensor, name: str) -> None:
        if tensor.dtype != self.h.dtype:
            raise ValueError(f"{name} must have dtype {self.h.dtype}, got {tensor.dtype}")
        if tensor.device != self.device:
            raise ValueError(f"{name} must live on {self.device}, got {tensor.device}")

    def _check_mask(self, mask: Tensor, shape: tuple[int, ...], name: str) -> None:
        if not isinstance(mask, Tensor):
            raise TypeError(f"{name} must be a bool torch.Tensor, got {type(mask).__name__}")
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must have dtype torch.bool, got {mask.dtype}")
        if mask.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}")
        if mask.device != self.device:
            raise ValueError(f"{name} must live on {self.device}, got {mask.device}")

    def _check_delta(self, cell_idx: Tensor, h_new: Tensor) -> Tensor:
        """Validate a sparse update and return ``cell_idx`` expanded to ``[B, K, d]`` for gather
        and scatter. Range and distinctness need the values, so this synchronises once."""
        B, N, d = self.batch_size, self.n_cells, self.width
        if not isinstance(cell_idx, Tensor) or not isinstance(h_new, Tensor):
            raise TypeError("cell_idx and h_new must be torch.Tensors")
        if cell_idx.dtype != _INDEX_DTYPE:
            raise ValueError(f"cell_idx must have dtype torch.int64, got {cell_idx.dtype}")
        if cell_idx.dim() != 2 or cell_idx.shape[0] != B or not 1 <= cell_idx.shape[1] <= N:
            raise ValueError(
                f"cell_idx must have shape [B, K] = [{B}, 1..{N}], got {tuple(cell_idx.shape)}"
            )
        K = int(cell_idx.shape[1])
        if h_new.shape != (B, K, d):
            raise ValueError(f"h_new must have shape {(B, K, d)}, got {tuple(h_new.shape)}")
        self._check_like_h(h_new, "h_new")
        if cell_idx.device != self.device:
            raise ValueError(f"cell_idx must live on {self.device}, got {cell_idx.device}")
        ordered = cell_idx.sort(dim=1).values
        duplicates = (ordered[:, 1:] == ordered[:, :-1]).sum()
        lowest, highest, n_duplicates = (
            int(v)
            for v in torch.stack([ordered[:, 0].min(), ordered[:, -1].max(), duplicates]).tolist()
        )
        if lowest < 0 or highest >= N:
            raise IndexError(f"cell_idx must lie in [0, {N}), got values in [{lowest}, {highest}]")
        if n_duplicates:
            raise ValueError(
                f"cell_idx must be distinct within each row ({n_duplicates} duplicate "
                f"{'entry' if n_duplicates == 1 else 'entries'})"
            )
        return cell_idx.unsqueeze(-1).expand(B, K, d)


def _check_layout(h: Tensor, q: Tensor, age: Tensor, pos: Tensor, *, where: str) -> None:
    """Shapes ``[B, N, d]``, ``[B, d]``, ``[B, N]``, ``[B]``; floating ``h`` and ``q`` of one dtype;
    int64 counters; one device; every dimension >= 1."""
    for name, tensor in zip(_TENSOR_FIELDS, (h, q, age, pos), strict=True):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{where}.{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if h.dim() != 3 or min(h.shape) < 1:
        raise ValueError(
            f"{where}.h must have shape [B, N, d] with B, N, d >= 1, got {tuple(h.shape)}"
        )
    if not h.is_floating_point():
        raise ValueError(f"{where}.h must be floating point, got {h.dtype}")
    B, N, d = (int(n) for n in h.shape)
    if q.shape != (B, d):
        raise ValueError(f"{where}.q must have shape [B, d] = {(B, d)}, got {tuple(q.shape)}")
    if q.dtype != h.dtype:
        raise ValueError(f"{where}.q must have dtype {h.dtype} like h, got {q.dtype}")
    if age.shape != (B, N):
        raise ValueError(f"{where}.age must have shape [B, N] = {(B, N)}, got {tuple(age.shape)}")
    if age.dtype != _INDEX_DTYPE:
        raise ValueError(f"{where}.age must have dtype torch.int64, got {age.dtype}")
    if pos.shape != (B,):
        raise ValueError(f"{where}.pos must have shape [B] = {(B,)}, got {tuple(pos.shape)}")
    if pos.dtype != _INDEX_DTYPE:
        raise ValueError(f"{where}.pos must have dtype torch.int64, got {pos.dtype}")
    devices = {str(tensor.device) for tensor in (h, q, age, pos)}
    if len(devices) != 1:
        raise ValueError(f"{where} tensors must share one device, got {sorted(devices)}")
