"""Selection schedules (P2.2): the gate mode, the temperature, the exploration rate and the
dense-fallback probability as functions of the optimiser step.

Plan §6 of the paper and chunk P2.2: training through hard selection warms up with soft gates, then
switches to the configured estimator (``train.gate_mode``: ``ste`` or ``reinforce``; ``soft`` keeps
the relaxation for the whole run), anneals the temperature of the relaxation, decays the exploration
rate and runs a fraction of samples dense. Every value is a pure function of the step the trainer
hands the model (:meth:`hsn.model.hsn.HSN.set_train_step`), so a resumed run continues the same
schedule and a probe (P2.3) can evaluate the schedule state of any step.

Config keys (Appendix A.3, ``train.*``)::

    gate_mode            soft | ste | reinforce      the estimator after the warm-up
    soft_until_step      steps trained with soft gates before gate_mode takes over (0: none)
    temperature_{start,end,steps}   τ: linear from start to end over `steps` steps, then held
    epsilon_{start,end,steps}       ε: the same shape
    p_dense              probability that a sample runs the dense step (constant)

A linear ramp is the gentlest start (the chunk's pitfall: a temperature annealed too fast kills
the gradient through the soft mask); ``steps = 0`` holds the start value and then ``end`` must equal
``start``, because an ``end`` that is silently never reached is exactly the "annealed too fast or
never" trap the config layer should refuse (errors over silence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

from hsn.config import GateMode, TrainConfig

__all__ = ["SelectionSchedule"]


def _check_step(step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be an int >= 0, got {step!r}")
    return step


def _ramp(start: float, end: float, steps: int, step: int) -> float:
    """``start`` at step 0, ``end`` from step ``steps`` on, linear in between; ``steps = 0`` is
    the constant ``start`` (the constructor guarantees ``end == start`` then)."""
    if steps <= 0 or step >= steps:
        return end if steps > 0 else start
    return start + (end - start) * (step / steps)


@dataclass(frozen=True)
class SelectionSchedule:
    """The P2.2 schedule values of a run (module docstring). Build it with :meth:`from_config` or
    :meth:`neutral`; every field is validated as ``train.*`` would be."""

    gate_mode: GateMode = "soft"
    soft_until: int = 0
    temperature_start: float = 1.0
    temperature_end: float = 1.0
    temperature_steps: int = 0
    epsilon_start: float = 0.0
    epsilon_end: float = 0.0
    epsilon_steps: int = 0
    p_dense_value: float = 0.0

    def __post_init__(self) -> None:
        if self.gate_mode not in ("soft", "ste", "reinforce"):
            raise ValueError(
                f"gate_mode must be soft, ste or reinforce, got {self.gate_mode!r} "
                "(train.gate_mode)"
            )
        for name in ("soft_until", "temperature_steps", "epsilon_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an int >= 0, got {value!r}")
        for name in ("temperature_start", "temperature_end"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
        for name in ("epsilon_start", "epsilon_end", "p_dense_value"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
        for what in ("temperature", "epsilon"):
            start, end = getattr(self, f"{what}_start"), getattr(self, f"{what}_end")
            if getattr(self, f"{what}_steps") == 0 and start != end:
                raise ValueError(
                    f"train.{what}_steps=0 holds train.{what}_start={start!r} for the whole run, "
                    f"but train.{what}_end={end!r} differs and would never be reached; give a "
                    f"step count or set the end equal to the start"
                )

    @classmethod
    def from_config(cls, train: TrainConfig) -> Self:
        """The schedule ``train.*`` describes."""
        if not isinstance(train, TrainConfig):
            raise TypeError(f"from_config needs a TrainConfig, got {type(train).__name__}")
        return cls(
            gate_mode=train.gate_mode,
            soft_until=train.soft_until_step,
            temperature_start=train.temperature_start,
            temperature_end=train.temperature_end,
            temperature_steps=train.temperature_steps,
            epsilon_start=train.epsilon_start,
            epsilon_end=train.epsilon_end,
            epsilon_steps=train.epsilon_steps,
            p_dense_value=train.p_dense,
        )

    @classmethod
    def neutral(cls) -> Self:
        """The config defaults: soft gates throughout, τ = 1, no exploration, no dense fallback."""
        return cls()

    # -- the schedule -----------------------------------------------------------------------------

    def gate(self, step: int) -> GateMode:
        """``"soft"`` while ``step < soft_until`` (the warm-up), then ``gate_mode``."""
        return "soft" if _check_step(step) < self.soft_until else self.gate_mode

    def temperature(self, step: int) -> float:
        """τ at ``step``: linear from ``temperature_start`` to ``temperature_end`` over
        ``temperature_steps`` steps, then held."""
        return _ramp(
            self.temperature_start,
            self.temperature_end,
            self.temperature_steps,
            _check_step(step),
        )

    def epsilon(self, step: int) -> float:
        """ε at ``step``: the probability that one selected cell is replaced by a random one."""
        return _ramp(self.epsilon_start, self.epsilon_end, self.epsilon_steps, _check_step(step))

    def p_dense(self, step: int) -> float:
        """The probability that a sample runs the dense step at ``step`` (``train.p_dense``;
        constant, but a function of the step like the others so callers treat it alike)."""
        _check_step(step)
        return self.p_dense_value

    def is_neutral(self) -> bool:
        """True when nothing stochastic is scheduled (no exploration, no dense fallback)."""
        return self.epsilon_start == 0.0 and self.epsilon_end == 0.0 and self.p_dense_value == 0.0
