"""A6: ``train.route_guard`` detects an established minority route being taken over by the
dominant router unit and halves that unit's applied router step (hsn.train.route_guard)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from hsn.config import RouteGuardConfig, canonical_json, config_hash
from hsn.train.route_guard import RouteGuard, lost_usage
from p1_11_support import restore_determinism  # noqa: F401
from p1_15_support import datasets, gru_config, hsn_config, hsn_trainer
from torch import nn

pytestmark = pytest.mark.usefixtures("restore_determinism")

A5 = (
    "model.routing=topk",
    "train.gate_mode=ste",
    "model.key_address=true",
    "model.selected_readout=true",
)
ON = "train.route_guard.enabled=true"
EVERY = ("train.eval_every=4", "train.ckpt_every=4")


class _Scorer(nn.Module):
    def __init__(self, units: int, features: int) -> None:
        super().__init__()
        self.net = nn.Linear(features, units)


class _Router(nn.Module):
    def __init__(self, units: int = 3, features: int = 5) -> None:
        super().__init__()
        self.scorer = _Scorer(units, features)


def _guard(**settings: object) -> tuple[RouteGuard, _Router]:
    """A guard whose routes are established after one reading and for which any loss above the
    lowest so far counts as not improving, unless ``settings`` say otherwise."""
    torch.manual_seed(0)
    router = _Router()
    cfg = RouteGuardConfig(
        enabled=True,
        **{"persistence": 1, "loss_rise": 0.0, "min_slots": 1, **settings},  # type: ignore[arg-type]
    )
    return RouteGuard(router, cfg, n_units=3, vocab=4), router


def _feed(
    guard: RouteGuard,
    table: list[list[int]],
    loss: float | None = None,
    read: tuple[int, int, int] = (100, 0, 0),
) -> None:
    """One interval's slots (``table[byte][unit]``), the slots per unit at supervised positions
    (unit 0 reads the answers unless given) and the training loss: 1.0 while the routes are
    intact, 2.0 (worse than the best so far) once they are taken, unless given."""
    guard.counts.copy_(torch.tensor(table, dtype=torch.int64).reshape(-1))
    guard.read_counts.copy_(torch.tensor(read, dtype=torch.int64))
    guard.note_loss((1.0 if table is INTACT else 2.0) if loss is None else loss)


#: byte 0: filler, all on the dominant unit 0; byte 1: a specialised byte, 80 % on unit 1
INTACT = [[1000, 0, 0], [20, 80, 0], [0, 0, 0], [0, 0, 0]]
#: the dominant unit has taken half of byte 1's minority slots
TAKEN = [[1000, 0, 0], [60, 40, 0], [0, 0, 0], [0, 0, 0]]
#: ... and then half of the rest
MORE = [[1000, 0, 0], [80, 20, 0], [0, 0, 0], [0, 0, 0]]


def test_default_is_off_and_keeps_the_hash() -> None:
    base = hsn_config(4, *A5)
    assert base.train.route_guard.enabled is False
    assert "route_guard" not in canonical_json(base)
    on = hsn_config(4, *A5, ON)
    assert config_hash(on) != config_hash(base)
    with pytest.raises(ValueError, match="HSN-only"):
        gru_config(4, ON)
    with pytest.raises(ValueError, match="learned routing"):
        hsn_config(4, ON)  # dense routing has no routes to guard


def test_lost_usage_reads_the_routes_a_reader_is_taking() -> None:
    reference = torch.tensor([[1.0, 0, 0], [0.2, 0.8, 0], [0, 0, 0], [0, 0, 0]]).double()
    established = reference >= 0.25  # the filler on unit 0, byte 1 on unit 1
    reader = torch.tensor([True, False, False])  # unit 0 serves the supervised positions
    lost, usage, shares, unit = lost_usage(torch.tensor(TAKEN), reference, established, reader)
    assert shares[1].tolist() == pytest.approx([0.6, 0.4, 0.0])
    assert lost == pytest.approx(0.5)  # byte 1's route on unit 1 kept half of its 80 %
    assert unit == 0  # the reader that took byte 1's slots
    assert usage == pytest.approx(100 * 0.8 / 1100)  # the other units' established routes
    # nothing established: no loss, whatever moved
    nothing = torch.zeros_like(established)
    assert lost_usage(torch.tensor(TAKEN), reference, nothing, reader)[::3] == (0.0, None)
    # the reader below its own record on a byte is not taking it
    grown = torch.tensor([[1.0, 0, 0], [0.6, 0.4, 0], [0, 0, 0], [0, 0, 0]]).double()
    assert lost_usage(torch.tensor(INTACT), grown, grown >= 0.25, reader)[0] == 0.0
    # a route that moves between units that do not read the answers is not this event
    moved = torch.tensor([[1000, 0, 0], [20, 30, 50], [0, 0, 0], [0, 0, 0]])
    assert lost_usage(moved, reference, established, reader)[::3] == (0.0, None)
    assert lost_usage(moved, reference, established, torch.tensor([False, False, True]))[::3] == (
        pytest.approx(0.625),
        2,
    )
    # a byte with too few slots is left out
    assert lost_usage(torch.tensor(TAKEN), reference, established, reader, min_slots=500)[0] == 0.0


def test_event_halves_the_dominant_units_step_only() -> None:
    guard, router = _guard(lost_fraction=0.1)
    _feed(guard, INTACT)
    assert guard.check(100) is None  # the first reading sets the reference
    _feed(guard, INTACT)
    assert guard.check(200) is None and guard.events == []
    _feed(guard, TAKEN)
    event = guard.check(300)
    assert event is not None and event["unit"] == 0 and event["lost"] == pytest.approx(0.5)
    assert event["scale_before"] == 1.0 and event["scale"] == 0.5
    assert guard.scale.tolist() == [0.5, 1.0, 1.0]
    assert int(guard.counts.sum()) == 0  # a check starts a new interval
    # the reference restarted from the current usage: the same usage is no further event
    _feed(guard, TAKEN)
    assert guard.check(400) is None
    # the scaled unit's rows take half a step, the others the whole step
    weight, bias = router.scorer.net.weight, router.scorer.net.bias
    before_w, before_b = weight.detach().clone(), bias.detach().clone()
    guard.capture()
    with torch.no_grad():
        weight.add_(1.0)
        bias.add_(2.0)
    guard.apply()
    torch.testing.assert_close(weight[0], before_w[0] + 0.5)
    torch.testing.assert_close(weight[1:], before_w[1:] + 1.0)
    torch.testing.assert_close(bias, before_b + torch.tensor([1.0, 2.0, 2.0]))
    assert guard.report()["scale"] == [0.5, 1.0, 1.0] and guard.report()["checks"] == 4
    assert guard.last is not None and guard.last["gaining_unit"] is None  # nothing lost at 400


def test_no_event_while_training_improves_or_before_a_route_has_persisted() -> None:
    # the dominant unit absorbing positions while the loss falls is learning
    falling, _ = _guard(lost_fraction=0.1)
    for step, (table, loss) in enumerate([(INTACT, 3.0), (INTACT, 2.0), (TAKEN, 1.0)]):
        _feed(falling, table, loss)
        assert falling.check(100 * (step + 1)) is None
    assert falling.last is not None and falling.last["lost"] == pytest.approx(0.5)
    assert falling.last["worse"] is False and falling.best_loss == 1.0
    # within the tolerance of the lowest loss so far: still not worse
    near, _ = _guard(lost_fraction=0.1, loss_rise=0.05)
    for step, (table, loss) in enumerate([(INTACT, 1.0), (INTACT, 1.0), (TAKEN, 1.04)]):
        _feed(near, table, loss)
        assert near.check(100 * (step + 1)) is None
    _feed(near, MORE, 1.06)  # the reference still holds the intact usage
    assert near.check(400) is not None
    # a route that has not persisted for `persistence` readings is not established yet
    young, _ = _guard(lost_fraction=0.1, persistence=3)
    for step, table in enumerate([INTACT, INTACT, TAKEN]):
        _feed(young, table)
        assert young.check(100 * (step + 1)) is None
    old, _ = _guard(lost_fraction=0.1, persistence=3)
    for step, table in enumerate([INTACT, INTACT, INTACT, INTACT]):
        _feed(old, table)
        assert old.check(100 * (step + 1)) is None
    _feed(old, TAKEN)
    assert old.check(500) is not None
    # a step without a finite loss is left out; an interval without any loss is never "worse"
    blind, _ = _guard(lost_fraction=0.1)
    for step, table in enumerate([INTACT, INTACT, TAKEN]):
        _feed(blind, table)
        blind._loss_sum, blind._loss_n = 0.0, 0
        blind.note_loss(float("nan"))
        assert blind.check(100 * (step + 1)) is None


def test_scale_has_a_floor_and_small_usage_is_ignored() -> None:
    guard, _ = _guard(lost_fraction=0.1, reduce_factor=0.5, min_scale=0.5)
    for step, table in enumerate([INTACT, TAKEN, TAKEN, MORE]):
        _feed(guard, table, loss=float(step + 1))
        guard.check(100 * (step + 1))
    # two events on unit 0 (a reader at its own record again is not taking more), floored
    assert guard.scale.tolist() == [0.5, 1.0, 1.0]
    assert [(e["step"], e["scale"]) for e in guard.events] == [(200, 0.5), (400, 0.5)]
    quiet, _ = _guard(lost_fraction=0.1, min_usage=0.99)
    for table in (INTACT, INTACT, TAKEN):
        _feed(quiet, table)
        assert quiet.check(100) is None  # the established routes hold under 99 % of the slots


def test_the_response_is_on_the_unit_that_gained_the_route() -> None:
    guard, _ = _guard(lost_fraction=0.1)
    for step in (100, 200):
        _feed(guard, INTACT)
        guard.check(step)
    # unit 2 takes byte 1's slots from unit 1 (the filler stays where it was)
    moved = [[1000, 0, 0], [20, 30, 50], [0, 0, 0], [0, 0, 0]]
    _feed(guard, moved, loss=2.0)  # unit 0 reads the answers: not its doing, no event
    assert guard.check(300) is None and guard.last is not None
    assert guard.last["readers"] == [0] and guard.last["gaining_unit"] is None
    other, _ = _guard(lost_fraction=0.1)
    for step in (100, 200):
        _feed(other, INTACT, read=(50, 0, 50))
        other.check(step)
    _feed(other, moved, loss=2.0, read=(50, 0, 50))  # units 0 and 2 read the answers
    event = other.check(300)
    assert event is not None and event["unit"] == 2 and other.last["readers"] == [0, 2]
    assert other.scale.tolist() == [1.0, 1.0, 0.5]


def test_restore_puts_the_dominant_units_rows_back() -> None:
    guard, router = _guard(lost_fraction=0.1, restore=True)
    weight = router.scorer.net.weight
    _feed(guard, INTACT)
    guard.check(100)
    _feed(guard, INTACT)
    guard.check(200)  # no event: the anchor is this reading's weights
    anchor = weight.detach().clone()
    with torch.no_grad():
        weight.add_(1.0)
    moved = weight.detach().clone()
    _feed(guard, TAKEN)
    event = guard.check(300)
    assert event is not None and event["restored_from"] == 200
    torch.testing.assert_close(weight[0], anchor[0])  # the gaining unit's row is back
    torch.testing.assert_close(weight[1:], moved[1:])  # every other row keeps its weights
    # the reference stays: a further loss is another event, restored from the same reading
    _feed(guard, MORE)
    again = guard.check(400)
    assert again is not None and again["restored_from"] == 200 and again["scale"] == 0.25


def test_state_round_trip() -> None:
    guard, _ = _guard(lost_fraction=0.1)
    for step, table in enumerate([INTACT, INTACT, TAKEN]):
        _feed(guard, table)
        guard.check(100 * (step + 1))
    _feed(guard, INTACT)  # a part-filled interval
    other, _ = _guard(lost_fraction=0.1)
    other.load_state_dict(guard.state_dict(), "x.pt")
    assert other.scale.tolist() == guard.scale.tolist() and other.events == guard.events
    assert torch.equal(other.counts, guard.counts)
    assert torch.equal(other.read_counts, guard.read_counts)
    assert torch.equal(other.previous, guard.previous)
    assert other.check(400) == guard.check(400)
    with pytest.raises(ValueError, match="missing"):
        other.load_state_dict(None, "x.pt")
    with pytest.raises(ValueError, match="does not fit"):
        other.load_state_dict({**guard.state_dict(), "scale": torch.ones(5)}, "x.pt")


# --------------------------------------------------------------------------------------------------
# the trainer
# --------------------------------------------------------------------------------------------------


def test_a_quiet_guard_changes_no_training_step(tmp_path: Path) -> None:
    train_ds, dev_ds = datasets()
    plain = hsn_trainer(hsn_config(8, *A5, *EVERY), tmp_path / "plain")
    expected = plain.fit(train_ds, dev_ds, 8)
    # a threshold no interval reaches: the guard watches and never acts
    quiet = (ON, "train.route_guard.interval=2", "train.route_guard.lost_fraction=0.999")
    guarded = hsn_trainer(hsn_config(8, *A5, *EVERY, *quiet), tmp_path / "guarded")
    history = guarded.fit(train_ds, dev_ds, 8)
    assert [s.loss for s in history.steps] == [s.loss for s in expected.steps]
    assert guarded.route_guard is not None and guarded.route_guard.checks == 4
    assert guarded.route_guard.events == [] and plain.route_guard is None
    assert history.measured["route_guard_events"] == 0.0
    assert history.measured["route_guard_scale"] == 1.0
    last = guarded.route_guard.last
    assert last is not None and last["step"] == 8 and 0.0 <= last["established_usage"] <= 1.0


def _rising_loss(trainer: Any) -> None:
    """Tell the guard a training loss that rises with the step (the tiny run's own loss falls),
    so that the run's ordinary route churn is an event; the same function of the step in a
    resumed trainer."""
    guard = trainer.route_guard
    original = guard.note_loss
    guard.note_loss = lambda loss: original(float(trainer.step))


def test_training_with_events_resumes_exactly(tmp_path: Path) -> None:
    train_ds, dev_ds = datasets()
    # thresholds that make the tiny run's ordinary churn an event
    eager = (
        ON,
        "train.route_guard.interval=1",
        "train.route_guard.lost_fraction=0.001",
        "train.route_guard.specialised=0.001",
        "train.route_guard.persistence=1",
        "train.route_guard.loss_rise=0.0",
        "train.route_guard.min_slots=1",
        "train.route_guard.min_usage=0.0",
        "train.lr=0.05",
    )
    cfg = hsn_config(8, *A5, *EVERY, *eager)
    full = hsn_trainer(cfg, tmp_path / "full")
    _rising_loss(full)
    expected = full.fit(train_ds, dev_ds, 8)
    assert full.route_guard is not None and full.route_guard.events, "no event in the tiny run"
    assert float(full.route_guard.scale.min()) < 1.0
    resumed = hsn_trainer(cfg, tmp_path / "resumed")
    _rising_loss(resumed)
    assert resumed.resume(tmp_path / "full" / "step_00000004.pt") == 4
    actual = resumed.fit(train_ds, dev_ds, 8)
    assert [s.loss for s in actual.steps] == [s.loss for s in expected.steps[4:]]
    assert resumed.route_guard is not None
    assert resumed.route_guard.events == full.route_guard.events
    assert resumed.route_guard.scale.tolist() == full.route_guard.scale.tolist()
