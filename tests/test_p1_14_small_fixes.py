"""P1.14 tests: the probe raises when no document is probed; the worker's card watch keeps a
finished arm's process group ours until no process is left in it.

Exps 56, 57, 72 and 73 were labelled ``shared (foreign: ...)`` by our own just-finished arms'
children (exp 55's PID 143244, for example), seen while they exited after their rows said done.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hsn.data.dataset import ByteDataset
from hsn.device import seed_everything
from hsn.experiments import db as dbm
from hsn.experiments import machine
from hsn.experiments.probe import edit_probe, edit_probe_reference
from hsn.experiments.worker import Worker, WorkerOptions, _Watch
from hsn.runner import build_model
from p1_11_support import (
    cpu_ctx,
    db_path,
    docs,
    dry_spec,
    enqueue_spec,
    isolated,  # noqa: F401  (fixture)
    tiny_hsn_config,
)


def test_p1_14_probe_raises_without_documents() -> None:
    cfg = tiny_hsn_config()
    seed_everything(0)
    model = build_model(cfg)
    kv_only = ByteDataset(docs("kv", "dev", 4))
    for probe in (edit_probe, edit_probe_reference):
        with pytest.raises(ValueError, match=r"no document probed among 4 dev documents"):
            probe(model, cpu_ctx(), kv_only)
    assert edit_probe(model, cpu_ctx(), ByteDataset(docs("delayed", "dev", 4)), batch_rows=4)


class _Machine:
    """A fake card: ``gpu`` lists the compute processes, ``groups`` the process groups."""

    def __init__(self) -> None:
        self.gpu: list[machine.GpuProcess] = []
        self.groups: dict[int, int] = {}

    host = staticmethod(machine.host)
    pid_alive = staticmethod(lambda pid, start=None: True)
    classify = staticmethod(machine.classify)
    cpu_hogs = staticmethod(lambda: [])

    def read_gpu(self) -> machine.GpuReading:
        used = sum(p.used_mib for p in self.gpu)
        return machine.GpuReading(True, True, 12_288.0, used, tuple(self.gpu))

    def process_groups(self) -> dict[int, int]:
        return dict(self.groups)


def _set(db: dbm.Database, exp_id: int, **fields: Any) -> None:
    sets = ", ".join(f"{k} = ?" for k in fields)
    with db.write() as conn:
        conn.execute(f"UPDATE experiments SET {sets} WHERE exp_id = ?", (*fields.values(), exp_id))


def _claim(db: dbm.Database) -> int:
    row = dbm.claim_next(db, lambda q, r: q[0]["exp_id"], worker="t", host=machine.host())
    assert row is not None
    return int(row["exp_id"])


@pytest.mark.usefixtures("isolated")
def test_p1_14_card_watch_counts_finishing_group(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    enqueue_spec(db, dry_spec(tmp_path, n=2))
    a, b = (_claim(db) for _ in range(2))
    _set(db, a, pid=1000, pgid=1000)
    _set(db, b, pid=2000, pgid=2000)
    fake = _Machine()
    worker = Worker(db, tmp_path / "runs", WorkerOptions(), machine=fake, log=lambda m: None)
    worker.watch[a], worker.watch[b] = _Watch(), _Watch()
    fake.gpu = [
        machine.GpuProcess(1000, 300.0, "/home/joe/hsn/.venv/bin/python"),
        machine.GpuProcess(2000, 400.0, "/home/joe/hsn/.venv/bin/python"),
        machine.GpuProcess(3000, 50.0, "Xorg"),  # a small desktop context: ignored
    ]
    fake.groups = {1000: 1000, 1001: 1000, 2000: 2000, 3000: 3000}
    card, _ = worker.observe()
    assert card.foreign == () and card.own_used_mib == 700.0
    assert worker.own_pgids == {1000: a, 2000: b}

    # a's child wrote its final status and is still exiting (its CUDA context, a compile worker)
    _set(db, a, status="done")
    card, _ = worker.observe()
    assert card.foreign == (), "our own finishing arm is never foreign"
    assert worker.watch[b].foreign == {} and a in worker.watch[b].own_with
    assert worker.watch[b].status() == f"shared (own: exp {a})"
    assert card.own_used_mib == 400.0  # a's memory is in use, but not one of our running arms'
    assert worker.idle_since is None, "not idle while a finishing group lives"

    # the leader is gone but a compile worker of its group still runs: still ours
    fake.gpu = [p for p in fake.gpu if p.pid != 1000]
    fake.groups.pop(1000)
    worker.observe()
    assert 1000 in worker.own_pgids

    # b finishes too and every process of a's group is gone: a is dropped, b is finishing
    _set(db, b, status="done")
    fake.groups.pop(1001)
    worker.observe()
    assert worker.own_pgids == {2000: b} and worker.idle_since is None
    # once b's group is empty the card is idle again
    fake.gpu = [p for p in fake.gpu if p.pid != 2000]
    fake.groups.pop(2000)
    worker.observe()
    assert worker.own_pgids == {} and worker.idle_since is not None
    # a process of another session is still foreign
    fake.gpu = [machine.GpuProcess(4000, 900.0, "python")]
    fake.groups[4000] = 4000
    card, _ = worker.observe()
    assert card.foreign == ("4000 python 900 MiB",) and worker.idle_since is None
