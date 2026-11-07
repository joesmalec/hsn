"""P1.11 tests: the scheduler's decision (simulated on the CPU), worker crash recovery, reaping,
process-group ownership, cancel safety, and two workers plus a reporter on one database.

No test touches the real card: the isolation fixture sets ``HSN_EXPERIMENTS_FAKE_MACHINE=1`` (no
nvidia-smi, an idle CPU) and hides CUDA from every child; the scheduler's GPU cases use fake cards.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from hsn.experiments import db as dbm
from hsn.experiments import machine
from hsn.experiments.report import build_report, format_markdown
from hsn.experiments.scheduler import (
    MAX_BYPASS,
    Card,
    Launch,
    Limits,
    QueueItem,
    RunningItem,
    Wait,
    decide,
)
from hsn.experiments.worker import Worker, WorkerOptions, kill_experiment
from p1_11_support import REPO, db_path, dry_spec, enqueue_spec, isolated  # noqa: F401

pytestmark = pytest.mark.usefixtures("isolated")
LIMITS = Limits(max_concurrent=3, reserve_mib=1024, idle_gap_s=60)
CARD = Card(present=True, ok=True, total_mib=12288, used_mib=500, own_used_mib=0)


def q(
    exp_id: int, *, device: str = "cuda", timing: bool = False, mem: float | None = 3000
) -> QueueItem:
    return QueueItem(exp_id, device, timing, mem)


def r(
    exp_id: int,
    *,
    device: str = "cuda",
    timing: bool = False,
    mem: float | None = 3000,
    used: float = 0,
) -> RunningItem:
    return RunningItem(exp_id, device, timing, mem, used)


def test_p1_11_scheduler_memory_and_concurrency_limits() -> None:
    assert decide([q(1)], [], CARD, limits=LIMITS) == Launch(1)
    # memory: 12288 - 1024 reserve - 500 foreign/desktop - running estimates
    assert isinstance(decide([q(2, mem=7000)], [r(1, mem=3000)], CARD, limits=LIMITS), Launch)
    wait = decide([q(2, mem=7000)], [r(1, mem=4000)], CARD, limits=LIMITS)
    assert isinstance(wait, Wait) and "MiB available" in wait.reason
    # a running arm counts with the larger of its estimate and its current use
    assert isinstance(
        decide([q(2, mem=5000)], [r(1, mem=3000, used=6000)], CARD, limits=LIMITS), Wait
    )
    # memory used by others (desktop, graphics) reduces the budget
    busy = Card(present=True, ok=True, total_mib=12288, used_mib=6000, own_used_mib=0)
    assert isinstance(decide([q(1, mem=6000)], [], busy, limits=LIMITS), Wait)
    # concurrency
    three = [r(1, mem=100), r(2, mem=100), r(3, mem=100)]
    assert "max_concurrent" in decide([q(4, mem=100)], three, CARD, limits=LIMITS).reason  # type: ignore[union-attr]
    # FIFO: the oldest that fits runs; a younger one may pass a blocked head, MAX_BYPASS times
    queue = [q(5, mem=9000), q(6, mem=1000)]
    running = [r(1, mem=3000)]
    assert decide(queue, running, CARD, limits=LIMITS) == Launch(6, bypassed_head=True)
    starved = decide(queue, running, CARD, limits=LIMITS, head_bypassed=MAX_BYPASS)
    assert isinstance(starved, Wait) and "waited" in starved.reasons[6]
    # an arm with no estimate runs alone; CPU arms only count against max_concurrent
    assert isinstance(decide([q(7, mem=None)], [r(1)], CARD, limits=LIMITS), Wait)
    assert decide([q(7, mem=None)], [], CARD, limits=LIMITS) == Launch(7)
    assert decide([q(8, device="cpu", mem=None)], [r(1), r(2)], Card(), limits=LIMITS) == Launch(8)
    # a foreign GPU process blocks CUDA arms, not CPU arms; so does an unreadable or absent card
    foreign = Card(
        present=True, ok=True, total_mib=12288, used_mib=3000, foreign=("123 python 2900 MiB",)
    )
    assert "foreign GPU process" in decide([q(1)], [], foreign, limits=LIMITS).reason  # type: ignore[union-attr]
    assert decide([q(1), q(2, device="cpu")], [], foreign, limits=LIMITS) == Launch(
        2, bypassed_head=True
    )
    failed = Card(present=True, ok=False, error="timeout")
    assert "nvidia-smi failed: timeout" in decide([q(1)], [], failed, limits=LIMITS).reason  # type: ignore[union-attr]
    assert "no nvidia-smi" in decide([q(1)], [], Card(), limits=LIMITS).reason  # type: ignore[union-attr]
    assert decide([q(1)], [], CARD, limits=LIMITS, paused=True) == Wait("paused")
    assert decide([], [], CARD, limits=LIMITS) == Wait("queue empty")


def test_p1_11_timing_arm_runs_alone() -> None:
    timing = q(1, timing=True)
    # it waits for our running arms to drain, then for two idle looks 60 s apart
    assert "nothing of ours" in decide([timing], [r(9)], CARD, limits=LIMITS).reason  # type: ignore[union-attr]
    assert (
        "idle card and CPU"
        in decide([timing], [], CARD, limits=LIMITS, idle_since=None, now=100).reason
    )  # type: ignore[union-attr]
    assert isinstance(decide([timing], [], CARD, limits=LIMITS, idle_since=50, now=100), Wait)
    assert decide([timing], [], CARD, limits=LIMITS, idle_since=40, now=100) == Launch(1)
    # while a timing arm runs nothing else launches, CPU arms included
    assert decide([q(2, device="cpu")], [r(1, timing=True)], CARD, limits=LIMITS) == Wait(
        "a timing arm is running"
    )
    # younger arms may pass a waiting timing arm only MAX_BYPASS times, then the queue drains
    queue = [timing, q(2)]
    assert decide(queue, [r(9)], CARD, limits=LIMITS) == Launch(2, bypassed_head=True)
    assert isinstance(decide(queue, [r(9)], CARD, limits=LIMITS, head_bypassed=MAX_BYPASS), Wait)


def _claim_all(db: dbm.Database) -> list[int]:
    ids = []
    while True:
        row = dbm.claim_next(db, lambda q_, r_: q_[0]["exp_id"], worker="test", host=machine.host())
        if row is None:
            return ids
        ids.append(int(row["exp_id"]))


def _set(db: dbm.Database, exp_id: int, **fields: Any) -> None:
    sets = ", ".join(f"{k} = ?" for k in fields)
    with db.write() as conn:
        conn.execute(f"UPDATE experiments SET {sets} WHERE exp_id = ?", (*fields.values(), exp_id))


def _worker(db: dbm.Database, tmp: Path, **options: Any) -> Worker:
    return Worker(db, tmp / "runs", WorkerOptions(**options), log=lambda m: None)


@pytest.mark.parametrize("requeue", [False, True])
def test_p1_11_worker_recovers_dead_running_row(tmp_path: Path, requeue: bool) -> None:
    db = dbm.Database(db_path())
    enqueue_spec(db, dry_spec(tmp_path, n=5))
    dead_pid, live = 0, os.getpid()
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    dead_pid = done.pid
    zombie = subprocess.Popen([sys.executable, "-c", "pass"])  # exits, not reaped: a zombie
    for _ in range(100):
        if not machine.pid_alive(zombie.pid):
            break
        time.sleep(0.05)
    ids = _claim_all(db)
    _set(db, ids[0], pid=dead_pid)
    _set(db, ids[1], pid=zombie.pid, pid_start=machine.pid_start(zombie.pid))
    _set(db, ids[2], pid=live, pid_start="not-the-start-time")  # a reused PID
    _set(db, ids[3], claimed_at="2026-01-01T00:00:00+00:00")  # claimed long ago, never started
    _set(db, ids[4], pid=live, pid_start=machine.pid_start(live))  # alive: kept
    worker = _worker(db, tmp_path, requeue_orphans=requeue)
    gone = worker.recover()
    zombie.wait()
    assert sorted(gone) == ids[:4]
    statuses = {e: db.experiment(e)["status"] for e in ids}
    assert statuses == {**{e: "failed" for e in ids[:4]}, ids[4]: "running"}
    assert (
        "gone" in db.experiment(ids[0])["error"] and "no process" in db.experiment(ids[3])["error"]
    )
    requeued = [row["requeue_of"] for row in db.queued()]
    assert requeued == (ids[:4] if requeue else [])
    assert worker.recover() == [], "recovery is idempotent"
    # a child whose row was recovered stops at its next guarded write
    assert not dbm.child_progress(db, ids[0], dead_pid, {"step": 1})


def test_p1_11_worker_reaps_children_without_a_result(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    enqueue_spec(db, dry_spec(tmp_path, n=2))
    a, b = _claim_all(db)
    worker = _worker(db, tmp_path)
    for exp_id, code in ((a, 0), (b, 3)):
        proc = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])
        proc.wait()
        _set(db, exp_id, pid=proc.pid)
        worker.children[exp_id] = proc
    worker.reap()
    assert not worker.children
    assert (
        db.experiment(a)["status"] == "failed"
        and "exited 0 without a result" in db.experiment(a)["error"]
    )
    assert db.experiment(b)["exit_code"] == 3 and "exited with 3" in db.experiment(b)["error"]


def test_p1_11_own_processes_by_group() -> None:
    # a child that leads its own group (as the worker starts it) and has a grandchild in it
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "time.sleep(30)",
        ],
        start_new_session=True,
    )
    try:
        for _ in range(100):
            groups = machine.process_groups()
            grand = [p for p, g in groups.items() if g == child.pid and p != child.pid]
            if grand:
                break
            time.sleep(0.05)
        assert grand, "the grandchild shares the child's process group"
        procs = [
            machine.GpuProcess(grand[0], 100.0, "python3"),
            machine.GpuProcess(os.getpid(), 50.0, "python"),  # another Python process: foreign
            machine.GpuProcess(999_999, 30.0, "nautilus"),  # a small desktop context: ignored
        ]
        own, foreign, ignored = machine.classify(procs, [child.pid], groups)
        assert [p.pid for p in own] == [grand[0]]
        assert [p.pid for p in foreign] == [os.getpid()]
        assert [p.name for p in ignored] == ["nautilus"]
    finally:
        os.killpg(child.pid, 15)
        child.wait(timeout=20)


def test_p1_11_cancel_only_signals_our_run_arm(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    enqueue_spec(db, dry_spec(tmp_path, n=2))
    a, _b = _claim_all(db)
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        _set(db, a, pid=sleeper.pid, pgid=sleeper.pid)
        assert not kill_experiment(db, a), "not a run-arm process of ours: never signalled"
        assert sleeper.poll() is None
    finally:
        sleeper.kill()
        sleeper.wait()


def test_p1_11_queue_side_imports_no_torch() -> None:
    code = (
        "import sys\n"
        "import hsn.experiments.cli, hsn.experiments.db, hsn.experiments.spec\n"
        "import hsn.experiments.rules\n"
        "import hsn.experiments.scheduler, hsn.experiments.machine, hsn.experiments.worker\n"
        "import hsn.experiments.report, hsn.experiments.importer, hsn.experiments.run_arm\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('torch'))\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120, cwd=REPO)


def test_p1_11_concurrent_workers_and_reporter(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    report = enqueue_spec(db, dry_spec(tmp_path, n=20, sleep_s=0.2))
    assert len(report.added) == 20
    root = tmp_path / "runs"
    cmd = [
        sys.executable,
        "-m",
        "hsn.experiments",
        "--db",
        str(db_path()),
        "--root",
        str(root),
        "worker",
        "--exit-when-empty",
        "--poll",
        "0.05",
        "--max-concurrent",
        "3",
    ]
    workers = [
        subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for _ in range(2)
    ]
    errors: list[BaseException] = []
    snapshots: list[int] = []
    stop = threading.Event()

    def reporter() -> None:
        reader = dbm.Database(db_path())
        try:
            while not stop.is_set():
                rows = build_report(reader, sweep="dry", per_seed=True).rows
                assert len(rows) == 20
                assert {row["status"] for row in rows} <= {"queued", "running", "done"}
                format_markdown(build_report(reader, sweep="dry"))
                snapshots.append(sum(row["status"] == "done" for row in rows))
                time.sleep(0.02)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=reporter)
    thread.start()
    try:
        outputs = [w.communicate(timeout=240)[0].decode() for w in workers]
    finally:
        stop.set()
        thread.join(timeout=30)
        for w in workers:
            if w.poll() is None:
                w.kill()
    assert not errors, errors
    assert [w.returncode for w in workers] == [0, 0], outputs
    rows = db.rows(
        "SELECT exp_id, arm_id, status, worker, pid, started_at FROM experiments ORDER BY exp_id"
    )
    assert len(rows) == 20, "no arm was claimed twice (no extra attempts)"
    assert {row["status"] for row in rows} == {"done"}
    assert all(row["pid"] and row["started_at"] for row in rows)
    assert len({row["worker"] for row in rows}) == 2, "both workers ran arms"
    assert db.row("SELECT COUNT(*) AS n FROM results")["n"] == 20  # type: ignore[index]
    assert snapshots == sorted(snapshots), "every snapshot is consistent (done never decreases)"
    assert snapshots[-1] <= 20


def test_p1_11_cancel_matches_the_exact_experiment(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    enqueue_spec(db, dry_spec(tmp_path, n=1))
    (a,) = _claim_all(db)
    sleep = "import time; time.sleep(30)"

    def fake_run_arm(exp: str) -> subprocess.Popen[bytes]:
        # a process whose command line reads like a run-arm child of experiment `exp`
        argv = [sys.executable, "-c", sleep, "hsn.experiments", "run-arm", "--exp", exp]
        return subprocess.Popen(argv, start_new_session=True)

    other, ours = fake_run_arm(f"{a}0"), fake_run_arm(str(a))
    try:
        time.sleep(0.3)
        _set(db, a, pid=other.pid, pgid=other.pid)
        assert not kill_experiment(db, a), "run-arm --exp N0 is another experiment"
        assert other.poll() is None
        _set(db, a, pid=ours.pid, pgid=ours.pid)
        assert kill_experiment(db, a)
        assert ours.wait(timeout=20) != 0
    finally:
        for proc in (other, ours):
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def test_p1_11_worker_watches_only_its_children_and_orphans(tmp_path: Path) -> None:
    db = dbm.Database(db_path())
    enqueue_spec(db, dry_spec(tmp_path, n=2))
    a, b = _claim_all(db)
    live = os.getpid()
    start = machine.pid_start(live)
    # a: launched by a worker that still runs (this test process); b: by a worker that is gone
    _set(db, a, pid=live, pid_start=start, worker=f"{machine.host()}:{live}")
    _set(db, b, pid=live, pid_start=start, worker=f"{machine.host()}:999999999")
    worker = _worker(db, tmp_path)
    assert worker.recover() == []
    assert set(worker.watch) == {b} and worker.adopted == {b}
    card, _by_group = worker.observe()
    assert card.present is False  # the fake machine: a CPU host
    assert worker.watch[b].samples == 1


def test_p1_11_foreign_sightings_are_counted() -> None:
    from hsn.experiments.worker import _seen

    table: dict[str, dict[str, Any]] = {}
    _seen(table, 42, 100.0, name="dracut", cpu=60.0)
    _seen(table, 42, 130.0, name="dracut", cpu=94.0)
    _seen(table, 7, 130.0, name="python", used_mib=800.0)
    assert table["42"] == {
        "pid": 42,
        "first": 100.0,
        "last": 130.0,
        "looks": 2,
        "name": "dracut",
        "cpu": 94.0,
    }
    assert table["7"]["looks"] == 1 and table["7"]["used_mib"] == 800.0


def test_p1_11_cpu_watch_counts_the_arms_process_group(tmp_path: Path) -> None:
    """A process our arm spawned between the process-group look and top's sample (an inductor
    compile worker, say) is in the arm's group: never a foreign CPU user."""
    db = dbm.Database(db_path())
    enqueue_spec(db, dry_spec(tmp_path, n=1))
    (a,) = _claim_all(db)
    _set(db, a, pid=1000, pgid=1000, timing=1)
    looks = {"n": 0}

    class Machine:
        host = staticmethod(machine.host)
        pid_alive = staticmethod(lambda pid, start=None: True)

        @staticmethod
        def read_gpu() -> machine.GpuReading:
            return machine.GpuReading(present=False, ok=True)

        @staticmethod
        def process_groups() -> dict[int, int]:
            looks["n"] += 1
            # the compile worker 2001 appears only in the second look (after top)
            return {1000: 1000} if looks["n"] == 1 else {1000: 1000, 2001: 1000}

        @staticmethod
        def classify(*args: Any, **kwargs: Any) -> Any:
            return machine.classify(*args, **kwargs)

        @staticmethod
        def cpu_hogs() -> list[tuple[int, float, str]]:
            return [(2001, 340.0, "python"), (3003, 95.0, "dracut")]

    worker = Worker(db, tmp_path / "runs", WorkerOptions(), machine=Machine(), log=lambda m: None)
    worker.watch[a] = __import__("hsn.experiments.worker", fromlist=["_Watch"])._Watch()
    worker.observe()
    assert worker.cpu_hogs == [(3003, 95.0, "dracut")]
    assert set(worker.watch[a].cpu_foreign) == {"3003"}
