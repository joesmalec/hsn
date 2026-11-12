"""The sweep worker (P1.11): a long-running scheduler that pops the FIFO queue, launches arms
detached and watches them. Imports without torch.

Every poll (default 5 s), in this order:

1. **Reap** its own children. A child that exited while its row still says ``running`` (it crashed
   before its final write, or exited 0 without a result) becomes ``failed`` with the exit code and
   the tail of its log.
2. **Recover.** Every ``running`` row on this host whose process is gone, a zombie or a reused PID
   (another start time), or that has had no PID for ``claim_grace_s`` after its claim, becomes
   ``failed`` ("process gone"); with ``--requeue-orphans`` a new attempt is queued too. A worker
   killed or a machine rebooted leaves such rows; children of a dead worker that are still alive
   keep running, finish their own rows and count in the next worker's budget.
3. **Watch** the card (nvidia-smi every poll) and, while a timing arm runs or waits, the CPU
   (``top`` every 30 s). Processes in our arms' process groups are ours; any other Python process,
   or any other process of at least 200 MiB, is foreign. Per running experiment the worker keeps
   the peak MiB of its group, the foreign processes and the other arms of ours seen during it,
   flushed to the row every 60 s and when the child is reaped: ``card_status`` is ``idle``,
   ``shared (own: ...)`` or ``shared (foreign: ...)``. **P1.14:** the process group of an arm this
   worker launched or adopted stays ours after its row leaves ``running`` (the child writes its
   final status, then exits, its CUDA context and inductor's compile workers last) until no process
   is left in it: such a *finishing* group is never foreign, counts as ``own`` for the arms it
   overlaps, keeps the card from being idle (a timing arm never starts beside it) and its memory
   counts as used, not as one of our running arms'.
4. **Decide** (:func:`hsn.experiments.scheduler.decide`) and **claim** inside one
   ``BEGIN IMMEDIATE`` transaction, so that several workers on one database never launch the same
   arm or overrun the memory budget, then **spawn** ``python -m hsn.experiments run-arm`` with
   ``start_new_session`` (its own process group), ``cwd`` at the repository root of the imported
   ``hsn``, its output in ``<run dir>/run.log``, and record its PID, group and start time.

The memory estimate of an arm is its row's reservation (``enqueue --mem-estimate-mib``), else the
largest recorded peak of an arm with the same memory key times :data:`MEM_MARGIN`, else unknown (the
arm runs alone, and its peak sets the estimate). Since the memory key covers what a compiled graph
depends on (shapes, widths, routing and gate, dtypes, compile mode), the first arm of each key also
compiles alone and fills the shared inductor cache (``TORCHINDUCTOR_CACHE_DIR``, default
``<db dir>/inductor-cache``, persistent, unlike inductor's ``/tmp`` default) for the arms after it.
``--exit-when-empty`` stops once nothing is queued and none of its children runs.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any

from hsn.config import REPO_ROOT
from hsn.experiments import db as dbm
from hsn.experiments import machine as machine_module
from hsn.experiments.db import Database, loads, parse_time
from hsn.experiments.run_arm import run_dir_for
from hsn.experiments.scheduler import (
    Card,
    Launch,
    Limits,
    QueueItem,
    RunningItem,
    decide,
)

__all__ = ["MEM_MARGIN", "Worker", "WorkerOptions"]

MEM_MARGIN = 1.1
CPU_EVERY_S = 30.0
FLUSH_EVERY_S = 60.0


@dataclass
class WorkerOptions:
    max_concurrent: int = 3
    reserve_mib: float = 1024.0
    idle_gap_s: float = 60.0
    requeue_orphans: bool = False
    require_clean: bool = False
    exit_when_empty: bool = False
    poll_s: float = 5.0
    omp_threads: int = 4
    claim_grace_s: float = 30.0
    inductor_cache: Path | None = (
        None  # None: $TORCHINDUCTOR_CACHE_DIR, else <db dir>/inductor-cache
    )
    turn_file: Path | None = None
    turn_owner: str | None = None
    telemetry_max_samples: int = 0  # opt-in, bounded per-run card time series


def turn_remaining(path: Path | None, owner: str | None, now: float) -> float:
    """Fail closed on a missing/expired/unclaimed compute turn. No clock-only handoff."""
    if path is None:
        return math.inf
    try:
        turn = json.loads(path.read_text())
        start, end = parse_time(turn.get("start")), parse_time(turn.get("end"))
        if (
            not owner
            or turn.get("owner") != owner
            or turn.get("status") != "active"
            or turn.get("released_at") is not None
            or start is None
            or end is None
            or not start.timestamp() <= now < end.timestamp()
        ):
            return 0.0
        return end.timestamp() - now
    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0


@dataclass
class _Watch:
    peak_mib: float = 0.0
    foreign: dict[str, dict[str, Any]] = field(default_factory=dict)
    cpu_foreign: dict[str, dict[str, Any]] = field(default_factory=dict)
    own_with: set[int] = field(default_factory=set)
    samples: int = 0
    first: float = field(default_factory=time.time)
    last: float = field(default_factory=time.time)
    flushed: float = 0.0
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    telemetry_dropped: int = 0

    def sample_telemetry(
        self,
        reading: machine_module.GpuReading,
        progress: dict[str, Any],
        process_mib: float | None,
        limit: int,
    ) -> None:
        available = reading.ok and reading.present
        self.telemetry.append(
            {
                "time": reading.time,
                "step": progress.get("step"),
                "phase": progress.get("phase", "training_or_setup"),
                "utilization_pct": reading.utilization_pct if available else None,
                "device_memory_mib": reading.used_mib if available else None,
                "process_memory_mib": process_mib if available else None,
            }
        )
        if len(self.telemetry) > limit:
            excess = len(self.telemetry) - limit
            del self.telemetry[:excess]
            self.telemetry_dropped += excess

    def card(self, adopted: bool) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "first": self.first,
            "last": self.last,
            "peak_mib": self.peak_mib,
            "foreign_gpu": list(self.foreign.values()),
            "foreign_cpu": list(self.cpu_foreign.values()),
            "own_with": sorted(self.own_with),
            "watch_incomplete": adopted,
            **(
                {"telemetry": self.telemetry, "telemetry_dropped": self.telemetry_dropped}
                if self.telemetry
                else {}
            ),
        }

    def status(self) -> str:
        if self.foreign or self.cpu_foreign:
            names = [
                f"{v['pid']} {v['name']}"
                for v in [*self.foreign.values(), *self.cpu_foreign.values()]
            ]
            return "shared (foreign: " + ", ".join(names[:5]) + ")"
        if self.own_with:
            return "shared (own: " + ", ".join(f"exp {e}" for e in sorted(self.own_with)) + ")"
        return "idle"


def _seen(table: dict[str, dict[str, Any]], pid: int, now: float, **fields: Any) -> None:
    """Record one sighting of a foreign process: first and last time seen, how many looks, and
    the largest memory / CPU figure seen."""
    entry = table.setdefault(str(pid), {"pid": pid, "first": now, "looks": 0, **fields})
    entry["last"] = now
    entry["looks"] += 1
    for key in ("used_mib", "cpu"):
        if key in fields:
            entry[key] = max(float(entry.get(key) or 0.0), float(fields[key]))


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} worker: {message}", flush=True)


class Worker:
    """One worker process (module docstring). ``machine`` is injectable for tests."""

    def __init__(
        self,
        db: Database,
        root: Path,
        options: WorkerOptions,
        *,
        machine: ModuleType | Any = machine_module,
        python: str = sys.executable,
        log: Callable[[str], None] = _log,
    ) -> None:
        self.db = db
        self.root = Path(root)
        self.opt = options
        if options.telemetry_max_samples < 0:
            raise ValueError("telemetry_max_samples must be nonnegative")
        if (options.turn_file is None) != (options.turn_owner is None):
            raise ValueError("turn_file and turn_owner must be supplied together")
        self.machine = machine
        self.python = python
        self.log = log
        self.host = machine.host()
        self.id = f"{self.host}:{os.getpid()}"
        self.children: dict[int, subprocess.Popen[bytes]] = {}
        self.watch: dict[int, _Watch] = {}
        # P1.14: process group -> experiment of every arm launched or adopted here, until no process
        # is left in the group (finishing groups stay ours after their row leaves running)
        self.own_pgids: dict[int, int] = {}
        self.adopted: set[int] = set()
        self.idle_since: float | None = None
        self.last_cpu = 0.0
        self.cpu_hogs: list[tuple[int, float, str]] | None = []
        self.head: int | None = None
        self.head_bypassed = 0
        self.stopping = False
        self.last_wait = ""
        self.arms: dict[str, dict[str, Any]] = {}
        self.deadlines_sent: dict[int, tuple[int, float]] = {}  # pgid -> (exp_id, signal time)

    # lifecycle -------------------------------------------------------------------------------

    def run(self) -> int:
        git = self.machine.git_state(REPO_ROOT)
        if git["dirty"]:
            if self.opt.require_clean:
                self.log(
                    f"refusing to run: modified tracked files {git['dirty']} (--require-clean)"
                )
                return 2
            self.log(f"WARNING: modified tracked files {git['dirty']}; each run saves dirty.patch")
        self.log(f"{self.id} on {self.db.path} (commit {git['sha']}, root {self.root})")

        def stop(signum: int, frame: FrameType | None) -> None:
            self.stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while not self.stopping:
            self.poll()
            if self.opt.exit_when_empty and not self.children and not self.db.queued():
                self.log("queue empty and no child running: exiting")
                break
            time.sleep(self.opt.poll_s)
        if self.stopping:
            self.log(f"stopping; {len(self.children)} children keep running detached")
        return 0

    def poll(self) -> None:
        self.enforce_deadlines()
        self.reap()
        self.recover()
        card = self.observe()
        self.launch(card)

    def enforce_deadlines(self) -> None:
        """Independent watchdog: timeout signal, then kill the owned group after ten seconds."""
        now = time.time()
        remaining = turn_remaining(self.opt.turn_file, self.opt.turn_owner, now)
        for exp in self.db.running(host=self.host):
            if int(exp["exp_id"]) not in self.children and int(exp["exp_id"]) not in self.adopted:
                continue
            protocol = loads(exp.get("protocol_json")) or {}
            if protocol.get("name") != "spike_diagnostic" or exp["pid"] is None:
                continue
            start = parse_time(exp["started_at"] or exp["claimed_at"])
            if start is None:
                continue
            expired = now >= start.timestamp() + float(protocol["timeout_s"])
            if not expired and remaining > 15:
                continue
            pid, pgid = int(exp["pid"]), int(exp["pgid"] or exp["pid"])
            if pgid in self.deadlines_sent:
                continue
            if self.machine.pid_alive(pid, exp["pid_start"]):
                self.db.set_paused(True)
                try:
                    os.kill(pid, signal.SIGALRM)
                except ProcessLookupError:
                    pass
                self.deadlines_sent[pgid] = (int(exp["exp_id"]), now)
                self.log(f"exp {exp['exp_id']}: diagnostic deadline; queue paused")
        # The child's own alarm may have recorded failure between polls. Its compiler children
        # can still own CUDA after the parent exits, so include those finishing groups as well.
        for pgid, exp_id in self.own_pgids.items():
            if pgid in self.deadlines_sent:
                continue
            exp = self.db.experiment(exp_id)
            protocol = loads(exp.get("protocol_json")) or {}
            if protocol.get("name") != "spike_diagnostic" or exp["status"] != "failed":
                continue
            if "TimeoutError" in (exp.get("error") or ""):
                self.db.set_paused(True)
                self.deadlines_sent[pgid] = (exp_id, now)
        if self.deadlines_sent:
            alive = set(self.machine.process_groups().values())
            for pgid, (exp_id, sent) in list(self.deadlines_sent.items()):
                if pgid not in alive:
                    del self.deadlines_sent[pgid]
                elif now - sent >= 10:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    dbm.mark_failed(self.db, exp_id, "diagnostic timeout: process group killed")
                    del self.deadlines_sent[pgid]

    # reap and recover ------------------------------------------------------------------------

    def _log_tail(self, exp: dict[str, Any]) -> str:
        path = exp.get("log_path")
        if not path or not Path(path).exists():
            return ""
        return Path(path).read_text(encoding="utf-8", errors="replace")[-2000:]

    def reap(self) -> None:
        for exp_id, proc in list(self.children.items()):
            code = proc.poll()
            if code is None:
                continue
            del self.children[exp_id]
            exp = self.db.experiment(exp_id)
            if exp["status"] == "running":
                why = (
                    "exited 0 without a result"
                    if code == 0
                    else f"exited with {code} without a final status"
                )
                dbm.mark_failed(self.db, exp_id, f"{why}\n{self._log_tail(exp)}", exit_code=code)
                self.log(f"exp {exp_id} {why}")
            else:
                self.log(f"exp {exp_id} finished: {exp['status']} (exit {code})")
            with self.db.write() as conn:
                conn.execute(
                    "UPDATE experiments SET exit_code = ? WHERE exp_id = ?", (code, exp_id)
                )
            self._flush(exp_id, final=True)
            protocol = loads(exp.get("protocol_json")) or {}
            if (
                protocol.get("name") == "spike_diagnostic"
                and self.db.experiment(exp_id)["status"] != "done"
            ):
                self.db.set_paused(True)

    def recover(self) -> list[int]:
        """Mark this host's running rows with no live process failed; returns their ids."""
        gone = []
        for row in self.db.running(host=self.host):
            exp_id = int(row["exp_id"])
            if exp_id in self.children:
                continue
            pid = row["pid"]
            if pid is None:
                claimed = parse_time(row["claimed_at"])
                age = (time.time() - claimed.timestamp()) if claimed else float("inf")
                if age < self.opt.claim_grace_s:
                    continue
                why = f"no process recorded {age:.0f} s after the claim"
            elif self.machine.pid_alive(int(pid), row["pid_start"]):
                if exp_id not in self.watch and not self._worker_alive(row["worker"]):
                    self.watch[exp_id] = _Watch()  # its worker is gone: watch it from here
                    self.adopted.add(exp_id)
                continue
            else:
                why = f"process {pid} gone (worker restart or crash)"
            new = dbm.mark_failed(
                self.db,
                exp_id,
                why,
                requeue=self.opt.requeue_orphans,
                expect_pid=None if pid is None else int(pid),
            )
            if self.db.experiment(exp_id)["status"] == "failed":
                if (loads(row.get("protocol_json")) or {}).get("name") == "spike_diagnostic":
                    self.db.set_paused(True)
                gone.append(exp_id)
                self.log(f"exp {exp_id}: {why}" + (f"; requeued as exp {new}" if new else ""))
                self._flush(exp_id, final=True)
        return gone

    def _worker_alive(self, worker: str | None) -> bool:
        """Whether the worker ``host:pid`` that launched a row still runs (on this host)."""
        if not worker or ":" not in worker:
            return False
        host, _, pid = worker.rpartition(":")
        return host == self.host and pid.isdigit() and self.machine.pid_alive(int(pid))

    # watch -----------------------------------------------------------------------------------

    def observe(self) -> tuple[Card, dict[int, float]]:
        now = time.time()
        running = self.db.running()
        reading = self.machine.read_gpu()
        for r in running:  # the groups of the arms watched here (our children, adopted orphans)
            if int(r["exp_id"]) in self.watch and r["pgid"] is not None:
                self.own_pgids[int(r["pgid"])] = int(r["exp_id"])
        look = bool(running or reading.processes or self.own_pgids)
        groups = self.machine.process_groups() if look else {}
        running_groups = {int(r["pgid"]) for r in running if r["pgid"] is not None}
        alive = set(groups.values())
        for pgid in [g for g in self.own_pgids if g not in running_groups and g not in alive]:
            del self.own_pgids[pgid]  # its row is finished and no process is left in its group
        finishing = {g: e for g, e in self.own_pgids.items() if g not in running_groups}
        own_groups = running_groups | set(finishing)
        own, foreign, _ignored = self.machine.classify(reading.processes, own_groups, groups)
        by_group: dict[int, float] = {}
        for p in own:
            g = groups.get(p.pid, p.pid)
            by_group[g] = by_group.get(g, 0.0) + p.used_mib
        timing_live = any(r["timing"] for r in running)
        head = self.db.queued()[:1]
        timing_waiting = bool(head) and bool(head[0]["timing"])
        fresh_cpu = False
        if (timing_live or timing_waiting) and now - self.last_cpu >= CPU_EVERY_S:
            fresh_cpu = True
            hogs = self.machine.cpu_hogs()
            # groups read after top's 3-s sample, so that a process our arm spawned meanwhile (an
            # inductor compile worker, in the arm's process group) is ours, not foreign
            groups = {**groups, **self.machine.process_groups()}
            self.cpu_hogs = (
                None
                if hogs is None
                else [
                    h
                    for h in hogs
                    if groups.get(h[0], h[0]) not in own_groups and h[0] != os.getpid()
                ]
            )
            self.last_cpu = now
        elif not (timing_live or timing_waiting):
            self.cpu_hogs = []
        idle = reading.ok and not foreign and self.cpu_hogs == [] and not finishing
        if idle and not running:
            self.idle_since = self.idle_since if self.idle_since is not None else now
        elif not idle:
            self.idle_since = None
        running_ids = [int(r["exp_id"]) for r in running]
        for r in running:
            exp_id = int(r["exp_id"])
            if exp_id not in self.watch:  # only our children and the orphans we adopted
                continue
            w = self.watch[exp_id]
            w.samples += 1
            w.last = now
            if self.opt.telemetry_max_samples > 0:
                progress = loads(r["progress_json"]) or {}
                w.sample_telemetry(
                    reading,
                    progress,
                    by_group.get(int(r["pgid"]), 0.0) if r["pgid"] is not None else None,
                    self.opt.telemetry_max_samples,
                )
            if r["pgid"] is not None:
                w.peak_mib = max(w.peak_mib, by_group.get(int(r["pgid"]), 0.0))
            for p in foreign:
                _seen(w.foreign, p.pid, now, name=p.name, used_mib=p.used_mib)
            if r["timing"] and fresh_cpu:
                for pid, pct, cmd in self.cpu_hogs or []:
                    _seen(w.cpu_foreign, pid, now, name=cmd, cpu=pct)
            w.own_with.update(e for e in running_ids if e != exp_id)
            w.own_with.update(e for e in finishing.values() if e != exp_id)
            if now - w.flushed >= FLUSH_EVERY_S:
                self._flush(exp_id)
        # a finishing group's memory is in use but belongs to none of our running arms: it counts
        # with the foreign use in the scheduler's memory arithmetic, never as a foreign process
        own_used = sum(v for g, v in by_group.items() if g not in finishing)
        return Card(
            present=reading.present,
            ok=reading.ok,
            total_mib=reading.total_mib,
            used_mib=reading.used_mib,
            own_used_mib=own_used,
            foreign=tuple(f"{p.pid} {p.name} {p.used_mib:.0f} MiB" for p in foreign),
            error=reading.error,
        ), by_group

    def _flush(self, exp_id: int, *, final: bool = False) -> None:
        w = self.watch.get(exp_id)
        if w is None:
            return
        w.flushed = time.time()
        dbm.update_card(
            self.db,
            exp_id,
            w.card(exp_id in self.adopted),
            status=w.status() if w.samples else None,
            peak_mib=w.peak_mib if w.peak_mib > 0 else None,
        )
        if final:
            self.watch.pop(exp_id, None)
            self.adopted.discard(exp_id)

    # decide, claim, spawn --------------------------------------------------------------------

    def _arm(self, arm_id: str) -> dict[str, Any]:
        if arm_id not in self.arms:
            arm = self.db.arm(arm_id)
            config = loads(arm["config_json"])
            self.arms[arm_id] = {
                "device": str(config["hardware"]["device"]),
                "memory_key": arm["memory_key"],
            }
        return self.arms[arm_id]

    def _estimate(self, row: dict[str, Any]) -> float | None:
        if row["mem_est_mib"] is not None:
            return float(row["mem_est_mib"])
        peak = self.db.memory_estimate(self._arm(row["arm_id"])["memory_key"])
        return None if peak is None else peak * MEM_MARGIN

    def launch(self, observed: tuple[Card, dict[int, float]]) -> int | None:
        card, by_group = observed
        limits = Limits(self.opt.max_concurrent, self.opt.reserve_mib, self.opt.idle_gap_s)
        decision_box: dict[str, Any] = {}

        def choose(queue: list[dict[str, Any]], running: list[dict[str, Any]]) -> int | None:
            if self.opt.turn_file is not None:
                protocol = loads(queue[0].get("protocol_json")) or {}
                budget = protocol.get("timeout_s")
                remaining = turn_remaining(self.opt.turn_file, self.opt.turn_owner, time.time())
                if budget is None or remaining < float(budget) + 30:
                    return None
            head = int(queue[0]["exp_id"])
            if head != self.head:
                self.head, self.head_bypassed = head, 0
            items = [
                QueueItem(
                    int(q["exp_id"]),
                    self._arm(q["arm_id"])["device"],
                    bool(q["timing"]),
                    self._estimate(q),
                )
                for q in queue
            ]
            if (loads(queue[0].get("protocol_json")) or {}).get("name") == "spike_diagnostic":
                items = items[:1]  # dependencies run strictly in insertion order; no bypass
            runs = [
                RunningItem(
                    int(r["exp_id"]),
                    self._arm(r["arm_id"])["device"],
                    bool(r["timing"]),
                    self._estimate(r),
                    by_group.get(int(r["pgid"]), 0.0) if r["pgid"] is not None else 0.0,
                )
                for r in running
            ]
            decision = decide(
                items,
                runs,
                card,
                limits=limits,
                paused=self.db.paused,
                idle_since=self.idle_since,
                now=time.time(),
                head_bypassed=self.head_bypassed,
            )
            decision_box["d"] = decision
            return decision.exp_id if isinstance(decision, Launch) else None

        if self.stopping:
            return None
        claimed = dbm.claim_next(self.db, choose, worker=self.id, host=self.host)
        decision = decision_box.get("d")
        if claimed is None:
            reason = getattr(decision, "reason", "")
            kind = re.sub(r"\d+", "#", reason)  # a changing MiB figure is not a new reason
            if reason and kind != self.last_wait and reason != "queue empty":
                self.log(f"waiting: {reason}")
            self.last_wait = kind
            return None
        if isinstance(decision, Launch) and decision.bypassed_head:
            self.head_bypassed += 1
        self.last_wait = ""
        self.idle_since = None  # a timing arm needs a fresh idle streak after any launch
        return self.spawn(claimed)

    def spawn(self, exp: dict[str, Any]) -> int | None:
        exp_id = int(exp["exp_id"])
        sweep = (
            self.db.row("SELECT name FROM sweeps WHERE id = ?", (exp["sweep_id"],))
            if exp["sweep_id"]
            else None
        )
        run_dir = run_dir_for(self.root, sweep["name"] if sweep else None, exp["run_id"], exp_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "run.log"
        cache = self.opt.inductor_cache or os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        env = {
            **os.environ,
            # P2.8: compiled artifacts persist across arms and reboots (inductor's default is /tmp)
            "TORCHINDUCTOR_CACHE_DIR": str(cache or self.db.path.parent / "inductor-cache"),
            "OMP_NUM_THREADS": str(self.opt.omp_threads),
            "HSN_EXPERIMENTS_DB": str(self.db.path),
            "HSN_EXPERIMENTS_ROOT": str(self.root),
            "PYTHONUNBUFFERED": "1",
        }
        cmd = [
            self.python,
            "-m",
            "hsn.experiments",
            "--db",
            str(self.db.path),
            "--root",
            str(self.root),
            "run-arm",
            "--exp",
            str(exp_id),
        ]
        try:
            with open(log_path, "ab") as out:
                proc = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
        except OSError as exc:
            dbm.mark_failed(self.db, exp_id, f"spawn failed: {exc}")
            self.log(f"exp {exp_id}: spawn failed: {exc}")
            return None
        dbm.set_launched(
            self.db,
            exp_id,
            pid=proc.pid,
            pgid=proc.pid,  # start_new_session: the child leads its own process group
            pid_start=self.machine.pid_start(proc.pid),
            run_dir=str(run_dir),
            log_path=str(log_path),
        )
        self.children[exp_id] = proc
        self.watch[exp_id] = _Watch()
        self.own_pgids[proc.pid] = exp_id
        self.log(f"launched exp {exp_id} ({exp['label']}) pid {proc.pid} in {run_dir}")
        return exp_id


def kill_experiment(db: Database, exp_id: int, machine: Any = machine_module) -> bool:
    """SIGTERM to a running experiment's process group, only when its PID's command line is our
    ``run-arm`` for that experiment (never a process the harness did not start)."""
    exp = db.experiment(exp_id)
    if exp["status"] != "running" or exp["pid"] is None:
        return False
    pid = int(exp["pid"])
    line = machine.cmdline(pid) or ""
    ours = re.search(rf"\brun-arm --exp {exp_id}(\s|$)", line) is not None
    if "hsn.experiments" not in line or not ours:
        return False
    try:
        os.killpg(int(exp["pgid"] or pid), signal.SIGTERM)
    except ProcessLookupError:
        return False
    return True
