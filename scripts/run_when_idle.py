#!/usr/bin/env python3
"""Run a P2.10 command after the shared experiment queue and host are idle.

Stdlib only: no torch import, CUDA context, database migration, or scheduler changes.
The database is authoritative for HSN/Claude experiment status. Both queued and running
rows block; a clock deadline never overrides them. GPU and CPU probes then establish
resource availability. Missing observations block, including an isolated PID namespace.

    python3 scripts/run_when_idle.py --check
    python3 scripts/run_when_idle.py --wait -- .venv/bin/python scripts/bench_kernels.py ...

Without --wait, a busy/unknown observation exits 75. Even an initially clear machine
must stay quiet for 60 seconds before a command starts. A per-user, physical-GPU lock
serializes these launchers across worktrees. Other schedulers do not honor that lock:
recheck the database before launch and monitor it while running. If work appears, stop
only our verified process tree and return 75; its timings must not be used.
SIGTERM, SIGINT and SIGHUP request cancellation. The launcher drains its owned processes
before releasing the reservation, including when cancellation arrives during spawn.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parents[1]
BUSY = 75
QUIET_SECONDS = 60.0
POLL_SECONDS = 5.0


class Unavailable(RuntimeError):
    """The resource gate could not establish an exclusive slot."""


class Cancelled(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"received {signal.Signals(signum).name}")


@dataclass
class Cancellation:
    signum: int | None = None

    def request(self, signum: int, _frame: object) -> None:
        # Do not raise asynchronously: that can interrupt Popen before we own its
        # returned handle, or interrupt cleanup while descendants still run.
        if self.signum is None:
            self.signum = signum

    def check(self) -> None:
        if self.signum is not None:
            raise Cancelled(self.signum)

    def wait(self, seconds: float) -> None:
        until = time.monotonic() + seconds
        while (remaining := until - time.monotonic()) > 0:
            self.check()
            time.sleep(min(remaining, 0.2))
        self.check()


@contextmanager
def cancellation_signals():
    """Main-thread signal handling; repeated signals cannot abort child cleanup."""
    cancellation = Cancellation()
    previous = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[sig] = signal.signal(sig, cancellation.request)
        yield cancellation
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@dataclass(frozen=True)
class Queue:
    path: str
    running: tuple[int, ...]
    queued: tuple[int, ...]

    @property
    def busy(self) -> bool:
        return bool(self.running or self.queued)


def read_queue(path: Path) -> Queue:
    """One consistent read from an existing database; never create or migrate it."""
    try:
        resolved = path.expanduser().resolve(strict=True)
        with closing(sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
            db.execute("PRAGMA query_only=ON")
            rows = db.execute(
                "SELECT exp_id,status FROM experiments "
                "WHERE status IN ('running','queued') ORDER BY exp_id"
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise Unavailable(f"cannot read experiment database {path}: {exc}") from exc
    return Queue(
        str(resolved),
        tuple(int(i) for i, status in rows if status == "running"),
        tuple(int(i) for i, status in rows if status == "queued"),
    )


def require_empty_queues(paths: list[Path]) -> list[Queue]:
    queues = [read_queue(p) for p in paths]
    busy = [q for q in queues if q.busy]
    if busy:
        raise Unavailable(
            "; ".join(
                f"{q.path}: {len(q.running)} running {list(q.running[:10])}, "
                f"{len(q.queued)} queued {list(q.queued[:10])}"
                for q in busy
            )
        )
    return queues


def require_host_visibility(proc: Path = Path("/proc")) -> None:
    if sys.platform != "linux":
        raise Unavailable("host CPU/process probes require Linux")
    try:
        init = (proc / "1/comm").read_text().strip()
        namespace = next(
            line.split()[1:]
            for line in (proc / "self/status").read_text().splitlines()
            if line.startswith("NSpid:")
        )
        if init not in {"systemd", "init"} or len(namespace) != 1:
            raise Unavailable(f"host process visibility unavailable (PID 1 is {init!r})")
        mounts = (proc / "mounts").read_text().splitlines()
        for line in mounts:
            fields = line.split()
            if len(fields) > 3 and fields[1] == "/proc":
                if any(x.startswith("hidepid=") and x != "hidepid=0" for x in fields[3].split(",")):
                    raise Unavailable("/proc hides other processes")
                return
        raise Unavailable("cannot identify the host /proc mount")
    except (OSError, StopIteration) as exc:
        raise Unavailable(f"cannot verify host process visibility: {exc}") from exc


@dataclass(frozen=True)
class Process:
    pid: int
    group: int
    start: int
    ticks: int
    name: str
    session: int
    parent: int = 0


def read_processes(proc: Path = Path("/proc")) -> dict[int, Process]:
    processes = {}
    try:
        for directory in proc.iterdir():
            if not directory.name.isdigit():
                continue
            try:
                raw = (directory / "stat").read_text()
            except FileNotFoundError:
                continue  # exited between enumeration and read
            fields = raw[raw.rindex(")") + 2 :].split()
            if fields[0] in {"Z", "X"}:
                continue
            pid = int(directory.name)
            processes[pid] = Process(
                pid,
                int(fields[2]),
                int(fields[19]),
                int(fields[11]) + int(fields[12]),
                raw[raw.index("(") + 1 : raw.rindex(")")],
                int(fields[3]),
                int(fields[1]),
            )
    except (OSError, ValueError, IndexError) as exc:
        raise Unavailable(f"incomplete host process listing: {exc}") from exc
    if not processes:
        raise Unavailable("empty host process listing")
    return processes


@dataclass
class Ownership:
    """Follow our session and observed descendants, including new child sessions.

    Keep process birth times so an orphan stays owned while a reused PID does not.
    Processes with neither an observed identity nor verified ancestry stay foreign.
    """

    root: int
    known: dict[int, int] = field(default_factory=dict)
    initialized: bool = False

    def members(self, processes: dict[int, Process]) -> dict[int, Process]:
        owned = {
            pid
            for pid, p in processes.items()
            if self.known.get(pid) == p.start or (not self.initialized and p.session == self.root)
        }
        while True:
            sessions = {processes[pid].session for pid in owned}
            added = {
                pid
                for pid, p in processes.items()
                if pid not in owned and (p.parent in owned or p.session in sessions)
            }
            if not added:
                break
            owned.update(added)
        self.known = {pid: processes[pid].start for pid in owned}
        self.initialized = True
        return {pid: processes[pid] for pid in owned}


@dataclass(frozen=True)
class Cpu:
    time: float
    total: int
    idle: int
    processes: dict[int, Process]


def read_cpu(proc: Path = Path("/proc")) -> Cpu:
    try:
        fields = (proc / "stat").read_text().splitlines()[0].split()
        if fields[0] != "cpu" or len(fields) < 9:
            raise ValueError("missing aggregate CPU counters")
        ticks = [int(v) for v in fields[1:9]]  # guest counters would double-count user time
        return Cpu(time.monotonic(), sum(ticks), ticks[3], read_processes(proc))
    except (OSError, ValueError, IndexError) as exc:
        raise Unavailable(f"cannot read host CPU counters: {exc}") from exc


def cpu_activity(before: Cpu, after: Cpu, owner: Ownership | None) -> tuple[float, list[int]]:
    elapsed = after.time - before.time
    total = after.total - before.total
    idle = after.idle - before.idle
    if elapsed <= 0 or total <= 0 or not 0 <= idle <= total:
        raise Unavailable("invalid CPU sample interval or counters")
    hz = os.sysconf("SC_CLK_TCK")
    busy = []
    if owner is not None:
        owner.members(before.processes)
    owned = owner.members(after.processes) if owner is not None else {}
    for pid, p in after.processes.items():
        if pid == os.getpid() or pid in owned:
            continue
        previous = before.processes.get(pid)
        ticks = p.ticks
        if previous is not None and previous.start == p.start:
            ticks -= previous.ticks
        if ticks < 0:
            raise Unavailable(f"CPU counters moved backwards for pid {pid}")
        if 100 * ticks / hz / elapsed > 50:
            busy.append(pid)
    return 100 * idle / total, busy


@dataclass(frozen=True)
class Gpu:
    uuid: str
    utilization: float
    pids: tuple[int, ...]


def smi(*args: str) -> list[list[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return [[value.strip() for value in row] for row in csv.reader(out.splitlines()) if row]
    except (OSError, subprocess.SubprocessError) as exc:
        raise Unavailable(f"GPU probe failed: {exc}") from exc


def read_gpu(device: str) -> Gpu:
    try:
        rows = smi(f"--id={device}", "--query-gpu=uuid,utilization.gpu")
        if len(rows) != 1 or len(rows[0]) != 2 or not rows[0][0].startswith("GPU-"):
            raise ValueError("expected one physical GPU with a UUID")
        uuid, utilization = rows[0]
        pct = float(utilization)
        if not math.isfinite(pct) or not 0 <= pct <= 100:
            raise ValueError("invalid GPU utilization")
        apps = smi("--query-compute-apps=gpu_uuid,pid")
        if any(len(row) != 2 or not row[1].isdigit() for row in apps):
            raise ValueError("unreadable GPU process list")
        return Gpu(uuid, pct, tuple(int(pid) for gpu, pid in apps if gpu == uuid))
    except (ValueError, IndexError) as exc:
        raise Unavailable(f"invalid GPU observation: {exc}") from exc


@dataclass(frozen=True)
class Observation:
    at: str
    reason: str | None = None
    gpu: Gpu | None = None
    cpu_idle: float | None = None

    @property
    def clear(self) -> bool:
        return self.reason is None and self.gpu is not None and self.cpu_idle is not None


def observe(databases: list[Path], device: str, owner: Ownership | None = None) -> Observation:
    at = datetime.now(UTC).isoformat()
    try:
        if os.environ.get("HSN_EXPERIMENTS_FAKE_MACHINE") == "1":
            raise Unavailable("fake-machine observations are forbidden for the live launcher")
        require_empty_queues(databases)  # first gate: no hardware reads while experiments wait
        require_host_visibility()
        before = read_cpu()
        time.sleep(1)
        after = read_cpu()
        idle, busy = cpu_activity(before, after, owner)
        gpu = read_gpu(device)
        owned = owner.members(after.processes) if owner is not None else {}
        foreign = [pid for pid in gpu.pids if pid not in owned]
        if foreign:
            raise Unavailable(f"GPU compute processes present: {foreign}")
        if owner is None and gpu.utilization > 2:
            raise Unavailable(f"GPU utilization {gpu.utilization:g}% exceeds idle limit 2%")
        if busy or (owner is None and idle < 95):
            details = [asdict(after.processes[pid]) for pid in busy]
            raise Unavailable(f"CPU busy: {idle:.1f}% idle; foreign busy PIDs {busy}: {details}")
        # Once our command runs, aggregate GPU/CPU utilization includes its work. Continue
        # checking foreign compute processes, per-process CPU, and the authoritative queue.
        require_empty_queues(databases)
        return Observation(at, gpu=gpu, cpu_idle=idle)
    except Unavailable as exc:
        return Observation(at, reason=str(exc))


@dataclass
class QuietWindow:
    since: float | None = None

    def ready(self, observation: Observation, now: float) -> bool:
        if not observation.clear:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= QUIET_SECONDS


@contextmanager
def reservation(uuid: str, directory: Path | None = None):
    """A cooperative lock only; the database/probe monitor remains mandatory."""
    directory = directory or Path(f"/tmp/hsn-resources-{os.getuid()}")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_uid != os.getuid():
        raise Unavailable(f"resource-lock directory is not owned by this user: {directory}")
    key = hashlib.sha256(uuid.encode()).hexdigest()
    try:
        fd = os.open(directory / f"{key}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "r+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Unavailable(f"another guarded workload holds GPU {uuid}") from exc
            # Do not unlink: waiters must continue locking the same inode.
            yield
    except OSError as exc:
        raise Unavailable(f"resource reservation failed: {exc}") from exc


def record(report: IO[str], event: str, **details: object) -> None:
    report.write(
        json.dumps({"at": datetime.now(UTC).isoformat(), "event": event, **details}) + "\n"
    )
    report.flush()


def stop_child_session(
    child: subprocess.Popen, report: IO[str], owner: Ownership | None = None
) -> None:
    """Drain verified descendants, including compiler groups and child sessions."""
    owner = owner or Ownership(child.pid)
    deadline = time.monotonic() + 5
    signalled = {}
    visibility_reported = False

    def stop(group: int, sig: signal.Signals) -> None:
        if signalled.get(group) == sig:
            return
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        signalled[group] = sig
        record(report, "signal_own_group", root_session=child.pid, group=group, signal=int(sig))

    while True:
        child.poll()  # reap the leader when it exits
        sig = signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL
        try:
            members = owner.members(read_processes())
        except Unavailable as exc:
            # Without a complete listing, only the still-live leader's group is
            # provably owned. Hold the reservation until session visibility returns.
            if child.returncode is None:
                stop(child.pid, sig)
            if not visibility_reported:
                record(report, "cleanup_visibility_unavailable", reason=str(exc))
                visibility_reported = True
        else:
            if not members:
                child.wait()
                return
            for group in sorted({p.group for p in members.values()}):
                stop(group, sig)
        # Keep the reservation after SIGKILL until every live session member is gone,
        # even if the leader exited first. Zombies have released their resources.
        time.sleep(0.2)


def run_command(command: list[str], databases: list[Path], gpu: str, report: IO[str]) -> int:
    # Last complete check closes the quiet-window/launch gap as far as an external
    # scheduler allows. No shared DB writes or scheduler pause are used for reservation.
    with cancellation_signals() as cancellation:
        final = observe(databases, gpu)
        cancellation.check()
        record(report, "before_launch", observation=asdict(final))
        if not final.clear:
            return BUSY
        require_empty_queues(databases)
        cancellation.check()
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = gpu  # child sees this physical GPU as cuda:0
        child = None
        owner = None
        finished = False
        try:
            child = subprocess.Popen(command, env=child_env, start_new_session=True)
            owner = Ownership(child.pid)
            cancellation.check()  # also catches signals received inside Popen
            record(report, "launched", pid=child.pid)
            while True:
                cancellation.check()
                status = child.poll()
                members = owner.members(read_processes())
                observation = observe(databases, gpu, owner=owner)
                cancellation.check()
                record(report, "monitor", observation=asdict(observation))
                if not observation.clear:
                    print(
                        f"Interrupted: {observation.reason}; discard this attempt's timings.",
                        flush=True,
                    )
                    return BUSY
                if status is not None and not members:
                    finished = True
                    record(report, "finished", exit_code=status)
                    return status if status >= 0 else 128 - status
                cancellation.wait(POLL_SECONDS)
        finally:
            if child is not None and not finished:
                stop_child_session(child, report, owner)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, action="append", help="database; repeat to check several"
    )
    parser.add_argument("--gpu", default="0", help="physical NVIDIA GPU index or UUID (default 0)")
    parser.add_argument("--check", action="store_true", help="one readiness snapshot, no command")
    parser.add_argument(
        "--wait", action="store_true", help="wait through busy/unknown observations"
    )
    parser.add_argument("--report", type=Path, help="JSONL report (default evidence/P2.10/idle/)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (args.check and command) or (not args.check and not command):
        parser.error("choose --check or provide a command after --")
    if args.check and args.wait:
        parser.error("--check is a single snapshot; omit --wait")
    databases = args.db or [
        Path(
            os.environ.get("HSN_EXPERIMENTS_DB", "~/hsn-experiments/experiments.sqlite")
        ).expanduser()
    ]
    if args.check:
        observation = observe(databases, args.gpu)
        print(json.dumps(asdict(observation), indent=2))
        return 0 if observation.clear else BUSY
    report_path = args.report or (
        ROOT / "evidence/P2.10/idle" / f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{os.getpid()}.jsonl"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as report:
        try:
            while True:
                initial = observe(databases, args.gpu)
                record(report, "waiting", observation=asdict(initial))
                if not initial.clear:
                    print(f"Waiting: {initial.reason}", flush=True)
                    if not args.wait:
                        return BUSY
                    time.sleep(POLL_SECONDS)
                    continue
                assert initial.gpu is not None
                try:
                    with reservation(initial.gpu.uuid):
                        window = QuietWindow()
                        while True:
                            observation = observe(databases, initial.gpu.uuid)
                            record(report, "quiet_window", observation=asdict(observation))
                            if window.ready(observation, time.monotonic()):
                                return run_command(command, databases, initial.gpu.uuid, report)
                            if not observation.clear:
                                print(f"Waiting: {observation.reason}", flush=True)
                                if not args.wait:
                                    return BUSY
                                break  # release the reservation while someone else works
                            time.sleep(POLL_SECONDS)
                except Unavailable as exc:
                    record(report, "reservation_blocked", reason=str(exc))
                    print(f"Waiting: {exc}", flush=True)
                    if not args.wait:
                        return BUSY
                time.sleep(POLL_SECONDS)
        except Cancelled as exc:
            record(report, "cancelled", signal=exc.signum, timings_usable=False)
            return 128 + exc.signum
        except KeyboardInterrupt:
            record(report, "cancelled")
            return 130
        except (Unavailable, OSError, subprocess.SubprocessError) as exc:
            record(report, "blocked", reason=str(exc))
            print(f"Blocked: {exc}", file=sys.stderr)
            return BUSY


if __name__ == "__main__":
    raise SystemExit(main())
