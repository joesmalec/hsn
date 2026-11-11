"""Machine probes of the sweep worker (P1.11): the GPU (nvidia-smi), the CPU (``top``), processes
(groups, liveness, start times, command lines), git state and versions. Imports without torch.

* :func:`read_gpu` separates three cases: no ``nvidia-smi`` on this host (a CPU host),
  ``nvidia-smi`` failed or timed out (the worker waits and logs the reason; it never reads a
  failure as an empty card), and a reading (``memory.total``, ``memory.used``, compute processes).
* :func:`cpu_hogs` reads ``top -b -n 2 -d 3`` and takes only the process rows after the second
  ``PID`` header (``%CPU`` is column 9); never the summary lines. ``None`` where ``top -b`` is not
  available (macOS).
* :func:`classify` applies the audit's card rule (``card_status`` of ``eff_overfit.py``): another
  process counts when it is a Python process (any memory) or holds at least 200 MiB; our own arms
  are recognised by their process group (the child leads its own group, and its descendants, such
  as inductor's compile workers, share it).
* :func:`pid_alive` treats a vanished process, a zombie and a reused PID (another start time) as
  gone.

``HSN_EXPERIMENTS_FAKE_MACHINE=1`` (tests) makes :func:`read_gpu` report a CPU host and
:func:`cpu_hogs` an idle CPU, so that no test reads or waits on the real card.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "MIN_FOREIGN_MIB",
    "GpuProcess",
    "GpuReading",
    "classify",
    "cmdline",
    "cpu_hogs",
    "git_state",
    "host",
    "pid_alive",
    "pid_start",
    "process_groups",
    "read_gpu",
    "versions",
]

MIN_FOREIGN_MIB = 200  # the audit's rule: a non-Python process below this is a desktop context


def _fake() -> bool:
    return os.environ.get("HSN_EXPERIMENTS_FAKE_MACHINE", "") == "1"


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    used_mib: float
    name: str

    def to_json(self) -> dict[str, Any]:
        return {"pid": self.pid, "used_mib": self.used_mib, "name": self.name}


@dataclass(frozen=True)
class GpuReading:
    """One look at the card. ``present`` False: no nvidia-smi (a CPU host); ``ok`` False: the
    reading failed."""

    present: bool
    ok: bool
    total_mib: float = 0.0
    used_mib: float = 0.0
    processes: tuple[GpuProcess, ...] = ()
    error: str | None = None
    time: float = field(default_factory=time.time)
    utilization_pct: float | None = None


def _smi(args: list[str], timeout: float) -> str:
    return subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout, check=True
    ).stdout


def read_gpu(timeout: float = 20.0) -> GpuReading:
    if _fake() or shutil.which("nvidia-smi") is None:
        return GpuReading(present=False, ok=True)
    try:
        mem = _smi(
            [
                "--query-gpu=memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout,
        )
        apps = _smi(
            ["--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"],
            timeout,
        )
        first = mem.strip().splitlines()[0].split(",")
        total, used = float(first[0]), float(first[1])
        procs = []
        for line in apps.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    mib = float(parts[1])
                except ValueError:
                    mib = float("inf")  # unreadable: counts as large
                procs.append(GpuProcess(int(parts[0]), mib, ",".join(parts[2:])))
        try:
            utilization = float(first[2]) if len(first) > 2 else None
        except ValueError:
            utilization = None  # unsupported telemetry must not make the card look unavailable
        return GpuReading(True, True, total, used, tuple(procs), utilization_pct=utilization)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return GpuReading(present=True, ok=False, error=f"{type(exc).__name__}: {exc}")


def cpu_hogs(threshold: float = 50.0) -> list[tuple[int, float, str]] | None:
    """Processes above ``threshold`` % CPU in ``top``'s second sample: ``(pid, %cpu, command)``."""
    if _fake():
        return []
    if sys.platform != "linux" or shutil.which("top") is None:
        return None
    try:
        out = subprocess.run(
            ["top", "-b", "-n", "2", "-d", "3"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    headers = 0
    hogs = []
    for line in out.splitlines():
        cols = line.split()
        if cols and cols[0] == "PID":
            headers += 1
            continue
        if headers < 2 or len(cols) < 12 or not cols[0].isdigit():
            continue
        try:
            pct = float(cols[8].replace(",", "."))
        except ValueError:
            continue
        if pct > threshold:
            hogs.append((int(cols[0]), pct, " ".join(cols[11:])))
    return hogs


def process_groups() -> dict[int, int]:
    """``{pid: pgid}`` of every process (``ps -eo pid=,pgid=``, Linux and macOS)."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,pgid="], capture_output=True, text=True, timeout=20, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    groups = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            groups[int(parts[0])] = int(parts[1])
    return groups


def _proc_stat(pid: int) -> list[str] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # the command may contain spaces and parentheses: split after the last ")"
    return text[text.rindex(")") + 2 :].split()


def _ps(pid: int, field_name: str) -> str | None:
    try:
        out = subprocess.run(
            ["ps", "-o", f"{field_name}=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.strip()
    return text or None


def pid_start(pid: int) -> str | None:
    """A token for the process's start time (clock ticks since boot on Linux, ``lstart`` elsewhere);
    ``None`` when there is no such process."""
    if Path("/proc").is_dir():
        stat = _proc_stat(pid)
        return None if stat is None else stat[19]  # field 22 of /proc/<pid>/stat
    return _ps(pid, "lstart")


def _zombie(pid: int) -> bool:
    if Path("/proc").is_dir():
        stat = _proc_stat(pid)
        return stat is not None and stat[0] == "Z"
    state = _ps(pid, "stat")
    return state is not None and state.startswith("Z")


def pid_alive(pid: int | None, start: str | None = None) -> bool:
    """The process exists, is not a zombie, and (when ``start`` is given) started then."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by someone else
    if _zombie(pid):
        return False
    if start is not None:
        current = pid_start(pid)
        if current is not None and current != start:
            return False
    return True


def cmdline(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return _ps(pid, "command")


def classify(
    processes: Iterable[GpuProcess],
    own_groups: Iterable[int],
    group_of: Mapping[int, int],
    *,
    min_mib: float = MIN_FOREIGN_MIB,
) -> tuple[list[GpuProcess], list[GpuProcess], list[GpuProcess]]:
    """``(own, foreign, ignored)``: own = in one of our arms' process groups; foreign = any other
    Python process, or any other process of at least ``min_mib``; ignored = small non-Python
    contexts (a desktop file manager, for example)."""
    groups = set(own_groups)
    own, foreign, ignored = [], [], []
    for p in processes:
        if group_of.get(p.pid, p.pid) in groups or p.pid in groups:
            own.append(p)
        elif "python" in p.name.lower() or p.used_mib >= min_mib:
            foreign.append(p)
        else:
            ignored.append(p)
    return own, foreign, ignored


def git_state(repo: Path) -> dict[str, Any]:
    """The checked-out commit and the tracked files that differ from it (the code a run loads)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()

    try:
        sha = git("rev-parse", "HEAD")
        dirty = [
            line
            for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
            if line
        ]
        diff = git("diff", "HEAD") if dirty else ""
    except (OSError, subprocess.SubprocessError):
        return {"sha": None, "dirty": None, "diff": ""}
    return {"sha": sha, "dirty": dirty, "diff": diff}


def host() -> str:
    return socket.gethostname()


def versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for pkg in ("hsn", "torch"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out
