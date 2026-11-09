"""P2.10 guard tests: SQLite fixtures, mocked probes and sleeping subprocesses.

No CUDA context, real experiment or hardware probe is started.

Run without a project environment or torch:
    python3 -m unittest discover -s tests -p test_p2_10_idle_guard.py
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import closing, suppress
from pathlib import Path
from unittest.mock import Mock, call, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_when_idle.py"
spec = importlib.util.spec_from_file_location("hsn_p2_10_idle_guard", SCRIPT)
assert spec is not None and spec.loader is not None
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)


def clear():
    return guard.Observation("test", gpu=guard.Gpu("GPU-test", 0, ()), cpu_idle=99)


def samples(*, foreign_ticks=0, idle=99, own_session=None, reused=False):
    previous = guard.Process(11, 11, 1, 1000 if reused else 0, "python", own_session or 11)
    current = guard.Process(11, 11, 2 if reused else 1, foreign_ticks, "python", own_session or 11)
    return (
        guard.Cpu(0, 1000, 500, {11: previous}),
        guard.Cpu(1, 1100, 500 + idle, {11: current}),
    )


class IdleGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db = self.root / "experiments.sqlite"
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("CREATE TABLE experiments(exp_id INTEGER PRIMARY KEY, status TEXT)")
        self.patch(guard.time, "sleep", return_value=None)
        self.patch(guard.os, "sysconf", return_value=100)
        env = patch.dict(os.environ, {"HSN_EXPERIMENTS_FAKE_MACHINE": "0"})
        env.start()
        self.addCleanup(env.stop)

    def patch(self, target, name, **kwargs):
        context = patch.object(target, name, **kwargs)
        self.addCleanup(context.stop)
        return context.start()

    def enqueue(self, status, exp_id=1):
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("INSERT INTO experiments VALUES (?,?)", (exp_id, status))

    def hardware(self, *, gpu=None, cpu=None):
        self.patch(guard, "require_host_visibility")
        self.patch(guard, "read_cpu", side_effect=cpu or samples())
        self.patch(guard, "read_gpu", return_value=gpu or guard.Gpu("GPU-test", 0, ()))

    def test_database_is_read_only_and_counts_all_pending_rows(self):
        for i, status in enumerate(["running", "queued", "done", "failed", "cancelled"], 1):
            self.enqueue(status, i)
        before = self.db.read_bytes()
        state = guard.read_queue(self.db)
        self.assertEqual(state.running, (1,))
        self.assertEqual(state.queued, (2,))
        self.assertTrue(state.busy)
        self.assertEqual(before, self.db.read_bytes())

    def test_missing_database_is_not_created_or_treated_as_empty(self):
        missing = self.root / "missing.sqlite"
        with self.assertRaisesRegex(guard.Unavailable, "cannot read"):
            guard.read_queue(missing)
        self.assertFalse(missing.exists())

    def test_bad_schema_blocks(self):
        self.db.write_bytes(b"invalid sqlite")
        with self.assertRaisesRegex(guard.Unavailable, "cannot read"):
            guard.read_queue(self.db)

    def test_running_database_row_blocks_before_hardware_reads(self):
        self.enqueue("running", 209)
        hardware = self.patch(guard, "require_host_visibility")
        observation = guard.observe([self.db], "0")
        self.assertFalse(observation.clear)
        self.assertIn("1 running [209]", observation.reason)
        hardware.assert_not_called()

    def test_queued_database_row_alone_blocks(self):
        self.enqueue("queued", 210)
        observation = guard.observe([self.db], "0")
        self.assertFalse(observation.clear)
        self.assertIn("1 queued [210]", observation.reason)

    def test_finished_queue_advances_to_hardware_checks(self):
        self.enqueue("done")
        self.hardware()
        self.assertTrue(guard.observe([self.db], "0").clear)

    def test_additional_database_is_also_checked(self):
        other = self.root / "other.sqlite"
        with closing(sqlite3.connect(other)) as db, db:
            db.execute("CREATE TABLE experiments(exp_id INTEGER, status TEXT)")
            db.execute("INSERT INTO experiments VALUES (999,'running')")
        with self.assertRaisesRegex(guard.Unavailable, "999"):
            guard.require_empty_queues([self.db, other])

    def test_fake_machine_cannot_enable_live_launch(self):
        with patch.dict(os.environ, {"HSN_EXPERIMENTS_FAKE_MACHINE": "1"}):
            observation = guard.observe([self.db], "0")
        self.assertIn("fake-machine", observation.reason)

    @unittest.mock.patch.object(guard.sys, "platform", "linux")
    def test_isolated_namespace_blocks(self):
        (self.root / "1").mkdir()
        (self.root / "self").mkdir()
        (self.root / "1/comm").write_text("codex\n")
        (self.root / "self/status").write_text("NSpid:\t2\n")
        with self.assertRaisesRegex(guard.Unavailable, "host process visibility"):
            guard.require_host_visibility(self.root)

    @unittest.mock.patch.object(guard.sys, "platform", "linux")
    def test_host_init_and_unrestricted_proc_are_required(self):
        (self.root / "1").mkdir()
        (self.root / "self").mkdir()
        (self.root / "1/comm").write_text("systemd\n")
        (self.root / "self/status").write_text("NSpid:\t1234\n")
        (self.root / "mounts").write_text("proc /proc proc rw,relatime 0 0\n")
        guard.require_host_visibility(self.root)
        (self.root / "mounts").write_text("proc /proc proc rw,hidepid=2 0 0\n")
        with self.assertRaisesRegex(guard.Unavailable, "hides"):
            guard.require_host_visibility(self.root)

    def test_non_linux_host_probes_remain_unsupported(self):
        for platform in ("darwin", "win32"):
            with self.subTest(platform=platform), patch.object(guard.sys, "platform", platform):
                with self.assertRaisesRegex(guard.Unavailable, "require Linux"):
                    guard.require_host_visibility(self.root)

    def test_gpu_process_with_any_memory_blocks(self):
        self.hardware(gpu=guard.Gpu("GPU-test", 0, (11,)))
        observation = guard.observe([self.db], "0")
        self.assertIn("GPU compute processes present", observation.reason)

    def test_busy_gpu_utilization_blocks_without_process_entry(self):
        self.hardware(gpu=guard.Gpu("GPU-test", 3, ()))
        self.assertIn("utilization", guard.observe([self.db], "0").reason)

    def test_gpu_probe_failure_is_not_an_empty_card(self):
        self.hardware()
        self.patch(guard, "read_gpu", side_effect=guard.Unavailable("driver unavailable"))
        self.assertIn("driver unavailable", guard.observe([self.db], "0").reason)

    def test_invalid_gpu_counters_block(self):
        self.patch(guard, "smi", return_value=[["GPU-test", "N/A"]])
        with self.assertRaisesRegex(guard.Unavailable, "invalid GPU observation"):
            guard.read_gpu("0")

    def test_gpu_selection_filters_by_physical_uuid(self):
        self.patch(
            guard,
            "smi",
            side_effect=[[["GPU-test", "0"]], [["GPU-other", "20"], ["GPU-test", "21"]]],
        )
        gpu = guard.read_gpu("0")
        self.assertEqual(gpu.pids, (21,))
        self.assertEqual(gpu.uuid, "GPU-test")

    def test_cpu_only_job_blocks_on_per_process_usage(self):
        self.hardware(cpu=samples(foreign_ticks=60))
        self.assertIn("foreign busy PIDs [11]", guard.observe([self.db], "0").reason)

    def test_aggregate_cpu_usage_blocks(self):
        self.hardware(cpu=samples(idle=94))
        self.assertIn("CPU busy", guard.observe([self.db], "0").reason)

    def test_cpu_samples_handle_reused_pid(self):
        idle, busy = guard.cpu_activity(*samples(foreign_ticks=60, reused=True), None)
        self.assertEqual(idle, 99)
        self.assertEqual(busy, [11])

    def test_our_session_groups_are_excluded_from_monitor_utilization(self):
        self.hardware(
            cpu=samples(foreign_ticks=100, idle=0, own_session=55),
            gpu=guard.Gpu("GPU-test", 100, (11,)),
        )
        self.assertTrue(guard.observe([self.db], "GPU-test", owner=guard.Ownership(55)).clear)

    def test_foreign_session_cpu_still_blocks_during_monitoring(self):
        self.hardware(cpu=samples(foreign_ticks=60))
        observation = guard.observe([self.db], "GPU-test", owner=guard.Ownership(55))
        self.assertIn("foreign busy PIDs [11]", observation.reason)

    def test_process_listing_preserves_group_and_session(self):
        process = self.root / "44"
        process.mkdir()
        fields = ["S", "1", "22", "33"] + ["0"] * 16
        fields[11], fields[12], fields[19] = "4", "5", "10"
        (process / "stat").write_text(f"44 (compiler worker) {' '.join(fields)}")
        self.assertEqual(
            guard.read_processes(self.root),
            {44: guard.Process(44, 22, 10, 9, "compiler worker", 33, parent=1)},
        )

    def test_owned_ancestry_tracks_new_sessions_and_orphans_without_pid_reuse(self):
        owner = guard.Ownership(10)
        root = guard.Process(10, 10, 1, 0, "root", 10, parent=1)
        worker = guard.Process(11, 11, 1, 0, "worker", 11, parent=10)
        job = guard.Process(12, 12, 1, 0, "job", 12, parent=11)
        foreign = guard.Process(99, 99, 1, 0, "foreign", 99, parent=1)
        processes = {p.pid: p for p in (root, worker, job, foreign)}
        self.assertEqual(set(owner.members(processes)), {10, 11, 12})
        orphan = guard.Process(12, 12, 1, 0, "job", 12, parent=1)
        reused = guard.Process(11, 11, 2, 100, "unrelated", 11, parent=1)
        self.assertEqual(owner.members({12: orphan, 11: reused, 99: foreign}), {12: orphan})

    def test_child_sessions_are_owned_during_cpu_and_gpu_monitoring(self):
        root = guard.Process(55, 55, 1, 0, "root", 55, parent=1)
        child = guard.Process(11, 11, 1, 100, "job", 11, parent=55)
        self.hardware(
            cpu=(
                guard.Cpu(0, 1000, 500, {55: root}),
                guard.Cpu(1, 1100, 500, {55: root, 11: child}),
            ),
            gpu=guard.Gpu("GPU-test", 100, (11,)),
        )
        self.assertTrue(guard.observe([self.db], "GPU-test", owner=guard.Ownership(55)).clear)

    def test_queue_appearing_during_hardware_probe_blocks(self):
        self.hardware()
        self.patch(
            guard,
            "require_empty_queues",
            side_effect=[[], guard.Unavailable("new queued experiment")],
        )
        self.assertIn("new queued experiment", guard.observe([self.db], "0").reason)

    def test_quiet_window_requires_full_minute_and_resets(self):
        window = guard.QuietWindow()
        self.assertFalse(window.ready(clear(), 0))
        self.assertFalse(window.ready(clear(), 59))
        self.assertTrue(window.ready(clear(), 60))
        self.assertFalse(window.ready(guard.Observation("test", reason="busy"), 61))
        self.assertFalse(window.ready(clear(), 120))
        self.assertTrue(window.ready(clear(), 180))

    def test_incomplete_observation_is_never_clear(self):
        self.assertFalse(guard.Observation("test").clear)
        self.assertFalse(guard.QuietWindow().ready(guard.Observation("test"), 100))

    def test_reservation_contention_and_release(self):
        with guard.reservation("GPU-test", self.root):
            with self.assertRaisesRegex(guard.Unavailable, "holds GPU"):
                with guard.reservation("GPU-test", self.root):
                    self.fail("overlapping reservation")
        with guard.reservation("GPU-test", self.root):
            pass

    def test_busy_final_probe_never_spawns_child(self):
        self.patch(guard, "observe", return_value=guard.Observation("test", reason="busy"))
        spawn = self.patch(guard.subprocess, "Popen")
        result = guard.run_command(["never-run"], [self.db], "GPU-test", io.StringIO())
        self.assertEqual(result, guard.BUSY)
        spawn.assert_not_called()

    def test_queue_change_immediately_before_spawn_blocks(self):
        self.patch(guard, "observe", return_value=clear())
        self.patch(guard, "require_empty_queues", side_effect=guard.Unavailable("new row"))
        spawn = self.patch(guard.subprocess, "Popen")
        with self.assertRaisesRegex(guard.Unavailable, "new row"):
            guard.run_command(["never-run"], [self.db], "GPU-test", io.StringIO())
        spawn.assert_not_called()

    def test_contention_after_launch_stops_only_our_group(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        spawn = self.patch(guard.subprocess, "Popen", return_value=child)
        self.patch(guard, "read_processes", return_value={})
        self.patch(
            guard, "observe", side_effect=[clear(), guard.Observation("test", reason="new row")]
        )
        stop = self.patch(guard, "stop_child_session")
        report = io.StringIO()
        with patch("sys.stdout", new=io.StringIO()):
            result = guard.run_command(["never-run"], [self.db], "GPU-test", report)
        self.assertEqual(result, guard.BUSY)
        stop.assert_called_once()
        self.assertEqual(stop.call_args.args[:2], (child, report))
        self.assertEqual(stop.call_args.args[2].root, child.pid)
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-test")

    def test_normal_exit_records_result_without_signalling(self):
        child = Mock(pid=123)
        child.poll.return_value = 0
        self.patch(guard.subprocess, "Popen", return_value=child)
        self.patch(guard, "read_processes", return_value={})
        self.patch(guard, "observe", return_value=clear())
        stop = self.patch(guard, "stop_child_session")
        report = io.StringIO()
        self.assertEqual(guard.run_command(["never-run"], [self.db], "GPU-test", report), 0)
        stop.assert_not_called()
        self.assertIn('"event": "finished"', report.getvalue())

    def test_termination_during_spawn_cleans_up_and_restores_handlers(self):
        child = Mock(pid=123)
        self.patch(guard, "observe", return_value=clear())
        report = io.StringIO()
        signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        previous = {sig: signal.getsignal(sig) for sig in signals}

        def finish_cleanup(*_args):
            # A second termination request must not interrupt draining the group.
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGINT)

        stop = self.patch(guard, "stop_child_session", side_effect=finish_cleanup)
        for sig in signals:
            with self.subTest(signal=sig):

                def spawn(*_args, requested_signal=sig, **_kwargs):
                    signal.raise_signal(requested_signal)
                    return child

                with patch.object(guard.subprocess, "Popen", side_effect=spawn):
                    with self.assertRaises(guard.Cancelled) as raised:
                        guard.run_command(["never-run"], [self.db], "GPU-test", report)
                self.assertEqual(raised.exception.signum, sig)
                stop.assert_called_once()
                self.assertEqual(stop.call_args.args[:2], (child, report))
                self.assertEqual(stop.call_args.args[2].root, child.pid)
                stop.reset_mock()
                self.assertEqual({sig: signal.getsignal(sig) for sig in signals}, previous)

    def test_termination_before_spawn_never_launches(self):
        def final_probe(*_args):
            signal.raise_signal(signal.SIGTERM)
            return clear()

        self.patch(guard, "observe", side_effect=final_probe)
        spawn = self.patch(guard.subprocess, "Popen")
        with self.assertRaises(guard.Cancelled):
            guard.run_command(["never-run"], [self.db], "GPU-test", io.StringIO())
        spawn.assert_not_called()

    def test_cleanup_waits_for_descendants_after_leader_exit(self):
        child = Mock(pid=123, returncode=-signal.SIGTERM)
        child.poll.return_value = -signal.SIGTERM
        descendant = {124: guard.Process(124, 124, 1, 0, "sleep", 123)}
        foreign = {456: guard.Process(456, 456, 1, 0, "foreign", 456)}
        kill = self.patch(guard.os, "killpg")
        # TERM grace expires with a live descendant; KILL must then drain it,
        # even though the original child has already exited.
        self.patch(guard.time, "monotonic", side_effect=[0, 0, 6, 7])
        self.patch(guard, "read_processes", side_effect=[descendant | foreign, descendant, {}])
        guard.stop_child_session(child, io.StringIO())
        self.assertEqual(
            kill.call_args_list, [call(124, signal.SIGTERM), call(124, signal.SIGKILL)]
        )
        self.assertEqual(child.poll.call_count, 3)
        child.wait.assert_called_once_with()

    def test_cleanup_retains_reservation_until_session_visibility_returns(self):
        child = Mock(pid=123, returncode=None)
        self.patch(guard.time, "monotonic", side_effect=[0, 0, 6, 7])
        descendant = {124: guard.Process(124, 124, 1, 0, "sleep", 123)}
        self.patch(
            guard,
            "read_processes",
            side_effect=[guard.Unavailable("missing /proc"), descendant, {}],
        )
        kill = self.patch(guard.os, "killpg")
        guard.stop_child_session(child, io.StringIO())
        self.assertEqual(
            kill.call_args_list,
            [call(123, signal.SIGTERM), call(124, signal.SIGKILL)],
        )
        self.assertEqual(child.poll.call_count, 3)
        child.wait.assert_called_once_with()

    def test_check_only_reports_busy_without_launching(self):
        self.enqueue("queued", 210)
        spawn = self.patch(guard.subprocess, "Popen")
        with patch("sys.stdout", new=io.StringIO()) as out:
            code = guard.main(["--db", str(self.db), "--check"])
        self.assertEqual(code, guard.BUSY)
        self.assertIn("1 queued [210]", out.getvalue())
        spawn.assert_not_called()


@unittest.skipUnless(sys.platform == "linux", "process-group drain test uses Linux /proc")
class SignalCleanupTests(unittest.TestCase):
    def test_sigterm_drains_owned_group_before_releasing_reservation(self):
        self.check_shutdown(separate_group=False)

    def test_sigterm_drains_compiler_groups_before_releasing_reservation(self):
        self.check_shutdown(separate_group=True)

    def test_sigterm_drains_child_sessions_before_releasing_reservation(self):
        self.check_shutdown(separate_group=False, new_session=True)

    def check_shutdown(self, *, separate_group, new_session=False):
        # Both descendants only sleep. The leader reaps its child and delays its
        # own exit so the test can check the reservation while cleanup is active.
        child_code = textwrap.dedent("""\
            import json, os, signal, subprocess, sys, time
            from pathlib import Path
            root = Path(sys.argv[1])
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                     process_group=int(sys.argv[2]),
                                     start_new_session=bool(int(sys.argv[3])))
            def terminate(signum, frame):
                (root / 'terminating').touch()
                child.wait(timeout=3)
                time.sleep(1)
                raise SystemExit(0)
            signal.signal(signal.SIGTERM, terminate)
            (root / 'ready.tmp').write_text(json.dumps([os.getpid(), child.pid]))
            (root / 'ready.tmp').replace(root / 'ready.json')
            time.sleep(60)
            """)
        launcher_code = textwrap.dedent("""\
            import importlib.util, sys
            from pathlib import Path
            spec = importlib.util.spec_from_file_location('idle_guard', sys.argv[1])
            guard = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = guard
            spec.loader.exec_module(guard)
            root = Path(sys.argv[2])
            guard.observe = lambda *a, **kw: guard.Observation(
                'test', gpu=guard.Gpu('GPU-test', 0, ()), cpu_idle=99)
            guard.require_empty_queues = lambda _: []
            guard.QUIET_SECONDS = 0
            guard.POLL_SECONDS = 0.02
            reserve = guard.reservation
            guard.reservation = lambda uuid: reserve(uuid, root / 'locks')
            raise SystemExit(guard.main(['--report', str(root / 'report.jsonl'), '--',
                                        sys.executable, '-c', sys.argv[3], str(root),
                                        sys.argv[4], sys.argv[5]]))
            """)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.jsonl"
            owner = None
            with subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    launcher_code,
                    str(SCRIPT),
                    str(root),
                    child_code,
                    "0" if separate_group else "-1",
                    "1" if new_session else "0",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            ) as launcher:

                def wait_for_file(path):
                    deadline = time.monotonic() + 5
                    while not path.exists():
                        self.assertIsNone(
                            launcher.poll(), "launcher exited before fixture was ready"
                        )
                        self.assertLess(time.monotonic(), deadline, f"timed out waiting for {path}")
                        time.sleep(0.02)

                try:
                    wait_for_file(root / "ready.json")
                    leader, descendant = json.loads((root / "ready.json").read_text())
                    owner = guard.Ownership(leader)
                    owner.members(guard.read_processes())
                    self.assertEqual(
                        os.getpgid(descendant),
                        descendant if separate_group or new_session else leader,
                    )
                    self.assertEqual(os.getsid(descendant), descendant if new_session else leader)
                    launcher.send_signal(signal.SIGTERM)
                    wait_for_file(root / "terminating")
                    with self.assertRaisesRegex(guard.Unavailable, "holds GPU"):
                        with guard.reservation("GPU-test", root / "locks"):
                            self.fail("reservation released before the child exited")
                    launcher.send_signal(signal.SIGTERM)  # must not abort cleanup
                    stdout, stderr = launcher.communicate(timeout=10)
                    self.assertEqual(launcher.returncode, 128 + signal.SIGTERM, stdout + stderr)
                    live = guard.read_processes()
                    self.assertNotIn(leader, live)
                    self.assertNotIn(descendant, live)
                    with guard.reservation("GPU-test", root / "locks"):
                        pass
                    events = [json.loads(line) for line in report.read_text().splitlines()]
                    signals = [e for e in events if e["event"] == "signal_own_group"]
                    expected = {leader, descendant} if separate_group or new_session else {leader}
                    self.assertEqual({e["group"] for e in signals}, expected)
                    self.assertTrue(all(e["signal"] == signal.SIGTERM for e in signals))
                    self.assertEqual(events[-1]["event"], "cancelled")
                    self.assertEqual(events[-1]["signal"], signal.SIGTERM)
                    self.assertFalse(events[-1]["timings_usable"])
                finally:
                    # Also clean up our detached fixture if an assertion fails.
                    if report.exists():
                        for line in report.read_text().splitlines():
                            with suppress(json.JSONDecodeError):
                                event = json.loads(line)
                                if event["event"] == "launched":
                                    owner = owner or guard.Ownership(event["pid"])
                                    groups = {
                                        p.group
                                        for p in owner.members(guard.read_processes()).values()
                                    }
                                    for group in groups:
                                        with suppress(ProcessLookupError):
                                            os.killpg(group, signal.SIGKILL)
                    if launcher.poll() is None:
                        launcher.kill()
                    launcher.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
