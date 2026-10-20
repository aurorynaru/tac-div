r"""Offline contract tests for the ComfyUI supervisor and the GPU/RAM monitor.

Operator command: .venv\Scripts\python.exe -m pytest tests/cat_factory/test_gpu_supervisor.py -q

The managed-mode tests spawn a plain ``sys.executable`` placeholder child and
always stop it; nothing here needs a GPU, a ComfyUI server, or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import pixelle_video.cat_factory.gpu.lease as lease_module
import pixelle_video.cat_factory.gpu.supervisor as supervisor_module
from pixelle_video.cat_factory.gpu.monitor import (
    GpuMonitor,
    NvidiaSmiSampler,
    RamGateBlocked,
    RamSampler,
    SamplerUnavailable,
    VramReleaseTimeout,
    VramSample,
)
from pixelle_video.cat_factory.gpu.supervisor import (
    ComfySupervisor,
    ComfySupervisorError,
    ComfySupervisorIndeterminate,
    ComfySupervisorNotAuthorized,
    ComfySupervisorUnavailable,
)

_SLEEP_FOREVER = "import time; time.sleep(10)"


def _supervisor_config(command, *, poll_interval_s=0.01):
    """Build the minimal ``cat_factory.gpu`` stand-in used by the supervisor."""
    return SimpleNamespace(
        comfy_command=list(command),
        supervisor_mode="managed",
        poll_interval_s=poll_interval_s,
        state_dir="state",
    )


def _monitor_config(
    *,
    poll_interval_s=1.0,
    memory_release_timeout_s=10.0,
    offload_min_ram_mb=None,
    ram_margin_mb=0,
    vram_margin_mb=0,
):
    """Build the minimal ``cat_factory.gpu`` stand-in used by the monitor."""
    return SimpleNamespace(
        poll_interval_s=poll_interval_s,
        memory_release_timeout_s=memory_release_timeout_s,
        offload_min_ram_mb=offload_min_ram_mb,
        ram_margin_mb=ram_margin_mb,
        vram_margin_mb=vram_margin_mb,
    )


class _SequenceVramSampler:
    """Return a scripted sequence of VRAM ``used_mb`` values, then repeat the last."""

    def __init__(self, used_values):
        self._used_values = list(used_values)
        self.calls = 0

    def sample(self):
        index = min(self.calls, len(self._used_values) - 1)
        self.calls += 1
        used_mb = self._used_values[index]
        return VramSample(
            gpu_name="Fake GPU",
            total_mb=24576,
            used_mb=used_mb,
            free_mb=24576 - used_mb,
        )


# --- supervisor: managed mode ----------------------------------------------


def test_start_fails_closed_without_command(tmp_path, monkeypatch):
    """An unconfigured comfy_command must never be guessed or spawned."""
    monkeypatch.chdir(tmp_path)
    supervisor = ComfySupervisor(
        "job-1",
        "managed",
        state_dir="state",
        gpu_config=_supervisor_config([]),
    )

    with pytest.raises(ComfySupervisorUnavailable):
        supervisor.start()


def test_managed_start_stop_writes_per_job_logs(tmp_path, monkeypatch):
    """A managed start records the owned child and routes logs to per-job files."""
    monkeypatch.chdir(tmp_path)
    command = [sys.executable, "-c", _SLEEP_FOREVER]
    supervisor = ComfySupervisor(
        "job-logs",
        "managed",
        state_dir="state",
        gpu_config=_supervisor_config(command),
    )

    record = supervisor.start()
    child_pid = record.pid
    try:
        assert record.pid > 0
        assert record.command == tuple(command)
        assert Path(record.log_stdout).exists()
        assert Path(record.log_stderr).exists()
        assert supervisor.owned_pid == child_pid

        assert supervisor.stop(2.0) is True
        assert supervisor.owned_pid is None
        assert lease_module._pid_alive(child_pid) is False
    finally:
        supervisor.stop(2.0)


def test_double_start_refused(tmp_path, monkeypatch):
    """A live owned child blocks a second start."""
    monkeypatch.chdir(tmp_path)
    supervisor = ComfySupervisor(
        "job-double",
        "managed",
        state_dir="state",
        gpu_config=_supervisor_config([sys.executable, "-c", _SLEEP_FOREVER]),
    )

    supervisor.start()
    try:
        with pytest.raises(ComfySupervisorError):
            supervisor.start()
    finally:
        supervisor.stop(2.0)


def test_relaunch_refused_while_live(tmp_path, monkeypatch):
    """Relaunch refuses while the previous owned child is still running."""
    monkeypatch.chdir(tmp_path)
    supervisor = ComfySupervisor(
        "job-relaunch",
        "managed",
        state_dir="state",
        gpu_config=_supervisor_config([sys.executable, "-c", _SLEEP_FOREVER]),
    )

    supervisor.start()
    try:
        with pytest.raises(ComfySupervisorError):
            supervisor.relaunch()
    finally:
        supervisor.stop(2.0)


def test_identity_mismatch_fails_closed(tmp_path, monkeypatch):
    """A changed process identity token stops termination instead of signalling."""
    monkeypatch.chdir(tmp_path)
    supervisor = ComfySupervisor(
        "job-identity",
        "managed",
        state_dir="state",
        gpu_config=_supervisor_config([sys.executable, "-c", _SLEEP_FOREVER]),
    )

    supervisor.start()
    real_identity = supervisor_module._process_identity
    try:
        monkeypatch.setattr(
            supervisor_module, "_process_identity", lambda pid: "bogus"
        )
        with pytest.raises(ComfySupervisorIndeterminate):
            supervisor.stop()
    finally:
        monkeypatch.setattr(
            supervisor_module, "_process_identity", real_identity
        )
        supervisor.stop(2.0)


# --- supervisor: attached mode ---------------------------------------------


def test_attached_requires_operator_authorization(tmp_path, monkeypatch):
    """Attached mode only attaches with an explicit exclusive-use grant."""
    monkeypatch.chdir(tmp_path)
    supervisor = ComfySupervisor(
        "job-attached",
        "attached",
        state_dir="state",
        gpu_config=_supervisor_config([]),
    )
    endpoint = "http://127.0.0.1:8188"

    with pytest.raises(ComfySupervisorNotAuthorized):
        supervisor.attach(endpoint)
    assert supervisor.is_attached is False

    record = supervisor.attach(endpoint, operator_authorized_exclusive=True)
    assert record.endpoint == endpoint
    assert supervisor.is_attached is True
    assert supervisor.attached_endpoint == endpoint


def test_attached_stop_detaches_without_signalling(tmp_path, monkeypatch):
    """Attached stop detaches and never terminates any process."""
    monkeypatch.chdir(tmp_path)

    def _forbidden_terminate(pid):
        raise AssertionError(f"attached mode must never terminate pid {pid}")

    monkeypatch.setattr(
        supervisor_module, "_windows_terminate_process", _forbidden_terminate
    )
    supervisor = ComfySupervisor(
        "job-detach",
        "attached",
        state_dir="state",
        gpu_config=_supervisor_config([]),
    )
    supervisor.attach(
        "http://127.0.0.1:8188", operator_authorized_exclusive=True
    )

    assert supervisor.stop() is False
    assert supervisor.is_attached is False
    assert supervisor.attached_endpoint is None


# --- supervisor: state-dir containment -------------------------------------


@pytest.mark.parametrize(
    "unsafe_state_dir",
    ["\\\\foo", "/foo", "C:foo", "..", "../escape", "C:\\abs"],
)
def test_state_dir_rejects_containment_escapes(tmp_path, monkeypatch, unsafe_state_dir):
    """Absolute, drive-relative, and ``..`` state dirs fail closed at construction."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError):
        ComfySupervisor(
            "job-escape",
            "managed",
            state_dir=unsafe_state_dir,
            gpu_config=_supervisor_config([]),
        )


def test_dotdot_job_id_slug_is_safe(tmp_path, monkeypatch):
    """A ``..`` job id is slugged to a contained directory name."""
    monkeypatch.chdir(tmp_path)
    supervisor = ComfySupervisor(
        "..",
        mode="managed",
        state_dir="state",
        gpu_config=_supervisor_config([]),
    )

    assert supervisor.state_dir.name == "job"


# --- monitor: nvidia-smi sampler -------------------------------------------


def test_nvidia_smi_sampler_parses_csv():
    """The ``name, total, used, free`` CSV row is parsed into a VRAM sample."""
    def _runner(args):
        if "--query-compute-apps" in args:
            return ""
        return "NVIDIA GeForce RTX 3090, 24576, 1000, 23576\n"

    sample = NvidiaSmiSampler(runner=_runner).sample()

    assert sample.gpu_name == "NVIDIA GeForce RTX 3090"
    assert sample.total_mb == 24576
    assert sample.used_mb == 1000
    assert sample.free_mb == 23576
    assert sample.processes == ()


def test_nvidia_smi_sampler_rejects_unparsable():
    """Empty output and runner failures surface as ``SamplerUnavailable``."""
    empty_sampler = NvidiaSmiSampler(runner=lambda args: "")

    with pytest.raises(SamplerUnavailable):
        empty_sampler.sample()

    def _failing_runner(args):
        raise OSError("nvidia-smi is not installed")

    with pytest.raises(SamplerUnavailable):
        NvidiaSmiSampler(runner=_failing_runner).sample()


def test_nvidia_smi_sampler_degrades_process_query():
    """A failing compute-apps query degrades to no processes, not a failed sample."""
    gpu_row = "NVIDIA GeForce RTX 3090, 24576, 1000, 23576\n"

    def _runner(args):
        if "--query-compute-apps" in args:
            raise OSError("compute apps query unsupported")
        return gpu_row

    sample = NvidiaSmiSampler(runner=_runner).sample()

    assert sample.processes == ()
    assert sample.gpu_name == "NVIDIA GeForce RTX 3090"
    assert sample.used_mb == 1000


# --- monitor: RAM sampler and gates ----------------------------------------


def test_ram_sampler_uses_injected_reader():
    """The injected reader supplies ``(free_mb, total_mb)`` verbatim."""
    sample = RamSampler(reader=lambda: (30000, 64000)).sample()

    assert sample.free_mb == 30000
    assert sample.total_mb == 64000


def test_wait_for_vram_release_measured():
    """Release is accepted only after a fresh measurement reaches the baseline."""
    sampler = _SequenceVramSampler([5000, 5000, 1000])
    clock = [0.0]
    sleep_calls = []

    def _sleep(seconds):
        sleep_calls.append(seconds)
        clock[0] += seconds

    monitor = GpuMonitor(
        vram_sampler=sampler,
        ram_sampler=RamSampler(reader=lambda: (64000, 64000)),
        config=_monitor_config(poll_interval_s=1.0),
        sleep=_sleep,
        clock=lambda: clock[0],
    )

    result = monitor.wait_for_vram_release(baseline_used_mb=1000, timeout_s=10.0)

    assert result.used_mb == 1000
    assert sampler.calls == 3
    assert len(sleep_calls) == 2
    assert sleep_calls == [1.0, 1.0]


def test_wait_for_vram_release_times_out():
    """VRAM that never returns to the baseline raises ``VramReleaseTimeout``."""
    sampler = _SequenceVramSampler([5000])
    clock = [0.0]

    def _sleep(seconds):
        clock[0] += seconds

    monitor = GpuMonitor(
        vram_sampler=sampler,
        ram_sampler=RamSampler(reader=lambda: (64000, 64000)),
        config=_monitor_config(poll_interval_s=1.0),
        sleep=_sleep,
        clock=lambda: clock[0],
    )

    with pytest.raises(VramReleaseTimeout):
        monitor.wait_for_vram_release(baseline_used_mb=1000, timeout_s=2.0)


def test_ram_gate_fail_closed_without_measured_minimum():
    """An unset offload minimum blocks video generation without sampling RAM."""
    ram_reads = []

    def _reader():
        ram_reads.append(1)
        return (64000, 64000)

    monitor = GpuMonitor(
        vram_sampler=_SequenceVramSampler([0]),
        ram_sampler=RamSampler(reader=_reader),
        config=_monitor_config(offload_min_ram_mb=None, ram_margin_mb=1000),
    )

    with pytest.raises(RamGateBlocked):
        monitor.check_video_ram_gate()

    assert ram_reads == []


def test_ram_gate_boundary():
    """Free RAM exactly at minimum + margin passes; one MiB less is blocked."""
    passing = GpuMonitor(
        vram_sampler=_SequenceVramSampler([0]),
        ram_sampler=RamSampler(reader=lambda: (9000, 64000)),
        config=_monitor_config(offload_min_ram_mb=8000, ram_margin_mb=1000),
    )
    sample = passing.check_video_ram_gate()
    assert sample.free_mb == 9000

    blocked = GpuMonitor(
        vram_sampler=_SequenceVramSampler([0]),
        ram_sampler=RamSampler(reader=lambda: (8999, 64000)),
        config=_monitor_config(offload_min_ram_mb=8000, ram_margin_mb=1000),
    )
    with pytest.raises(RamGateBlocked):
        blocked.check_video_ram_gate()


# --- supervisor: bounded startup readiness ---------------------------------


def test_startup_probe_success_returns_record(tmp_path, monkeypatch):
    """A probe that reports ready returns the owned child record."""
    monkeypatch.chdir(tmp_path)
    command = [sys.executable, "-c", "import time; time.sleep(5)"]
    supervisor = ComfySupervisor(
        "job-probe-ok",
        "managed",
        state_dir="state",
        gpu_config=_supervisor_config(command, poll_interval_s=0.01),
        startup_probe=lambda: True,
        startup_timeout_s=1.0,
    )

    record = supervisor.start()
    try:
        assert record.pid > 0
    finally:
        supervisor.stop(2.0)


def test_startup_probe_timeout_fails_closed_and_stops_owned_child(
    tmp_path, monkeypatch
):
    """A probe that never reports ready fails closed and stops the owned child."""
    monkeypatch.chdir(tmp_path)
    command = [sys.executable, "-c", "import time; time.sleep(5)"]
    supervisor = ComfySupervisor(
        "job-probe-timeout",
        "managed",
        state_dir="state",
        gpu_config=_supervisor_config(command, poll_interval_s=0.01),
        startup_probe=lambda: False,
        startup_timeout_s=0.05,
    )

    try:
        with pytest.raises(ComfySupervisorUnavailable) as exc:
            supervisor.start()
        assert "did not become ready" in str(exc.value)
        assert supervisor.owned_pid is None
    finally:
        supervisor.stop(2.0)


def test_startup_child_exit_fails_closed(tmp_path, monkeypatch):
    """A child that exits during startup fails closed with a clear reason."""
    monkeypatch.chdir(tmp_path)
    command = [sys.executable, "-c", "pass"]
    supervisor = ComfySupervisor(
        "job-probe-exit",
        "managed",
        state_dir="state",
        gpu_config=_supervisor_config(command, poll_interval_s=0.01),
        startup_probe=lambda: False,
        startup_timeout_s=1.0,
    )

    try:
        with pytest.raises(ComfySupervisorUnavailable) as exc:
            supervisor.start()
        assert "exited during startup" in str(exc.value)
        assert supervisor.owned_pid is None
    finally:
        supervisor.stop(2.0)


def test_startup_failure_log_contains_command_and_no_environment(
    tmp_path, monkeypatch, caplog
):
    """A startup failure logs the command array but never the environment."""
    import logging

    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.ERROR)
    command = [sys.executable, "-c", "import time; time.sleep(5)"]
    supervisor = ComfySupervisor(
        "job-probe-log",
        "managed",
        state_dir="state",
        gpu_config=_supervisor_config(command, poll_interval_s=0.01),
        startup_probe=lambda: False,
        startup_timeout_s=0.05,
    )

    try:
        with pytest.raises(ComfySupervisorUnavailable):
            supervisor.start()
        assert "time.sleep" in caplog.text
        assert "PATH=" not in caplog.text
    finally:
        supervisor.stop(2.0)
