"""Operator command: .venv\\Scripts\\python.exe -m pytest tests/cat_factory/test_gpu_cli.py -q."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pixelle_video.cat_factory import storage
from pixelle_video.cat_factory.gpu import cli, reconcile
from pixelle_video.cat_factory.gpu.reconcile import GpuBusyError
from web.utils import streamlit_helpers


class FakeLease:
    """Offline stand-in for the single-GPU lease manager."""

    def __init__(self, held: bool = True) -> None:
        self._held = held
        self.phases: list[str] = []
        self.release_calls = 0

    def held(self) -> bool:
        """Report whether the lease is currently owned."""
        return self._held

    def acquire(self, job_id: str, phase: str) -> SimpleNamespace:
        """Record the acquired phase and flip ownership on."""
        self.phases.append(phase)
        self._held = True
        return SimpleNamespace(job_id=job_id, lock_generation=len(self.phases))

    def heartbeat(self, phase: str) -> None:
        """Record a heartbeat phase."""
        self.phases.append(phase)

    def release(self) -> bool:
        """Release the lease and report success."""
        self.release_calls += 1
        self._held = False
        return True


def _shot(label: str = "one") -> dict[str, object]:
    """Build one manifest shot entry."""
    return {
        "duration_s": 2.0,
        "image_prompt": f"a cat on a counter {label} " * 2,
        "motion_prompt": f"the cat turns its head {label} " * 2,
    }


def _write_manifest(root, job_id="job-1", shots=None, complete=True):
    """Write a planning manifest (optionally without a complete checkpoint)."""
    shots = shots or [_shot()]
    manifest = storage.build_planning_manifest(
        job_id=job_id,
        brief={},
        selected_idea={"shots": shots},
    )
    if complete:
        manifest["checkpoint"] = storage.planning_complete_checkpoint(manifest)
    storage.write_manifest(manifest, output_root=root)
    return manifest


_MISSING = object()


def _find(payload, key):
    """Recursively locate ``key`` inside a decoded JSON document."""
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find(value, key)
            if found is not _MISSING:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find(value, key)
            if found is not _MISSING:
                return found
    return _MISSING


def _parse_stdout(text: str):
    """Decode the JSON object printed by the CLI."""
    start = text.index("{")
    end = text.rindex("}")
    return json.loads(text[start : end + 1])


def _explode(*args, **kwargs):
    """Fail if a dry-run touches a runtime builder."""
    raise AssertionError("dry-run must not build runtime adapters")


def test_dry_run_proves_zero_destructive_calls(tmp_path, monkeypatch, capsys):
    """Report every destructive action as false without constructing adapters."""
    _write_manifest(tmp_path)
    monkeypatch.setattr(cli, "_build_lease", _explode)
    monkeypatch.setattr(cli, "_build_monitor", _explode)
    monkeypatch.setattr(cli, "_build_client", _explode)
    monkeypatch.setattr(cli, "_build_supervisor", _explode)

    exit_code = cli.main(
        ["dry-run", "--job", "job-1", "--output-root", str(tmp_path)]
    )

    assert exit_code == 0
    payload = _parse_stdout(capsys.readouterr().out)
    assert _find(payload, "destructive_calls") == []
    for flag in ("will_post", "will_spawn", "will_terminate"):
        assert _find(payload, flag) is False

    assert _find(payload, "shots_needing_keyframe") == 1
    assert _find(payload, "shots_needing_video") == 1


def test_dry_run_refuses_incomplete_planning(tmp_path, capsys):
    """Exit non-zero and explain the planning blocker."""
    _write_manifest(tmp_path, complete=False)

    exit_code = cli.main(
        ["dry-run", "--job", "job-1", "--output-root", str(tmp_path)]
    )

    assert exit_code == 3
    captured = capsys.readouterr()
    assert "planning" in (captured.err + captured.out).lower()


def test_dry_run_refuses_non_list_shots(tmp_path):
    """Exit non-zero when the manifest shots field is not a list."""
    _write_manifest(tmp_path, complete=True)
    manifest = storage.read_manifest(tmp_path / "job-1")
    manifest["shots"] = {}
    storage.write_manifest(manifest, output_root=tmp_path)

    exit_code = cli.main(
        ["dry-run", "--job", "job-1", "--output-root", str(tmp_path)]
    )

    assert exit_code == 3


def test_release_requires_owned_only():
    """Refuse an unbounded release request."""
    assert cli.main(["release", "--job", "job-1"]) == 3


def test_release_refuses_unowned_lease(monkeypatch):
    """Never release a lease this process does not own."""
    lease = FakeLease(held=False)
    monkeypatch.setattr(cli, "_build_lease", lambda *args, **kwargs: lease)

    exit_code = cli.main(["release", "--job", "job-1", "--owned-only"])

    assert exit_code == 3
    assert lease.release_calls == 0


def test_release_releases_owned_lease(monkeypatch):
    """Release exactly once when the lease is owned."""
    lease = FakeLease(held=True)
    monkeypatch.setattr(cli, "_build_lease", lambda *args, **kwargs: lease)

    exit_code = cli.main(["release", "--job", "job-1", "--owned-only"])

    assert exit_code == 0
    assert lease.release_calls == 1


def test_status_is_read_only(tmp_path, capsys):
    """Print a JSON status report for an existing job."""
    _write_manifest(tmp_path)

    exit_code = cli.main(
        ["status", "--job", "job-1", "--output-root", str(tmp_path)]
    )

    assert exit_code == 0
    payload = _parse_stdout(capsys.readouterr().out)
    assert isinstance(payload, dict)


def test_exclusive_lease_held_is_read_only(tmp_path):
    """Probing another process' lease must not create any file."""
    assert reconcile.exclusive_lease_held(gpu_id=0, lock_root=tmp_path) is False
    assert list(tmp_path.iterdir()) == []


def test_exclusive_lease_warning_flag_gate(tmp_path, monkeypatch):
    """Only warn when the feature is enabled and a live lease record exists."""
    from pixelle_video.config import config_manager

    monkeypatch.setattr(
        config_manager, "config",
        SimpleNamespace(cat_factory=SimpleNamespace(enabled=False)),
    )
    assert reconcile.exclusive_lease_warning(gpu_id=0) is None

    lease_dir = tmp_path / "lease"
    lease_dir.mkdir()
    record = {
        "job_id": "job-other", "owner_pid": 1234,
        "start_time": "2026-01-01T00:00:00+00:00",
        "heartbeat": "2026-01-01T00:00:00+00:00",
        "phase": "JOB_ACQUIRE", "lock_generation": 1,
    }
    (lease_dir / "gpu-0.lease.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        config_manager, "config",
        SimpleNamespace(cat_factory=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(
        reconcile, "_resolve_gpu_config",
        lambda: SimpleNamespace(stale_heartbeat_seconds=300, state_dir=str(tmp_path)),
    )
    assert (
        reconcile.exclusive_lease_warning(gpu_id=0)
        == reconcile.CONCURRENT_USE_WARNING
    )


def test_media_service_for_lease_requires_ownership():
    """Refuse to build a media service without an owned lease."""
    factory_calls: list[int] = []

    def factory():
        factory_calls.append(1)
        return "service"

    with pytest.raises(GpuBusyError):
        reconcile.media_service_for_lease(
            FakeLease(held=False), service_factory=factory
        )
    assert factory_calls == []

    assert (
        reconcile.media_service_for_lease(
            FakeLease(held=True), service_factory=lambda: "service"
        )
        == "service"
    )


def test_warn_if_gpu_lease_held_is_non_breaking(monkeypatch):
    """A failing lease probe never breaks the page and a warning surfaces."""
    streamlit = pytest.importorskip("streamlit")
    warnings: list[str] = []

    def record_warning(message) -> None:
        warnings.append(str(message))

    monkeypatch.setattr(streamlit, "warning", record_warning)
    if hasattr(streamlit_helpers, "st"):
        monkeypatch.setattr(streamlit_helpers.st, "warning", record_warning)

    def _raise(*args, **kwargs):
        raise RuntimeError("lease probe failed")

    monkeypatch.setattr(reconcile, "exclusive_lease_warning", _raise)
    assert streamlit_helpers.warn_if_gpu_lease_held() is False

    monkeypatch.setattr(
        reconcile, "exclusive_lease_warning", lambda *args, **kwargs: "WARN"
    )
    assert streamlit_helpers.warn_if_gpu_lease_held() is True
    assert len(warnings) == 1


def test_cli_source_has_no_process_control():
    """The CLI must never spawn or terminate another process."""
    source_path = (
        Path(__file__).resolve().parents[2]
        / "pixelle_video"
        / "cat_factory"
        / "gpu"
        / "cli.py"
    )
    source = source_path.read_text(encoding="utf-8")

    for marker in ("subprocess", "os.kill", "signal"):
        assert marker not in source
