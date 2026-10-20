r"""Offline contract tests for the fail-closed GPU lease manager.

Operator command: .venv\Scripts\python.exe -m pytest tests/cat_factory/test_gpu_lease.py -q

Every test runs against ``tmp_path`` and the real OS-level lock primitives. No
GPU, no ComfyUI server, no subprocess, and no network access is required.
"""

from __future__ import annotations

import json

import pytest

import pixelle_video.cat_factory.gpu.lease as lease_module
from pixelle_video.cat_factory.gpu.lease import (
    GpuLeaseError,
    GpuLeaseManager,
    LeaseRecord,
    StaleAssessment,
)


def test_acquire_records_lease_and_release_preserves_record(tmp_path):
    """Acquiring publishes a durable record bound to this process."""
    manager = GpuLeaseManager(0, tmp_path)

    handle = manager.acquire("job-1", "JOB_ACQUIRE")

    assert manager.held() is True
    assert handle.job_id == "job-1"
    assert manager.lock_path.exists()

    record = manager.inspect()
    assert isinstance(record, LeaseRecord)
    assert record.job_id == "job-1"
    assert record.phase == "JOB_ACQUIRE"
    assert record.lock_generation == 1
    assert record.owner_pid > 0
    assert record.start_time

    assert manager.release() is True
    assert manager.held() is False
    # The durable record is preserved after release for lock-generation continuity.
    durable = manager.inspect()
    assert isinstance(durable, LeaseRecord)
    assert durable.job_id == "job-1"


def test_second_manager_fails_closed_while_lease_held(tmp_path):
    """A competing manager cannot take an already-held GPU lease."""
    holder = GpuLeaseManager(0, tmp_path)
    holder.acquire("job-holder", "JOB_ACQUIRE")
    contender = GpuLeaseManager(0, tmp_path)

    try:
        with pytest.raises(GpuLeaseError):
            contender.acquire("job-contender", "JOB_ACQUIRE")
        assert contender.held() is False
    finally:
        holder.release()


def test_release_without_holding_is_idempotent(tmp_path):
    """A manager that never acquired reports a harmless no-op release."""
    manager = GpuLeaseManager(0, tmp_path)

    assert manager.release() is True
    assert manager.held() is False
    assert manager.inspect() is None


def test_heartbeat_updates_phase(tmp_path):
    """Heartbeats persist the new phase and refresh the heartbeat timestamp."""
    manager = GpuLeaseManager(0, tmp_path)
    manager.acquire("job-1", "JOB_ACQUIRE")
    try:
        updated = manager.heartbeat("VIDEO_GEN")

        assert updated.phase == "VIDEO_GEN"
        assert updated.heartbeat
        assert manager.inspect().phase == "VIDEO_GEN"
        assert manager.inspect().heartbeat == updated.heartbeat
    finally:
        manager.release()


def test_generation_increments_across_acquire_cycles(tmp_path):
    """Each acquire after a clean release bumps the lock generation."""
    manager = GpuLeaseManager(0, tmp_path)

    first = manager.acquire("job-1", "JOB_ACQUIRE")
    assert first.lock_generation == 1
    manager.release()

    second = manager.acquire("job-2", "JOB_ACQUIRE")
    try:
        assert second.lock_generation == 2
        assert manager.inspect().lock_generation == 2
    finally:
        manager.release()


def test_classify_stale_fresh_lease_is_not_recoverable(tmp_path):
    """A lease held by this live process never reports stale evidence."""
    manager = GpuLeaseManager(0, tmp_path)
    manager.acquire("job-1", "JOB_ACQUIRE")
    try:
        assessment = manager.classify_stale()

        assert isinstance(assessment, StaleAssessment)
        assert assessment.owner_alive is True
        assert assessment.heartbeat_stale is False
        assert assessment.classification != "stale-evidence"
    finally:
        manager.release()


def test_recover_stale_requires_stale_assessment_instance(tmp_path):
    """Recovery is hard-disabled without a classify_stale() assessment."""
    manager = GpuLeaseManager(0, tmp_path)

    with pytest.raises(GpuLeaseError):
        manager.recover_stale(None)


def test_recover_stale_refuses_live_owner(tmp_path, monkeypatch):
    """Even a stale-looking record is kept while the owner process is alive."""
    monkeypatch.setattr(lease_module, "_pid_alive", lambda pid: True)
    manager = GpuLeaseManager(0, tmp_path)
    manager.acquire("job-1", "JOB_ACQUIRE")
    try:
        assessment = manager.classify_stale()
        assert assessment.owner_alive is True

        with pytest.raises(GpuLeaseError):
            manager.recover_stale(assessment)

        assert manager.inspect() is not None
    finally:
        manager.release()


def test_recover_stale_clears_stale_record(tmp_path, monkeypatch):
    """A dead owner plus a stale heartbeat lets recovery delete only the record."""
    manager = GpuLeaseManager(0, tmp_path)
    stale_record = LeaseRecord(
        job_id="ghost-job",
        owner_pid=999999,
        start_time="2000-01-01T00:00:00+00:00",
        heartbeat="2000-01-01T00:00:00+00:00",
        phase="JOB_ACQUIRE",
        lock_generation=1,
    )
    manager.record_path.write_text(
        json.dumps(stale_record.to_dict()), encoding="utf-8"
    )
    assert not manager.lock_path.exists()
    monkeypatch.setattr(lease_module, "_pid_alive", lambda pid: False)

    assessment = manager.classify_stale()
    assert assessment.owner_alive is False
    assert assessment.heartbeat_stale is True
    assert assessment.lock_held_elsewhere is False
    assert assessment.classification == "stale-evidence"

    assert manager.recover_stale(assessment) is True
    assert manager.inspect() is None
    assert not manager.lock_path.exists()


def test_lease_record_roundtrip():
    """A lease record survives a dict encode/decode cycle unchanged."""
    record = LeaseRecord(
        job_id="job-roundtrip",
        owner_pid=4242,
        start_time="2026-10-20T00:00:00+00:00",
        heartbeat="2026-10-20T00:05:00+00:00",
        phase="VIDEO_GEN",
        lock_generation=7,
    )

    restored = LeaseRecord.from_dict(record.to_dict())

    assert restored.job_id == record.job_id
    assert restored.owner_pid == record.owner_pid
    assert restored.start_time == record.start_time
    assert restored.heartbeat == record.heartbeat
    assert restored.phase == record.phase
    assert restored.lock_generation == record.lock_generation
