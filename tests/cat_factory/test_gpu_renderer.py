"""Operator command: .venv\\Scripts\\python.exe -m pytest tests/cat_factory/test_gpu_renderer.py -q."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pixelle_video.cat_factory import storage
from pixelle_video.cat_factory.gpu.comfy_client import (
    ComfyClient,
    ComfyRequestError,
    QueueState,
)
from pixelle_video.cat_factory.gpu.mock_comfy import MockComfyClient, MockTransport
from pixelle_video.cat_factory.gpu.renderer import (
    GpuRenderer,
    RenderBlocked,
    RenderState,
    RenderStateError,
)


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


class FakeMonitor:
    """Offline stand-in for the GPU/CPU resource monitor."""

    def __init__(
        self,
        used_mb: int = 1000,
        release_timeout: bool = False,
        ram_blocked: bool = False,
    ) -> None:
        self._used = used_mb
        self.release_timeout = release_timeout
        self.ram_blocked = ram_blocked
        self.release_calls = 0

    def sample_vram(self):
        """Return a deterministic VRAM sample."""
        from pixelle_video.cat_factory.gpu.monitor import VramSample

        return VramSample(
            "RTX 3090", 24576, self._used, 24576 - self._used, ()
        )

    def vram_ceiling_mb(self) -> int:
        """Return the configured VRAM ceiling."""
        return 23552

    def wait_for_vram_release(self, *, baseline_used_mb: int, timeout_s: float | None = None):
        """Simulate a measured VRAM release or a release timeout."""
        from pixelle_video.cat_factory.gpu.monitor import VramReleaseTimeout, VramSample

        self.release_calls += 1
        if self.release_timeout:
            raise VramReleaseTimeout("VRAM still occupied")
        self._used = 0
        return VramSample("RTX 3090", 24576, 0, 24576, ())

    def check_video_ram_gate(self):
        """Return the RAM gate sample or block the video phase."""
        from pixelle_video.cat_factory.gpu.monitor import RamGateBlocked, RamSample

        if self.ram_blocked:
            raise RamGateBlocked("free RAM below measured offload minimum")
        return RamSample(32000, 64000)


def _config(**overrides):
    """Build a simple GPU config object with the offline test defaults."""
    values = {
        "generation_timeout_s": 5,
        "poll_interval_s": 0.01,
        "memory_release_timeout_s": 2,
        "oom_max_retries": 0,
        "vram_margin_mb": 1024,
        "ram_margin_mb": 2048,
        "offload_min_ram_mb": 4096,
        "image_workflow": "workflows/selfhost/image_flux3.json",
        "reference_workflow": "workflows/selfhost/image_flux3_ref.json",
        "video_workflow": "workflows/selfhost/video_wan2_2_i2v.json",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def _posts(client) -> list[tuple[str, str, object]]:
    """Return every recorded POST /prompt call."""
    return [call for call in client.calls if call[0] == "POST" and call[1] == "/prompt"]


def _build_renderer(tmp_path, *, client=None, monitor=None, lease=None, **config):
    """Construct a renderer wired to the offline doubles."""
    return GpuRenderer(
        lease=FakeLease() if lease is None else lease,
        monitor=FakeMonitor() if monitor is None else monitor,
        client=MockComfyClient("success") if client is None else client,
        gpu_config=_config(**config),
        output_root=tmp_path,
    )


def _is_subsequence(expected: list[RenderState], actual: list[RenderState]) -> bool:
    """Return True when ``expected`` appears in ``actual`` in order."""
    index = 0
    for state in actual:
        if index < len(expected) and state == expected[index]:
            index += 1
    return index == len(expected)


def test_render_fails_closed_without_planning_checkpoint(tmp_path):
    """Refuse to touch the GPU when the planning checkpoint is incomplete."""
    _write_manifest(tmp_path, complete=False)
    lease = FakeLease()
    client = MockComfyClient("success")
    renderer = _build_renderer(tmp_path, client=client, lease=lease)

    with pytest.raises(RenderBlocked):
        renderer.render("job-1")

    assert lease.phases == []
    assert client.calls == []


def test_render_happy_path_reaches_done_and_persists_artifacts(tmp_path):
    """Drive two shots to DONE and persist every artifact while releasing once."""
    _write_manifest(tmp_path, shots=[_shot("one"), _shot("two")])
    lease = FakeLease()
    renderer = _build_renderer(tmp_path, client=MockComfyClient("success"), lease=lease)

    report = renderer.render("job-1")

    assert report.state == RenderState.DONE
    manifest = storage.read_manifest(tmp_path / "job-1")
    for shot in manifest["shots"]:
        assets = shot["assets"]
        assert assets["keyframe_status"] == "completed"
        assert assets["keyframe_ref"]
        assert assets["video_status"] == "completed"
        assert assets["video_ref"]

    job_dir = tmp_path / "job-1"
    assert (job_dir / "render_report.json").is_file()
    assert (job_dir / "assembly.json").is_file()
    assert lease.release_calls == 1


def test_render_state_order_matches_canonical_chain(tmp_path):
    """Walk the documented state chain without reordering a single phase."""
    _write_manifest(tmp_path, shots=[_shot("one"), _shot("two")])
    renderer = _build_renderer(tmp_path)
    recorded: list[RenderState] = []
    original = renderer._transition

    def record(target: RenderState) -> None:
        recorded.append(target)
        original(target)

    renderer._transition = record
    renderer.render("job-1")

    expected = [
        RenderState.JOB_ACQUIRE,
        RenderState.VRAM_VERIFY,
        RenderState.IMAGE_GEN,
        RenderState.CHECKPOINT,
        RenderState.IMAGE_RELEASE,
        RenderState.VRAM_VERIFY,
        RenderState.VIDEO_ACQUIRE,
        RenderState.VRAM_VERIFY,
        RenderState.RAM_VERIFY,
        RenderState.VIDEO_GEN,
        RenderState.CHECKPOINT,
        RenderState.VIDEO_RELEASE,
        RenderState.VRAM_VERIFY,
        RenderState.CPU_ASSEMBLY,
        RenderState.DONE,
    ]
    assert _is_subsequence(expected, recorded), recorded


def test_render_resume_does_not_rerun_completed_shots(tmp_path):
    """Reuse completed assets so a resume submits only the unfinished shot."""
    completed_assets = {
        "keyframe_ref": "done.png",
        "keyframe_status": "completed",
        "video_ref": "done.mp4",
        "video_status": "completed",
    }
    first = _shot("one")
    first["assets"] = dict(completed_assets)
    _write_manifest(tmp_path, shots=[first, _shot("two")])
    client = MockComfyClient("success")
    renderer = _build_renderer(tmp_path, client=client)

    renderer.render("job-1")

    assert len(_posts(client)) == 2
    manifest = storage.read_manifest(tmp_path / "job-1")
    assert manifest["shots"][0]["assets"] == completed_assets


def test_render_timeout_marks_failed_and_releases_lease(tmp_path):
    """A submit timeout fails the shot closed and still releases the lease."""
    _write_manifest(tmp_path)
    lease = FakeLease()
    renderer = _build_renderer(
        tmp_path,
        client=MockComfyClient("timeout"),
        lease=lease,
        oom_max_retries=0,
    )

    with pytest.raises(RenderBlocked):
        renderer.render("job-1")

    manifest = storage.read_manifest(tmp_path / "job-1")
    assert manifest["shots"][0]["assets"]["keyframe_status"] == "failed"
    assert lease.release_calls == 1


class _OomClient:
    """Client whose prompt submission always reports an OOM rejection."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []
        self.post_calls = 0
        self.free_calls = 0

    def post_prompt(self, workflow, client_id=None) -> str:
        """Record the attempt and raise a classified OOM request error."""
        self.post_calls += 1
        self.calls.append(("POST", "/prompt", {"prompt": workflow}))
        raise ComfyRequestError("CUDA out of memory")

    def get_queue(self) -> QueueState:
        """Return an empty queue so OOM recovery can drain it."""
        return QueueState()

    def get_history(self, prompt_id: str) -> dict[str, object]:
        """Return no history; the prompt never completes."""
        return {}

    def free(self, **kwargs) -> bool:
        """Record a free call and acknowledge it."""
        self.free_calls += 1
        self.calls.append(("POST", "/free", dict(kwargs)))
        return True


def test_render_bounded_oom_retry_then_failure(tmp_path):
    """Retry exactly once after freeing VRAM, then fail closed."""
    _write_manifest(tmp_path)
    client = _OomClient()
    lease = FakeLease()
    renderer = _build_renderer(
        tmp_path,
        client=client,
        lease=lease,
        oom_max_retries=1,
    )

    with pytest.raises(RenderBlocked):
        renderer.render("job-1")

    assert client.post_calls == 2
    assert client.free_calls >= 1
    assert lease.release_calls == 1


def test_vram_release_not_observed_aborts_before_video(tmp_path):
    """Abort after keyframes when the measured VRAM release never happens."""
    _write_manifest(tmp_path, shots=[_shot("one"), _shot("two")])
    client = MockComfyClient("success")
    monitor = FakeMonitor(release_timeout=True)
    renderer = _build_renderer(tmp_path, client=client, monitor=monitor)

    with pytest.raises(RenderBlocked):
        renderer.render("job-1")

    assert len(_posts(client)) < 2 * 2


def test_ram_gate_blocks_video_path(tmp_path):
    """Abort before the video phase when the RAM gate rejects it."""
    _write_manifest(tmp_path, shots=[_shot("one"), _shot("two")])
    client = MockComfyClient("success")
    monitor = FakeMonitor(ram_blocked=True)
    renderer = _build_renderer(tmp_path, client=client, monitor=monitor)

    with pytest.raises(RenderBlocked):
        renderer.render("job-1")

    assert len(_posts(client)) < 2 * 2


def test_client_free_payload_is_exact():
    """Send the documented /free payload with no extra fields."""
    transport = MockTransport([(200, "{}")])
    client = ComfyClient(base_url="http://127.0.0.1:8188", transport=transport)

    client.free()

    assert transport.calls[-1] == (
        "POST",
        "/free",
        {"unload_models": True, "free_memory": True},
    )


def test_client_prompt_errors_fail_closed():
    """Reject node errors and non-200 responses instead of reporting success."""
    node_error_transport = MockTransport(
        [
            (
                200,
                json.dumps(
                    {
                        "prompt_id": "prompt-1",
                        "node_errors": {"3": {"errors": ["missing model"]}},
                    }
                ),
            )
        ]
    )
    node_error_client = ComfyClient(
        base_url="http://127.0.0.1:8188", transport=node_error_transport
    )
    with pytest.raises(ComfyRequestError):
        node_error_client.post_prompt({"3": {}})

    status_transport = MockTransport([(500, "boom")])
    status_client = ComfyClient(
        base_url="http://127.0.0.1:8188", transport=status_transport
    )
    with pytest.raises(ComfyRequestError):
        status_client.post_prompt({"3": {}})


def test_probe_is_read_only():
    """Probe the queue and VRAM without ever issuing a POST."""
    monitor = FakeMonitor()
    transport = MockTransport(
        [(200, json.dumps({"queue_running": [], "queue_pending": []}))],
        default=(200, "{}"),
    )
    client = ComfyClient(base_url="http://127.0.0.1:8188", transport=transport)

    result = client.probe(
        workflow_paths=(), vram_sampler=lambda: monitor.sample_vram()
    )

    assert result.vram is not None
    assert int(result.vram.used_mb) == 1000
    assert transport.calls
    assert all(
        method == "GET" and path == "/queue"
        for method, path, _payload in transport.calls
    )


def test_renderer_source_has_no_process_control():
    """The renderer must never spawn or terminate another process."""
    source_path = (
        Path(__file__).resolve().parents[2]
        / "pixelle_video"
        / "cat_factory"
        / "gpu"
        / "renderer.py"
    )
    source = source_path.read_text(encoding="utf-8")

    for marker in ("subprocess", "os.kill", "taskkill", "signal"):
        assert marker not in source


def test_illegal_transition_raises_render_state_error(tmp_path):
    """Requesting a non-adjacent phase fails closed with a state error."""
    _write_manifest(tmp_path)
    renderer = _build_renderer(tmp_path, client=MockComfyClient("success"))

    assert renderer.state == RenderState.IDLE
    with pytest.raises(RenderStateError):
        renderer._transition(RenderState.IMAGE_GEN)


def test_partial_success_marks_shot_failed(tmp_path):
    """A partial ComfyUI result fails the shot closed and releases the lease."""
    _write_manifest(tmp_path)
    lease = FakeLease()
    renderer = _build_renderer(
        tmp_path,
        client=MockComfyClient("partial_success"),
        lease=lease,
        oom_max_retries=0,
    )

    with pytest.raises(RenderBlocked):
        renderer.render("job-1")

    manifest = storage.read_manifest(tmp_path / "job-1")
    assert manifest["shots"][0]["status"] == "failed"
    assert lease.release_calls == 1


def test_request_error_is_bounded(tmp_path):
    """A rejected submit is attempted at least once and then fails closed."""
    _write_manifest(tmp_path)
    lease = FakeLease()
    client = MockComfyClient("request_error")
    renderer = _build_renderer(
        tmp_path,
        client=client,
        lease=lease,
        oom_max_retries=0,
    )

    with pytest.raises(RenderBlocked):
        renderer.render("job-1")

    assert len(_posts(client)) >= 1
    assert lease.release_calls == 1


def test_long_running_pending_times_out(tmp_path):
    """A still-pending prompt times out, marks the shot failed, releases once."""
    _write_manifest(tmp_path)
    lease = FakeLease()
    client = MockComfyClient("long_running_pending", settle_after_polls=10**6)
    renderer = _build_renderer(
        tmp_path,
        client=client,
        lease=lease,
        generation_timeout_s=0.2,
        poll_interval_s=0.01,
        oom_max_retries=0,
    )

    with pytest.raises(RenderBlocked):
        renderer.render("job-1")

    manifest = storage.read_manifest(tmp_path / "job-1")
    assert manifest["shots"][0]["status"] == "failed"
    assert lease.release_calls == 1


class _BusyQueueClient:
    """Client whose queue always holds another client's pending work."""

    def __init__(self) -> None:
        from pixelle_video.cat_factory.gpu.comfy_client import QueueState
        from pixelle_video.cat_factory.gpu.mock_comfy import MockComfyClient

        self._inner = MockComfyClient("success")
        self._queue_state = QueueState
        self.free_calls = 0
        self.calls: list[tuple[str, str, object]] = []

    def get_queue(self):
        """Report another client's work as always queued."""
        return self._queue_state(running=("other-user",), pending=())

    def post_prompt(self, workflow, client_id=None):
        """Delegate submission to the inner successful mock."""
        self.calls.append(("POST", "/prompt", workflow))
        return self._inner.post_prompt(workflow)

    def get_history(self, prompt_id):
        """Delegate history lookups to the inner successful mock."""
        return self._inner.get_history(prompt_id)

    def free(self):
        """Record a /free call and acknowledge it."""
        self.free_calls += 1
        return True

    def probe(self, **kwargs):
        """Refuse to probe a busy queue."""
        raise AssertionError("busy queue client must not be probed")

    def is_local_endpoint(self):
        """Report a loopback endpoint."""
        return True


def test_queue_active_refuses_release(tmp_path):
    """Never free VRAM while another client's work is still queued."""
    _write_manifest(tmp_path)
    client = _BusyQueueClient()
    renderer = _build_renderer(
        tmp_path,
        client=client,
        generation_timeout_s=0.2,
        poll_interval_s=0.01,
    )

    with pytest.raises(RenderBlocked):
        renderer.render("job-1")

    assert client.free_calls == 0


def test_resume_revalidates_missing_artifact_file(tmp_path):
    """A persisted ref whose backing file vanished forces a re-render."""
    manifest = _write_manifest(tmp_path)
    manifest["shots"][0]["assets"] = {
        "keyframe_ref": "k.png",
        "keyframe_status": "completed",
        "keyframe_path": str(tmp_path / "missing-keyframe.png"),
        "video_ref": "v.mp4",
        "video_status": "completed",
        "video_path": str(tmp_path / "missing-video.mp4"),
    }
    storage.write_manifest(manifest, output_root=tmp_path)
    client = MockComfyClient("success")
    renderer = _build_renderer(tmp_path, client=client)

    report = renderer.render("job-1")

    assert report.state == RenderState.DONE
    assert len(_posts(client)) == 2


def test_second_resume_after_revalidation_performs_no_generation(tmp_path):
    """Revalidation rewrites the stale refs, so later resumes generate nothing."""
    manifest = _write_manifest(tmp_path)
    manifest["shots"][0]["assets"] = {
        "keyframe_ref": "k.png",
        "keyframe_status": "completed",
        "keyframe_path": str(tmp_path / "missing-k.png"),
        "video_ref": "v.mp4",
        "video_status": "completed",
        "video_path": str(tmp_path / "missing-v.mp4"),
    }
    storage.write_manifest(manifest, output_root=tmp_path)
    client = MockComfyClient("success")
    renderer = _build_renderer(tmp_path, client=client)

    first_report = renderer.render("job-1")
    first = len(_posts(client))
    assert first_report.state == RenderState.DONE
    assert first == 2

    second_report = renderer.render("job-1")

    assert second_report.state == RenderState.DONE
    assert len(_posts(client)) == first

    third_report = renderer.render("job-1")

    assert third_report.state == RenderState.DONE
    assert len(_posts(client)) == first


def test_vram_occupied_pauses_before_generation(tmp_path):
    """Pause with a measured reason when another app still holds the GPU."""
    _write_manifest(tmp_path)
    lease = FakeLease()
    client = MockComfyClient("success")
    renderer = _build_renderer(
        tmp_path,
        client=client,
        monitor=FakeMonitor(used_mb=24000),
        lease=lease,
    )

    with pytest.raises(RenderBlocked):
        renderer.render("job-1")

    assert len(_posts(client)) == 0
    assert lease.release_calls == 1


def test_vram_ceiling_applies_configured_margin():
    """Derive the ceiling from the measured total minus the configured margin."""
    from pixelle_video.cat_factory.gpu.monitor import GpuMonitor, VramSample

    monitor = GpuMonitor(
        vram_sampler=SimpleNamespace(
            sample=lambda: VramSample("RTX 3090", 1000, 100, 900, ())
        ),
        config=SimpleNamespace(
            vram_margin_mb=200,
            ram_margin_mb=0,
            offload_min_ram_mb=None,
            poll_interval_s=0.01,
            memory_release_timeout_s=1,
        ),
    )

    assert monitor.vram_ceiling_mb() == 800
