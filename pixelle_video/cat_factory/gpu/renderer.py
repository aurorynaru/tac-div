# Copyright (C) 2024 Pixelle-Video contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Durable, fail-closed single-GPU render orchestrator.

The renderer drives one job at a time (``CONCURRENCY == 1``) through the
ordered state machine

``IDLE -> JOB_ACQUIRE -> VRAM_VERIFY -> IMAGE_GEN -> CHECKPOINT ->
IMAGE_RELEASE -> VRAM_VERIFY -> VIDEO_ACQUIRE -> VRAM_VERIFY -> RAM_VERIFY ->
VIDEO_GEN -> CHECKPOINT -> VIDEO_RELEASE -> VRAM_VERIFY -> CPU_ASSEMBLY ->
DONE``.

Only one model family is resident at any moment: every image keyframe is
generated, checkpointed to the job manifest, and released (measured VRAM
release) before the video model is acquired. The renderer never calls an LLM,
never falls back to a paid/cloud provider, never deletes artifacts, and never
force-releases or removes another process' GPU lease. Any condition it cannot
verify (incomplete planning checkpoint, high VRAM, slow queue drain, missing
workflow, exhausted retries) fails closed with :class:`RenderBlocked`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pixelle_video.cat_factory.gpu.comfy_client import ComfyClient
    from pixelle_video.cat_factory.gpu.lease import GpuLeaseManager
    from pixelle_video.cat_factory.gpu.monitor import GpuMonitor
    from pixelle_video.cat_factory.gpu.supervisor import ComfySupervisor
    from pixelle_video.config import CatFactoryGpuConfig

__all__ = [
    "CONCURRENCY",
    "RenderState",
    "RenderError",
    "RenderBlocked",
    "RenderStateError",
    "ShotRenderResult",
    "RenderReport",
    "GpuRenderer",
]

#: The orchestrator is strictly serial: one job, one model family at a time.
CONCURRENCY = 1

#: Durable side-car files written inside ``output/cat_factory/<job_id>/``.
RENDER_STATE_FILENAME = "render_state.json"
RENDER_REPORT_FILENAME = "render_report.json"
ASSEMBLY_FILENAME = "assembly.json"

#: Repository root, used to resolve project-relative workflow paths.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_STATE_VALUES: dict[str, "RenderState"] = {}


class RenderState(str, Enum):
    """Ordered render phases; only adjacent phases may follow each other."""

    IDLE = "idle"
    JOB_ACQUIRE = "job_acquire"
    VRAM_VERIFY = "vram_verify"
    IMAGE_GEN = "image_gen"
    CHECKPOINT = "checkpoint"
    IMAGE_RELEASE = "image_release"
    VIDEO_ACQUIRE = "video_acquire"
    RAM_VERIFY = "ram_verify"
    VIDEO_GEN = "video_gen"
    VIDEO_RELEASE = "video_release"
    CPU_ASSEMBLY = "cpu_assembly"
    DONE = "done"
    FAILED = "failed"


# FAILED is reachable from every non-terminal phase so that any verification
# or generation failure can be recorded without violating the state machine.
_ALLOWED_TRANSITIONS: dict[RenderState, frozenset[RenderState]] = {
    RenderState.IDLE: frozenset({RenderState.JOB_ACQUIRE}),
    RenderState.JOB_ACQUIRE: frozenset({RenderState.VRAM_VERIFY}),
    RenderState.VRAM_VERIFY: frozenset(
        {
            RenderState.IMAGE_GEN,
            RenderState.IMAGE_RELEASE,
            RenderState.VIDEO_ACQUIRE,
            RenderState.RAM_VERIFY,
            RenderState.CPU_ASSEMBLY,
        }
    ),
    RenderState.IMAGE_GEN: frozenset({RenderState.CHECKPOINT}),
    RenderState.CHECKPOINT: frozenset(
        {
            RenderState.IMAGE_GEN,
            RenderState.IMAGE_RELEASE,
            RenderState.VIDEO_GEN,
            RenderState.VIDEO_RELEASE,
            RenderState.CPU_ASSEMBLY,
        }
    ),
    RenderState.IMAGE_RELEASE: frozenset({RenderState.VRAM_VERIFY}),
    RenderState.VIDEO_ACQUIRE: frozenset({RenderState.VRAM_VERIFY}),
    RenderState.RAM_VERIFY: frozenset({RenderState.VIDEO_GEN}),
    RenderState.VIDEO_GEN: frozenset({RenderState.CHECKPOINT}),
    RenderState.VIDEO_RELEASE: frozenset({RenderState.VRAM_VERIFY}),
    RenderState.CPU_ASSEMBLY: frozenset({RenderState.DONE}),
    RenderState.DONE: frozenset(),
    RenderState.FAILED: frozenset(),
}

_TERMINAL_STATES = frozenset({RenderState.DONE, RenderState.FAILED})

for _state in RenderState:
    _STATE_VALUES[_state.value] = _state
    if _state not in _TERMINAL_STATES:
        _ALLOWED_TRANSITIONS[_state] = _ALLOWED_TRANSITIONS[_state] | frozenset(
            {RenderState.FAILED}
        )
del _state


class RenderError(Exception):
    """Base class for render orchestration failures."""


class RenderBlocked(RenderError):
    """A fail-closed condition stopped the render; artifacts are preserved."""


class RenderStateError(RenderError):
    """An illegal state transition was requested."""


@dataclass(frozen=True)
class ShotRenderResult:
    """Durable per-shot outcome recorded in ``render_state.json``."""

    shot_id: str
    status: str
    keyframe_ref: str | None = None
    video_ref: str | None = None
    attempts: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this result."""
        return {
            "shot_id": self.shot_id,
            "status": self.status,
            "keyframe_ref": self.keyframe_ref,
            "video_ref": self.video_ref,
            "attempts": self.attempts,
            "error": self.error,
        }


@dataclass(frozen=True)
class RenderReport:
    """Summary returned by a successful render and persisted to disk."""

    job_id: str
    state: RenderState
    shots: tuple[ShotRenderResult, ...] = ()
    errors: tuple[str, ...] = ()
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this report."""
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "shots": [shot.to_dict() for shot in self.shots],
            "errors": list(self.errors),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class _ShotFailure(Exception):
    """Internal carrier for a classified per-shot generation failure."""

    def __init__(
        self,
        message: str,
        *,
        transient: bool,
        oom: bool = False,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.transient = transient
        self.oom = oom
        self.attempts = attempts


def _config_value(config: object | None, name: str, default: Any) -> Any:
    """Read a GPU config field, falling back to ``default`` when unset."""
    if config is None:
        return default
    value = getattr(config, name, None)
    return default if value is None else value


def _is_oom_message(message: str) -> bool:
    """Return True when a ComfyUI error message denotes GPU out-of-memory."""
    lowered = message.lower()
    if "out of memory" in lowered or "outofmemory" in lowered or "cuda oom" in lowered:
        return True
    normalized = lowered.replace(":", " ").replace(".", " ").replace("[", " ").replace("]", " ")
    return "oom" in normalized.split()


def _history_entry(history: object, prompt_id: str) -> dict[str, Any]:
    """Normalise the ``get_history`` payload into the entry for ``prompt_id``."""
    if not isinstance(history, dict):
        return {}
    entry = history.get(prompt_id)
    if isinstance(entry, dict):
        return entry
    if "outputs" in history or "status" in history:
        return history
    return {}


def _history_failure(entry: dict[str, Any]) -> str | None:
    """Return a failure description when a history entry is terminal-failed."""
    status = entry.get("status")
    if not isinstance(status, dict):
        return None
    status_str = str(status.get("status_str") or "").strip().lower()
    if status_str in ("error", "failed"):
        return f"ComfyUI reported status {status_str!r}"
    messages = status.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, (list, tuple)) or not message:
                continue
            kind = str(message[0]).strip().lower()
            if kind not in ("execution_error", "execution_interrupted"):
                continue
            detail = message[1] if len(message) > 1 else {}
            if isinstance(detail, dict):
                text = detail.get("exception_message") or detail.get("message")
                if text:
                    return str(text)
            return f"ComfyUI reported {kind}"
    return None


def _extract_output_refs(entry: dict[str, Any]) -> list[str]:
    """Collect ``subfolder/filename`` references from a ComfyUI history entry."""
    refs: list[str] = []
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return refs
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for key in ("images", "gifs", "videos"):
            items = node_output.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename")
                if not isinstance(filename, str) or not filename.strip():
                    continue
                subfolder = item.get("subfolder")
                if isinstance(subfolder, str) and subfolder.strip():
                    refs.append(f"{subfolder.strip('/')}/{filename.strip()}")
                else:
                    refs.append(filename.strip())
    return refs


def _history_is_terminal(entry: dict[str, Any]) -> bool:
    """Return True when a non-empty history entry cannot change any further."""
    if _extract_output_refs(entry):
        return True
    status = entry.get("status")
    if isinstance(status, dict):
        status_str = str(status.get("status_str") or "").strip().lower()
        if status_str in ("success", "completed", "error", "failed"):
            return True
    return False


class GpuRenderer:
    """Serial, self-checkpointing ComfyUI image/video render orchestrator."""

    def __init__(
        self,
        *,
        lease: "GpuLeaseManager",
        monitor: "GpuMonitor",
        client: "ComfyClient",
        supervisor: "ComfySupervisor | None" = None,
        gpu_config: "CatFactoryGpuConfig | None" = None,
        output_root: str | Path | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        # Lazy imports keep the module importable without the wider project.
        from pixelle_video.cat_factory import storage as storage_module
        from pixelle_video.cat_factory.gpu import comfy_client as comfy_module
        from pixelle_video.cat_factory.gpu import lease as lease_module
        from pixelle_video.cat_factory.gpu import monitor as monitor_module

        self._storage = storage_module
        self._comfy_timeout = comfy_module.ComfyTimeoutError
        self._comfy_request = comfy_module.ComfyRequestError
        self._monitor_error = monitor_module.MonitorError
        self._vram_timeout = monitor_module.VramReleaseTimeout
        self._lease_error = lease_module.GpuLeaseError

        self._lease = lease
        self._monitor = monitor
        self._client = client
        self._supervisor = supervisor
        self._gpu_config = gpu_config
        self._resolved_gpu_config: "CatFactoryGpuConfig | None" = None
        root = storage_module.CAT_FACTORY_OUTPUT_ROOT if output_root is None else output_root
        self._output_root = Path(root)
        self._clock = time.monotonic if clock is None else clock
        self._sleep = time.sleep if sleep is None else sleep

        self._job_id = ""
        self._job_dir: Path | None = None
        self._state = RenderState.IDLE
        self._results: list[ShotRenderResult] = []
        self._errors: list[str] = []
        self._started_at: float | None = None
        self._durable = False
        self._active = False
        self._owns_lease = False
        self._started_supervisor = False
        self._temp_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def job_id(self) -> str:
        """Identifier of the job most recently driven by this renderer."""
        return self._job_id

    @property
    def state(self) -> RenderState:
        """Current render phase (``IDLE`` before the first render call)."""
        return self._state

    def render(self, job_id: str) -> RenderReport:
        """Run (or resume) one render job and return its final report.

        The manifest must already contain a complete planning checkpoint;
        otherwise :class:`RenderBlocked` is raised without touching GPU state.
        """
        if self._active:
            raise RenderBlocked(
                f"renderer already running job {self._job_id!r}; "
                f"concurrency is limited to {CONCURRENCY}"
            )

        self._active = True
        self._job_id = str(job_id)
        self._job_dir = self._output_root / self._job_id
        self._state = RenderState.IDLE
        self._results = []
        self._errors = []
        self._started_at = self._clock()
        self._durable = False
        self._owns_lease = False
        self._started_supervisor = False

        try:
            manifest = self._load_manifest()
            self._seed_results(manifest)
            config = self._resolve_config()

            self._transition(RenderState.JOB_ACQUIRE)
            self._acquire_lease()
            self._start_supervisor_if_requested()

            self._transition(RenderState.VRAM_VERIFY)
            self._verify_vram()

            self._ensure_keyframes(manifest, config)
            self._release_gpu(config, RenderState.IMAGE_RELEASE)

            self._transition(RenderState.VIDEO_ACQUIRE)
            self._transition(RenderState.VRAM_VERIFY)
            self._verify_vram()

            self._transition(RenderState.RAM_VERIFY)
            self._verify_ram(config)

            self._ensure_videos(manifest, config)
            self._release_gpu(config, RenderState.VIDEO_RELEASE)

            self._transition(RenderState.CPU_ASSEMBLY)
            self._write_assembly(manifest)

            self._transition(RenderState.DONE)
            finished_at = self._clock()
            report = RenderReport(
                job_id=self._job_id,
                state=RenderState.DONE,
                shots=tuple(self._results),
                errors=tuple(self._errors),
                started_at=self._started_at,
                finished_at=finished_at,
            )
            self._write_report(report)
            return report
        except RenderBlocked as exc:
            self._mark_failed(str(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - converted to a fail-closed block
            message = f"{type(exc).__name__}: {exc}"
            self._mark_failed(message)
            raise RenderBlocked(message) from exc
        finally:
            self._cleanup()
            self._active = False

    def resume(self, job_id: str) -> RenderReport:
        """Resume a job, re-validating every persisted output before reuse."""
        return self.render(job_id)

    def status(self, job_id: str) -> RenderReport:
        """Return the persisted status of ``job_id`` without side effects.

        Only ``render_state.json`` and the manifest are read. No lease is
        acquired, no process is spawned, no ComfyUI call is made and no file
        is created, modified or removed.
        """
        job_id_text = str(job_id or "")
        if not job_id_text:
            raise RenderStateError("job_id is required")

        job_dir = self._output_root / job_id_text
        state = RenderState.IDLE
        results: list[ShotRenderResult] = []
        errors: list[str] = []

        state_path = job_dir / RENDER_STATE_FILENAME
        if state_path.is_file():
            payload = self._read_json(state_path)
            if isinstance(payload, dict):
                raw_state = str(payload.get("state") or "")
                state = _STATE_VALUES.get(raw_state, RenderState.IDLE)
                results = self._results_from_payload(payload.get("shots"))
                raw_errors = payload.get("errors")
                if isinstance(raw_errors, list):
                    errors = [str(item) for item in raw_errors]
        elif not (job_dir / self._storage.MANIFEST_FILENAME).is_file():
            return RenderReport(job_id=job_id_text, state=RenderState.IDLE)

        if not results:
            manifest = self._read_manifest_safely(job_dir)
            if manifest is not None:
                results = self._results_from_manifest(manifest)

        return RenderReport(
            job_id=job_id_text,
            state=state,
            shots=tuple(results),
            errors=tuple(errors),
        )

    # ------------------------------------------------------------------
    # Manifest, config and state plumbing
    # ------------------------------------------------------------------
    def _load_manifest(self) -> dict[str, Any]:
        job_dir = self._require_job_dir()
        try:
            manifest = self._storage.read_manifest(job_dir)
        except Exception as exc:  # noqa: BLE001 - any read failure is fatal
            raise RenderBlocked(
                f"cannot read manifest for job {self._job_id!r}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise RenderBlocked(f"manifest for job {self._job_id!r} is malformed")

        self._durable = True
        try:
            planning_complete = bool(self._storage.is_planning_complete(manifest))
        except Exception as exc:  # noqa: BLE001 - validation failure is fatal
            raise RenderBlocked(
                f"cannot validate the planning checkpoint for job "
                f"{self._job_id!r}: {exc}"
            ) from exc
        if not planning_complete:
            raise RenderBlocked(
                f"planning checkpoint is incomplete for job {self._job_id!r}; "
                "refusing to acquire the GPU"
            )
        return manifest

    def _resolve_config(self) -> "CatFactoryGpuConfig":
        if self._gpu_config is not None:
            return self._gpu_config
        if self._resolved_gpu_config is None:
            try:
                from pixelle_video.config import config_manager
            except Exception as exc:  # noqa: BLE001 - config import is optional
                raise RenderBlocked(
                    f"cat_factory.gpu configuration is unavailable: {exc}"
                ) from exc
            config = getattr(config_manager, "config", None)
            cat_factory = getattr(config, "cat_factory", None)
            gpu_config = getattr(cat_factory, "gpu", None)
            if gpu_config is None:
                raise RenderBlocked(
                    "cat_factory.gpu configuration is not available"
                )
            self._resolved_gpu_config = gpu_config
        resolved = self._resolved_gpu_config
        if resolved is None:  # pragma: no cover - guarded by the branch above
            raise RenderBlocked("cat_factory.gpu configuration is not available")
        return resolved

    def _transition(self, target: RenderState) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(self._state, frozenset())
        if target not in allowed:
            raise RenderStateError(
                f"illegal render transition {self._state.value!r} -> {target.value!r}"
            )
        self._state = target
        self._persist_render_state()

    def _mark_failed(self, message: str) -> None:
        self._record_error(message)
        if self._state in _TERMINAL_STATES:
            return
        try:
            self._transition(RenderState.FAILED)
        except RenderStateError:
            self._state = RenderState.FAILED
        except OSError as exc:
            self._record_error(f"failed to persist failed state: {exc}")
            self._state = RenderState.FAILED

    def _persist_render_state(self) -> None:
        if not self._durable:
            return
        job_dir = self._job_dir
        if job_dir is None:
            return
        payload = {
            "job_id": self._job_id,
            "state": self._state.value,
            "updated_at": time.time(),
            "shots": [result.to_dict() for result in self._results],
            "errors": list(self._errors),
        }
        self._atomic_write_json(job_dir / RENDER_STATE_FILENAME, payload)

    def _persist_manifest(self, manifest: dict[str, Any]) -> None:
        self._storage.write_manifest(manifest, output_root=self._output_root)

    def _write_report(self, report: RenderReport) -> None:
        job_dir = self._require_job_dir()
        self._atomic_write_json(job_dir / RENDER_REPORT_FILENAME, report.to_dict())

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._temp_counter += 1
        temp_name = (
            f".{path.name}.{os.getpid()}.{time.monotonic_ns()}."
            f"{self._temp_counter}.tmp"
        )
        temp_path = path.with_name(temp_name)
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    @staticmethod
    def _read_json(path: Path) -> object | None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def _read_manifest_safely(self, job_dir: Path) -> dict[str, Any] | None:
        if not (job_dir / self._storage.MANIFEST_FILENAME).is_file():
            return None
        try:
            manifest = self._storage.read_manifest(job_dir)
        except Exception:  # noqa: BLE001 - status must stay read-only and safe
            return None
        return manifest if isinstance(manifest, dict) else None

    # ------------------------------------------------------------------
    # Lease and supervisor handling
    # ------------------------------------------------------------------
    def _acquire_lease(self) -> None:
        try:
            self._lease.acquire(self._job_id, "JOB_ACQUIRE")
        except Exception as exc:  # noqa: BLE001 - any lease failure blocks the run
            if isinstance(exc, self._lease_error):
                raise RenderBlocked(
                    f"GPU lease unavailable for job {self._job_id!r}: {exc}"
                ) from exc
            raise
        self._owns_lease = True

    def _start_supervisor_if_requested(self) -> None:
        if self._supervisor is None:
            return
        try:
            mode = str(getattr(self._supervisor, "mode", "") or "").strip().lower()
            if mode == "managed":
                self._supervisor.start()
                self._started_supervisor = True
                return
            if mode == "attached":
                if not bool(getattr(self._supervisor, "is_attached", False)):
                    raise RenderBlocked(
                        "ComfyUI supervisor is in attached mode but is not "
                        "attached to an operator-authorized endpoint"
                    )
                return
            raise RenderBlocked(f"unknown ComfyUI supervisor mode {mode!r}")
        except RenderBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - supervisor failure blocks the run
            raise RenderBlocked(
                f"ComfyUI supervisor could not be prepared: {exc}"
            ) from exc

    def _cleanup(self) -> None:
        if self._owns_lease:
            try:
                if bool(self._lease.held()):
                    self._lease.release()
            except Exception as exc:  # noqa: BLE001 - never mask the render failure
                self._record_error(f"GPU lease release failed: {exc}")
            self._owns_lease = False

        if self._started_supervisor and self._supervisor is not None:
            try:
                self._supervisor.stop()
            except Exception as exc:  # noqa: BLE001 - never mask the render failure
                self._record_error(f"ComfyUI supervisor stop failed: {exc}")
            self._started_supervisor = False

        if self._durable:
            try:
                self._persist_render_state()
            except OSError as exc:
                self._record_error(f"failed to persist final render state: {exc}")

    # ------------------------------------------------------------------
    # GPU verification and release
    # ------------------------------------------------------------------
    def _verify_vram(self) -> None:
        try:
            sample = self._monitor.sample_vram()
            ceiling_mb = int(self._monitor.vram_ceiling_mb())
        except Exception as exc:  # noqa: BLE001 - unverifiable VRAM blocks
            raise RenderBlocked(f"VRAM verification failed: {exc}") from exc
        if int(sample.used_mb) > ceiling_mb:
            raise RenderBlocked(
                f"GPU {sample.gpu_name!r} is using {sample.used_mb} MiB, above "
                f"the configured ceiling of {ceiling_mb} MiB"
            )

    def _verify_ram(self, config: "CatFactoryGpuConfig") -> None:
        try:
            sample = self._monitor.check_video_ram_gate()
        except Exception as exc:  # noqa: BLE001 - unverifiable RAM blocks
            raise RenderBlocked(f"RAM gate rejected the video phase: {exc}") from exc
        minimum_mb = _config_value(config, "offload_min_ram_mb", None)
        if minimum_mb is not None and int(sample.free_mb) < int(minimum_mb):
            raise RenderBlocked(
                f"only {sample.free_mb} MiB RAM free, below the configured "
                f"offload minimum of {int(minimum_mb)} MiB"
            )

    def _release_gpu(
        self,
        config: "CatFactoryGpuConfig",
        release_state: RenderState,
    ) -> None:
        self._transition(release_state)
        self._drain_queue(config)

        try:
            pre_free = self._monitor.sample_vram()
        except Exception as exc:  # noqa: BLE001 - unverifiable baseline blocks
            raise RenderBlocked(
                f"cannot measure VRAM before releasing {release_state.value}: {exc}"
            ) from exc

        freed = self._client.free()
        if freed is False:
            self._record_error(
                f"{release_state.value}: ComfyUI /free returned False; "
                "waiting for measured release anyway"
            )

        baseline_mb = 0 if int(pre_free.used_mb) <= 0 else max(0, int(pre_free.used_mb) - 1)
        timeout_s = float(_config_value(config, "memory_release_timeout_s", 60.0))
        try:
            self._monitor.wait_for_vram_release(
                baseline_used_mb=baseline_mb,
                timeout_s=timeout_s,
            )
        except self._vram_timeout as exc:
            raise RenderBlocked(
                f"VRAM did not return to the measured baseline of {baseline_mb} MiB "
                f"within {timeout_s:.1f}s after {release_state.value}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - unmeasured release blocks
            raise RenderBlocked(
                f"VRAM release could not be measured after "
                f"{release_state.value}: {exc}"
            ) from exc

        self._transition(RenderState.VRAM_VERIFY)

    def _drain_queue(self, config: "CatFactoryGpuConfig") -> None:
        timeout_s = float(_config_value(config, "generation_timeout_s", 600.0))
        poll_s = max(0.05, float(_config_value(config, "poll_interval_s", 2.0)))
        deadline = self._clock() + timeout_s
        last_error: str | None = None

        while self._clock() < deadline:
            try:
                queue_state = self._client.get_queue()
            except Exception as exc:  # noqa: BLE001 - poll again until deadline
                last_error = str(exc)
                self._sleep(min(poll_s, max(0.0, deadline - self._clock())))
                continue

            running = getattr(queue_state, "running", None)
            pending = getattr(queue_state, "pending", None)
            if not running and not pending:
                return
            self._sleep(min(poll_s, max(0.0, deadline - self._clock())))

        detail = f" (last error: {last_error})" if last_error else ""
        raise RenderBlocked(
            f"ComfyUI queue did not drain within {timeout_s:.1f}s; refusing to "
            f"discard another client's work{detail}"
        )

    # ------------------------------------------------------------------
    # Image phase
    # ------------------------------------------------------------------
    def _ensure_keyframes(
        self,
        manifest: dict[str, Any],
        config: "CatFactoryGpuConfig",
    ) -> None:
        shots = self._manifest_shots(manifest)
        if not shots:
            raise RenderBlocked("manifest contains no shots; nothing to render")

        for index, shot in enumerate(shots):
            shot_id = self._shot_id(shot, index)
            assets = self._ensure_assets(shot)
            if self._has_completed_asset(assets, "keyframe"):
                continue

            self._transition(RenderState.IMAGE_GEN)
            uses_reference = self._uses_identity_reference(manifest, shot)
            workflow_spec = (
                _config_value(config, "reference_workflow", None)
                if uses_reference
                else _config_value(config, "image_workflow", None)
            )
            workflow = self._load_workflow(workflow_spec, "image keyframe")

            try:
                refs, attempts = self._generate_with_budget(
                    shot_id=shot_id,
                    workflow=workflow,
                    config=config,
                    kind="keyframe",
                )
            except _ShotFailure as exc:
                self._record_shot_failure(
                    manifest, shot, shot_id, "keyframe", exc
                )
                raise RenderBlocked(
                    f"keyframe generation failed for shot {shot_id!r}: {exc}"
                ) from exc

            ref = refs[0]
            assets["keyframe_ref"] = ref
            assets["keyframe_status"] = "completed"
            assets.pop("keyframe_error", None)
            assets.pop("keyframe_path", None)
            shot["status"] = "image_done"
            self._record_attempt(manifest, shot_id, attempts)
            self._store_result(
                ShotRenderResult(
                    shot_id=shot_id,
                    status="image_done",
                    keyframe_ref=ref,
                    video_ref=assets.get("video_ref"),
                    attempts=attempts,
                )
            )

            self._transition(RenderState.CHECKPOINT)
            self._persist_manifest(manifest)

    # ------------------------------------------------------------------
    # Video phase
    # ------------------------------------------------------------------
    def _ensure_videos(
        self,
        manifest: dict[str, Any],
        config: "CatFactoryGpuConfig",
    ) -> None:
        shots = self._manifest_shots(manifest)
        if not shots:
            raise RenderBlocked("manifest contains no shots; nothing to render")

        generated_any = False
        for index, shot in enumerate(shots):
            shot_id = self._shot_id(shot, index)
            assets = self._ensure_assets(shot)
            if self._has_completed_asset(assets, "video"):
                continue

            self._transition(RenderState.VIDEO_GEN)
            generated_any = True

            workflow = self._load_workflow(
                _config_value(config, "video_workflow", None),
                "video",
            )
            try:
                refs, attempts = self._generate_with_budget(
                    shot_id=shot_id,
                    workflow=workflow,
                    config=config,
                    kind="video",
                )
            except _ShotFailure as exc:
                self._record_shot_failure(manifest, shot, shot_id, "video", exc)
                raise RenderBlocked(
                    f"video generation failed for shot {shot_id!r}: {exc}"
                ) from exc

            ref = refs[0]
            assets["video_ref"] = ref
            assets["video_status"] = "completed"
            assets.pop("video_error", None)
            assets.pop("video_path", None)
            shot["status"] = "video_done"
            self._record_attempt(manifest, shot_id, attempts)
            self._store_result(
                ShotRenderResult(
                    shot_id=shot_id,
                    status="video_done",
                    keyframe_ref=assets.get("keyframe_ref"),
                    video_ref=ref,
                    attempts=attempts,
                )
            )

            self._transition(RenderState.CHECKPOINT)
            self._persist_manifest(manifest)

        if not generated_any:
            # No work to do on resume: traverse the canonical chain so the
            # VIDEO_RELEASE transition stays legal, without submitting a prompt.
            self._transition(RenderState.VIDEO_GEN)
            self._transition(RenderState.CHECKPOINT)

    # ------------------------------------------------------------------
    # Generation driver
    # ------------------------------------------------------------------
    def _generate_with_budget(
        self,
        *,
        shot_id: str,
        workflow: dict[str, Any],
        config: "CatFactoryGpuConfig",
        kind: str,
    ) -> tuple[list[str], int]:
        retries = int(_config_value(config, "oom_max_retries", 0))
        max_attempts = max(1, 1 + retries)
        attempts = 0
        last_failure: _ShotFailure | None = None

        while attempts < max_attempts:
            attempts += 1
            try:
                return self._submit_and_wait(workflow, config, kind, shot_id), attempts
            except _ShotFailure as exc:
                last_failure = exc
                if not exc.transient or attempts >= max_attempts:
                    break
                if exc.oom:
                    try:
                        self._release_after_oom(config)
                    except RenderBlocked as release_exc:
                        last_failure = _ShotFailure(
                            f"OOM recovery failed: {release_exc}",
                            transient=False,
                            oom=True,
                        )
                        break
                poll_s = max(0.0, float(_config_value(config, "poll_interval_s", 2.0)))
                self._sleep(poll_s)

        if last_failure is None:  # pragma: no cover - loop always sets it on failure
            last_failure = _ShotFailure(
                f"{kind} generation for shot {shot_id!r} failed",
                transient=False,
            )
        last_failure.attempts = attempts
        raise last_failure

    def _submit_and_wait(
        self,
        workflow: dict[str, Any],
        config: "CatFactoryGpuConfig",
        kind: str,
        shot_id: str,
    ) -> list[str]:
        try:
            prompt_id = str(self._client.post_prompt(workflow))
        except Exception as exc:  # noqa: BLE001 - classified below
            transient, oom = self._classify_failure(exc)
            raise _ShotFailure(
                f"{kind} submit failed for shot {shot_id!r}: {exc}",
                transient=transient,
                oom=oom,
            ) from exc

        if not prompt_id.strip():
            raise _ShotFailure(
                f"{kind} submit for shot {shot_id!r} returned an empty prompt id",
                transient=False,
            )

        timeout_s = float(_config_value(config, "generation_timeout_s", 600.0))
        poll_s = max(0.05, float(_config_value(config, "poll_interval_s", 2.0)))
        deadline = self._clock() + timeout_s

        while self._clock() < deadline:
            try:
                self._client.get_queue()
                history = self._client.get_history(prompt_id)
            except Exception as exc:  # noqa: BLE001 - classified below
                transient, oom = self._classify_failure(exc)
                raise _ShotFailure(
                    f"{kind} polling failed for shot {shot_id!r}: {exc}",
                    transient=transient,
                    oom=oom,
                ) from exc

            entry = _history_entry(history, prompt_id)
            if entry:
                failure = _history_failure(entry)
                if failure is not None:
                    oom = _is_oom_message(failure)
                    raise _ShotFailure(
                        f"{kind} generation failed for shot {shot_id!r}: {failure}",
                        transient=oom,
                        oom=oom,
                    )
                refs = _extract_output_refs(entry)
                if refs:
                    return refs
                if _history_is_terminal(entry):
                    raise _ShotFailure(
                        f"{kind} generation for shot {shot_id!r} finished without "
                        "any output reference",
                        transient=False,
                    )

            self._sleep(min(poll_s, max(0.0, deadline - self._clock())))

        raise _ShotFailure(
            f"{kind} generation for shot {shot_id!r} timed out after "
            f"{timeout_s:.1f}s (prompt {prompt_id!r})",
            transient=True,
        )

    def _classify_failure(self, exc: Exception) -> tuple[bool, bool]:
        message = str(exc)
        oom = _is_oom_message(message)
        if isinstance(exc, self._comfy_timeout):
            return True, oom
        if isinstance(exc, self._comfy_request):
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429:
                return False, oom
            return True, oom
        if isinstance(exc, self._monitor_error):
            return False, oom
        return oom, oom

    def _release_after_oom(self, config: "CatFactoryGpuConfig") -> None:
        try:
            self._drain_queue(config)
            pre_free = self._monitor.sample_vram()
            self._client.free()
            baseline_mb = (
                0
                if int(pre_free.used_mb) <= 0
                else max(0, int(pre_free.used_mb) - 1)
            )
            self._monitor.wait_for_vram_release(
                baseline_used_mb=baseline_mb,
                timeout_s=float(_config_value(config, "memory_release_timeout_s", 60.0)),
            )
        except RenderBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - recovery failure is fatal
            raise RenderBlocked(f"OOM recovery release failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Workflow loading and shot helpers
    # ------------------------------------------------------------------
    def _load_workflow(self, spec: object, label: str) -> dict[str, Any]:
        if isinstance(spec, dict):
            return spec
        if spec is None or (isinstance(spec, str) and not spec.strip()):
            raise RenderBlocked(f"no {label} workflow is configured")

        try:
            raw_path = Path(str(spec))
        except (TypeError, ValueError) as exc:
            raise RenderBlocked(f"invalid {label} workflow path {spec!r}: {exc}") from exc

        if raw_path.is_absolute():
            candidates = [raw_path]
        else:
            candidates = [_PROJECT_ROOT / raw_path, Path.cwd() / raw_path]

        selected = next((item for item in candidates if item.is_file()), None)
        if selected is None:
            joined = ", ".join(str(item) for item in candidates)
            raise RenderBlocked(
                f"{label} workflow file was not found (looked for {joined})"
            )

        try:
            with selected.open("r", encoding="utf-8") as handle:
                workflow = json.load(handle)
        except OSError as exc:
            raise RenderBlocked(
                f"cannot read {label} workflow {selected}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RenderBlocked(
                f"cannot parse {label} workflow {selected}: {exc}"
            ) from exc

        if not isinstance(workflow, dict):
            raise RenderBlocked(
                f"{label} workflow {selected} is not a JSON object"
            )
        return workflow

    def _manifest_shots(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        shots = manifest.get("shots")
        if not isinstance(shots, list):
            raise RenderBlocked("manifest shots are missing or malformed")
        normalised: list[dict[str, Any]] = []
        for shot in shots:
            if not isinstance(shot, dict):
                raise RenderBlocked("manifest contains a malformed shot entry")
            normalised.append(shot)
        return normalised

    @staticmethod
    def _shot_id(shot: dict[str, Any], index: int) -> str:
        value = shot.get("shot_id")
        if isinstance(value, str) and value.strip():
            return value
        return f"shot_{index + 1}"

    @staticmethod
    def _ensure_assets(shot: dict[str, Any]) -> dict[str, Any]:
        assets = shot.get("assets")
        if not isinstance(assets, dict):
            assets = {}
            shot["assets"] = assets
        return assets

    def _has_completed_asset(self, assets: dict[str, Any], kind: str) -> bool:
        prefix = "keyframe" if kind == "keyframe" else "video"
        if assets.get(f"{prefix}_status") != "completed":
            return False
        ref = assets.get(f"{prefix}_ref")
        if not isinstance(ref, str) or not ref.strip():
            return False
        path_value = assets.get(f"{prefix}_path")
        if path_value:
            return self._file_is_nonempty(path_value)
        return True

    def _file_is_nonempty(self, value: object) -> bool:
        try:
            raw_path = Path(str(value))
        except (TypeError, ValueError) as exc:
            self._record_error(f"invalid persisted artifact path {value!r}: {exc}")
            return False

        job_dir = self._job_dir
        if raw_path.is_absolute():
            candidates = [raw_path]
        elif job_dir is not None:
            candidates = [job_dir / raw_path, _PROJECT_ROOT / raw_path, raw_path]
        else:
            candidates = [_PROJECT_ROOT / raw_path, raw_path]

        for candidate in candidates:
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return True
            except OSError as exc:
                self._record_error(f"cannot stat artifact {candidate}: {exc}")
        return False

    @staticmethod
    def _uses_identity_reference(
        manifest: dict[str, Any],
        shot: dict[str, Any],
    ) -> bool:
        keys = ("identity_reference", "character_id", "reference_image")
        for key in keys:
            if shot.get(key):
                return True

        selected = manifest.get("selected_idea")
        if not isinstance(selected, dict):
            return False
        for key in keys:
            if selected.get(key):
                return True
        for list_key in ("shots", "shot_plan", "shot_list", "beats", "storyboard"):
            entries = selected.get(list_key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for key in keys:
                    if entry.get(key):
                        return True
        return False

    def _record_attempt(
        self,
        manifest: dict[str, Any],
        shot_id: str,
        attempts: int,
    ) -> None:
        attempts_map = manifest.get("attempts")
        if not isinstance(attempts_map, dict):
            attempts_map = {}
            manifest["attempts"] = attempts_map
        attempts_map[shot_id] = int(attempts)

    def _record_shot_failure(
        self,
        manifest: dict[str, Any],
        shot: dict[str, Any],
        shot_id: str,
        kind: str,
        failure: _ShotFailure,
    ) -> None:
        assets = self._ensure_assets(shot)
        assets[f"{kind}_status"] = "failed"
        assets[f"{kind}_error"] = str(failure)
        shot["status"] = "failed"
        attempts = int(failure.attempts)
        self._record_attempt(manifest, shot_id, attempts)
        message = f"{kind} generation failed for shot {shot_id!r}: {failure}"
        self._record_error(message)
        self._store_result(
            ShotRenderResult(
                shot_id=shot_id,
                status="failed",
                keyframe_ref=assets.get("keyframe_ref"),
                video_ref=assets.get("video_ref"),
                attempts=attempts,
                error=str(failure),
            )
        )
        self._persist_manifest(manifest)

    def _store_result(self, result: ShotRenderResult) -> None:
        for index, existing in enumerate(self._results):
            if existing.shot_id == result.shot_id:
                self._results[index] = result
                return
        self._results.append(result)

    def _seed_results(self, manifest: dict[str, Any]) -> None:
        self._results = self._results_from_manifest(manifest)

    def _results_from_manifest(
        self, manifest: dict[str, Any]
    ) -> list[ShotRenderResult]:
        shots = manifest.get("shots")
        if not isinstance(shots, list):
            return []
        attempts_map = manifest.get("attempts")
        if not isinstance(attempts_map, dict):
            attempts_map = {}
        results: list[ShotRenderResult] = []
        for index, shot in enumerate(shots):
            if not isinstance(shot, dict):
                continue
            shot_id = self._shot_id(shot, index)
            assets = shot.get("assets")
            if not isinstance(assets, dict):
                assets = {}
            raw_attempts = attempts_map.get(shot_id, 0)
            try:
                attempts = int(raw_attempts)
            except (TypeError, ValueError):
                attempts = 0
            results.append(
                ShotRenderResult(
                    shot_id=shot_id,
                    status=str(shot.get("status") or "planned"),
                    keyframe_ref=assets.get("keyframe_ref"),
                    video_ref=assets.get("video_ref"),
                    attempts=attempts,
                )
            )
        return results

    def _results_from_payload(self, payload: object) -> list[ShotRenderResult]:
        if not isinstance(payload, list):
            return []
        results: list[ShotRenderResult] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            raw_attempts = item.get("attempts", 0)
            try:
                attempts = int(raw_attempts)
            except (TypeError, ValueError):
                attempts = 0
            results.append(
                ShotRenderResult(
                    shot_id=str(item.get("shot_id") or ""),
                    status=str(item.get("status") or "unknown"),
                    keyframe_ref=item.get("keyframe_ref"),
                    video_ref=item.get("video_ref"),
                    attempts=attempts,
                    error=item.get("error"),
                )
            )
        return results

    def _write_assembly(self, manifest: dict[str, Any]) -> None:
        shots = self._manifest_shots(manifest)
        ordered: list[dict[str, Any]] = []
        for index, shot in enumerate(shots):
            shot_id = self._shot_id(shot, index)
            assets = shot.get("assets")
            if not isinstance(assets, dict):
                assets = {}
            ref = assets.get("video_ref")
            if assets.get("video_status") != "completed" or not (
                isinstance(ref, str) and ref.strip()
            ):
                raise RenderBlocked(
                    f"shot {shot_id!r} has no completed video reference"
                )
            ordered.append(
                {
                    "shot_id": shot_id,
                    "duration_s": shot.get("duration_s"),
                    "video_ref": ref,
                }
            )

        job_dir = self._require_job_dir()
        self._atomic_write_json(
            job_dir / ASSEMBLY_FILENAME,
            {
                "job_id": self._job_id,
                "shots": ordered,
                "created_at": time.time(),
            },
        )

    def _record_error(self, message: str) -> None:
        if message and message not in self._errors:
            self._errors.append(message)

    def _require_job_dir(self) -> Path:
        if self._job_dir is None:
            raise RenderStateError("renderer has no active job directory")
        return self._job_dir
