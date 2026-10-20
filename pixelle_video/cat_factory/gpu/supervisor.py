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

"""Fail-closed ComfyUI process supervisor for the Cat Factory GPU layer.

This module owns the *process* lifecycle that :mod:`pixelle_video.cat_factory.gpu.lease`
deliberately does not touch. It exposes a single :class:`ComfySupervisor` with two
operating modes:

* ``managed`` - launch ComfyUI as a direct child via
  :class:`subprocess.Popen` using an ARGUMENT ARRAY (never a shell string, never
  interpolated user text), route the child's stdout/stderr into per-job log
  files, and terminate only the child this supervisor positively owns.
* ``attached`` - dial an already-running ComfyUI server that an operator has
  explicitly authorized for exclusive use. Attached mode never spawns and never
  force-kills; :meth:`ComfySupervisor.stop` only detaches.

Design invariants:

* Fail closed. An empty ``comfy_command`` list, a missing ownership record, or an
  indeterminate process-identity token stops the operation with a
  :class:`ComfySupervisorError` subclass instead of guessing.
* Identity-checked termination. A process identity token (Windows process
  creation time, POSIX ``/proc`` start time) is captured at launch and
  re-verified immediately before any termination, so a reused PID is never
  signalled.
* Bounded, verified shutdown. ``stop`` sends a graceful ``terminate()``, waits a
  bounded interval, and only escalates to ``kill()`` after re-verifying the
  owned child's identity. Termination is never *claimed* without observing the
  child's ``returncode``.
* No global kills. Termination is restricted to the recorded owned PID and its
  positively verified descendants; it never targets processes by image name and
  never signals an unrelated process.
* No GPU runtime, no ComfyUI HTTP calls, no ``torch`` import. Only process
  management primitives from the standard library are used.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from pixelle_video.config.schema import CatFactoryGpuConfig


# --- Errors -----------------------------------------------------------------


class ComfySupervisorError(Exception):
    """Base class for all ComfyUI supervisor failures."""


class ComfySupervisorNotAuthorized(ComfySupervisorError):
    """Raised when an attached-mode operation lacks operator authorization."""


class ComfySupervisorUnavailable(ComfySupervisorError):
    """Raised when the supervisor cannot launch or confirm termination."""


class ComfySupervisorIndeterminate(ComfySupervisorError):
    """Raised when process ownership or identity cannot be positively verified."""


# Module logger. Only sanitized launch metadata (command array, pid, log paths)
# is ever logged here; environment variables and credentials never are.
logger = logging.getLogger(__name__)


# Default bounded startup-readiness window, in seconds.
DEFAULT_STARTUP_TIMEOUT_S = 30.0


# --- Records ----------------------------------------------------------------

SupervisorMode = Literal["managed", "attached"]


@dataclass(frozen=True)
class ManagedChildRecord:
    """Immutable snapshot of the owned managed ComfyUI child."""

    pid: int
    identity: str
    command: tuple[str, ...]
    log_stdout: str
    log_stderr: str
    started_at: float
    config: dict[str, object]


@dataclass(frozen=True)
class AttachRecord:
    """Immutable snapshot of an attached (externally managed) ComfyUI endpoint."""

    endpoint: str
    attached_at: float


# --- Config resolution ------------------------------------------------------


def _resolve_gpu_config() -> "CatFactoryGpuConfig":
    """Resolve the live ``cat_factory.gpu`` config section at call time.

    Imported lazily so this module carries no import-time dependency on the
    configuration package, and never hardcodes a machine-specific path.
    """
    from pixelle_video.config import config_manager

    return config_manager.config.cat_factory.gpu


def _slugify(value: str) -> str:
    """Return a filesystem-safe slug for a job identifier."""
    raw = value.strip()
    slug = "".join(
        char if (char.isalnum() or char in "-_.") else "_" for char in raw
    )
    if slug in (".", ".."):
        return "job"
    return slug or "job"


def _validate_project_relative_dir(value: str | Path, *, source: str) -> Path:
    """Resolve ``value`` as a project-relative directory, failing closed.

    Rejects empty values, absolute paths (POSIX, driveless-rooted Windows, and
    fully qualified Windows paths), drive-relative Windows paths, and any
    ``..`` segment. The joined target is resolved and must remain inside the
    current working directory.
    """
    raw = str(value)
    if not raw.strip():
        raise ValueError(f"{source} must be a non-empty path, got {raw!r}")

    windows_path = PureWindowsPath(raw)
    if windows_path.root:
        raise ValueError(f"{source} must be project-relative but is rooted: {raw!r}")
    if windows_path.drive:
        raise ValueError(
            f"{source} must be project-relative but is drive-relative: {raw!r}"
        )

    posix_path = PurePosixPath(raw)
    if posix_path.is_absolute() or Path(raw).is_absolute():
        raise ValueError(f"{source} must be project-relative but is absolute: {raw!r}")
    if ".." in windows_path.parts or ".." in posix_path.parts:
        raise ValueError(f"{source} must not contain '..' segments: {raw!r}")

    cwd = Path.cwd()
    try:
        resolved_cwd = cwd.resolve()
        resolved_target = (cwd / raw).resolve()
    except OSError as exc:
        raise ValueError(
            f"cannot resolve {source} {raw!r} against {str(cwd)!r}: {exc}"
        ) from exc
    if resolved_target != resolved_cwd and resolved_cwd not in resolved_target.parents:
        raise ValueError(
            f"{source} {raw!r} escapes the project directory "
            f"{str(resolved_cwd)!r}"
        )
    return resolved_target


# --- Process identity -------------------------------------------------------


def _process_identity(pid: int) -> str | None:
    """Return a stable identity token for ``pid``, or ``None`` if indeterminate.

    The token is the process creation time: it changes when a PID is reused, so
    a mismatch proves the original process is gone. ``None`` means identity
    cannot be established and callers must fail closed.
    """
    if pid <= 0:
        return None
    if os.name == "nt":
        return _windows_creation_token(pid)
    return _posix_creation_token(pid)


def _windows_creation_token(pid: int) -> str | None:
    """Windows process creation time via query-only ``GetProcessTimes``."""
    import ctypes
    from ctypes import wintypes

    class _FileTime(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.restype = wintypes.HANDLE
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    get_process_times = kernel32.GetProcessTimes
    get_process_times.restype = wintypes.BOOL
    get_process_times.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    ]
    close_handle = kernel32.CloseHandle
    close_handle.restype = wintypes.BOOL
    close_handle.argtypes = [wintypes.HANDLE]

    handle = open_process(process_query_limited_information, False, int(pid))
    if not handle:
        return None
    try:
        creation = _FileTime()
        exit_time = _FileTime()
        kernel_time = _FileTime()
        user_time = _FileTime()
        if not get_process_times(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None
        token = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
        return str(token)
    finally:
        close_handle(handle)


def _posix_creation_token(pid: int) -> str | None:
    """POSIX (Linux) process start time from ``/proc/<pid>/stat`` field 22."""
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        content = stat_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    close_paren = content.rfind(")")
    if close_paren == -1:
        return None
    fields = content[close_paren + 2 :].split()
    # Field 3 is ``state``; ``starttime`` is field 22 -> index 19 after ")".
    if len(fields) < 20:
        return None
    return fields[19]


def _windows_terminate_process(pid: int) -> bool:
    """Terminate exactly ``pid`` via ``TerminateProcess``; never by image name."""
    import ctypes
    from ctypes import wintypes

    process_terminate = 0x0001
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.restype = wintypes.HANDLE
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    terminate_process = kernel32.TerminateProcess
    terminate_process.restype = wintypes.BOOL
    terminate_process.argtypes = [wintypes.HANDLE, wintypes.UINT]
    close_handle = kernel32.CloseHandle
    close_handle.restype = wintypes.BOOL
    close_handle.argtypes = [wintypes.HANDLE]

    handle = open_process(process_terminate, False, int(pid))
    if not handle:
        return False
    try:
        return bool(terminate_process(handle, 1))
    finally:
        close_handle(handle)


def _windows_process_parent_map() -> dict[int, int]:
    """Return a ``{pid: parent_pid}`` map from a Toolhelp32 process snapshot."""
    import ctypes
    from ctypes import wintypes

    class _ProcessEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    th32cs_snapprocess = 0x00000002
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.restype = wintypes.HANDLE
    create_snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    process32_first = kernel32.Process32First
    process32_first.restype = wintypes.BOOL
    process32_first.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32)]
    process32_next = kernel32.Process32Next
    process32_next.restype = wintypes.BOOL
    process32_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32)]
    close_handle = kernel32.CloseHandle
    close_handle.restype = wintypes.BOOL
    close_handle.argtypes = [wintypes.HANDLE]

    snapshot = create_snapshot(th32cs_snapprocess, 0)
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return {}
    try:
        parent_of: dict[int, int] = {}
        entry = _ProcessEntry32()
        entry.dwSize = ctypes.sizeof(_ProcessEntry32)
        has_entry = process32_first(snapshot, ctypes.byref(entry))
        while has_entry:
            parent_of[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            has_entry = process32_next(snapshot, ctypes.byref(entry))
        return parent_of
    finally:
        close_handle(snapshot)


def _windows_owned_descendants(root_pid: int) -> list[tuple[int, str | None]]:
    """Return ``(pid, identity)`` pairs reachable from ``root_pid``.

    Only processes whose recorded parent chain leads back to ``root_pid`` are
    returned, so termination is bounded to the verified owned tree. Each
    descendant's identity token is captured here and re-verified immediately
    before it is signalled.
    """
    parent_of = _windows_process_parent_map()
    children_of: dict[int, list[int]] = {}
    for pid, parent_pid in parent_of.items():
        children_of.setdefault(parent_pid, []).append(pid)
    ordered: list[tuple[int, str | None]] = []
    queue = list(children_of.get(root_pid, []))
    seen = set(queue)
    while queue:
        pid = queue.pop(0)
        ordered.append((pid, _process_identity(pid)))
        for child in children_of.get(pid, []):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return ordered


def _best_effort_terminate(proc: "subprocess.Popen[bytes]") -> None:
    """Best-effort kill of a freshly spawned child whose identity is unknown."""
    try:
        proc.kill()
    except OSError:
        return


# --- Supervisor -------------------------------------------------------------


class ComfySupervisor:
    """Owns the ComfyUI process lifecycle for a single Cat Factory job.

    The operating mode defaults to ``cat_factory.gpu.supervisor_mode`` and can
    be overridden explicitly for tests or advanced callers.
    """

    def __init__(
        self,
        job_id: str,
        mode: SupervisorMode | None = None,
        *,
        state_dir: str | Path | None = None,
        gpu_config: "CatFactoryGpuConfig | None" = None,
        terminate_timeout_s: float = 10.0,
        startup_probe: Callable[[], bool] | None = None,
        startup_timeout_s: float | None = None,
    ) -> None:
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        self._job_id: str = job_id.strip()
        self._slug: str = _slugify(self._job_id)

        self._config: CatFactoryGpuConfig = (
            gpu_config if gpu_config is not None else _resolve_gpu_config()
        )

        resolved_mode = mode if mode is not None else self._config.supervisor_mode
        if resolved_mode not in ("managed", "attached"):
            raise ValueError(f"unsupported supervisor mode: {resolved_mode!r}")
        self._mode: SupervisorMode = resolved_mode

        if terminate_timeout_s <= 0:
            raise ValueError("terminate_timeout_s must be positive")
        self._terminate_timeout_s: float = float(terminate_timeout_s)
        self._poll_interval_s: float = float(self._config.poll_interval_s)
        self._startup_probe: Callable[[], bool] | None = startup_probe
        if startup_timeout_s is not None and startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        self._startup_timeout_s: float | None = (
            float(startup_timeout_s) if startup_timeout_s is not None else None
        )
        self._state_dir: Path = self._resolve_state_dir(state_dir)

        self._child: subprocess.Popen[bytes] | None = None
        self._child_identity: str | None = None
        self._record: ManagedChildRecord | None = None
        self._stdout_handle: IO[bytes] | None = None
        self._stderr_handle: IO[bytes] | None = None
        self._attach_record: AttachRecord | None = None

    # -- introspection -------------------------------------------------------

    @property
    def job_id(self) -> str:
        """Return the job identifier this supervisor is bound to."""
        return self._job_id

    @property
    def mode(self) -> SupervisorMode:
        """Return the resolved operating mode (``managed`` or ``attached``)."""
        return self._mode

    @property
    def resolved_config(self) -> "CatFactoryGpuConfig":
        """Return the resolved ``cat_factory.gpu`` config used by this instance."""
        return self._config

    @property
    def state_dir(self) -> Path:
        """Return the per-job state directory (logs and supervisor records)."""
        return self._state_dir

    @property
    def owned_pid(self) -> int | None:
        """Return the owned child PID, or ``None`` when nothing is owned."""
        return self._child.pid if self._child is not None else None

    @property
    def launch_record(self) -> ManagedChildRecord | None:
        """Return the latest managed-child record, or ``None`` if never launched."""
        return self._record

    @property
    def is_attached(self) -> bool:
        """Return whether an attached endpoint has been recorded."""
        return self._attach_record is not None

    @property
    def attached_endpoint(self) -> str | None:
        """Return the attached ComfyUI endpoint, or ``None`` when detached."""
        return self._attach_record.endpoint if self._attach_record is not None else None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> ManagedChildRecord:
        """Launch the managed ComfyUI child and return its record.

        Fails closed when ``comfy_command`` is empty (not configured). The child
        is spawned as an argument array with ``shell=False``; its stdout/stderr
        are redirected into per-job log files and never inherited by the parent.
        """
        self._require_managed()
        if self._owned_child_is_live():
            raise ComfySupervisorError(
                "a managed ComfyUI child is already running; call stop() before start()"
            )
        if self._child is not None:
            self._release_owned_child()
        record = self._spawn(tuple(self._config.comfy_command))
        self._await_startup(record)
        return record

    def relaunch(self) -> ManagedChildRecord:
        """Restart the managed child, reusing the recorded command and config.

        Refuses when a live owned child still exists; the caller must stop it
        first so a controlled restart never leaves two ComfyUI servers running.
        """
        self._require_managed()
        if self._owned_child_is_live():
            raise ComfySupervisorError(
                "refusing to relaunch: a live owned ComfyUI child is still running"
            )
        command = (
            self._record.command
            if self._record is not None
            else tuple(self._config.comfy_command)
        )
        if self._child is not None:
            self._release_owned_child()
        record = self._spawn(command)
        self._await_startup(record)
        return record

    def stop(self, timeout_s: float | None = None) -> bool:
        """Stop the owned child (managed) or detach (attached).

        In ``managed`` mode this sends a graceful ``terminate()``, waits a
        bounded interval, verifies the child's identity again, and escalates to
        ``kill()`` only when required. Returns ``True`` only when the child's
        exit was positively observed. In ``attached`` mode it detaches the
        external endpoint and never signals any process, returning ``False``.
        """
        if self._mode == "attached":
            self._attach_record = None
            return False
        return self._stop_managed(timeout_s)

    def attach(
        self,
        endpoint: str,
        *,
        operator_authorized_exclusive: bool = False,
    ) -> AttachRecord:
        """Record an operator-authorized attachment to a running ComfyUI server.

        The explicit ``operator_authorized_exclusive=True`` grant is mandatory;
        without it this raises :class:`ComfySupervisorNotAuthorized`. Attached
        mode never spawns a process and never force-kills one.
        """
        self._require_attached()
        if not operator_authorized_exclusive:
            raise ComfySupervisorNotAuthorized(
                "attached mode requires an explicit "
                "operator_authorized_exclusive=True grant; refusing to attach"
            )
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a non-empty string")
        record = AttachRecord(endpoint=endpoint.strip(), attached_at=time.time())
        self._attach_record = record
        return record

    # -- internals: spawn / stop --------------------------------------------

    def _spawn(self, command: tuple[str, ...]) -> ManagedChildRecord:
        """Launch ``command`` as an owned child and capture its identity token."""
        if not command:
            raise ComfySupervisorUnavailable(
                "comfy_command is not configured (empty); refusing to guess a "
                "ComfyUI executable"
            )

        log_dir = self._state_dir / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ComfySupervisorUnavailable(
                f"cannot create supervisor state directory {log_dir}: {exc}"
            ) from exc

        stdout_path = log_dir / f"comfy.{self._slug}.stdout.log"
        stderr_path = log_dir / f"comfy.{self._slug}.stderr.log"

        stdout_handle: IO[bytes] | None = None
        stderr_handle: IO[bytes] | None = None
        logger.info("launching ComfyUI: %s", list(command))
        try:
            stdout_handle = open(stdout_path, "ab")
            stderr_handle = open(stderr_path, "ab")
            proc = subprocess.Popen(
                list(command),
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        except OSError as exc:
            logger.error("failed to launch ComfyUI (%r): %s", command[0], exc)
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()
            raise ComfySupervisorUnavailable(
                f"failed to launch ComfyUI ({command[0]!r}): {exc}"
            ) from exc

        identity = _process_identity(proc.pid)
        if identity is None:
            _best_effort_terminate(proc)
            stdout_handle.close()
            stderr_handle.close()
            raise ComfySupervisorIndeterminate(
                f"launched pid {proc.pid} but could not establish a process "
                "identity token; failing closed"
            )

        record = ManagedChildRecord(
            pid=proc.pid,
            identity=identity,
            command=tuple(command),
            log_stdout=str(stdout_path),
            log_stderr=str(stderr_path),
            started_at=time.time(),
            config=self._config_snapshot(),
        )
        self._child = proc
        self._child_identity = identity
        self._stdout_handle = stdout_handle
        self._stderr_handle = stderr_handle
        self._record = record
        return record

    def _stop_managed(self, timeout_s: float | None) -> bool:
        """Terminate the owned child with bounded, identity-verified escalation."""
        proc = self._child
        if proc is None:
            self._close_logs()
            return False

        effective_timeout = (
            self._terminate_timeout_s if timeout_s is None else float(timeout_s)
        )
        if effective_timeout <= 0:
            raise ValueError("timeout_s must be positive")

        if proc.poll() is None:
            self._require_owned_identity()
            try:
                proc.terminate()
            except OSError as exc:
                raise ComfySupervisorUnavailable(
                    f"failed to terminate ComfyUI child pid {proc.pid}: {exc}"
                ) from exc
            self._wait_for_exit(proc, effective_timeout)
            if proc.poll() is None:
                self._require_owned_identity()
                self._escalate_kill(proc)
                self._wait_for_exit(proc, effective_timeout)

        if proc.poll() is None:
            raise ComfySupervisorUnavailable(
                f"ComfyUI child pid {proc.pid} did not exit after termination "
                "and kill"
            )

        self._release_owned_child()
        return True

    def _escalate_kill(self, proc: "subprocess.Popen[bytes]") -> None:
        """Force-terminate the identity-verified child.

        On Windows this also terminates the verified owned descendant tree. On
        POSIX only the direct child is signalled; descendants are not tracked.
        """
        if os.name == "nt":
            self._terminate_windows_owned_tree(proc.pid)
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                return

    def _terminate_windows_owned_tree(self, root_pid: int) -> None:
        """Terminate only the verified owned tree on Windows (root + descendants)."""
        self._require_owned_identity()
        descendants = _windows_owned_descendants(root_pid)
        for pid, captured_identity in reversed(descendants):
            current = _process_identity(pid)
            if current is None or current != captured_identity:
                continue
            _windows_terminate_process(pid)
        self._require_owned_identity()
        _windows_terminate_process(root_pid)

    def _wait_for_exit(
        self, proc: "subprocess.Popen[bytes]", timeout_s: float
    ) -> bool:
        """Poll ``proc`` for a bounded interval; return whether it has exited."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            if proc.poll() is not None:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return proc.poll() is not None
            time.sleep(min(self._poll_interval_s, remaining))

    # -- internals: ownership / verification --------------------------------

    def _owned_child_is_live(self) -> bool:
        """Return whether the owned child process is still running."""
        return self._child is not None and self._child.poll() is None

    def _require_owned_identity(self) -> None:
        """Verify the owned child's identity, failing closed on any doubt."""
        proc = self._child
        if proc is None or self._child_identity is None:
            raise ComfySupervisorIndeterminate(
                "no owned child identity is recorded; refusing to act"
            )
        current = _process_identity(proc.pid)
        if current is None:
            raise ComfySupervisorIndeterminate(
                f"process identity for pid {proc.pid} is indeterminate; "
                "refusing to act"
            )
        if current != self._child_identity:
            raise ComfySupervisorIndeterminate(
                f"pid {proc.pid} identity mismatch (possible PID reuse); "
                "refusing to signal"
            )

    def _require_managed(self) -> None:
        """Raise unless the supervisor is in ``managed`` mode."""
        if self._mode != "managed":
            raise ComfySupervisorError(
                f"operation requires managed mode, supervisor is in {self._mode!r} mode"
            )

    def _require_attached(self) -> None:
        """Raise unless the supervisor is in ``attached`` mode."""
        if self._mode != "attached":
            raise ComfySupervisorError(
                f"operation requires attached mode, supervisor is in {self._mode!r} mode"
            )

    # -- internals: state helpers -------------------------------------------

    def _resolve_state_dir(self, override: str | Path | None) -> Path:
        """Resolve the per-job state directory, never hardcoding a machine path.

        Both the config-derived value and a caller-supplied override are
        enforced as project-relative and fail closed when they escape the
        current working directory.
        """
        if override is not None:
            root = _validate_project_relative_dir(
                override, source="state_dir override"
            )
        elif self._config.state_dir:
            root = _validate_project_relative_dir(
                self._config.state_dir, source="cat_factory.gpu.state_dir"
            )
        else:
            root = Path.cwd() / "state" / "cat_factory" / "gpu"
        candidate = root / self._slug
        cwd = Path.cwd()
        try:
            resolved_cwd = cwd.resolve()
            resolved_candidate = candidate.resolve()
        except OSError as exc:
            raise ValueError(
                f"cannot resolve composed state directory {str(candidate)!r} "
                f"against {str(cwd)!r}: {exc}"
            ) from exc
        if (
            resolved_candidate != resolved_cwd
            and resolved_cwd not in resolved_candidate.parents
        ):
            raise ValueError(
                f"composed state directory {str(candidate)!r} escapes the "
                f"project directory {str(resolved_cwd)!r}"
            )
        return candidate

    def _config_snapshot(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot of the resolved config, if available."""
        dump = getattr(self._config, "model_dump", None)
        if callable(dump):
            snapshot = dump()
            if isinstance(snapshot, dict):
                return dict(snapshot)
        return {}

    def _sanitized_launch_summary(self) -> dict[str, object]:
        """Return a log-safe launch summary; never includes environment variables."""
        record = self._record
        if record is None:
            return {
                "command": [],
                "pid": None,
                "log_stdout": None,
                "log_stderr": None,
                "state_dir": str(self._state_dir),
            }
        return {
            "command": list(record.command),
            "pid": record.pid,
            "log_stdout": record.log_stdout,
            "log_stderr": record.log_stderr,
            "state_dir": str(self._state_dir),
        }

    def _await_startup(self, record: ManagedChildRecord) -> None:
        """Wait (bounded) for the injected startup probe to confirm readiness.

        With no probe configured this is a no-op, preserving the historical
        behavior. Cleanup on failure uses only the identity-verified owned-child
        path; the logged summary contains no environment variables or secrets.
        """
        probe = self._startup_probe
        if probe is None:
            return
        timeout = (
            self._startup_timeout_s
            if self._startup_timeout_s is not None
            else DEFAULT_STARTUP_TIMEOUT_S
        )
        deadline = time.monotonic() + timeout
        while True:
            if self._child is None or self._child.poll() is not None:
                summary = self._sanitized_launch_summary()
                self._release_owned_child()
                logger.error("ComfyUI exited during startup: %s", summary)
                raise ComfySupervisorUnavailable(
                    f"ComfyUI child exited during startup: {summary}"
                )
            if probe():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                summary = self._sanitized_launch_summary()
                try:
                    self._stop_managed(self._terminate_timeout_s)
                except ComfySupervisorError as exc:
                    logger.error("cleanup after startup timeout failed: %s", exc)
                logger.error(
                    "ComfyUI did not become ready within %ss: %s", timeout, summary
                )
                raise ComfySupervisorUnavailable(
                    f"ComfyUI did not become ready within {timeout}s: {summary}"
                )
            time.sleep(min(self._poll_interval_s, remaining))

    def _close_logs(self) -> None:
        """Close the per-job log handles if they are open."""
        for attribute in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, attribute)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
                setattr(self, attribute, None)

    def _release_owned_child(self) -> None:
        """Forget the owned child after its exit has been observed."""
        self._close_logs()
        self._child = None
        self._child_identity = None
        self._record = None


__all__ = [
    "DEFAULT_STARTUP_TIMEOUT_S",
    "AttachRecord",
    "ComfySupervisor",
    "ComfySupervisorError",
    "ComfySupervisorIndeterminate",
    "ComfySupervisorNotAuthorized",
    "ComfySupervisorUnavailable",
    "ManagedChildRecord",
    "SupervisorMode",
]
