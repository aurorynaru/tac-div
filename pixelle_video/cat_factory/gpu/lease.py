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

"""Windows-capable, fail-closed GPU lease manager for the Cat Factory.

The manager serializes exclusive access to a single physical GPU id by taking
an OS-level exclusive lock on a per-GPU lock file (``msvcrt.locking`` on
Windows, ``fcntl.flock`` on POSIX, chosen at runtime) and persisting a durable
companion lease record.

Design invariants:

* Fail closed. If exclusivity cannot be established, ``acquire`` raises
  :class:`GpuLeaseUnavailable` and never touches another owner's lock/record.
* Atomic records. The lease record is written to a temp file in the same
  directory and published with :func:`os.replace` (never ``os.rename``).
* Ownership checked on every mutation. ``heartbeat`` and ``release`` verify the
  on-disk ``owner_pid`` and ``lock_generation`` before acting.
* Evidence, never force. Stale handling is report-only via
  :meth:`GpuLeaseManager.classify_stale`; any recovery is explicit and
  hard-disabled unless the caller passes freshly observed evidence.
* No subprocesses, no process spawning, no GPU runtime imports, and no process
  termination. Liveness probes are strictly non-destructive.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if os.name == "nt":  # pragma: no cover - platform selection
    import msvcrt
else:  # pragma: no cover - platform selection
    import fcntl


# --- Errors -----------------------------------------------------------------


class GpuLeaseError(Exception):
    """Base class for all GPU lease failures."""


class GpuLeaseUnavailable(GpuLeaseError):
    """Raised when OS-level exclusivity for the GPU cannot be established."""


class GpuLeaseNotHeld(GpuLeaseError):
    """Raised when an operation requires a lease this process does not hold."""


class GpuLeaseLost(GpuLeaseError):
    """Raised when on-disk ownership no longer matches the held lease."""


# --- Time helpers -----------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _utc_from_epoch(epoch_seconds: float) -> str:
    """Return an ISO-8601 UTC string for a POSIX epoch timestamp."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp, returning UTC-aware datetime or ``None``."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# --- Records ----------------------------------------------------------------


@dataclass
class LeaseRecord:
    """Durable on-disk lease record for one physical GPU."""

    job_id: str
    owner_pid: int
    start_time: str
    heartbeat: str
    phase: str
    lock_generation: int

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serializable representation of this record."""
        return {
            "job_id": self.job_id,
            "owner_pid": int(self.owner_pid),
            "start_time": self.start_time,
            "heartbeat": self.heartbeat,
            "phase": self.phase,
            "lock_generation": int(self.lock_generation),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LeaseRecord":
        """Build a record from a decoded JSON mapping, validating field types."""
        try:
            return cls(
                job_id=str(payload["job_id"]),
                owner_pid=int(payload["owner_pid"]),  # type: ignore[arg-type]
                start_time=str(payload["start_time"]),
                heartbeat=str(payload["heartbeat"]),
                phase=str(payload["phase"]),
                lock_generation=int(payload["lock_generation"]),  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GpuLeaseError(f"malformed lease record: {exc}") from exc


@dataclass(frozen=True)
class GpuLease:
    """Immutable snapshot of a successfully acquired lease handle."""

    gpu_id: "int | str"
    job_id: str
    owner_pid: int
    lock_generation: int
    lock_path: str
    record_path: str
    start_time: str


@dataclass(frozen=True)
class StaleAssessment:
    """Read-only evidence gathered about a GPU lease. Never mutates state."""

    gpu_id: "int | str"
    lock_path: str
    record_path: str
    lock_file_exists: bool
    lock_mtime_iso: str | None
    record: LeaseRecord | None
    record_corrupt: bool
    heartbeat_age_seconds: float | None
    stale_heartbeat_seconds: int
    heartbeat_stale: bool
    owner_pid: int | None
    owner_alive: bool | None
    lock_held_elsewhere: bool
    classification: str


# --- Atomic persistence -----------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish ``payload`` as JSON at ``path`` (temp + os.replace)."""
    handle = None
    tmp_name: str | None = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
        )
        handle = os.fdopen(tmp_fd, "w", encoding="utf-8")
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(tmp_name, str(path))
        tmp_name = None
    finally:
        if handle is not None:
            handle.close()
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


# --- Liveness probe (non-destructive) --------------------------------------


def _pid_alive(pid: int | None) -> bool | None:
    """Return whether ``pid`` is alive, or ``None`` when indeterminate.

    The probe is strictly non-destructive: POSIX uses ``os.kill(pid, 0)`` (never
    a terminating signal) and Windows uses ``OpenProcess`` with query-only
    access followed by ``GetExitCodeProcess``. No process is ever signalled or
    terminated.
    """
    if pid is None or pid <= 0:
        return None
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _pid_alive_windows(pid: int) -> bool | None:
    """Windows liveness probe via query-only ``OpenProcess`` (no termination)."""
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.restype = wintypes.HANDLE
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    close_handle = kernel32.CloseHandle
    close_handle.restype = wintypes.BOOL
    close_handle.argtypes = [wintypes.HANDLE]
    get_exit_code_process = kernel32.GetExitCodeProcess
    get_exit_code_process.restype = wintypes.BOOL
    get_exit_code_process.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]

    handle = open_process(process_query_limited_information, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        if error == error_access_denied:
            return True
        return False
    try:
        exit_code = wintypes.DWORD()
        if get_exit_code_process(handle, ctypes.byref(exit_code)):
            return exit_code.value == still_active
        return None
    finally:
        close_handle(handle)


# --- OS-level exclusive lock ------------------------------------------------


class _OsExclusiveLock:
    """Thin, non-blocking OS-level exclusive lock over a single lock file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> None:
        """Take an exclusive non-blocking lock or raise ``GpuLeaseUnavailable``."""
        fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self._lock_fd(fd)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    @staticmethod
    def _lock_fd(fd: int) -> None:
        if os.name == "nt":
            # msvcrt.locking requires a byte range within the file to exist.
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\x00")
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise GpuLeaseUnavailable(
                    "GPU lock is held by another owner (msvcrt.locking failed)"
                ) from exc
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise GpuLeaseUnavailable(
                    "GPU lock is held by another owner (fcntl.flock failed)"
                ) from exc

    def release(self) -> None:
        """Release the lock (if held) and close the underlying descriptor."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            if os.name == "nt":
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(fd)

    def held(self) -> bool:
        return self._fd is not None


# --- Manager ----------------------------------------------------------------


def _slugify_gpu_id(gpu_id: int | str) -> str:
    """Return a filesystem-safe slug for a physical GPU id."""
    raw = str(gpu_id).strip()
    if not raw:
        raise ValueError("gpu_id must be a non-empty int or str")
    slug = "".join(
        char if (char.isalnum() or char in "-_.") else "_" for char in raw
    )
    return slug or "unknown"


def _read_record(path: Path) -> LeaseRecord | None:
    """Read and validate a lease record; return ``None`` only when absent."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise GpuLeaseError(f"cannot read lease record {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GpuLeaseError(f"lease record {path} is corrupt: {exc}") from exc
    if not isinstance(payload, dict):
        raise GpuLeaseError(f"lease record {path} is not a JSON object")
    return LeaseRecord.from_dict(payload)


def _classify(
    record: LeaseRecord | None,
    record_corrupt: bool,
    owner_alive: bool | None,
    lock_free: bool | None,
    heartbeat_stale: bool,
) -> str:
    """Derive a conservative classification from observed evidence."""
    if record_corrupt:
        return "indeterminate"
    if record is None:
        if lock_free is False:
            return "indeterminate"
        return "vacant"
    if owner_alive is True:
        return "owned-by-live-process"
    if lock_free is False:
        return "indeterminate"
    if owner_alive is False and heartbeat_stale:
        return "stale-evidence"
    return "indeterminate"


class GpuLeaseManager:
    """Fail-closed, single-GPU lease manager backed by an OS-level file lock."""

    def __init__(
        self,
        gpu_id: int | str,
        lock_root: Path,
        stale_heartbeat_seconds: int = 300,
    ) -> None:
        if stale_heartbeat_seconds <= 0:
            raise ValueError("stale_heartbeat_seconds must be a positive integer")
        self.gpu_id: int | str = gpu_id
        self.lock_root = Path(lock_root)
        self.stale_heartbeat_seconds = int(stale_heartbeat_seconds)
        self._slug = _slugify_gpu_id(gpu_id)
        self.lock_path = self.lock_root / f"gpu-{self._slug}.lock"
        self.record_path = self.lock_root / f"gpu-{self._slug}.lease.json"
        self._lock: _OsExclusiveLock | None = None
        self._handle: GpuLease | None = None

    # -- introspection -------------------------------------------------------

    def held(self) -> bool:
        """Return whether this manager currently holds a lease."""
        return self._handle is not None and self._lock is not None

    def read_lease(self) -> LeaseRecord | None:
        """Return the on-disk lease record, or ``None`` when none exists."""
        return _read_record(self.record_path)

    def inspect(self) -> LeaseRecord | None:
        """Read-only alias of :meth:`read_lease` returning the on-disk record."""
        return self.read_lease()

    # -- lifecycle -----------------------------------------------------------

    def acquire(self, job_id: str, phase: str) -> GpuLease:
        """Acquire the exclusive GPU lease, failing closed on contention."""
        if not job_id:
            raise ValueError("job_id must be a non-empty string")
        if not phase:
            raise ValueError("phase must be a non-empty string")
        if self.held():
            raise GpuLeaseError("this manager already holds a GPU lease")

        self.lock_root.mkdir(parents=True, exist_ok=True)
        lock = _OsExclusiveLock(self.lock_path)
        lock.acquire()

        try:
            generation = self._next_generation()
            now = _utc_now_iso()
            record = LeaseRecord(
                job_id=job_id,
                owner_pid=os.getpid(),
                start_time=now,
                heartbeat=now,
                phase=phase,
                lock_generation=generation,
            )
            _atomic_write_json(self.record_path, record.to_dict())
        except BaseException:
            lock.release()
            raise

        handle = GpuLease(
            gpu_id=self.gpu_id,
            job_id=job_id,
            owner_pid=os.getpid(),
            lock_generation=generation,
            lock_path=str(self.lock_path),
            record_path=str(self.record_path),
            start_time=now,
        )
        self._lock = lock
        self._handle = handle
        return handle

    def heartbeat(self, phase: str) -> LeaseRecord:
        """Update heartbeat + phase for the lease this owner holds."""
        if not phase:
            raise ValueError("phase must be a non-empty string")
        current = self._require_owned_record()
        current.heartbeat = _utc_now_iso()
        current.phase = phase
        _atomic_write_json(self.record_path, current.to_dict())
        return current

    def release(self) -> bool:
        """Release only the lease this owner holds; idempotent."""
        if self._lock is None and self._handle is None:
            return True

        handle = self._handle
        owned = self._owns_exclusivity(handle) if handle is not None else False

        if self._lock is not None:
            self._lock.release()
        self._lock = None
        self._handle = None

        if not owned:
            raise GpuLeaseLost(
                "on-disk lease no longer matches this owner; os-level lock "
                "released without touching the foreign record"
            )
        return True

    # -- stale evidence ------------------------------------------------------

    def classify_stale(self) -> StaleAssessment:
        """Report observed stale evidence without ever mutating state."""
        lock_file_exists = self.lock_path.exists()
        lock_mtime_iso: str | None = None
        if lock_file_exists:
            try:
                lock_mtime_iso = _utc_from_epoch(self.lock_path.stat().st_mtime)
            except OSError:
                lock_mtime_iso = None

        record_corrupt = False
        record: LeaseRecord | None = None
        try:
            record = self.read_lease()
        except GpuLeaseError:
            record_corrupt = True

        heartbeat_age_seconds: float | None = None
        heartbeat_stale = False
        owner_pid = record.owner_pid if record is not None else None
        if record is not None:
            heartbeat_dt = _parse_iso(record.heartbeat)
            if heartbeat_dt is not None:
                age = (datetime.now(timezone.utc) - heartbeat_dt).total_seconds()
                heartbeat_age_seconds = max(0.0, age)
                heartbeat_stale = heartbeat_age_seconds > self.stale_heartbeat_seconds

        owner_alive = _pid_alive(owner_pid) if owner_pid else None
        lock_free = self._probe_lock_free()
        lock_held_elsewhere = lock_free is False

        classification = _classify(
            record=record,
            record_corrupt=record_corrupt,
            owner_alive=owner_alive,
            lock_free=lock_free,
            heartbeat_stale=heartbeat_stale,
        )

        return StaleAssessment(
            gpu_id=self.gpu_id,
            lock_path=str(self.lock_path),
            record_path=str(self.record_path),
            lock_file_exists=lock_file_exists,
            lock_mtime_iso=lock_mtime_iso,
            record=record,
            record_corrupt=record_corrupt,
            heartbeat_age_seconds=heartbeat_age_seconds,
            stale_heartbeat_seconds=self.stale_heartbeat_seconds,
            heartbeat_stale=heartbeat_stale,
            owner_pid=owner_pid,
            owner_alive=owner_alive,
            lock_held_elsewhere=lock_held_elsewhere,
            classification=classification,
        )

    def recover_stale(self, evidence: StaleAssessment) -> bool:
        """Explicitly clear a stale *record*, gated on freshly observed evidence.

        Hard-disabled unless the caller passes a :class:`StaleAssessment`. The
        evidence is re-observed at call time and recovery proceeds only when it
        classifies as ``stale-evidence`` (owner not alive, heartbeat stale, lock
        not held elsewhere). The lock file is intentionally preserved; this
        never deletes a lock whose owner is live. Returns ``True`` when a stale
        record was removed, ``False`` when there was nothing to remove.
        """
        if not isinstance(evidence, StaleAssessment):
            raise GpuLeaseError(
                "recover_stale requires a StaleAssessment from classify_stale()"
            )

        fresh = self.classify_stale()
        if fresh.owner_alive is True:
            raise GpuLeaseError(
                f"refusing stale recovery: owner pid {fresh.owner_pid} is alive"
            )
        if fresh.lock_held_elsewhere:
            raise GpuLeaseError(
                "refusing stale recovery: the OS-level lock is currently held"
            )
        if fresh.record is None:
            raise GpuLeaseError("refusing stale recovery: no lease record present")
        if not fresh.heartbeat_stale:
            raise GpuLeaseError("refusing stale recovery: heartbeat is still fresh")
        if fresh.classification != "stale-evidence":
            raise GpuLeaseError(
                f"refusing stale recovery: classification={fresh.classification!r}"
            )

        try:
            os.unlink(str(self.record_path))
        except FileNotFoundError:
            return False
        return True

    # -- internals -----------------------------------------------------------

    def _next_generation(self) -> int:
        """Return the next lock generation, based on the prior on-disk record."""
        try:
            previous = self.read_lease()
        except GpuLeaseError:
            previous = None
        if previous is None:
            return 1
        return int(previous.lock_generation) + 1

    def _require_owned_record(self) -> LeaseRecord:
        """Return the on-disk record only if it is owned by this manager."""
        if self._handle is None:
            raise GpuLeaseNotHeld("no GPU lease is held by this manager")
        record = _read_record(self.record_path)
        if (
            record is None
            or record.owner_pid != os.getpid()
            or record.lock_generation != self._handle.lock_generation
        ):
            raise GpuLeaseLost(
                "lease ownership lost: on-disk record no longer matches this owner"
            )
        return record

    def _owns_exclusivity(self, handle: GpuLease | None) -> bool:
        """Return whether ``handle`` still matches the on-disk record."""
        if handle is None:
            return False
        try:
            record = _read_record(self.record_path)
        except GpuLeaseError:
            return False
        return bool(
            record is not None
            and record.owner_pid == os.getpid()
            and record.lock_generation == handle.lock_generation
        )

    def _probe_lock_free(self) -> bool | None:
        """Non-destructively test whether the lock file is currently lockable."""
        if not self.lock_path.exists():
            return True
        try:
            fd = os.open(str(self.lock_path), os.O_RDWR)
        except OSError:
            return None
        try:
            if os.name == "nt":
                if os.fstat(fd).st_size == 0:
                    return True
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError:
                    return False
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                return True
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        finally:
            os.close(fd)
