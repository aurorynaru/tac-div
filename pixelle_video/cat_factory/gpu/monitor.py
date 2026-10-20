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

"""Read-only GPU VRAM and system RAM monitoring for local ComfyUI jobs.

VRAM is sampled through ``nvidia-smi`` (Windows and POSIX) and system RAM
through the platform memory API.  The module performs no mutations: it never
unloads models, never calls ComfyUI, and never trusts a ``POST /free``
acknowledgement.  A VRAM release is only accepted once a fresh measurement
shows that the previous allocation is actually gone.

Configuration is resolved lazily from the opt-in ``cat_factory.gpu`` config
section, so importing this module has no side effects and does not require
optional runtime dependencies of the parent package.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pixelle_video.config import CatFactoryGpuConfig

__all__ = [
    "GpuMonitor",
    "MonitorError",
    "NvidiaSmiSampler",
    "ProcessVram",
    "RamGateBlocked",
    "RamSample",
    "RamSampler",
    "SamplerUnavailable",
    "VramReleaseTimeout",
    "VramSample",
]

_NVIDIA_SMI_TIMEOUT_S = 30.0
_BYTES_PER_MIB = 1024 * 1024
_KIB_PER_MIB = 1024
_GPU_QUERY_ARGS = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
]
_PROCESS_QUERY_ARGS = [
    "nvidia-smi",
    "--query-compute-apps=pid,used_memory",
    "--format=csv,noheader,nounits",
]


class MonitorError(Exception):
    """Base error for GPU and RAM monitoring failures."""


class SamplerUnavailable(MonitorError):
    """A sampler could not produce a trustworthy measurement."""


class VramReleaseTimeout(MonitorError):
    """VRAM did not fall back to the recorded baseline before the deadline."""


class RamGateBlocked(MonitorError):
    """System RAM is below the configured floor for video generation."""


@dataclass(frozen=True)
class ProcessVram:
    """VRAM held by one compute process, when ``nvidia-smi`` exposes it."""

    pid: int
    used_mb: int
    name: str | None = None


@dataclass(frozen=True)
class VramSample:
    """A measured snapshot of one GPU."""

    gpu_name: str
    total_mb: int
    used_mb: int
    free_mb: int
    processes: tuple[ProcessVram, ...] = ()


@dataclass(frozen=True)
class RamSample:
    """A measured snapshot of system RAM in mebibytes."""

    free_mb: int
    total_mb: int


def _resolve_gpu_config() -> "CatFactoryGpuConfig":
    """Resolve the opt-in ``cat_factory.gpu`` config section lazily."""
    from pixelle_video.config import config_manager

    return config_manager.config.cat_factory.gpu


def _parse_mib(value: str) -> int:
    """Parse an ``nvidia-smi`` memory field that may carry a ``MiB`` suffix."""
    token = value.strip()
    if token.lower().endswith("mib"):
        token = token[:-3].strip()
    if not token:
        raise ValueError("empty memory value")
    return int(float(token))


def _parse_first_gpu_line(line: str) -> tuple[str, int, int, int]:
    """Parse ``name, total, used, free`` from the first CSV row."""
    fields = [field.strip() for field in line.split(",")]
    if len(fields) < 4:
        raise ValueError(f"expected 4 fields, received {len(fields)}")
    gpu_name = fields[0]
    total_mb = _parse_mib(fields[1])
    used_mb = _parse_mib(fields[2])
    free_mb = _parse_mib(fields[3])
    if not gpu_name:
        raise ValueError("empty GPU name")
    if total_mb <= 0 or used_mb < 0 or free_mb < 0:
        raise ValueError("memory values are outside the measurable range")
    return gpu_name, total_mb, used_mb, free_mb


def _parse_process_lines(output: str) -> tuple[ProcessVram, ...]:
    """Parse ``pid, used_memory`` rows, skipping malformed entries."""
    processes: list[ProcessVram] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2 or not fields[0]:
            continue
        try:
            pid = int(fields[0])
            used_mb = _parse_mib(fields[1])
        except ValueError:
            continue
        processes.append(ProcessVram(pid=pid, used_mb=used_mb))
    return tuple(processes)


def _default_nvidia_runner(args: list[str]) -> str:
    """Run ``nvidia-smi`` as an argument array and return stdout."""
    try:
        completed = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SamplerUnavailable(
            f"could not execute nvidia-smi ({args[0]}): {exc}"
        ) from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip() or "no stderr output"
        raise SamplerUnavailable(
            f"nvidia-smi exited with code {completed.returncode}: {stderr[:300]}"
        )
    return completed.stdout or ""


class NvidiaSmiSampler:
    """Measure GPU memory through ``nvidia-smi`` on Windows and POSIX."""

    def __init__(self, runner: Callable[[list[str]], str] | None = None) -> None:
        """Create a sampler, optionally injecting the command runner."""
        self._runner = runner if runner is not None else _default_nvidia_runner

    def sample(self) -> VramSample:
        """Return a measured VRAM sample for the first GPU."""
        try:
            output = self._runner(list(_GPU_QUERY_ARGS))
        except SamplerUnavailable:
            raise
        except Exception as exc:
            raise SamplerUnavailable(f"nvidia-smi GPU query failed: {exc}") from exc

        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            raise SamplerUnavailable("nvidia-smi GPU query returned no rows")
        try:
            gpu_name, total_mb, used_mb, free_mb = _parse_first_gpu_line(lines[0])
        except (IndexError, ValueError) as exc:
            raise SamplerUnavailable(
                f"nvidia-smi GPU row is unparsable: {exc}"
            ) from exc

        return VramSample(
            gpu_name=gpu_name,
            total_mb=total_mb,
            used_mb=used_mb,
            free_mb=free_mb,
            processes=self._sample_processes(),
        )

    def _sample_processes(self) -> tuple[ProcessVram, ...]:
        """Return per-process VRAM, degrading to an empty tuple on failure."""
        try:
            output = self._runner(list(_PROCESS_QUERY_ARGS))
        except Exception:
            return ()
        return _parse_process_lines(output)


class _MemoryStatusEx(ctypes.Structure):
    """Mirror of the Win32 ``MEMORYSTATUSEX`` structure."""

    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _read_windows_memory() -> tuple[int, int]:
    """Return ``(free_mb, total_mb)`` through ``GlobalMemoryStatusEx``."""
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise SamplerUnavailable("Windows memory API is unavailable")
    try:
        kernel32 = win_dll("kernel32", use_last_error=True)
    except OSError as exc:
        raise SamplerUnavailable(f"could not load kernel32: {exc}") from exc
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise SamplerUnavailable(
            "GlobalMemoryStatusEx failed with winerror "
            f"{ctypes.get_last_error()}"
        )
    return (
        int(status.ullAvailPhys // _BYTES_PER_MIB),
        int(status.ullTotalPhys // _BYTES_PER_MIB),
    )


def _parse_kib(value: str) -> int:
    """Parse a ``/proc/meminfo`` value expressed in kibibytes."""
    return int(value.split()[0])


def _read_proc_meminfo() -> tuple[int, int]:
    """Return ``(free_mb, total_mb)`` from ``/proc/meminfo``."""
    try:
        content = Path("/proc/meminfo").read_text(encoding="ascii", errors="replace")
    except OSError as exc:
        raise SamplerUnavailable(f"could not read /proc/meminfo: {exc}") from exc

    entries: dict[str, str] = {}
    for line in content.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        entries[key.strip()] = value.strip()

    try:
        available_kb = _parse_kib(entries["MemAvailable"])
        total_kb = _parse_kib(entries["MemTotal"])
    except KeyError as exc:
        raise SamplerUnavailable(
            f"/proc/meminfo is missing the field {exc}"
        ) from exc
    except (IndexError, ValueError) as exc:
        raise SamplerUnavailable(
            f"/proc/meminfo value is unparsable: {exc}"
        ) from exc

    if total_kb <= 0:
        raise SamplerUnavailable(f"/proc/meminfo reports total RAM {total_kb} kB")
    return available_kb // _KIB_PER_MIB, total_kb // _KIB_PER_MIB


def _default_ram_reader() -> tuple[int, int]:
    """Read system memory using the platform-native API."""
    if sys.platform.startswith("win"):
        return _read_windows_memory()
    return _read_proc_meminfo()


class RamSampler:
    """Measure system RAM through the platform API or an injected reader."""

    def __init__(self, reader: Callable[[], tuple[int, int]] | None = None) -> None:
        """Create a sampler whose reader returns ``(free_mb, total_mb)``."""
        self._reader = reader if reader is not None else _default_ram_reader

    def sample(self) -> RamSample:
        """Return a validated system RAM sample in mebibytes."""
        try:
            free_mb, total_mb = self._reader()
        except SamplerUnavailable:
            raise
        except Exception as exc:
            raise SamplerUnavailable(f"system RAM sample failed: {exc}") from exc

        free_value = int(free_mb)
        total_value = int(total_mb)
        if total_value <= 0:
            raise SamplerUnavailable(
                f"system RAM total is not usable: {total_value} MiB"
            )
        if free_value < 0 or free_value > total_value:
            raise SamplerUnavailable(
                f"system RAM free value {free_value} MiB is outside "
                f"0..{total_value} MiB"
            )
        return RamSample(free_mb=free_value, total_mb=total_value)


class GpuMonitor:
    """Coordinate VRAM measurements and RAM gating for local generation."""

    def __init__(
        self,
        vram_sampler: NvidiaSmiSampler | None = None,
        ram_sampler: RamSampler | None = None,
        config: "CatFactoryGpuConfig | None" = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Create a monitor, resolving configuration only when first needed."""
        self._config = config
        self._vram_sampler = (
            vram_sampler if vram_sampler is not None else NvidiaSmiSampler()
        )
        self._ram_sampler = ram_sampler if ram_sampler is not None else RamSampler()
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic

    def _get_config(self) -> "CatFactoryGpuConfig":
        """Return the configured GPU settings, resolving them lazily."""
        if self._config is None:
            self._config = _resolve_gpu_config()
        return self._config

    def sample_vram(self) -> VramSample:
        """Delegate to the VRAM sampler."""
        return self._vram_sampler.sample()

    def sample_ram(self) -> RamSample:
        """Delegate to the RAM sampler."""
        return self._ram_sampler.sample()

    def vram_ceiling_mb(self) -> int:
        """Return measured total VRAM minus the configured safety margin."""
        config = self._get_config()
        sample = self.sample_vram()
        margin_mb = int(config.vram_margin_mb)
        if margin_mb < 0:
            raise MonitorError(f"vram_margin_mb must not be negative, got {margin_mb}")
        ceiling_mb = sample.total_mb - margin_mb
        if ceiling_mb <= 0:
            raise MonitorError(
                f"vram_margin_mb={margin_mb} leaves no usable VRAM of "
                f"{sample.total_mb} MiB on {sample.gpu_name!r}"
            )
        return ceiling_mb

    def wait_for_vram_release(
        self,
        *,
        baseline_used_mb: int,
        timeout_s: float | None = None,
    ) -> VramSample:
        """Poll until measured VRAM returns to the recorded baseline.

        The previous allocation is considered released only when a fresh
        measurement reports ``used_mb <= baseline_used_mb``.  A ComfyUI
        ``/free`` response is never consulted here.
        """
        config = self._get_config()
        interval_s = float(config.poll_interval_s)
        if interval_s <= 0:
            raise MonitorError(
                f"poll_interval_s must be positive, got {interval_s}"
            )
        if baseline_used_mb < 0:
            raise MonitorError(
                f"baseline_used_mb must not be negative, got {baseline_used_mb}"
            )
        if timeout_s is None:
            timeout_s = float(config.memory_release_timeout_s)
        if timeout_s < 0:
            raise MonitorError(f"timeout_s must not be negative, got {timeout_s}")

        deadline = self._clock() + timeout_s
        last_sample = self.sample_vram()
        while last_sample.used_mb > baseline_used_mb:
            if self._clock() >= deadline:
                raise VramReleaseTimeout(
                    f"GPU {last_sample.gpu_name!r} still holds "
                    f"{last_sample.used_mb} MiB after {timeout_s:.1f}s; "
                    f"baseline is {baseline_used_mb} MiB "
                    f"(total {last_sample.total_mb} MiB, "
                    f"free {last_sample.free_mb} MiB)"
                )
            self._sleep(interval_s)
            last_sample = self.sample_vram()
        return last_sample

    def check_video_ram_gate(self) -> RamSample:
        """Fail closed unless measured free RAM meets the configured floor."""
        config = self._get_config()
        minimum_mb = config.offload_min_ram_mb
        if minimum_mb is None:
            raise RamGateBlocked(
                "offload_min_ram_mb is not measured or configured; "
                "refusing to start video generation without a measured floor"
            )
        sample = self.sample_ram()
        margin_mb = int(config.ram_margin_mb)
        required_mb = int(minimum_mb) + margin_mb
        if sample.free_mb < required_mb:
            raise RamGateBlocked(
                f"free RAM {sample.free_mb} MiB is below the required "
                f"{required_mb} MiB (offload minimum {int(minimum_mb)} MiB + "
                f"margin {margin_mb} MiB; total {sample.total_mb} MiB)"
            )
        return sample
