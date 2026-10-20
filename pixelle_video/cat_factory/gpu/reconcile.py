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

"""Reconcile the Cat Factory GPU lease with the stock media generation path.

The Cat Factory owns a single GPU through an exclusive lease. The stock
``MediaService`` / ComfyKit path may be reused **only** while the CPU render
runner owns that lease; this module exposes the guard helpers that make that
rule explicit, plus a read-only probe for operator-facing warnings.

Everything first-party is imported lazily so this module stays importable even
when the ``pixelle_video`` package parent is unavailable. Nothing here forks or
modifies ``comfykit``, and the stock ComfyUI ``default_workflow`` semantics are
untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pixelle_video.cat_factory.gpu.lease import GpuLease
    from pixelle_video.config.schema import CatFactoryGpuConfig


__all__ = [
    "CONCURRENT_USE_WARNING",
    "GpuBusyError",
    "ReconcileError",
    "exclusive_lease_held",
    "exclusive_lease_warning",
    "media_service_for_lease",
    "require_owned_lease",
]


CONCURRENT_USE_WARNING = (
    "A Cat Factory GPU render currently holds the exclusive GPU lease. "
    "Running another ComfyUI generation at the same time is unsafe and may "
    "fail or corrupt results."
)


class ReconcileError(Exception):
    """Base class for GPU/media reconciliation failures."""


class GpuBusyError(ReconcileError):
    """Raised when stock media reuse is attempted without an owned GPU lease."""


def _resolve_gpu_config() -> "CatFactoryGpuConfig":
    """Return the opt-in Cat Factory GPU config (lazy import)."""
    from pixelle_video.config import config_manager

    return config_manager.config.cat_factory.gpu


def _lock_root(config: "CatFactoryGpuConfig") -> Path:
    """Return the lease lock directory for ``config``."""
    base = (
        Path(config.state_dir)
        if config.state_dir
        else Path("state") / "cat_factory" / "gpu"
    )
    return base / "lease"


def exclusive_lease_held(
    *,
    gpu_id: int = 0,
    lock_root: str | Path | None = None,
    stale_heartbeat_seconds: int | None = None,
) -> bool:
    """Return whether an exclusive GPU lease record exists (read-only probe).

    The probe never acquires, releases or recovers a lease; it only inspects the
    on-disk record. Any failure is reported as ``False`` so callers can treat
    this helper as non-raising.
    """
    try:
        from pixelle_video.cat_factory.gpu.lease import GpuLeaseManager

        config = _resolve_gpu_config()
        seconds = (
            int(config.stale_heartbeat_seconds)
            if stale_heartbeat_seconds is None
            else int(stale_heartbeat_seconds)
        )
        manager = GpuLeaseManager(
            gpu_id,
            Path(lock_root) if lock_root is not None else _lock_root(config),
            stale_heartbeat_seconds=seconds,
        )
        return manager.inspect() is not None
    except Exception:
        return False


def exclusive_lease_warning(*, gpu_id: int = 0) -> str | None:
    """Return the concurrent-use warning when a GPU lease is held, else ``None``.

    Non-raising: disabled feature flag, absent config, or any probe failure all
    yield ``None``.
    """
    try:
        from pixelle_video.config import config_manager

        config = config_manager.config
        if not getattr(config.cat_factory, "enabled", False):
            return None
        if exclusive_lease_held(gpu_id=gpu_id):
            return CONCURRENT_USE_WARNING
        return None
    except Exception:
        return None


def require_owned_lease(lease: object) -> None:
    """Refuse stock media reuse unless ``lease`` is currently held by us."""
    held_check = getattr(lease, "held", None)
    if not callable(held_check) or not held_check():
        raise GpuBusyError(
            "the CPU render runner does not own the exclusive GPU lease; "
            "reusing the stock media path is refused"
        )


def media_service_for_lease(
    lease: "GpuLease",
    *,
    service_factory: Callable[[], object] | None = None,
) -> object:
    """Return a stock ``MediaService`` bound to the GPU lease this runner owns.

    Reuse is gated on ownership first: the stock
    ``MediaService.__call__`` / ``comfy_base_service`` path is reused ONLY while
    the CPU runner owns the lease. Nothing in ``comfykit`` is forked or
    modified, and stock ComfyUI ``default_workflow`` semantics are untouched.
    """
    require_owned_lease(lease)
    if service_factory is None:
        from pixelle_video.config import config_manager
        from pixelle_video.services.media import MediaService

        service_factory = lambda: MediaService(config_manager.config.model_dump())
    return service_factory()
